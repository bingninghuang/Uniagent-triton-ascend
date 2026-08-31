import torch
import torch_npu
import triton
import triton.language as tl


_LOG2E = 1.4426950408889634


# ---------------------------------------------------------------------------
# Weight preprocessing
# ---------------------------------------------------------------------------

@triton.jit
def _fold_w2_kernel(w2_ptr, b2_ptr, w2f_ptr, b2f_ptr,
                    C, mid, K2,
                    HAS_B: tl.constexpr,
                    BC: tl.constexpr, BM: tl.constexpr,
                    NUMP: tl.constexpr):
    # w2   : (K2*C, mid)   -> w2f: (C, mid), w2f[c, m] = (1/K2) * sum_u w2[c*K2+u, m]
    # b2   : (K2*C,)       -> b2f: (C,)   , b2f[c]     = (1/K2) * sum_u b2[c*K2+u]
    ncb = tl.cdiv(C, BC)
    nmb = tl.cdiv(mid, BM)
    total = ncb * nmb
    inv = 1.0 / K2
    c_arr = tl.arange(0, BC)
    m_arr = tl.arange(0, BM)
    for item in range(tl.program_id(0), total, NUMP):
        cb = item // nmb
        mb = item % nmb
        c_offs = cb * BC + c_arr
        cm = c_offs < C
        m_offs = mb * BM + m_arr
        mm = m_offs < mid
        wm2 = cm[:, None] & mm[None, :]
        wbase = (c_offs * K2)[:, None] * mid + m_offs[None, :]
        acc = tl.zeros((BC, BM), dtype=tl.float32)
        for u in range(0, K2):
            w = tl.load(w2_ptr + wbase + u * mid, mask=wm2, other=0.0)
            acc += w
        acc = acc * inv
        tl.store(w2f_ptr + c_offs[:, None] * mid + m_offs[None, :],
                 acc.to(w2f_ptr.dtype.element_ty), mask=wm2)
        if HAS_B == 1:
            if mb == 0:
                accb = tl.zeros((BC,), dtype=tl.float32)
                for u2 in range(0, K2):
                    bb = tl.load(b2_ptr + c_offs * K2 + u2, mask=cm, other=0.0)
                    accb += bb.to(tl.float32)
                tl.store(b2f_ptr + c_offs, accb * inv, mask=cm)


@triton.jit
def _tap_w_kernel(wk_ptr, wkt_ptr,
                  C, ICg, K2,
                  BC: tl.constexpr, BG: tl.constexpr,
                  NUMP: tl.constexpr):
    # wk  : (C, ICg, K2) row-major, K2 = K*K, u in [0, K2)
    # wkt : (C, K2*ICg), wkt[c, u*ICg + gi] = wk[c, gi*K2 + u]
    ncb = tl.cdiv(C, BC)
    ngb = tl.cdiv(ICg, BG)
    total = ncb * ngb
    c_arr = tl.arange(0, BC)
    g_arr = tl.arange(0, BG)
    for item in range(tl.program_id(0), total, NUMP):
        cb = item % ncb
        gb = item // ncb
        c_offs = cb * BC + c_arr
        cm = c_offs < C
        g_offs = gb * BG + g_arr
        gm = g_offs < ICg
        mm = cm[:, None] & gm[None, :]
        src = c_offs[:, None] * (ICg * K2) + g_offs[None, :] * K2
        dst = c_offs[:, None] * (K2 * ICg) + g_offs[None, :]
        for u in range(0, K2):
            a = tl.load(wk_ptr + src + u, mask=mm, other=0.0)
            tl.store(wkt_ptr + dst + u * ICg, a.to(wkt_ptr.dtype.element_ty), mask=mm)


# ---------------------------------------------------------------------------
# Grouped k x k convolution (implicit GEMM) with bias + relu
# ---------------------------------------------------------------------------

@triton.jit
def _k1_conv_kernel(x_ptr, wkt_ptr, kb_ptr, k1o_ptr,
                    B, C, G, ICg, H, W, N,
                    K,
                    BOC: tl.constexpr, BIC: tl.constexpr,
                    BII: tl.constexpr, BJJ: tl.constexpr,
                    HAS_B: tl.constexpr,
                    NUMP: tl.constexpr):
    PAD = K // 2
    K2 = K * K
    SP = BII * BJJ
    n_oc = tl.cdiv(ICg, BOC)
    n_j = tl.cdiv(W, BJJ)
    n_i = tl.cdiv(H, BII)
    per_b = G * n_i * n_j * n_oc
    total = B * per_b
    ii0 = tl.arange(0, BII)
    jj0 = tl.arange(0, BJJ)
    oc0 = tl.arange(0, BOC)
    ic0 = tl.arange(0, BIC)
    for item in range(tl.program_id(0), total, NUMP):
        b = item // per_b
        r = item % per_b
        g = r // (n_i * n_j * n_oc)
        r2 = r % (n_i * n_j * n_oc)
        ti = r2 // (n_j * n_oc)
        r3 = r2 % (n_j * n_oc)
        tj = r3 // n_oc
        to = r3 % n_oc
        oc_offs = g * ICg + to * BOC + oc0
        om = oc_offs < C
        i0 = ti * BII
        j0 = tj * BJJ
        ii = i0 + ii0
        jj = j0 + jj0
        abase = oc_offs[:, None] * (K2 * ICg)
        acc = tl.zeros((BOC, SP), dtype=tl.float32)
        for u in range(0, K2):
            th = u // K
            tw = u % K
            ih = ii + (th - PAD)
            jw = jj + (tw - PAD)
            vm = ((ih >= 0) & (ih < H))[:, None] & ((jw >= 0) & (jw < W))[None, :]
            xb = (b * C + g * ICg) * N
            for icb in range(0, ICg, BIC):
                kmask = (icb + ic0) < ICg
                A = tl.load(wkt_ptr + abase + u * ICg + (icb + ic0)[None, :],
                            mask=om[:, None] & kmask[None, :], other=0.0)
                B3 = tl.load(x_ptr + (xb + icb + ic0)[:, None, None] * N
                             + ih[None, :, None] * W + jw[None, None, :],
                             mask=kmask[:, None, None] & vm[None, :, :], other=0.0)
                acc = tl.dot(A, tl.reshape(B3, (BIC, SP)), acc)
        if HAS_B == 1:
            bv = tl.load(kb_ptr + oc_offs, mask=om, other=0.0).to(tl.float32)
            acc += bv[:, None]
        acc = tl.maximum(acc, 0.0)
        acc3 = tl.reshape(acc, (BOC, BII, BJJ))
        omask = om[:, None, None] & ((ii < H)[None, :, None]) & ((jj < W)[None, None, :])
        tl.store(k1o_ptr + (b * C + oc_offs)[:, None, None] * N
                 + ii[None, :, None] * W + jj[None, None, :],
                 acc3.to(k1o_ptr.dtype.element_ty), mask=omask)


# ---------------------------------------------------------------------------
# Pointwise (1x1) conv = GEMM: out (M, N) = W (M, K) @ act (K, N)
# ---------------------------------------------------------------------------

@triton.jit
def _pw_gemm_kernel(x_ptr, w_ptr, bias_ptr, out_ptr,
                    Bch, Cout, Cin, Nsp,
                    HAS_B: tl.constexpr,
                    BOC: tl.constexpr, BIC: tl.constexpr, BSN: tl.constexpr,
                    NUMP: tl.constexpr):
    # x_ptr  : (Bch, Cin, Nsp) activations
    # w_ptr  : (Cout, Cin) row-major weight
    # out_ptr: (Bch, Cout, Nsp)
    n_b = tl.cdiv(Nsp, BSN)
    n_m = tl.cdiv(Cout, BOC)
    per_b = n_m * n_b
    total = Bch * per_b
    xb = Cin * Nsp
    ob_ = Cout * Nsp
    oc_offs = tl.arange(0, BOC)
    k_offs = tl.arange(0, BIC)
    n_offs = tl.arange(0, BSN)
    for item in range(tl.program_id(0), total, NUMP):
        bi = item // per_b
        r = item % per_b
        oblock = r // n_b
        nb = r % n_b
        x_base = x_ptr + bi * xb
        o_base = out_ptr + bi * ob_
        oc_abs = oblock * BOC + oc_offs
        om = oc_abs < Cout
        n_abs = nb * BSN + n_offs
        nm = n_abs < Nsp
        acc = tl.zeros((BOC, BSN), dtype=tl.float32)
        for k in range(0, Cin, BIC):
            kmask = (k + k_offs) < Cin
            A = tl.load(w_ptr + oc_abs[:, None] * Cin + (k + k_offs)[None, :],
                        mask=om[:, None] & kmask[None, :], other=0.0)
            Bt = tl.load(x_base + (k + k_offs)[:, None] * Nsp + n_abs[None, :],
                         mask=kmask[:, None] & nm[None, :], other=0.0)
            acc = tl.dot(A, Bt, acc)
        if HAS_B == 1:
            bv = tl.load(bias_ptr + oc_abs, mask=om, other=0.0).to(tl.float32)
            acc += bv[:, None]
        tl.store(o_base + oc_abs[:, None] * Nsp + n_abs[None, :],
                 acc.to(out_ptr.dtype.element_ty),
                 mask=om[:, None] & nm[None, :])


# att1 = relu( att_w1 @ cat([k1, x]) + b1 ) ; K = 2*C, first C rows from k1, rest from x
@triton.jit
def _att1_gemm_kernel(k1_ptr, x_ptr, w_ptr, bias_ptr, out_ptr,
                      Bch, C, Cout, Nsp,
                      HAS_B: tl.constexpr,
                      BOC: tl.constexpr, BIC: tl.constexpr, BSN: tl.constexpr,
                      NUMP: tl.constexpr):
    n_b = tl.cdiv(Nsp, BSN)
    n_m = tl.cdiv(Cout, BOC)
    per_b = n_m * n_b
    total = Bch * per_b
    cs = C * Nsp
    os_ = Cout * Nsp
    oc_offs = tl.arange(0, BOC)
    k_offs = tl.arange(0, BIC)
    n_offs = tl.arange(0, BSN)
    for item in range(tl.program_id(0), total, NUMP):
        bi = item // per_b
        r = item % per_b
        oblock = r // n_b
        nb = r % n_b
        k1_base = k1_ptr + bi * cs
        x_base = x_ptr + bi * cs
        o_base = out_ptr + bi * os_
        oc_abs = oblock * BOC + oc_offs
        om = oc_abs < Cout
        n_abs = nb * BSN + n_offs
        nm = n_abs < Nsp
        acc = tl.zeros((BOC, BSN), dtype=tl.float32)
        for k in range(0, C, BIC):
            kmask = (k + k_offs) < C
            A = tl.load(w_ptr + oc_abs[:, None] * (2 * C) + (k + k_offs)[None, :],
                        mask=om[:, None] & kmask[None, :], other=0.0)
            Bt = tl.load(k1_base + (k + k_offs)[:, None] * Nsp + n_abs[None, :],
                         mask=kmask[:, None] & nm[None, :], other=0.0)
            acc = tl.dot(A, Bt, acc)
        for k in range(C, 2 * C, BIC):
            kk = k - C
            kmask = (kk + k_offs) < C
            A = tl.load(w_ptr + oc_abs[:, None] * (2 * C) + (k + k_offs)[None, :],
                        mask=om[:, None] & kmask[None, :], other=0.0)
            Bt = tl.load(x_base + (kk + k_offs)[:, None] * Nsp + n_abs[None, :],
                         mask=kmask[:, None] & nm[None, :], other=0.0)
            acc = tl.dot(A, Bt, acc)
        if HAS_B == 1:
            bv = tl.load(bias_ptr + oc_abs, mask=om, other=0.0).to(tl.float32)
            acc += bv[:, None]
        acc = tl.maximum(acc, 0.0)
        tl.store(o_base + oc_abs[:, None] * Nsp + n_abs[None, :],
                 acc.to(out_ptr.dtype.element_ty),
                 mask=om[:, None] & nm[None, :])


# ---------------------------------------------------------------------------
# softmax over n, per (b, c) row:
#   kernel 1: mrow = max_n(att), lrow = sum_n exp2((att - mrow) * L2E)
#   kernel 2: out = k1 + exp2((att - mrow) * L2E) / lrow * v
# ---------------------------------------------------------------------------

@triton.jit
def _row_ms_kernel(att_ptr, m_ptr, l_ptr,
                   Bch, C, Nsp,
                   BOR: tl.constexpr, BSN: tl.constexpr,
                   NUMP: tl.constexpr):
    L2E: tl.constexpr = 1.4426950408889634
    n_b = tl.cdiv(Nsp, BSN)
    n_c = tl.cdiv(C, BOR)
    total = Bch * n_c
    bstride = C * Nsp
    c0 = tl.arange(0, BOR)
    n0 = tl.arange(0, BSN)
    neginf = -float("inf")
    for item in range(tl.program_id(0), total, NUMP):
        bi = item // n_c
        cb = item % n_c
        base = bi * bstride
        c_abs = cb * BOR + c0
        cm = c_abs < C
        mrow = tl.full((BOR,), neginf, dtype=tl.float32)
        lrow = tl.zeros((BOR,), dtype=tl.float32)
        for nb0 in range(0, n_b):
            n_cur = nb0 * BSN + n0
            nm_cur = n_cur < Nsp
            a = tl.load(att_ptr + base + c_abs[:, None] * Nsp + n_cur[None, :],
                        mask=cm[:, None] & nm_cur[None, :], other=neginf)
            amax = tl.max(a, axis=1)
            mnew = tl.maximum(mrow, amax)
            p = tl.exp2((a - mnew[:, None]) * L2E)
            s = tl.sum(p, axis=1)
            lrow = lrow * tl.exp2((mrow - mnew) * L2E) + s
            mrow = mnew
        tl.store(m_ptr + bi * C + c_abs, mrow, mask=cm)
        tl.store(l_ptr + bi * C + c_abs, lrow, mask=cm)


@triton.jit
def _out_combine_kernel(att_ptr, v_ptr, k1_ptr, m_ptr, l_ptr, out_ptr,
                        Bch, C, Nsp,
                        BOR: tl.constexpr, BSN: tl.constexpr,
                        NUMP: tl.constexpr):
    L2E: tl.constexpr = 1.4426950408889634
    n_b = tl.cdiv(Nsp, BSN)
    n_c = tl.cdiv(C, BOR)
    total = Bch * n_c
    bstride = C * Nsp
    c0 = tl.arange(0, BOR)
    n0 = tl.arange(0, BSN)
    for item in range(tl.program_id(0), total, NUMP):
        bi = item // n_c
        cb = item % n_c
        base = bi * bstride
        c_abs = cb * BOR + c0
        cm = c_abs < C
        mrow = tl.load(m_ptr + bi * C + c_abs, mask=cm, other=0.0)
        lrow = tl.load(l_ptr + bi * C + c_abs, mask=cm, other=0.0)
        linv = 1.0 / tl.where(lrow == 0.0, 1.0, lrow)
        for nb1 in range(0, n_b):
            n_cur = nb1 * BSN + n0
            nm_cur = n_cur < Nsp
            m2 = cm[:, None] & nm_cur[None, :]
            a = tl.load(att_ptr + base + c_abs[:, None] * Nsp + n_cur[None, :],
                        mask=m2, other=0.0)
            vv = tl.load(v_ptr + base + c_abs[:, None] * Nsp + n_cur[None, :],
                         mask=m2, other=0.0)
            kk = tl.load(k1_ptr + base + c_abs[:, None] * Nsp + n_cur[None, :],
                         mask=m2, other=0.0)
            p = tl.exp2((a - mrow[:, None]) * L2E) * linv[:, None]
            o = kk + p * vv
            tl.store(out_ptr + base + c_abs[:, None] * Nsp + n_cur[None, :],
                     o.to(out_ptr.dtype.element_ty), mask=m2)


# ---------------------------------------------------------------------------
# Host code
# ---------------------------------------------------------------------------

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.CUBE_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except Exception:
            self.CUBE_CORE_NUM = 20
            self.VEC_CORE_NUM = 40

    def forward(self, x, kernel_size,
                key_embed_w, key_embed_b,
                value_embed_w, value_embed_b,
                att_w1, att_b1, att_w2, att_b2):
        b, c, h, w = x.shape
        n = h * w
        if c % 4 == 0:
            groups = 4
        elif c % 2 == 0:
            groups = 2
        else:
            groups = 1
        icg = c // groups
        mid = c // 4
        if mid < 1:
            mid = 1
        k = int(kernel_size)
        k2 = k * k
        dtype = x.dtype
        dev = x.device
        is_f32 = dtype == torch.float32

        if not x.is_contiguous():
            x = x.contiguous()

        zero = torch.zeros(1, dtype=dtype, device=dev)

        # ---- weight prep buffers ----
        w2f = torch.empty((c, mid), dtype=dtype, device=dev)
        b2f = torch.zeros((c,), dtype=torch.float32, device=dev)
        wkt = torch.empty((c, k2 * icg), dtype=dtype, device=dev)

        has_w2b = 1 if att_b2 is not None else 0
        BC, BM = 32, 64
        tf = triton.cdiv(c, BC) * triton.cdiv(mid, BM)
        if tf > self.CUBE_CORE_NUM:
            tf = self.CUBE_CORE_NUM
        grid_f = (tf,)
        _fold_w2_kernel[grid_f](att_w2, att_b2 if att_b2 is not None else zero,
                                w2f, b2f, c, mid, k2, has_w2b, BC, BM,
                                grid_f[0])

        BC2, BG2 = 64, 64
        tt = triton.cdiv(c, BC2) * triton.cdiv(icg, BG2)
        if tt > self.CUBE_CORE_NUM:
            tt = self.CUBE_CORE_NUM
        grid_t = (tt,)
        _tap_w_kernel[grid_t](key_embed_w, wkt, c, icg, k2, BC2, BG2, grid_t[0])

        # ---- intermediate buffers (per batch channel-first: (b, c, n)) ----
        k1o = torch.empty((b, c, n), dtype=dtype, device=dev)
        vo = torch.empty((b, c, n), dtype=dtype, device=dev)
        a1o = torch.empty((b, mid, n), dtype=dtype, device=dev)
        att_o = torch.empty((b, c, n), dtype=dtype, device=dev)
        out = torch.empty((b, c, h, w), dtype=dtype, device=dev)

        has_kb = 1 if key_embed_b is not None else 0
        has_vb = 1 if value_embed_b is not None else 0
        has_ab1 = 1 if att_b1 is not None else 0

        # ---- k1: grouped k x k conv + relu ----
        if is_f32:
            BOC, BIC, BII, BJJ = 32, 32, 8, 8
        else:
            BOC, BIC, BII, BJJ = 64, 32, 8, 8
        n_oc = triton.cdiv(icg, BOC)
        n_j = triton.cdiv(w, BJJ)
        n_i = triton.cdiv(h, BII)
        total_k1 = b * groups * n_i * n_j * n_oc
        if total_k1 > self.CUBE_CORE_NUM:
            total_k1 = self.CUBE_CORE_NUM
        grid_k1 = (total_k1,)
        _k1_conv_kernel[grid_k1](x, wkt, key_embed_b if key_embed_b is not None else zero,
                                 k1o, b, c, groups, icg, h, w, n,
                                 k, BOC, BIC, BII, BJJ, has_kb, grid_k1[0])

        # ---- v = pw conv(x) ----
        if is_f32:
            BOCV, BICV, BSNV = 32, 32, 64
        else:
            BOCV, BICV, BSNV = 32, 32, 64
        n_m = triton.cdiv(c, BOCV)
        n_bsp = triton.cdiv(n, BSNV)
        tv = b * n_m * n_bsp
        if tv > self.CUBE_CORE_NUM:
            tv = self.CUBE_CORE_NUM
        grid_v = (tv,)
        _pw_gemm_kernel[grid_v](x, value_embed_w,
                                value_embed_b if value_embed_b is not None else zero,
                                vo, b, c, c, n, has_vb,
                                BOCV, BICV, BSNV, grid_v[0])

        # ---- att1 = relu(pw 2C -> mid) ----
        if is_f32:
            BOCa, BICa, BSNa = 32, 32, 64
        else:
            BOCa, BICa, BSNa = 32, 32, 64
        total_a1 = b * triton.cdiv(mid, BOCa) * n_bsp
        if total_a1 > self.CUBE_CORE_NUM:
            total_a1 = self.CUBE_CORE_NUM
        grid_a1 = (total_a1,)
        _att1_gemm_kernel[grid_a1](k1o, x, att_w1,
                                   att_b1 if att_b1 is not None else zero,
                                   a1o, b, c, mid, n, has_ab1,
                                   BOCa, BICa, BSNa, grid_a1[0])

        # ---- att = pw(mid -> C) with folded weight + bias ----
        if is_f32:
            BOCt, BICt, BSNa = 32, 32, 64
        else:
            BOCt, BICt, BSNa = 32, 32, 64
        total_att = b * triton.cdiv(c, BOCt) * n_bsp
        if total_att > self.CUBE_CORE_NUM:
            total_att = self.CUBE_CORE_NUM
        grid_att = (total_att,)
        _pw_gemm_kernel[grid_att](a1o, w2f, b2f, att_o, b, c, mid, n, 1,
                                  BOCt, BICt, BSNa, grid_att[0])

        # ---- softmax over n (per (b,c)) then out = k1 + softmax* v ----
        # pass 1: row max + row sum-of-exp ;  pass 2: combine
        BSR = 16
        if n <= 16:
            BSNs = 16
        elif is_f32:
            BSNs = 32 if n <= 32 else 64
        else:
            BSNs = 32 if n <= 32 else (64 if n <= 64 else 128)
        if BSNs > n:
            BSNs = 16
        total_s = b * triton.cdiv(c, BSR)
        if total_s > self.VEC_CORE_NUM:
            total_s = self.VEC_CORE_NUM
        grid_s = (total_s,)
        mrow_buf = torch.empty((b, c), dtype=torch.float32, device=dev)
        lrow_buf = torch.empty((b, c), dtype=torch.float32, device=dev)
        _row_ms_kernel[grid_s](att_o, mrow_buf, lrow_buf, b, c, n,
                               BSR, BSNs, grid_s[0])
        _out_combine_kernel[grid_s](att_o, vo, k1o, mrow_buf, lrow_buf, out,
                                    b, c, n, BSR, BSNs, grid_s[0])

        return out 
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _dot3f(a, b):
    ah = a.to(tl.float16)
    al = (a - ah.to(tl.float32)).to(tl.float16)
    bh = b.to(tl.float16)
    bl = (b - bh.to(tl.float32)).to(tl.float16)
    r = tl.dot(ah, bl)
    r = tl.dot(al, bh, r)
    r = tl.dot(ah, bh, r)
    return r


@triton.jit
def _dot3a(a, b, acc):
    ah = a.to(tl.float16)
    al = (a - ah.to(tl.float32)).to(tl.float16)
    bh = b.to(tl.float16)
    bl = (b - bh.to(tl.float32)).to(tl.float16)
    r = tl.dot(ah, bl, acc)
    r = tl.dot(al, bh, r)
    r = tl.dot(ah, bh, r)
    return r


@triton.jit
def _pool_kernel(q_ptr, k_ptr, ql_ptr, kl_ptr,
                 S, L,
                 D: tl.constexpr,
                 BD: tl.constexpr):
    pid = tl.program_id(0)
    l = pid % L
    bh = pid // L
    start = (l * S) // L
    end = ((l + 1) * S) // L
    d_offs = tl.arange(0, BD)
    dmask = d_offs < D
    base = bh * S * D
    acc_q = tl.zeros((BD,), dtype=tl.float32)
    acc_k = tl.zeros((BD,), dtype=tl.float32)
    for _ in range(0, end - start):
        idx = base + start * D + d_offs
        qv = tl.load(q_ptr + idx, mask=dmask, other=0.0).to(tl.float32)
        kv = tl.load(k_ptr + idx, mask=dmask, other=0.0).to(tl.float32)
        acc_q += qv
        acc_k += kv
        start += 1
    inv_cnt = 1.0 / (end - (l * S) // L)
    obase = bh * L * D + l * D
    tl.store(ql_ptr + obase + d_offs, acc_q * inv_cnt, mask=dmask)
    tl.store(kl_ptr + obase + d_offs, acc_k * inv_cnt, mask=dmask)


@triton.jit
def _k1_kernel(q_ptr, kl_ptr, k1_ptr,
               S, L, BH, num_pids,
               scale,
               D: tl.constexpr,
               BD: tl.constexpr,
               LN: tl.constexpr,
               BS: tl.constexpr):
    pid = tl.program_id(0)
    n_stiles = tl.cdiv(S, BS)
    n_tiles = BH * n_stiles
    d_offs = tl.arange(0, BD)
    dmask = d_offs < D
    l_offs = tl.arange(0, LN)
    lmask = l_offs < L
    for tile in range(pid, n_tiles, num_pids):
        bh = tile // n_stiles
        st = tile % n_stiles
        s0 = st * BS
        s_offs = s0 + tl.arange(0, BS)
        smask = s_offs < S
        q = tl.load(q_ptr + bh * S * D + s_offs[:, None] * D + d_offs[None, :],
                    mask=smask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        kl = tl.load(kl_ptr + bh * L * D + l_offs[:, None] * D + d_offs[None, :],
                     mask=lmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        scores = _dot3f(q, tl.trans(kl)) * scale
        scores = tl.where(lmask[None, :], scores, float('-inf'))
        rm = tl.max(scores, axis=1)
        e = tl.exp(scores - rm[:, None])
        e = tl.where(lmask[None, :], e, 0.0)
        rs = tl.sum(e, axis=1)
        k1 = e / rs[:, None]
        tl.store(k1_ptr + bh * S * L + s_offs[:, None] * L + l_offs[None, :],
                 k1, mask=smask[:, None] & lmask[None, :])


@triton.jit
def _k2inv_kernel(ql_ptr, kl_ptr, ainv_ptr,
                  L, BH, num_pids,
                  scale,
                  D: tl.constexpr,
                  BD: tl.constexpr,
                  LN: tl.constexpr):
    pid = tl.program_id(0)
    d_offs = tl.arange(0, BD)
    dmask = d_offs < D
    l_offs = tl.arange(0, LN)
    lmask = l_offs < L
    idmat = (l_offs[:, None] == l_offs[None, :]).to(tl.float32)
    seven = 7.0 * idmat
    fifteen = 15.0 * idmat
    thirteen = 13.0 * idmat
    for bh in range(pid, BH, num_pids):
        base = bh * L * D
        ql = tl.load(ql_ptr + base + l_offs[:, None] * D + d_offs[None, :],
                     mask=lmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        kl = tl.load(kl_ptr + base + l_offs[:, None] * D + d_offs[None, :],
                     mask=lmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        k2 = tl.where(lmask[None, :], _dot3f(ql, tl.trans(kl)) * scale,
                      float('-inf'))
        rm = tl.max(k2, axis=1)
        e = tl.exp(k2 - rm[:, None])
        e = tl.where(lmask[None, :], e, 0.0)
        rs = tl.sum(e, axis=1)
        k2s = e / rs[:, None]
        k2s = tl.where(lmask[:, None], k2s, 0.0)
        a2 = tl.abs(k2s)
        rsum = tl.sum(a2, axis=1)
        csum = tl.sum(a2, axis=0)
        denom = tl.maximum(tl.max(rsum) * tl.max(csum), 1e-6)
        inv = tl.trans(k2s) / denom
        for _ in range(6):
            prod = _dot3f(k2s, inv)
            inner = _dot3f(prod, seven - prod)
            mid = _dot3f(prod, fifteen - inner)
            inv = 0.25 * _dot3f(inv, thirteen - mid)
        tl.store(ainv_ptr + bh * LN * LN + l_offs[:, None] * LN + l_offs[None, :],
                 inv)


@triton.jit
def _k3v_kernel(ql_ptr, k_ptr, v_ptr, ainv_ptr, tmp2_ptr,
                S, L, BH, num_pids,
                scale,
                D: tl.constexpr,
                BD: tl.constexpr,
                LN: tl.constexpr,
                BN: tl.constexpr):
    pid = tl.program_id(0)
    d_offs = tl.arange(0, BD)
    dmask = d_offs < D
    l_offs = tl.arange(0, LN)
    lmask = l_offs < L
    for bh in range(pid, BH, num_pids):
        kbase = k_ptr + bh * S * D
        vbase = v_ptr + bh * S * D
        ql = tl.load(ql_ptr + bh * L * D + l_offs[:, None] * D + d_offs[None, :],
                     mask=lmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        row_max = tl.full((LN,), float('-inf'), dtype=tl.float32)
        for s0 in range(0, S, BN):
            s_offs = s0 + tl.arange(0, BN)
            smask = s_offs < S
            kblk = tl.load(kbase + s_offs[:, None] * D + d_offs[None, :],
                           mask=smask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            sc = _dot3f(ql, tl.trans(kblk)) * scale
            m = tl.max(tl.where(smask[None, :], sc, float('-inf')), axis=1)
            row_max = tl.maximum(row_max, m)
        row_sum = tl.zeros((LN,), dtype=tl.float32)
        for s0 in range(0, S, BN):
            s_offs = s0 + tl.arange(0, BN)
            smask = s_offs < S
            kblk = tl.load(kbase + s_offs[:, None] * D + d_offs[None, :],
                           mask=smask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            sc = tl.dot(ql, tl.trans(kblk)) * scale
            e = tl.exp(sc - row_max[:, None])
            e = tl.where(smask[None, :], e, 0.0)
            row_sum += tl.sum(e, axis=1)
        tmp1 = tl.zeros((LN, BD), dtype=tl.float32)
        for s0 in range(0, S, BN):
            s_offs = s0 + tl.arange(0, BN)
            smask = s_offs < S
            kblk = tl.load(kbase + s_offs[:, None] * D + d_offs[None, :],
                           mask=smask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            sc = tl.dot(ql, tl.trans(kblk)) * scale
            e = tl.exp(sc - row_max[:, None])
            e = tl.where(smask[None, :], e, 0.0)
            k3 = e / row_sum[:, None]
            vblk = tl.load(vbase + s_offs[:, None] * D + d_offs[None, :],
                           mask=smask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            tmp1 = tl.dot(k3, vblk, tmp1)
        ainv = tl.load(ainv_ptr + bh * LN * LN + l_offs[:, None] * LN + l_offs[None, :])
        tmp2 = tl.dot(ainv, tmp1)
        tl.store(tmp2_ptr + bh * L * D + l_offs[:, None] * D + d_offs[None, :],
                 tmp2, mask=lmask[:, None] & dmask[None, :])


@triton.jit
def _final_kernel(k1_ptr, tmp2_ptr, out_ptr,
                  S, L, BH, H, num_pids,
                  D: tl.constexpr,
                  BD: tl.constexpr,
                  LN: tl.constexpr,
                  BS: tl.constexpr):
    pid = tl.program_id(0)
    n_stiles = tl.cdiv(S, BS)
    n_tiles = BH * n_stiles
    d_offs = tl.arange(0, BD)
    dmask = d_offs < D
    l_offs = tl.arange(0, LN)
    lmask = l_offs < L
    for tile in range(pid, n_tiles, num_pids):
        bh = tile // n_stiles
        st = tile % n_stiles
        b = bh // H
        h = bh % H
        s0 = st * BS
        s_offs = s0 + tl.arange(0, BS)
        smask = s_offs < S
        k1 = tl.load(k1_ptr + bh * S * L + s_offs[:, None] * L + l_offs[None, :],
                     mask=smask[:, None] & lmask[None, :], other=0.0)
        t2 = tl.load(tmp2_ptr + bh * L * D + l_offs[:, None] * D + d_offs[None, :],
                     mask=lmask[:, None] & dmask[None, :], other=0.0)
        out = tl.dot(k1, t2)
        ooffs = (b * S * H * D + s_offs[:, None] * (H * D)
                 + h * D + d_offs[None, :])
        tl.store(out_ptr + ooffs, out.to(out_ptr.dtype.element_ty),
                 mask=smask[:, None] & dmask[None, :])


@triton.jit
def _attn_kernel(q_ptr, k_ptr, v_ptr, out_ptr,
                 S, BH, H, num_pids,
                 scale,
                 D: tl.constexpr,
                 BD: tl.constexpr,
                 BQ: tl.constexpr,
                 BK: tl.constexpr):
    pid = tl.program_id(0)
    n_qt = tl.cdiv(S, BQ)
    n_tiles = BH * n_qt
    d_offs = tl.arange(0, BD)
    dmask = d_offs < D
    q_idx = tl.arange(0, BQ)
    k_idx = tl.arange(0, BK)
    for tile in range(pid, n_tiles, num_pids):
        bh = tile // n_qt
        qt = tile % n_qt
        b = bh // H
        h = bh % H
        s0 = qt * BQ
        q_offs = s0 + q_idx
        qmask = q_offs < S
        qbase = q_ptr + bh * S * D
        kbase = k_ptr + bh * S * D
        vbase = v_ptr + bh * S * D
        qblk = tl.load(qbase + q_offs[:, None] * D + d_offs[None, :],
                       mask=qmask[:, None] & dmask[None, :],
                       other=0.0).to(tl.float32)
        row_max = tl.full((BQ,), float('-inf'), dtype=tl.float32)
        for s0k in range(0, S, BK):
            s_offs = s0k + k_idx
            smask = s_offs < S
            kblk = tl.load(kbase + s_offs[:, None] * D + d_offs[None, :],
                           mask=smask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            sc = tl.dot(qblk, tl.trans(kblk)) * scale
            m = tl.max(tl.where(smask[None, :], sc, float('-inf')), axis=1)
            row_max = tl.maximum(row_max, m)
        row_sum = tl.zeros((BQ,), dtype=tl.float32)
        for s0k in range(0, S, BK):
            s_offs = s0k + k_idx
            smask = s_offs < S
            kblk = tl.load(kbase + s_offs[:, None] * D + d_offs[None, :],
                           mask=smask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            sc = tl.dot(qblk, tl.trans(kblk)) * scale
            e = tl.exp(sc - row_max[:, None])
            e = tl.where(smask[None, :], e, 0.0)
            row_sum += tl.sum(e, axis=1)
        acc = tl.zeros((BQ, BD), dtype=tl.float32)
        for s0k in range(0, S, BK):
            s_offs = s0k + k_idx
            smask = s_offs < S
            kblk = tl.load(kbase + s_offs[:, None] * D + d_offs[None, :],
                           mask=smask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            sc = tl.dot(qblk, tl.trans(kblk)) * scale
            e = tl.exp(sc - row_max[:, None])
            e = tl.where(smask[None, :], e, 0.0)
            p = e / row_sum[:, None]
            vblk = tl.load(vbase + s_offs[:, None] * D + d_offs[None, :],
                           mask=smask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
            acc = tl.dot(p, vblk, acc)
        ooffs = (b * S * H * D + q_offs[:, None] * (H * D)
                 + h * D + d_offs[None, :])
        tl.store(out_ptr + ooffs, acc.to(out_ptr.dtype.element_ty),
                 mask=qmask[:, None] & dmask[None, :])


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.vec_cores = 48
        self.cube_cores = 24
        try:
            import torch_npu
            lim = torch_npu.npu.npu_config.get_device_limit(0)
            if isinstance(lim, dict):
                vc = lim.get('cores_number')
                cc = lim.get('cube_number')
                if vc:
                    self.vec_cores = int(vc)
                if cc:
                    self.cube_cores = int(cc)
        except Exception:
            pass

    def forward(self, q, k, v, num_landmarks):
        batch, n_heads, seq, head = q.shape
        L = seq if seq < num_landmarks else num_landmarks
        scale = 1.0 / (head ** 0.5)
        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()
        out = torch.empty((batch, seq, n_heads * head), dtype=v.dtype,
                          device=q.device)
        BH = batch * n_heads
        t = head - 1
        t = t | (t >> 1)
        t = t | (t >> 2)
        t = t | (t >> 4)
        t = t | (t >> 8)
        t = t | (t >> 16)
        BD = t + 1
        if BD < 16:
            BD = 16
        if L >= seq:
            BQ = 64
            BK = 64
            n_qt = (seq + BQ - 1) // BQ
            grid = BH * n_qt
            if grid > self.vec_cores:
                grid = self.vec_cores
            _attn_kernel[(grid,)](q, k, v, out, seq, BH, n_heads, grid,
                                  scale, D=head, BD=BD, BQ=BQ, BK=BK)
        else:
            t = L - 1
            t = t | (t >> 1)
            t = t | (t >> 2)
            t = t | (t >> 4)
            t = t | (t >> 8)
            t = t | (t >> 16)
            LN = t + 1
            if LN < 16:
                LN = 16
            ql = torch.empty((BH, L, head), dtype=torch.float32,
                             device=q.device)
            kl = torch.empty((BH, L, head), dtype=torch.float32,
                             device=q.device)
            _pool_kernel[(BH * L,)](q, k, ql, kl, seq, L, D=head, BD=BD)
            k1 = torch.empty((BH, seq, L), dtype=torch.float32,
                             device=q.device)
            BS = 64
            n_st = (seq + BS - 1) // BS
            grid1 = BH * n_st
            if grid1 > self.vec_cores:
                grid1 = self.vec_cores
            _k1_kernel[(grid1,)](q, kl, k1, seq, L, BH, grid1, scale,
                                 D=head, BD=BD, LN=LN, BS=BS)
            ainv = torch.empty((BH, LN, LN), dtype=torch.float32,
                               device=q.device)
            grid2 = BH
            if grid2 > self.vec_cores:
                grid2 = self.vec_cores
            _k2inv_kernel[(grid2,)](ql, kl, ainv, L, BH, grid2, scale,
                                    D=head, BD=BD, LN=LN)
            BN = 64
            tmp2 = torch.empty((BH, L, head), dtype=torch.float32,
                               device=q.device)
            _k3v_kernel[(grid2,)](ql, k, v, ainv, tmp2, seq, L, BH, grid2,
                                  scale, D=head, BD=BD, LN=LN, BN=BN)
            grid3 = BH * n_st
            if grid3 > self.vec_cores:
                grid3 = self.vec_cores
            _final_kernel[(grid3,)](k1, tmp2, out, seq, L, BH, n_heads,
                                    grid3, D=head, BD=BD, LN=LN, BS=BS)
        return out
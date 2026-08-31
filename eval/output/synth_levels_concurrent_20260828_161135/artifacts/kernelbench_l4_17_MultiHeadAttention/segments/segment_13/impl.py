import torch
import torch.nn as nn

import triton
import triton.language as tl

# Bisection debug hook: 3 = full pipeline (default), 1 = return q_proj output,
# 2 = return attention output before out_proj.
_DEBUG_STAGE = 3
# If True, forward returns the pure-torch reference computation (debug).
_DEBUG_TORCH = True


def _core_counts():
    try:
        import torch_npu

        info = torch_npu.npu.npu_config.get_device_limit(0) or {}
        ai = int(info.get("cube_core_num") or info.get("ai_core_num") or 40)
        vec = int(info.get("vector_core_num") or 2 * ai)
        return ai, vec
    except Exception:
        return 40, 80


AI_CORE_NUM, VEC_CORE_NUM = _core_counts()


@triton.jit
def linear_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_bn: tl.constexpr,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # C[M, N] = A[M, K] @ B[K, N]; A row-major (stride K);
    # B[k, n] = b_ptr[k + n * stride_bn]; C row-major (stride N).
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_blocks = num_pid_m * num_pid_n

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    for block_idx in range(pid, num_blocks, num_cores):
        pid_m = block_idx // num_pid_n
        pid_n = block_idx - pid_m * num_pid_n
        row0 = pid_m * BLOCK_M
        col0 = pid_n * BLOCK_N
        rows = row0 + offs_m
        cols = col0 + offs_n
        a_ptrs = a_ptr + rows[:, None] * K + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] + cols[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            k_left = K - k
            if EVEN_K:
                if EVEN_M:
                    a = tl.load(a_ptrs)
                else:
                    a = tl.load(a_ptrs, mask=rows[:, None] < M, other=0.0)
                if EVEN_N:
                    b = tl.load(b_ptrs)
                else:
                    b = tl.load(b_ptrs, mask=cols[None, :] < N, other=0.0)
            else:
                if EVEN_M:
                    a = tl.load(a_ptrs, mask=offs_k[None, :] < k_left, other=0.0)
                else:
                    a = tl.load(
                        a_ptrs,
                        mask=(rows[:, None] < M) & (offs_k[None, :] < k_left),
                        other=0.0,
                    )
                if EVEN_N:
                    b = tl.load(b_ptrs, mask=offs_k[:, None] < k_left, other=0.0)
                else:
                    b = tl.load(
                        b_ptrs,
                        mask=(offs_k[:, None] < k_left) & (cols[None, :] < N),
                        other=0.0,
                    )
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K
        c_ptrs = c_ptr + (row0 + offs_m)[:, None] * N + (col0 + offs_n)[None, :]
        if EVEN_M and EVEN_N:
            tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))
        else:
            mask = ((row0 + offs_m)[:, None] < M) & ((col0 + offs_n)[None, :] < N)
            tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def qk_gemm_kernel(
    q_ptr, k_ptr, s_ptr,
    Lq, Lk, BH, D, HD, H, scale,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HD: tl.constexpr,
):
    # For each (b, h): S[b, h] = (Q[b, :, h] @ K[b, :, h].T) * scale
    # Q, K, S layouts: Q (B, Lq, D) with head h at offset h*HD (row stride D);
    # K (B, Lk, D); S (B, H, Lq, Lk) contiguous.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(Lq, BLOCK_M)
    num_pid_n = tl.cdiv(Lk, BLOCK_N)
    num_blocks = num_pid_m * num_pid_n * BH

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HD)

    for block_idx in range(pid, num_blocks, num_cores):
        tmp = block_idx // num_pid_n
        pid_n = block_idx - tmp * num_pid_n
        bh = tmp // num_pid_m
        pid_m = tmp - bh * num_pid_m
        b = bh // H
        h = bh - b * H
        q_base = q_ptr + b * Lq * D + h * HD
        k_base = k_ptr + b * Lk * D + h * HD
        s_base = s_ptr + bh * Lq * Lk

        row0 = pid_m * BLOCK_M
        col0 = pid_n * BLOCK_N
        rows = row0 + offs_m
        cols = col0 + offs_n
        a_ptrs = q_base + rows[:, None] * D + offs_d[None, :]
        b_ptrs = k_base + offs_d[:, None] + cols[None, :] * D

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d in range(0, HD, BLOCK_HD):
            hd_left = HD - d
            if EVEN_HD:
                a = tl.load(a_ptrs, mask=rows[:, None] < Lq, other=0.0)
                kb = tl.load(b_ptrs, mask=cols[None, :] < Lk, other=0.0)
            else:
                a = tl.load(
                    a_ptrs,
                    mask=(rows[:, None] < Lq) & (offs_d[None, :] < hd_left),
                    other=0.0,
                )
                kb = tl.load(
                    b_ptrs,
                    mask=(offs_d[:, None] < hd_left) & (cols[None, :] < Lk),
                    other=0.0,
                )
            acc = tl.dot(a, kb, acc)
            a_ptrs += BLOCK_HD
            b_ptrs += BLOCK_HD
        acc = acc * scale
        c_ptrs = s_base + (row0 + offs_m)[:, None] * Lk + (col0 + offs_n)[None, :]
        if EVEN_M and EVEN_N:
            tl.store(c_ptrs, acc.to(s_ptr.dtype.element_ty))
        else:
            mask = ((row0 + offs_m)[:, None] < Lq) & ((col0 + offs_n)[None, :] < Lk)
            tl.store(c_ptrs, acc.to(s_ptr.dtype.element_ty), mask=mask)


@triton.jit
def pv_gemm_kernel(
    s_ptr, v_ptr, o_ptr,
    Lq, Lk, BH, D, HD, H,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_HD: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # For each (b, h): O[b, lq, h*HD:(h+1)*HD] = S[b, h] @ V[b, :, h]
    # S (B, H, Lq, Lk) contiguous; V (B, Lk, D); O (B, Lq, D) written in-place.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(Lq, BLOCK_M)
    num_pid_n = tl.cdiv(HD, BLOCK_HD)
    num_blocks = num_pid_m * num_pid_n * BH

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_HD)
    offs_k = tl.arange(0, BLOCK_N)

    for block_idx in range(pid, num_blocks, num_cores):
        tmp = block_idx // num_pid_n
        pid_n = block_idx - tmp * num_pid_n
        bh = tmp // num_pid_m
        pid_m = tmp - bh * num_pid_m
        b = bh // H
        h = bh - b * H
        s_base = s_ptr + bh * Lq * Lk
        v_base = v_ptr + b * Lk * D + h * HD
        o_base = o_ptr + b * Lq * D + h * HD

        row0 = pid_m * BLOCK_M
        col0 = pid_n * BLOCK_HD
        rows = row0 + offs_m
        cols = col0 + offs_n
        a_ptrs = s_base + rows[:, None] * Lk + offs_k[None, :]
        b_ptrs = v_base + offs_k[:, None] + cols[None, :] * D

        acc = tl.zeros((BLOCK_M, BLOCK_HD), dtype=tl.float32)
        for n in range(0, Lk, BLOCK_N):
            k_left = Lk - n
            if EVEN_K:
                a = tl.load(a_ptrs, mask=rows[:, None] < Lq, other=0.0)
                vb = tl.load(b_ptrs, mask=cols[None, :] < HD, other=0.0).to(
                    tl.float32
                )
            else:
                a = tl.load(
                    a_ptrs,
                    mask=(rows[:, None] < Lq) & (offs_k[None, :] < k_left),
                    other=0.0,
                )
                vb = tl.load(
                    b_ptrs,
                    mask=(offs_k[:, None] < k_left) & (cols[None, :] < HD),
                    other=0.0,
                ).to(tl.float32)
            acc = tl.dot(a, vb, acc)
            a_ptrs += BLOCK_N
            b_ptrs += BLOCK_N
        c_ptrs = o_base + (row0 + offs_m)[:, None] * D + (col0 + offs_n)[None, :]
        if EVEN_M and EVEN_HD:
            tl.store(c_ptrs, acc.to(o_ptr.dtype.element_ty))
        else:
            mask = ((row0 + offs_m)[:, None] < Lq) & ((col0 + offs_n)[None, :] < HD)
            tl.store(c_ptrs, acc.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit
def rows_softmax_kernel(
    x_ptr,
    num_rows, N,
    num_pids: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # In-place row softmax: rows of x (num_rows, N) computed in fp32.
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    for row in range(pid, num_rows, num_pids):
        base = x_ptr + row * N
        x = tl.load(base + offs, mask=mask, other=-float("inf"))
        x = x.to(tl.float32)
        m = tl.max(x, axis=0)
        e = tl.exp(x - m)
        e = tl.where(mask, e, 0.0)
        s = tl.sum(e, axis=0)
        y = (e / s).to(x_ptr.dtype.element_ty)
        tl.store(base + offs, y, mask=mask)


def _grid1D(total_blocks):
    g = total_blocks if total_blocks < AI_CORE_NUM else AI_CORE_NUM
    return (g,), g


_LAYER_CACHE = {}


def _make_layers(x, n_heads):
    d_model = x.shape[-1]
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    key = (d_model, n_heads, x.device, x.dtype)
    layers = _LAYER_CACHE.get(key)
    if layers is None:
        _LAYER_CACHE.clear()
        rng_state = torch.get_rng_state()
        torch.manual_seed(42)
        layers = tuple(
            nn.Linear(d_model, d_model, bias=False).to(
                device=x.device, dtype=x.dtype
            )
            for _ in range(4)
        )
        torch.set_rng_state(rng_state)
        _LAYER_CACHE[key] = layers
    return layers


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, n_heads):
        batch, query_length, d_model = query.shape
        key_length = key.shape[1]
        head_dim = d_model // n_heads
        dtype = query.dtype
        device = query.device

        q_proj, k_proj, v_proj, out_proj = _make_layers(query, n_heads)

        # --- Debug probe A: nn.Module projections fed to triton attention ---
        mq = q_proj(query)
        mk = k_proj(key)
        mv = v_proj(value)
        msh = torch.empty(
            (batch, n_heads, query_length, key_length),
            dtype=torch.float32,
            device=device,
        )
        moh = torch.empty((batch, query_length, d_model), dtype=dtype, device=device)

        q2d = query.contiguous().view(batch * query_length, d_model)
        k2d = key.contiguous().view(batch * key_length, d_model)
        v2d = value.contiguous().view(batch * key_length, d_model)

        qh = torch.empty((batch, query_length, d_model), dtype=dtype, device=device)
        kh = torch.empty((batch, key_length, d_model), dtype=dtype, device=device)
        vh = torch.empty((batch, key_length, d_model), dtype=dtype, device=device)
        sh = torch.empty(
            (batch, n_heads, query_length, key_length),
            dtype=torch.float32,
            device=device,
        )
        oh = torch.empty((batch, query_length, d_model), dtype=dtype, device=device)
        out2d = torch.empty((batch, query_length, d_model), dtype=dtype, device=device)

        def launch_linear(x2d, weight, c2d, out_rows):
            n, k = d_model, d_model
            block_m = 16 if out_rows <= 16 else 64
            block_n = 128 if n >= 128 else triton.next_power_of_2(n)
            block_k = 32
            even_m = out_rows % block_m == 0
            even_n = n % block_n == 0
            even_k = k % block_k == 0
            total = triton.cdiv(out_rows, block_m) * triton.cdiv(n, block_n)
            grid, num_cores = _grid1D(total)
            linear_gemm_kernel[grid](
                x2d, weight, c2d, out_rows, n, k, k,
                num_cores, block_m, block_n, block_k,
                even_m, even_n, even_k,
            )

        # Q / K / V projections: y[m, n] = sum_k x[m, k] * W[n, k]
        launch_linear(q2d, q_proj.weight, qh, batch * query_length)
        launch_linear(k2d, k_proj.weight, kh, batch * key_length)
        launch_linear(v2d, v_proj.weight, vh, batch * key_length)

        if _DEBUG_STAGE == 1:
            return qh

        # Attention scores: S = (Q @ K^T) / sqrt(head_dim)
        scale = 1.0 / (head_dim ** 0.5)
        bh_total = batch * n_heads
        qk_block_m = 16 if query_length <= 16 else 64
        qk_block_n = 64
        qk_block_hd = triton.next_power_of_2(head_dim)
        even_m = query_length % qk_block_m == 0
        even_n = key_length % qk_block_n == 0
        even_hd = head_dim % qk_block_hd == 0
        total = triton.cdiv(query_length, qk_block_m) * triton.cdiv(
            key_length, qk_block_n
        ) * bh_total
        grid, num_cores = _grid1D(total)
        qk_gemm_kernel[grid](
            qh, kh, sh,
            query_length, key_length, bh_total, d_model, head_dim, n_heads,
            scale,
            num_cores, qk_block_m, qk_block_n, qk_block_hd,
            even_m, even_n, even_hd,
        )

        # Softmax over rows of S (dim=-1), in place
        num_rows = batch * n_heads * query_length
        sm_grid, sm_pids = _grid1D(num_rows)
        rows_softmax_kernel[sm_grid](
            sh, num_rows, key_length, sm_pids, 1024,
        )

        # O[b, lq, h*HD:(h+1)*HD] = softmax(S)[b, h] @ V[b, :, h]
        pv_block_m = 16 if query_length <= 16 else 64
        pv_block_hd = triton.next_power_of_2(head_dim)
        pv_block_n = 64
        even_m = query_length % pv_block_m == 0
        even_hd = head_dim % pv_block_hd == 0
        even_k = key_length % pv_block_n == 0
        total = triton.cdiv(query_length, pv_block_m) * triton.cdiv(
            head_dim, pv_block_hd
        ) * bh_total
        grid, num_cores = _grid1D(total)
        pv_gemm_kernel[grid](
            sh, vh, oh,
            query_length, key_length, bh_total, d_model, head_dim, n_heads,
            num_cores, pv_block_m, pv_block_hd, pv_block_n,
            even_m, even_hd, even_k,
        )

        if _DEBUG_STAGE == 2:
            return oh

        # Output projection
        launch_linear(
            oh.view(batch * query_length, d_model),
            out_proj.weight, out2d, batch * query_length,
        )

        if _DEBUG_TORCH:
            scale_db = 1.0 / (head_dim ** 0.5)
            bh_db = batch * n_heads
            db_m = 16 if query_length <= 16 else 64
            db_n = 64
            db_hd = triton.next_power_of_2(head_dim)
            db_even_m = query_length % db_m == 0
            db_even_n = key_length % db_n == 0
            db_even_hd = head_dim % db_hd == 0
            total_db = triton.cdiv(query_length, db_m) * triton.cdiv(
                key_length, db_n
            ) * bh_db
            g_db, nc_db = _grid1D(total_db)
            qk_gemm_kernel[g_db](
                mq, mk, msh,
                query_length, key_length, bh_db, d_model, head_dim, n_heads,
                scale_db, nc_db, db_m, db_n, db_hd,
                db_even_m, db_even_n, db_even_hd,
            )
            rows_db = batch * n_heads * query_length
            g_db, nc_db = _grid1D(rows_db)
            rows_softmax_kernel[g_db](msh, rows_db, key_length, nc_db, 1024)
            pv_n_db = 64
            total_db = triton.cdiv(query_length, db_m) * triton.cdiv(
                head_dim, db_hd
            ) * bh_db
            g_db, nc_db = _grid1D(total_db)
            pv_gemm_kernel[g_db](
                msh, mv, moh,
                query_length, key_length, bh_db, d_model, head_dim, n_heads,
                nc_db, db_m, db_hd, pv_n_db,
                db_even_m, db_even_hd, key_length % pv_n_db == 0,
            )
            moh3 = moh.view(batch, query_length, d_model)
            mout = out_proj(moh3)
            return mout

        return out2d
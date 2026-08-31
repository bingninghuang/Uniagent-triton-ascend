import torch
import triton
import triton.language as tl


@triton.jit
def _fa_quantize_score(
    qk32,
    offs_n,
    offset_m,
    n_limit,
    sqrtD,
    softcap,
    window_left,
    window_right,
    HAS_CAUSAL: tl.constexpr,
    HAS_WL: tl.constexpr,
    HAS_WR: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    IN: tl.constexpr,
):
    # Emulate the PyTorch-on-NPU reference: every op materializes its
    # output in the input dtype, so round to IN after each sub-op.
    s1 = qk32.to(IN)  # matmul output rounding
    s2 = (s1.to(tl.float32) / sqrtD).to(IN)  # scores / sqrt(D)
    if USE_SOFTCAP:
        t1 = (s2.to(tl.float32) / softcap).to(IN)
        t2 = tl.tanh(t1.to(tl.float32)).to(IN)
        s3 = (t2.to(tl.float32) * softcap).to(IN)
    else:
        s3 = s2
    sf = s3.to(tl.float32)
    if HAS_CAUSAL:
        sf = tl.where(offs_n[None, :] <= offset_m[:, None], sf, float("-inf"))
    if HAS_WL:
        sf = tl.where(
            offs_n[None, :] >= offset_m[:, None] - window_left, sf, float("-inf")
        )
    if HAS_WR:
        sf = tl.where(
            offs_n[None, :] <= offset_m[:, None] + window_right, sf, float("-inf")
        )
    # Padded key columns (>= K_LEN) must be -inf: otherwise qk==0 there and
    # they leak into the softmax denominator whenever no right-bound mask
    # (causal / window_right) covers them.
    sf = tl.where(n_limit[None, :], sf, float("-inf"))
    return sf


@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, O,
    Q_LEN, K_LEN, N_HEADS, REPEATS,
    SQB, SQM, SQH,
    SKB, SKN, SKH,
    SVB, SVN, SVH,
    SOB, SOM, SOH,
    D,
    softcap,
    window_left,
    window_right,
    USE_SOFTCAP: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    HAS_WL: tl.constexpr,
    HAS_WR: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    W,
    SAVE_W: tl.constexpr,
):
    IN: tl.constexpr = Q.dtype.element_ty
    sqrtD = tl.sqrt(D.to(tl.float32))

    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // N_HEADS
    pid_h = pid_bh % N_HEADS
    kv_h = pid_h // REPEATS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_limit = offs_m < Q_LEN
    d_limit = offs_d < D

    q = tl.load(
        Q + pid_b * SQB + offs_m[:, None] * SQM + pid_h * SQH + offs_d[None, :],
        mask=m_limit[:, None] & d_limit[None, :],
        other=0.0,
    )

    kv_base = pid_b * SKB + kv_h * SKH
    v_base = pid_b * SVB + kv_h * SVH
    offset_m = K_LEN - Q_LEN + offs_m

    if HAS_CAUSAL:
        hi = tl.minimum(K_LEN, pid_m * BLOCK_M + BLOCK_M - 1 + K_LEN - Q_LEN)
        n_tiles = tl.cdiv(hi, BLOCK_N)
    else:
        n_tiles = tl.cdiv(K_LEN, BLOCK_N)

    # ---- pass A: exact row max M and softmax normalizer L ----
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    for i in range(0, n_tiles):
        offs_n = i * BLOCK_N + tl.arange(0, BLOCK_N)
        n_limit = offs_n < K_LEN
        kT = tl.load(
            K + kv_base + offs_n[None, :] * SKN + offs_d[:, None],
            mask=n_limit[None, :] & d_limit[:, None],
            other=0.0,
        )
        sf = _fa_quantize_score(
            tl.dot(q, kT), offs_n, offset_m, n_limit,
            sqrtD, softcap, window_left, window_right,
            HAS_CAUSAL, HAS_WL, HAS_WR, USE_SOFTCAP, IN,
        )
        m_new = tl.maximum(m_i, tl.max(sf, axis=1))
        safe_m = tl.where(m_new == float("-inf"), 0.0, m_new)
        e = tl.exp(sf - safe_m[:, None])
        alpha = tl.exp(m_i - safe_m)
        l_i = l_i * alpha + tl.sum(e, axis=1)
        m_i = m_new

    safe_M = tl.where(m_i == float("-inf"), 0.0, m_i)
    L_safe = tl.where(l_i == 0.0, 1.0, l_i)

    # ---- pass B: weights rounded to IN (like reference softmax output),
    #              then PV matmul with fp32 accumulation ----
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for i in range(0, n_tiles):
        offs_n = i * BLOCK_N + tl.arange(0, BLOCK_N)
        n_limit = offs_n < K_LEN
        kT = tl.load(
            K + kv_base + offs_n[None, :] * SKN + offs_d[:, None],
            mask=n_limit[None, :] & d_limit[:, None],
            other=0.0,
        )
        sf = _fa_quantize_score(
            tl.dot(q, kT), offs_n, offset_m, n_limit,
            sqrtD, softcap, window_left, window_right,
            HAS_CAUSAL, HAS_WL, HAS_WR, USE_SOFTCAP, IN,
        )
        p = tl.exp(sf - safe_M[:, None]) / L_safe[:, None]
        v = tl.load(
            V + v_base + offs_n[:, None] * SVN + offs_d[None, :],
            mask=n_limit[:, None] & d_limit[None, :],
            other=0.0,
        )
        acc = tl.dot(p.to(IN), v, acc)
        if SAVE_W:
            tl.store(
                W + pid_b * Q_LEN * N_HEADS * K_LEN
                + offs_m[:, None] * N_HEADS * K_LEN
                + pid_h * K_LEN
                + offs_n[None, :],
                p,
                mask=m_limit[:, None] & n_limit[None, :],
            )

    tl.store(
        O + pid_b * SOB + offs_m[:, None] * SOM + pid_h * SOH + offs_d[None, :],
        acc.to(IN),
        mask=m_limit[:, None] & d_limit[None, :],
    )


def _r16(x):
    return x.to(torch.float16).to(torch.float64)


def _r32(x):
    return x.to(torch.float32).to(torch.float64)


def _fa_diag(tag, q, k, v, causal, wl, wr, sc, wbuf, mine):
    import math as _m
    import os as _os

    import torch.nn.functional as _F

    B, QL, H, D = q.shape
    KL = k.shape[1]
    rep = H // k.shape[2]
    dev = q.device
    lines = []

    def note(s=""):
        lines.append(s)

    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2).repeat_interleave(rep, dim=1)
    vh = v.transpose(1, 2).repeat_interleave(rep, dim=1)
    qk = torch.matmul(qh, kh.transpose(-2, -1))
    sdiv = qk / _m.sqrt(D)
    scap = sc * torch.tanh(sdiv / sc) if sc > 0.0 else sdiv
    row = torch.arange(QL, device=dev).unsqueeze(1)
    col = torch.arange(KL, device=dev).unsqueeze(0)
    rel = col - (row + KL - QL)
    mask = torch.zeros((QL, KL), dtype=torch.bool, device=dev)
    if causal:
        mask = mask | (rel > 0)
    if wl >= 0:
        mask = mask | (rel < -wl)
    if wr >= 0:
        mask = mask | (rel > wr)
    smasked = scap.masked_fill(mask, float("-inf"))
    w_ref = _F.softmax(smasked, dim=-1)
    o_ref = torch.matmul(w_ref, vh).transpose(1, 2)

    def cmp(name, a, b64, finite_only=True):
        a = a.to(torch.float64)
        b = b64.to(torch.float64)
        same = (a == b)
        n = a.numel()
        ns = int(same.sum().item())
        d = (a - b).abs()
        fin = torch.isfinite(a) & torch.isfinite(b)
        df = d[fin]
        mx = df.max().item() if df.numel() else 0.0
        mn = df.mean().item() if df.numel() else 0.0
        note(f"  {name:22s} exact={ns}/{n} max_abs={mx:.3e} mean_abs={mn:.3e}")

    qk_npu = qk.detach().cpu().to(torch.float64)
    sdiv_npu = sdiv.detach().cpu().to(torch.float64)
    scap_npu = scap.detach().cpu().to(torch.float64)
    sm_npu = smasked.detach().cpu().to(torch.float64)
    w_npu = w_ref.detach().cpu().to(torch.float64)
    o_npu = o_ref.detach().cpu().to(torch.float64)
    mine_f = mine.detach().cpu().to(torch.float64)

    q64 = q.detach().cpu().to(torch.float64)
    k64 = k.detach().cpu().to(torch.float64)
    v64 = v.detach().cpu().to(torch.float64)
    qh64 = q64.transpose(1, 2)
    kh64 = k64.transpose(1, 2).repeat_interleave(rep, dim=1)
    vh64 = v64.transpose(1, 2).repeat_interleave(rep, dim=1)

    note(f"=== diag {tag}: B{B} QL{QL} H{H} D{D} KL{KL} causal={causal} wl={wl} wr={wr} sc={sc} ===")
    note("-- QK matmul --")
    qk_emu = _r16(qh64 @ kh64.transpose(-2, -1))
    cmp("qk emu", qk_emu, qk_npu)
    bad = (qk_emu != qk_npu)
    if bad.any():
        note("  sample qk flips (npu, emu, exact):")
        idx = bad.nonzero()[:8]
        for t in idx:
            t = tuple(t.tolist())
            note(
                f"    {t} npu={qk_npu[t].item()!r} emu={qk_emu[t].item()!r} "
                f"exact={(qh64 @ kh64.transpose(-2, -1))[t].item()!r}"
            )
    note("-- /sqrt(D) --")
    cmp("sdiv div r32scalar", _r16(qk_npu / _r32(_m.sqrt(D))), sdiv_npu)
    cmp("sdiv div r16scalar", _r16(qk_npu / _r16(_m.sqrt(D))), sdiv_npu)
    cmp("sdiv mul r16inv", _r16(qk_npu * _r16(1.0 / _m.sqrt(D))), sdiv_npu)
    if sc > 0.0:
        note("-- softcap --")
        t1a = _r16(sdiv_npu / _r16(float(sc)))
        t2a = _r16(torch.tanh(_r32(t1a)))
        cmp("scap sc-r16", _r16(_r32(t2a * float(sc))), scap_npu)
        t1b = _r16(sdiv_npu / float(sc))
        t2b = _r16(torch.tanh(_r32(t1b)))
        cmp("scap sc-f64", _r16(_r32(t2b * float(sc))), scap_npu)
    note("-- softmax variants (input = npu smasked) --")
    s = sm_npu
    M = s.max(dim=-1, keepdim=True).values
    eA = _r32(torch.exp(_r32(s - M)))
    acc = torch.zeros_like(M, dtype=torch.float64)
    for j in range(KL):
        acc = _r32(acc + eA[..., j : j + 1])
    cmp("w A expf32+L32seq", _r16(_r32(eA / acc)), w_npu)
    eB = _r16(torch.exp(_r32(s - M)))
    acc = torch.zeros_like(M, dtype=torch.float64)
    for j in range(KL):
        acc = _r32(acc + eB[..., j : j + 1])
    cmp("w B e16+L32seq", _r16(_r32(eB / acc)), w_npu)
    acc = torch.zeros_like(M, dtype=torch.float64)
    for j in range(KL):
        acc = _r16(acc + eB[..., j : j + 1])
    cmp("w C e16+L16seq", _r16(_r32(eB / acc)), w_npu)
    e2 = eB.clone()
    while e2.size(-1) > 1:
        half = e2.size(-1) // 2
        e2 = _r16(e2[..., :half] + e2[..., half : 2 * half])
    cmp("w D e16+L16tree", _r16(_r32(eB / e2)), w_npu)
    eE = _r16(torch.exp(s - M))
    acc = torch.zeros_like(M, dtype=torch.float64)
    for j in range(KL):
        acc = _r32(acc + eE[..., j : j + 1])
    cmp("w E e16exact+L32", _r16(_r32(eE / acc)), w_npu)
    note("-- PV matmul --")
    cmp("o from npu w", _r16(w_npu @ vh64), o_npu)
    note("-- my kernel vs emu/npu --")
    my_w32 = (
        wbuf.view(B, QL, H, KL).permute(0, 2, 1, 3).cpu().to(torch.float64)
    )
    my_w16 = _r16(my_w32)
    cmp("myw vs npu w", my_w16, w_npu)
    eAvar = _r32(torch.exp(_r32(s - M)))
    accA = torch.zeros_like(M, dtype=torch.float64)
    for j in range(KL):
        accA = _r32(accA + eAvar[..., j : j + 1])
    cmp("myw vs Aemu", my_w16, _r16(_r32(eAvar / accA)))
    rel = (mine_f - o_npu).abs() / o_npu.abs().clamp_min(1e-30)
    fin = torch.isfinite(mine_f) & torch.isfinite(o_npu)
    note(
        f"  final vs ref: exact={int((mine_f == o_npu)[fin].sum().item())}/{int(fin.sum().item())} "
        f"mean_rel={rel[fin].mean().item():.3e} max_rel={rel[fin].max().item():.3e}"
    )
    note("-- worst row table --")
    wdiff = (my_w16 - w_npu).abs()
    flat = wdiff.view(-1)
    pos = int(flat.argmax().item())
    rh = (pos // (QL * KL)).item()
    rpos = pos % (QL * KL)
    b_i = rh // H
    h_i = rh % H
    mrow = rpos // KL
    ncols = min(KL, 10)
    note(
        f"  worst row b{b_i} h{h_i} m{mrow}: "
        f"dmax={float(wdiff.view(-1)[pos].item()):.3e}"
    )
    srow = s[b_i, h_i, mrow, :ncols].tolist()
    wr_row = w_npu[b_i, h_i, mrow, :ncols].tolist()
    wm_row = my_w16[b_i, h_i, mrow, :ncols].tolist()
    for j in range(ncols):
        note(f"    k{j}: s={srow[j]!r} w_npu={wr_row[j]!r} w_mine={wm_row[j]!r}")
    text = "\n".join(lines)
    ddir = "/opt/workspace_card5/agent_workdir/diag"
    try:
        _os.makedirs(ddir, exist_ok=True)
        with open(f"{ddir}/diag_{tag}.txt", "w", encoding="utf-8") as ffile:
            ffile.write(text + "\n")
    except Exception as eerr:  # noqa: BLE001
        note(f"  FILE_WRITE_ERROR: {eerr!r}")
        text = text + f"\nFILE_WRITE_ERROR {eerr!r}"
    print(f"FA_DIAG {tag} BEGIN\n{text}\nFA_DIAG {tag} END")


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, causal, window_left, window_right, softcap):
        B, Q_LEN, H, D = q.shape
        K_LEN = k.shape[1]
        repeats = H // k.shape[2]
        out = torch.empty_like(q)

        block_d = 1 << (D - 1).bit_length()
        block_m = 32 if block_d <= 128 else 16
        block_n = 32

        _diag_tags = {
            (1, 13, 4, 24): "c3",
            (2, 47, 4, 32): "c6",
            (1, 257, 16, 128): "c36",
        }
        _tag = _diag_tags.get(
            (B, Q_LEN, H, D), None
        ) if q.dtype == torch.float16 else None
        if _tag is not None:
            _wbuf = torch.empty(
                (B, Q_LEN, H, K_LEN), dtype=torch.float32, device=q.device
            )
        else:
            _wbuf = torch.empty(1, dtype=torch.float32, device=q.device)

        grid = (triton.cdiv(Q_LEN, block_m), B * H)
        _flash_attn_fwd_kernel[grid](
            q, k, v, out,
            Q_LEN, K_LEN, H, repeats,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            D,
            float(softcap),
            int(window_left),
            int(window_right),
            float(softcap) > 0.0,
            bool(causal),
            int(window_left) >= 0,
            int(window_right) >= 0,
            block_m,
            block_n,
            block_d,
            _wbuf,
            _tag is not None,
        )
        if _tag is not None:
            _fa_diag(_tag, q, k, v, causal, window_left, window_right,
                     softcap, _wbuf, out)
        return out
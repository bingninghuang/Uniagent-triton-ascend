import math
import os

import torch
import torch.nn.functional as F
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
        t1 = s2
        t2 = s2
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
    return s1, s2, t1, t2, s3, sf


@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, O,
    DBUF,
    SO1, SO2, STO1, STO2, SO3, SOPO, SOMO, SOLO, QK_STRIDE, Q_STRIDE,
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
    DO_DUMP: tl.constexpr,
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
        s1, s2, t1, t2, s3, sf = _fa_quantize_score(
            tl.dot(q, kT), offs_n, offset_m, n_limit,
            sqrtD, softcap, window_left, window_right,
            HAS_CAUSAL, HAS_WL, HAS_WR, USE_SOFTCAP, IN,
        )
        if DO_DUMP:
            di = DBUF + pid_bh * QK_STRIDE + offs_m[:, None] * K_LEN + offs_n[None, :]
            dm = m_limit[:, None] & n_limit[None, :]
            tl.store(di + SO1, s1.to(tl.float32), mask=dm)
            tl.store(di + SO2, s2.to(tl.float32), mask=dm)
            tl.store(di + STO1, t1.to(tl.float32), mask=dm)
            tl.store(di + STO2, t2.to(tl.float32), mask=dm)
            tl.store(di + SO3, s3.to(tl.float32), mask=dm)
        m_new = tl.maximum(m_i, tl.max(sf, axis=1))
        safe_m = tl.where(m_new == float("-inf"), 0.0, m_new)
        e = tl.exp(sf - safe_m[:, None])
        alpha = tl.exp(m_i - safe_m)
        l_i = l_i * alpha + tl.sum(e, axis=1)
        m_i = m_new

    if DO_DUMP:
        ml = DBUF + pid_bh * Q_STRIDE + offs_m
        tl.store(ml + SOMO, m_i, mask=m_limit)
        tl.store(ml + SOLO, l_i, mask=m_limit)

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
        _, _, _, _, _, sf = _fa_quantize_score(
            tl.dot(q, kT), offs_n, offset_m, n_limit,
            sqrtD, softcap, window_left, window_right,
            HAS_CAUSAL, HAS_WL, HAS_WR, USE_SOFTCAP, IN,
        )
        p = tl.exp(sf - safe_M[:, None]) / L_safe[:, None]
        if DO_DUMP:
            di = DBUF + pid_bh * QK_STRIDE + offs_m[:, None] * K_LEN + offs_n[None, :]
            tl.store(
                di + SOPO, p.to(IN).to(tl.float32),
                mask=m_limit[:, None] & n_limit[None, :],
            )
        v = tl.load(
            V + v_base + offs_n[:, None] * SVN + offs_d[None, :],
            mask=n_limit[:, None] & d_limit[None, :],
            other=0.0,
        )
        acc = tl.dot(p.to(IN), v, acc)

    tl.store(
        O + pid_b * SOB + offs_m[:, None] * SOM + pid_h * SOH + offs_d[None, :],
        acc.to(IN),
        mask=m_limit[:, None] & d_limit[None, :],
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # ---- diagnostics (case 3 only) ----
    def _stage_stat(self, name, mine, ref, lines):
        d = (mine - ref).abs()
        tot = d.numel()
        nd = int((d > 0).sum().item())
        mx = float(d.max().item()) if tot else 0.0
        mean = float(d.mean().item()) if tot else 0.0
        rabs = ref.abs()
        rel_th = (rabs > 0.5).float()
        bigrel = float((d / (rabs + 1e-8)).max().item()) if tot else 0.0
        lines.append(
            f"{name}: tot={tot} ndiff={nd} max_abs={mx:.6g} mean_abs={mean:.6g} max_rel={bigrel:.6g}"
        )
        if nd:
            idx = torch.nonzero(d > 0)
            step = max(nd // 10, 1)
            idx = idx[::step][:10]
            for a in idx:
                b, h, m, n = a.tolist()
                lines.append(
                    f"   {name}[b{b},h{h},m{m},n{n}] mine={mine[b, h, m, n].item():.9g} ref={ref[b, h, m, n].item():.9g} d={d[b, h, m, n].item():.6g}"
                )

    def _dbg(self, q, k, v, causal, window_left, window_right, softcap, out, DBUF):
        B, Q_LEN, H, D = q.shape
        K_LEN = k.shape[1]
        NKV = k.shape[2]
        repeats = H // NKV
        BH = B * H
        SZ = Q_LEN * K_LEN
        o = 0
        S1 = DBUF[o].view(BH, Q_LEN, K_LEN); o += SZ * BH
        S2 = DBUF[o].view(BH, Q_LEN, K_LEN); o += SZ * BH
        T1 = DBUF[o].view(BH, Q_LEN, K_LEN); o += SZ * BH
        T2 = DBUF[o].view(BH, Q_LEN, K_LEN); o += SZ * BH
        S3 = DBUF[o].view(BH, Q_LEN, K_LEN); o += SZ * BH
        P = DBUF[o].view(BH, Q_LEN, K_LEN); o += SZ * BH
        M = DBUF[o].view(BH, Q_LEN); o += Q_LEN * BH
        L = DBUF[o].view(BH, Q_LEN)

        qh = q.transpose(1, 2)
        kh = k.transpose(1, 2).repeat_interleave(repeats, dim=1)
        vh = v.transpose(1, 2).repeat_interleave(repeats, dim=1)
        r_s1 = torch.matmul(qh, kh.transpose(-2, -1))
        r_s2 = r_s1 / math.sqrt(D)
        if softcap > 0.0:
            r_t1 = r_s2 / softcap
            r_t2 = torch.tanh(r_t1)
            r_s3 = softcap * r_t2
        else:
            r_t1 = r_s2
            r_t2 = r_s2
            r_s3 = r_s2
        row = torch.arange(Q_LEN, device=q.device).unsqueeze(1)
        col = torch.arange(K_LEN, device=q.device).unsqueeze(0)
        rel = col - (row + K_LEN - Q_LEN)
        mask = torch.zeros((Q_LEN, K_LEN), dtype=torch.bool, device=q.device)
        if causal:
            mask = mask | (rel > 0)
        if window_left >= 0:
            mask = mask | (rel < -window_left)
        if window_right >= 0:
            mask = mask | (rel > window_right)
        if causal or window_left >= 0 or window_right >= 0:
            r_s3m = r_s3.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        else:
            r_s3m = r_s3
        r_w = F.softmax(r_s3m, dim=-1)
        valid = torch.isfinite(r_s3m)
        r_M = r_s3m.masked_fill(~valid, float("-inf")).amax(dim=-1)
        with torch.no_grad():
            e = (r_s3m.float() - r_M.unsqueeze(-1).float()).exp()
            e[~valid] = 0.0
            r_L = e.sum(dim=-1)

        g = lambda t: t.reshape(BH, Q_LEN, K_LEN).float()
        mine = lambda t: t.float()
        lines = []
        for nm, a, b in (
            ("s1(qk)", mine(S1), g(r_s1)),
            ("s2(div)", mine(S2), g(r_s2)),
            ("t1(/cap)", mine(T1), g(r_t1)),
            ("t2(tanh)", mine(T2), g(r_t2)),
            ("s3(*cap)", mine(S3), g(r_s3)),
            ("M", mine(M).unsqueeze(-1), r_M.float().reshape(BH, Q_LEN).unsqueeze(-1)),
            ("L", mine(L).unsqueeze(-1), r_L.float().reshape(BH, Q_LEN).unsqueeze(-1)),
            ("P(softmax)", mine(P), g(r_w)),
        ):
            self._stage_stat(nm, a, b, lines)

        # per-row P diff counts
        dp = (mine(P) - g(r_w)).abs()
        per_bhm = (dp > 0).sum(dim=-1)
        cnt = {}
        for bh in range(BH):
            for m in range(Q_LEN):
                c = int(per_bhm[bh, m].item())
                k2 = (bh // H, bh % H, m)
                if c:
                    cnt[k2] = c
        lines.append(f"P per (b,h,m) diffs (top20): {sorted(cnt.items(), key=lambda x: -x[1])[:20]}")
        lines.append(f"P per m aggregate (all bh): {[(m, int(dp[:, m].sum().item())) for m in range(Q_LEN)]}")

        # final output comparison
        r_out = torch.matmul(r_w, vh).transpose(1, 2).reshape(BH, Q_LEN, D)
        mo = out.float().reshape(BH, Q_LEN, D)
        lines.append(f"OUT: max_abs={(mo - r_out).abs().max().item():.6g} match%={100.0 * (torch.isclose(mo, r_out, atol=0.0, rtol=0.0)).float().mean().item():.3f}")

        txt = "\n".join(lines)
        os.makedirs(os.path.dirname("/opt/workspace_card5/agent_workdir/diag_fa.txt"), exist_ok=True)
        with open("/opt/workspace_card5/agent_workdir/diag_fa.txt", "w") as f:
            f.write(txt)

    def forward(self, q, k, v, causal, window_left, window_right, softcap):
        B, Q_LEN, H, D = q.shape
        K_LEN = k.shape[1]
        repeats = H // k.shape[2]
        out = torch.empty_like(q)

        block_d = 1 << (D - 1).bit_length()
        block_m = 32 if block_d <= 128 else 16
        block_n = 32

        dbg = (
            q.dtype == torch.float16
            and (B, Q_LEN, H, D) == (1, 13, 4, 24)
            and (k.shape[1], k.shape[2]) == (19, 2)
            and float(softcap) == 20.0
        )
        if dbg:
            BUFSZ = 6 * Q_LEN * K_LEN + 2 * Q_LEN
            DBUF = torch.zeros(
                B * H * BUFSZ, dtype=torch.float32, device=q.device
            )
            o = 0
            off = []
            for _ in range(6):
                off.append(o)
                o += B * H * Q_LEN * K_LEN
            off.append(o)  # M
            o += B * H * Q_LEN
            off.append(o)  # L
        else:
            DBUF = torch.empty(1, dtype=torch.float32, device=q.device)
            off = [0] * 8

        grid = (triton.cdiv(Q_LEN, block_m), B * H)
        _flash_attn_fwd_kernel[grid](
            q, k, v, out,
            DBUF,
            off[0], off[1], off[2], off[3], off[4], off[5], off[6], off[7],
            B * H * Q_LEN * K_LEN, B * H * Q_LEN,
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
            dbg,
        )
        if dbg:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            self._dbg(q, k, v, causal, window_left, window_right,
                      float(softcap), out, DBUF)
        return out
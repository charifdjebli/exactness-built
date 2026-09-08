#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
#  ANCHORED COMPOSITION TEST v2 -- BOUNDED residual.
#
#  v1 failure (measured): the zero-init residual became a hostile
#  takeover -- |LM correction| hit 113-118 on zero-shot n=64 (LM's
#  degenerate yes-prior amplified 30x via unbounded linear extrapolation
#  on distribution-shifted features).  D1@n64 = "yes to everything".
#
#  v2 fix -- the design law, enforced architecturally:
#    * the channel logits are the ONLY unbounded path into the decision
#    * the LM writes a bounded correction:  BOUND * tanh(MLP(H))
#    * standardized LM features are clipped to [-4, 4]  (no z-score
#      explosion on OOD inputs)
#  The LM may nudge; it can never override.
#
#  python smollm_anchored.py          (~2 min; uses disk-cached hidden
#                                      states from smollm_holo_test.py)
# =====================================================================
import argparse, hashlib, json, math, os, pickle, random, sys
import torch
import torch.nn as nn
import torch.nn.functional as Fn

CAP = 8
T_MAX = 5
BOUND = 1.0          # max |LM correction| -- the LM's entire budget

# ---------- copied verbatim from smollm_holo_test (cache compat) ---- #
def serialize(edges, s, t, cap=CAP):
    es = " ".join(f"{u}->{v}" for u, v in edges)
    return (f"Edges: {es}\nQuestion: starting at {s} and following the "
            f"arrows, can you reach {t} in at most {cap} steps? "
            f"Answer (Yes/No):")

def cap_label(k):
    return 1 if k <= CAP else 0

def cnorm_vec(v, eps=1e-6):
    return v / v.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(eps)

def cnorm_mat(C, eps=1e-6):
    return C / C.abs().pow(2).sum((-2, -1), keepdim=True).sqrt().clamp_min(eps)

def fourier_codebook(n, D):
    ph = 2 * math.pi * torch.outer(torch.arange(n).float(),
                                   torch.arange(D).float()) / D
    return torch.polar(torch.full_like(ph, 1.0 / math.sqrt(D)), ph)

class HoloChannel(nn.Module):
    def __init__(self, n_nodes, d_vsa=256):
        super().__init__()
        V = fourier_codebook(n_nodes, d_vsa)
        self.register_buffer("theta", torch.angle(V))
    def codebook(self):
        return torch.polar(torch.ones_like(self.theta), self.theta)
    def forward(self, onehot, dst, s_idx, T):
        V = self.codebook()
        bundles = torch.matmul(onehot.transpose(1, 2).to(V.dtype), V[dst])
        bundles = cnorm_vec(bundles + V.unsqueeze(0))
        O = cnorm_mat(torch.einsum("bui,uj->bij", bundles, V.conj()))
        seed = cnorm_vec(V[s_idx])
        outs = []
        for _ in range(T):
            O = cnorm_mat(torch.matmul(O, O))
            F = torch.matmul(O, seed.unsqueeze(-1)).squeeze(-1)
            outs.append(torch.einsum("bd,xd->bx", F, V.conj()).real)
        return torch.stack(outs, dim=1)

def batch_of(items, n):
    B = len(items)
    e_max = max(4, max(len(it[0]) for it in items))
    src = torch.zeros(B, e_max, dtype=torch.long)
    dst = torch.zeros(B, e_max, dtype=torch.long)
    msk = torch.zeros(B, e_max)
    s_idx = torch.zeros(B, dtype=torch.long)
    t_idx = torch.zeros(B, dtype=torch.long)
    for b, (edges, s, t, lab, k) in enumerate(items):
        for j, (u, v) in enumerate(edges):
            src[b, j], dst[b, j], msk[b, j] = u, v, 1.0
        s_idx[b], t_idx[b] = s, t
    onehot = Fn.one_hot(src, n).float() * msk.unsqueeze(-1)
    return onehot, dst, s_idx, t_idx

@torch.no_grad()
def channel_features(items, n, d_vsa=256, t_max=T_MAX):
    ch = HoloChannel(n, d_vsa)
    feats = []
    for i0 in range(0, len(items), 64):
        chunk = items[i0:i0 + 64]
        onehot, dst, s_idx, t_idx = batch_of(chunk, n)
        eq = ch(onehot, dst, s_idx, t_max)
        for b in range(len(chunk)):
            t = t_idx[b].item()
            row = []
            for T in range(1, t_max + 1):
                a = eq[b, T - 1]
                rel = (a[t] / a.abs().max().clamp_min(1e-12)).item()
                row += [rel, math.log10(max(rel, 1e-16)),
                        1.0 if rel > 1e-4 else 0.0]
            feats.append(row)
    return torch.tensor(feats, dtype=torch.float32)

def load_hidden(outdir, tag, texts):
    key = hashlib.sha1(("|".join(texts[:2]) + f"#{len(texts)}")
                       .encode()).hexdigest()[:12]
    path = os.path.join(outdir, f"hid_{tag}_{key}.pt")
    if not os.path.exists(path):
        sys.exit(f"[cache] missing {path}\n  -> run once: python "
                 f"smollm_holo_test.py --stage probe   (full, not --quick)")
    blob = torch.load(path)
    if blob["n"] != len(texts) or blob["first"] != texts[0]:
        sys.exit(f"[cache] mismatch for {path} (quick vs full run?)")
    print(f"[cache] {tag}: {len(texts)} hidden states")
    return blob["H"].float()

# -------------------------- probes -------------------------------- #
def train_probe(Xtr, ytr, seed=0, width=64, iters=600, lr=3e-3):
    g = torch.Generator().manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    X = (Xtr - mu) / sd
    net = nn.Sequential(nn.Linear(X.shape[1], width), nn.GELU(),
                        nn.Linear(width, 2))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    y = torch.tensor(ytr, dtype=torch.long)
    for _ in range(iters):
        idx = torch.randint(0, X.shape[0], (256,), generator=g)
        loss = Fn.cross_entropy(net(X[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return net, mu, sd

@torch.no_grad()
def probe_logits(net, mu, sd, X):
    return net((X - mu) / sd)                       # (N, 2)

@torch.no_grad()
def bucket_acc(logit, y, idxs):
    pred = (logit[idxs] > 0).long()
    return (pred == y[idxs]).float().mean().item()

class BoundedResidual(nn.Module):
    """The LM's ENTIRE influence: BOUND * tanh(MLP(H)), zero-init.
    It may nudge the decision; it can never override the anchor."""
    def __init__(self, din, w=32):
        super().__init__()
        self.l1 = nn.Linear(din, w)
        self.l2 = nn.Linear(w, 1)
        nn.init.zeros_(self.l2.weight); nn.init.zeros_(self.l2.bias)
    def forward(self, x):
        return BOUND * torch.tanh(self.l2(Fn.gelu(self.l1(x)))).squeeze(-1)

# -------------------------- main ---------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--data-dir", default="v3_results")
    ap.add_argument("--eval-dir", default="v5_results")
    ap.add_argument("--lm-outdir", default="smollm_results")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    print("=" * 68)
    print(f" ANCHORED COMPOSITION v2 -- channel anchors, LM bounded to +/-{BOUND}")
    print("=" * 68)

    # ---- data, identical construction to stage_probe ----
    p = os.path.join(args.data_dir,
                     f"v3data_n32_lam{args.lam}_tr12000_ep40_s{args.seed}.pkl")
    with open(p, "rb") as f:
        tr, ep32, en32 = pickle.load(f)
    with open(os.path.join(args.eval_dir, "v5eval_a_n64_p25.pkl"), "rb") as f:
        ep64, en64 = pickle.load(f)
    tr = random.Random(args.seed).sample(tr, 1200)

    tr_tt = [(it, serialize(it[0], it[1], it[2])) for it in tr]
    sets32 = [(it, serialize(it[0], it[1], it[2]))
              for k in range(1, 13) for it in ep32.get(k, [])]
    sets64 = ([(it, serialize(it[0], it[1], it[2]))
               for k in range(1, 17) for it in ep64.get(k, [])] +
              [(it, serialize(it[0], it[1], it[2]))
               for k in (18, 21, 24, 28) for it in en64.get(k, [])])
    y_tr = torch.tensor([cap_label(it[4]) for it, _ in tr_tt])
    y32 = torch.tensor([cap_label(it[4]) for it, _ in sets32])
    y64 = torch.tensor([cap_label(it[4]) for it, _ in sets64])

    texts_tr = [s for _, s in tr_tt]
    texts_32 = [s for _, s in sets32]
    texts_64 = [s for _, s in sets64]

    Xch_tr = channel_features([it for it, _ in tr_tt], 32)
    Xch_32 = channel_features([it for it, _ in sets32], 32)
    Xch_64 = channel_features([it for it, _ in sets64], 64)
    H_tr = load_hidden(args.lm_outdir, "tr32", texts_tr)
    H_32 = load_hidden(args.lm_outdir, "ev32", texts_32)
    H_64 = load_hidden(args.lm_outdir, "ev64", texts_64)

    # ---- buckets ----
    b = {}
    for i, (it, _) in enumerate(sets32):
        b.setdefault(("n32", "pos" if it[4] <= 6 else
                      ("ood7-8" if it[4] <= 8 else "neg9-12")), []).append(i)
    for i, (it, _) in enumerate(sets64):
        kk = it[4]
        g = "pos1-8" if kk <= 8 else ("neg9-16" if kk <= 16 else "far18-28")
        b.setdefault(("n64", g), []).append(i)

    def report(name, logit32, logit64):
        row = {}
        for (nset, gname), idxs in b.items():
            L = logit32 if nset == "n32" else logit64
            y = y32 if nset == "n32" else y64
            row[f"{nset}:{gname}"] = bucket_acc(L, y, torch.tensor(idxs))
        print(f"  {name:26s} " + "  ".join(f"{k}={v:.2f}"
                                           for k, v in row.items()))
        return row

    results = {}

    # A0: channel-only anchor
    net_c, mu_c, sd_c = train_probe(Xch_tr, y_tr.tolist())
    L32 = probe_logits(net_c, mu_c, sd_c, Xch_32)[:, 1]
    L64 = probe_logits(net_c, mu_c, sd_c, Xch_64)[:, 1]
    results["A0 channel-anchor"] = report("A0 channel-anchor", L32, L64)

    # B0: LM-only
    net_h, mu_h, sd_h = train_probe(H_tr, y_tr.tolist())
    results["B0 SmolLM-only"] = report("B0 SmolLM-only",
        probe_logits(net_h, mu_h, sd_h, H_32)[:, 1],
        probe_logits(net_h, mu_h, sd_h, H_64)[:, 1])

    # C0: naive concat (the V4 failure, replicated)
    Xc_tr = torch.cat([H_tr, Xch_tr], 1)
    net_x, mu_x, sd_x = train_probe(Xc_tr, y_tr.tolist())
    results["C0 concat"] = report("C0 concat",
        probe_logits(net_x, mu_x, sd_x, torch.cat([H_32, Xch_32], 1))[:, 1],
        probe_logits(net_x, mu_x, sd_x, torch.cat([H_64, Xch_64], 1))[:, 1])

    # D1v2: ANCHORED + BOUNDED -- frozen channel logits + tanh-bounded
    # zero-init residual on clipped standardized SmolLM features.
    with torch.no_grad():
        chL_tr = probe_logits(net_c, mu_c, sd_c, Xch_tr)[:, 1]
        chL_32 = probe_logits(net_c, mu_c, sd_c, Xch_32)[:, 1]
        chL_64 = probe_logits(net_c, mu_c, sd_c, Xch_64)[:, 1]
    mu_r, sd_r = H_tr.mean(0), H_tr.std(0).clamp_min(1e-6)
    def zsc(H):                       # clip: no z-score explosion on OOD
        return ((H - mu_r) / sd_r).clamp(-4, 4)
    res = BoundedResidual(H_tr.shape[1])
    opt = torch.optim.AdamW(res.parameters(), lr=1e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(0)
    for it in range(400):
        idx = torch.randint(0, H_tr.shape[0], (256,), generator=g)
        logit = chL_tr[idx].detach() + res(zsc(H_tr[idx]))
        loss = Fn.binary_cross_entropy_with_logits(logit, y_tr[idx].float())
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        d32 = chL_32 + res(zsc(H_32))
        d64 = chL_64 + res(zsc(H_64))
    results["D1v2 anchored-bounded"] = report(
        "D1v2 anchored+bounded", d32, d64)
    mags = {}
    for (nset, gname), idxs in b.items():
        d = d32 if nset == "n32" else d64
        c = chL_32 if nset == "n32" else chL_64
        idxs_t = torch.tensor(idxs)
        mags[f"{nset}:{gname}"] = (d[idxs_t] - c[idxs_t]).abs().mean().item()
    print("  D1v2 |LM correction| (bound=%.1f): " % BOUND +
          "  ".join(f"{k}={v:.3f}" for k, v in mags.items()))

    # ---- verdicts (W4 stage) ----
    def g(r, k): return results[r].get(k, float("nan"))
    ood_keys = ["n32:ood7-8", "n64:pos1-8", "n64:neg9-16", "n64:far18-28"]
    w1 = all(g("D1v2 anchored-bounded", k) >= 0.99 for k in ood_keys)
    w2 = (max(g("C0 concat", "n64:neg9-16"), g("C0 concat", "n64:far18-28")) <= 0.30
          and all(g("D1v2 anchored-bounded", k) >= 0.95
                  for k in ("n64:neg9-16", "n64:far18-28")))
    maxmag = max(mags.values())
    print(f"""
[VERDICT W4a - bounded anchoring preserves OOD exactness] {'YES' if w1 else 'NO'}
[VERDICT W4b - concat destroys / bounded-anchoring saves]
   concat  n64 neg/far = {g('C0 concat','n64:neg9-16'):.2f}/{g('C0 concat','n64:far18-28'):.2f}
   bounded n64 neg/far = {g('D1v2 anchored-bounded','n64:neg9-16'):.2f}/{g('D1v2 anchored-bounded','n64:far18-28'):.2f}
   -> {'YES' if w2 else 'NO'}
[VERDICT W4c - division of labor] max |LM correction| = {maxmag:.3f} (bound {BOUND})
   -> the LM {'writes only bounded nudges where the primitive is exact (obedient residual)' if maxmag <= BOUND + 0.05 else 'unexpected: correction exceeds bound -- report back'}""")
    with open(os.path.join(args.lm_outdir, "anchored_v2.json"), "w") as f:
        json.dump(dict(results=results, mags=mags, W1=w1, W2=w2,
                       maxmag=maxmag, bound=BOUND), f, indent=2, default=str)
    print(f"[saved] {args.lm_outdir}/anchored_v2.json")

if __name__ == "__main__":
    main()
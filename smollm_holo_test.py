#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
#  SMOLLM-135M x HOLOGRAPHIC CHANNEL -- composition test (CPU)
#
#  What this tests (honestly):
#    NOT: "SmolLM got smarter" -- no LM weights are changed.
#    YES: (1) does SmolLM-135M natively solve capped multi-hop
#         reachability from text?  (V1: we predict it collapses ~k<=3)
#         (2) do the channel's exact, scale-free features COMPOSE with
#         frozen SmolLM features via a small probe, and transfer
#         zero-shot to 2x larger graphs where the LM's own features
#         do not?  (V2/V3/V4)
#
#  Task everywhere: "is t within 8 hops of s?" (cap-8). Train n=32
#  (pos dist 1-6, neg 10-14), eval OOD n=32 (7-8 pos, 9-12 neg) and
#  zero-shot n=64 (1-8 pos, 9-16 out-of-cap neg, 18-28 far neg).
#
#  python smollm_holo_test.py --stage all          (~20-30 min first run)
#  python smollm_holo_test.py --stage native       (~4 min)
#  python smollm_holo_test.py --stage probe        (~15-20 min)
#  --quick for a ~8-min smoke.  Needs: pip install transformers
#  First run downloads ~540MB (HuggingFaceTB/SmolLM-135M). Internet needed once.
# =====================================================================
import argparse, hashlib, json, math, os, pickle, random, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as Fn

DEVICE = torch.device("cpu")
CAP = 8          # the reachability cap (T=3 doubling ball)
T_MAX = 5        # channel steps -> features at balls 2,4,8,16,32

# ------------------------- graph machinery (standalone) ----------- #
def poisson(lam, rng):
    L = math.exp(-lam); k, p = 0, 1.0
    while True:
        p *= rng.random()
        if p <= L: return k
        k += 1

def bfs_dist(adj, src, n):
    dist = [-1] * n; dist[src] = 0; q = [src]
    while q:
        u = q.pop(0)
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1; q.append(v)
    return dist

def gen_far_pair(n, k, lam, rng, label):
    for _att in range(60):
        edges = set(); adj = [set() for _ in range(n)]
        def add(u, v):
            if u != v and (u, v) not in edges:
                edges.add((u, v)); adj[u].add(v); return True
            return False
        perm = list(range(n)); rng.shuffle(perm)
        chain = perm[:k + 1]; s, t = chain[0], chain[-1]
        for i in range(k): add(chain[i], chain[i + 1])
        budget, tries = int(round(lam * n)), 0
        while budget > 0 and tries < 300:
            tries += 1
            u, v = rng.randrange(n), rng.randrange(n)
            if add(u, v):
                if bfs_dist(adj, s, n)[t] != k:
                    edges.discard((u, v)); adj[u].discard(v)
                else:
                    budget -= 1
        d = bfs_dist(adj, s, n); guard = 0
        while guard < 200 and any(x < 0 for x in d):
            guard += 1
            x = rng.randrange(n)
            if d[x] >= 0: continue
            tg = [y for y in range(n) if d[y] >= 0]
            y = tg[rng.randrange(len(tg))]
            if add(y, x):
                nd = bfs_dist(adj, s, n)
                if nd[t] != k:
                    edges.discard((y, x)); adj[y].discard(x)
                else:
                    d = nd
        if all(x >= 0 for x in d) and bfs_dist(adj, s, n)[t] == k:
            return sorted(edges), s, t, label, k
    return None

def gen_set(n, ks, count, lam, rng, label):
    out, att = [], 0
    while len(out) < count and att < count * 60:
        att += 1
        g = gen_far_pair(n, ks[rng.randrange(len(ks))], lam, rng, label)
        if g: out.append(g)
    return out

# ------------------------- holographic channel -------------------- #
def cnorm_vec(v, eps=1e-6):
    return v / v.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(eps)

def cnorm_mat(C, eps=1e-6):
    return C / C.abs().pow(2).sum((-2, -1), keepdim=True).sqrt().clamp_min(eps)

def fourier_codebook(n, D):
    assert D >= n
    ph = 2 * math.pi * torch.outer(torch.arange(n).float(),
                                   torch.arange(D).float()) / D
    return torch.polar(torch.full_like(ph, 1.0 / math.sqrt(D)), ph)

class HoloChannel(nn.Module):
    """Frozen Fourier codebook, anchored doubling; eq (scale-free) view."""
    def __init__(self, n_nodes, d_vsa=256):
        super().__init__()
        self.n = n_nodes
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
        return torch.stack(outs, dim=1)                     # (B, T, n)

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
    """15 dims: per T in 1..t_max -> [rel, log10(rel), sup] at target t."""
    ch = HoloChannel(n, d_vsa)
    feats = []
    for i0 in range(0, len(items), 64):
        chunk = items[i0:i0 + 64]
        onehot, dst, s_idx, t_idx = batch_of(chunk, n)
        eq = ch(onehot, dst, s_idx, t_max)                  # (B, T, n)
        for b in range(len(chunk)):
            t = t_idx[b].item()
            row = []
            for T in range(1, t_max + 1):
                a = eq[b, T - 1]
                rel = (a[t] / a.abs().max().clamp_min(1e-12)).item()
                relc = max(rel, 1e-16)
                row += [rel, math.log10(relc), 1.0 if rel > 1e-4 else 0.0]
            feats.append(row)
    return torch.tensor(feats, dtype=torch.float32)

# ------------------------- serialization -------------------------- #
def serialize(edges, s, t, cap=CAP):
    es = " ".join(f"{u}->{v}" for u, v in edges)
    return (f"Edges: {es}\nQuestion: starting at {s} and following the "
            f"arrows, can you reach {t} in at most {cap} steps? "
            f"Answer (Yes/No):")

def cap_label(k):
    return 1 if k <= CAP else 0

# ------------------------- LM wrapper ----------------------------- #
class LM:
    def __init__(self, model_id, outdir):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.eval()
        print(f"[lm] loaded {model_id}  "
              f"({sum(p.numel() for p in self.model.parameters())/1e6:.0f}M params)")
        self.yes_id = self.tok(" Yes", add_special_tokens=False).input_ids[0]
        self.no_id = self.tok(" No", add_special_tokens=False).input_ids[0]
        self.outdir = outdir
        # smoke
        enc = self.tok("The capital of France is", return_tensors="pt")
        with torch.no_grad():
            g = self.model.generate(**enc, max_new_tokens=6, do_sample=False)
        print("[lm] smoke: " + repr(self.tok.decode(g[0][enc.input_ids.shape[1]:])))

    def _encode(self, texts):
        return self.tok(texts, padding=True, truncation=True,
                        max_length=512, return_tensors="pt")

    @torch.no_grad()
    def hidden(self, texts, bs=12, tag="x"):
        """Final-layer hidden state at the last real token, disk-cached."""
        key = hashlib.sha1(("|".join(texts[:2]) + f"#{len(texts)}")
                           .encode()).hexdigest()[:12]
        path = os.path.join(self.outdir, f"hid_{tag}_{key}.pt")
        if os.path.exists(path):
            blob = torch.load(path)
            if blob["n"] == len(texts) and blob["first"] == texts[0]:
                print(f"    [lm-hidden {tag}] loaded cache ({len(texts)})")
                return blob["H"]
        Hs = []
        t0 = time.time()
        for i0 in range(0, len(texts), bs):
            enc = self._encode(texts[i0:i0 + bs])
            out = self.model(**enc, output_hidden_states=True,
                             use_cache=False)
            h = out.hidden_states[-1]                       # (B, L, H)
            lens = enc["attention_mask"].sum(1) - 1
            Hs.append(h[torch.arange(h.shape[0]), lens])
            if (i0 // bs) % 10 == 0:
                print(f"    [lm-hidden {tag}] {i0 + bs}/{len(texts)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        H = torch.cat(Hs, 0).float()
        torch.save({"n": len(texts), "first": texts[0], "H": H}, path)
        return H

    @torch.no_grad()
    def p_yes(self, texts, bs=12):
        ps = []
        for i0 in range(0, len(texts), bs):
            enc = self._encode(texts[i0:i0 + bs])
            out = self.model(**enc, use_cache=False)
            lens = enc["attention_mask"].sum(1) - 1
            logits = out.logits[torch.arange(out.logits.shape[0]), lens]
            ly = logits[:, self.yes_id]; ln = logits[:, self.no_id]
            ps.append(torch.sigmoid(ly - ln))
        return torch.cat(ps)

# ------------------------- probe ---------------------------------- #
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

def probe_acc(net, mu, sd, X, y):
    with torch.no_grad():
        pred = net((X - mu) / sd).argmax(-1)
    return (pred == torch.tensor(y)).float().mean().item()

# ------------------------- stages --------------------------------- #
def stage_native(lm, args):
    print("""
[V1] NATIVE FRONTIER: SmolLM zero-shot, cap-8 reachability from raw text
     prediction: high acc at k=1-2, collapse by k=3-5; far pairs over-called
""")
    rng = random.Random(101)
    texts, ys, ks = [], [], []
    per_k = args.native_pairs
    for k in range(1, 9):
        items = gen_set(16, [k], per_k, 2.0, rng, 1)
        for e, s, t, lab, kk in items:
            texts.append(serialize(e, s, t)); ys.append(1); ks.append(kk)
    for e, s, t, lab, kk in gen_set(16, [10, 11, 12], per_k, 2.0, rng, 0):
        texts.append(serialize(e, s, t)); ys.append(0); ks.append(kk)
    with torch.no_grad():
        p = lm.p_yes(texts, bs=args.batch)
    pred = (p > 0.5).long()
    print("    k :  acc(pos)      n")
    for k in range(1, 9):
        sel = [i for i in range(len(ks)) if ks[i] == k]
        acc = sum(int(pred[i] == 1) for i in sel) / max(1, len(sel))
        print(f"    {k} :   {acc:.2f}        {len(sel)}")
    farsel = [i for i in range(len(ks)) if ks[i] >= 10]
    fpyes = sum(int(pred[i] == 1) for i in farsel) / max(1, len(farsel))
    fpos = [i for i in range(len(ks)) if 1 <= ks[i] <= 8]
    pos_acc = sum(int(pred[i] == 1) for i in fpos) / max(1, len(fpos))
    collapse = max([k for k in range(1, 9)
                    if sum(int(pred[i] == 1) for i in
                           [j for j in range(len(ks)) if ks[j] == k])
                    / max(1, len([j for j in range(len(ks)) if ks[j] == k])) >= 0.7]
                   or [0])
    print(f"    far(10-12) yes-rate: {fpyes:.2f}  (should be ~0; "
          f"high = overcalling reachability)")
    print(f"[VERDICT V1] positive acc={pos_acc:.2f}, collapse depth={collapse} "
          f"(prediction: <=4) -> "
          f"{'LM NATIVELY LIMITED: YES' if collapse <= 4 else 'LM stronger than predicted'}")

def load_or_gen_data(args):
    """Reuse v3 (n32) and v5 (n64) caches when present."""
    tr = ep32 = en32 = None
    p = os.path.join(args.data_dir,
                     f"v3data_n32_lam{args.lam}_tr12000_ep40_s{args.seed}.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f: tr, ep32, en32 = pickle.load(f)
        print(f"[data] loaded v3 n=32 cache (train={len(tr)})")
    p64 = os.path.join(args.eval_dir, "v5eval_a_n64_p25.pkl")
    if os.path.exists(p64):
        with open(p64, "rb") as f: ep64, en64 = pickle.load(f)
        print("[data] loaded v5 n=64 eval cache")
    else:
        rng = random.Random(7001)
        pairs = 6 if args.quick else 10
        ep64 = {k: gen_set(64, [k], pairs, args.lam, rng, 1) for k in range(1, 17)}
        en64 = {k: gen_set(64, [k], pairs, args.lam, rng, 0)
                for k in (18, 21, 24, 28)}
        print("[data] generated n=64 eval set")
    if tr is None:
        sys.exit("[data] v3 n=32 cache not found -- run campaign_v5.py --exp "
                 "sets once from this folder (it builds v3_results/).")
    # subsample train for LM-forward budget
    rng = random.Random(args.seed)
    if len(tr) > args.n32_train:
        tr = rng.sample(tr, args.n32_train)
    return tr, ep32, en32, ep64, en64

def stage_probe(lm, args):
    print("""
[V2-V4] FROZEN-FEATURE PROBES at n=32 -> zero-shot n=64 (task: cap-8)
  P1 channel-only (15 dims, no LM)   prediction: ~exact everywhere (V2)
  P2 SmolLM-only (576 dims, frozen)  prediction: OOD weak (V3)
  P3 SmolLM+channel (591 dims)       prediction: ~= channel OOD (V4)
""")
    tr, ep32, en32, ep64, en64 = load_or_gen_data(args)
    # ----- splits -----
    def items_texts(items):
        return [(it, serialize(it[0], it[1], it[2])) for it in items]
    tr_tt = items_texts(tr)
    y_tr = [cap_label(it[4]) for it, _ in tr_tt]
    sets32 = []
    for k in range(1, 13):
        sets32 += [(it, serialize(it[0], it[1], it[2])) for it in ep32.get(k, [])]
    y32 = [cap_label(it[4]) for it, _ in sets32]
    sets64, y64 = [], []
    for k in range(1, 17):
        for it in ep64.get(k, []):
            sets64.append((it, serialize(it[0], it[1], it[2])))
            y64.append(cap_label(it[4]))
    for k in (18, 21, 24, 28):
        for it in en64.get(k, []):
            sets64.append((it, serialize(it[0], it[1], it[2])))
            y64.append(0)

    # ----- channel features (fast, no LM) -----
    print("[feats] channel features...")
    Xch_tr = channel_features([it for it, _ in tr_tt], 32)
    Xch_32 = channel_features([it for it, _ in sets32], 32)
    Xch_64 = channel_features([it for it, _ in sets64], 64)

    # ----- LM features -----
    texts_tr = [s for _, s in tr_tt]
    texts_32 = [s for _, s in sets32]
    texts_64 = [s for _, s in sets64]
    H_tr = lm.hidden(texts_tr, bs=args.batch, tag="tr32")
    H_32 = lm.hidden(texts_32, bs=args.batch, tag="ev32")
    H_64 = lm.hidden(texts_64, bs=args.batch, tag="ev64")

    # ----- probes -----
    print("[probes] training 3 probes on identical splits...")
    probes = {
        "P1 channel-only": (Xch_tr, Xch_32, Xch_64),
        "P2 SmolLM-only": (H_tr, H_32, H_64),
        "P3 SmolLM+chan": (torch.cat([H_tr, Xch_tr], 1),
                           torch.cat([H_32, Xch_32], 1),
                           torch.cat([H_64, Xch_64], 1)),
    }
    y32t = torch.tensor(y32); y64t = torch.tensor(y64)
    # buckets
    b = {}
    for i, (it, _) in enumerate(sets32):
        b.setdefault(("n32", "pos" if it[4] <= 6 else
                      ("ood7-8" if it[4] <= 8 else "neg9-12")), []).append(i)
    for i, (it, _) in enumerate(sets64):
        kk = it[4]
        g = "pos1-8" if kk <= 8 else ("neg9-16" if kk <= 16 else "far18-28")
        b.setdefault(("n64", g), []).append(i)

    results = {}
    for name, (Xtr, X32, X64) in probes.items():
        net, mu, sd = train_probe(Xtr, y_tr, seed=0)
        r = {}
        for (nset, g), idxs in b.items():
            X = X32 if nset == "n32" else X64
            y = (y32t if nset == "n32" else y64t)[idxs]
            r[f"{nset}:{g}"] = probe_acc(net, mu, sd, X[idxs], y)
        results[name] = r
        print(f"  {name:18s} " + "  ".join(f"{k}={v:.2f}"
                                           for k, v in r.items()))

    def g(r, key): return results[r].get(key, float("nan"))
    v2 = max(g("P1 channel-only", "n32:ood7-8"),
             g("P1 channel-only", "n64:pos1-8")) >= 0.95
    v3 = (g("P2 SmolLM-only", "n32:ood7-8") < 0.85
          or g("P2 SmolLM-only", "n64:pos1-8") < 0.75)
    ood_p3 = 0.5 * (g("P3 SmolLM+chan", "n32:ood7-8")
                    + g("P3 SmolLM+chan", "n64:pos1-8"))
    ood_p2 = 0.5 * (g("P2 SmolLM-only", "n32:ood7-8")
                    + g("P2 SmolLM-only", "n64:pos1-8"))
    v4 = ood_p3 >= 0.90 and ood_p3 >= ood_p2 + 0.10
    print(f"""
[VERDICT V2 - channel transfers through a probe] {'YES' if v2 else 'NO'}
[VERDICT V3 - frozen SmolLM features lack multi-hop structure] {'YES' if v3 else 'NO'}
[VERDICT V4 - composition: LM+channel ~= channel OOD, beats LM-only] OOD
   P2={ood_p2:.2f} -> P3={ood_p3:.2f}  -> {'YES: the primitive composes with a pretrained LM it never trained with' if v4 else 'NO'}""")
    save = dict(native=None, probes=results, V2=v2, V3=v3, V4=v4)
    with open(os.path.join(args.outdir, "smollm_probe.json"), "w") as f:
        json.dump(save, f, indent=2, default=str)
    print(f"[saved] {args.outdir}/smollm_probe.json")

# ----------------------------- main ------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["native", "probe", "all"], default="all")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM-135M")
    ap.add_argument("--instruct", action="store_true",
                    help="use HuggingFaceTB/SmolLM-135M-Instruct")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-lm", action="store_true",
                    help="channel-only probe without the LM (no transformers needed)")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--n32-train", type=int, default=1200)
    ap.add_argument("--native-pairs", type=int, default=12)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data-dir", default="v3_results")
    ap.add_argument("--eval-dir", default="v5_results")
    ap.add_argument("--outdir", default="smollm_results")
    args = ap.parse_args()
    if args.instruct:
        args.model = "HuggingFaceTB/SmolLM-135M-Instruct"
    if args.quick:
        args.n32_train, args.native_pairs = 300, 6
    torch.manual_seed(args.seed); random.seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    os.makedirs(args.outdir, exist_ok=True)
    print("=" * 68)
    print(" SMOLLM-135M x HOLOGRAPHIC CHANNEL -- composition test (CPU)")
    print("=" * 68)
    print(f"threads={torch.get_num_threads()} stage={args.stage}"
          f"{' QUICK' if args.quick else ''} model={args.model}")
    lm = None
    if not args.skip_lm:
        try:
            import transformers  # noqa
        except ImportError:
            sys.exit("pip install transformers   (then rerun; first run "
                     "downloads ~540MB)")
        lm = LM(args.model, args.outdir)
    t0 = time.time()
    if args.stage in ("native", "all"):
        if lm is None:
            print("[native] skipped (--skip-lm)")
        else:
            stage_native(lm, args)
    if args.stage in ("probe", "all"):
        stage_probe(lm, args)
    print(f"\n[done in {time.time()-t0:.0f}s] Paste the full console output back.")

if __name__ == "__main__":
    main()
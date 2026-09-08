#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
#  CAMPAIGN v5.3 -- BALL SEMANTICS: the closing experiment.
#
#  New in v5.3 (on top of v5.2):
#   * collate() now carries TRUE graph distances ("ks")
#   * train_model(label_mode="ball")  : label_T = [dist <= 2^T] -- the
#     channel's NATIVE semantics. Far negatives teach the swallow:
#     "no" at T<=4, "yes" at T=5 (32 >= 18).
#   * Reasoner(feat_mode="eq")        : eq-only features (bounded,
#     scale-free) -- fixes the k~2^T boundary-shell miscalibration that
#     raw-decay features caused (k=16@T=5 = 0.20 in the final run).
#   * eval_ball_acc()                 : ball-accuracy per T
#   * exp_ball stage: train n=64 (T<=5, negs 18-24, ball labels,
#     eq features) -> zero-shot n=128 with the FULL moving-edge sweep:
#     FP(34-40)=0 at T<=5 (32<34), swallow-acc HIGH at T=6 (64>=34).
#
#  python campaign_v5.py --exp sets     (~1 min, no training)
#  python campaign_v5.py --exp ball     (~35-45 min)  <-- THE experiment
#  python campaign_v5.py --exp scale    (~35 min)
#  python campaign_v5.py --exp origin   (~19 min)
#  python campaign_v5.py --exp final    (~17 min, reuses n64 cache)
#  python campaign_v5.py --exp all
#  Smoke test: --quick.  Data cache: --data-dir v3_results.
# =====================================================================
import argparse, json, math, os, pickle, random, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as Fn

# ----------------------------- utils ------------------------------ #
def chunks(lst, sz):
    for i in range(0, len(lst), sz):
        yield lst[i:i + sz]

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

def bfs_ball(edges, s, radius):
    adj = {}
    for u, v in edges: adj.setdefault(u, []).append(v)
    seen = {s}; frontier = [s]
    for _ in range(radius):
        nxt = []
        for u in frontier:
            for v in adj.get(u, []):
                if v not in seen:
                    seen.add(v); nxt.append(v)
        frontier = nxt
        if not frontier: break
    return seen

def pick_symbols():
    try:
        s = "\u2588\u2593\u2592\u2591\u00b7"
        s.encode(sys.stdout.encoding or "utf-8")
        return list(s)
    except Exception:
        return list("#+-. ")

# ------------------------- data ----------------------------------- #
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

TRAIN_KS = [1, 2, 3, 4, 5, 6]
NEG_KS = [10, 11, 12, 13, 14]
EVAL_KS = list(range(1, 13))
OOD78 = [7, 8]

def get_train_data(args, seed):
    key = (f"v3data_n{args.n_nodes}_lam{args.lam}_tr{args.train_n}"
           f"_ep{args.eval_pairs}_s{seed}")
    path = os.path.join(args.data_dir, key + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            train, eval_pos, eval_neg = pickle.load(f)
        print(f"[data] loaded v3 cache ({len(train)} train, n={args.n_nodes})")
        return train, eval_pos, eval_neg
    sys.exit(f"[data] missing {path} -- run from the folder that has v3_results/")

def get_train_data_n64(args):
    """n=64 training set: positives dist 1-6, negatives dist 18-24.
    Shared with the 'final' stage cache (same file)."""
    ntr = args.train_n_final
    p64 = os.path.join(args.data_dir,
                       f"v52F_data_n64_lam{args.lam}_tr{ntr}_s{args.seed}.pkl")
    if os.path.exists(p64):
        with open(p64, "rb") as f:
            train64 = pickle.load(f)
        print(f"[data] loaded n=64 train cache ({len(train64)})")
        return train64
    rng = random.Random(args.seed * 31 + 7)
    train64, att = [], 0
    while len(train64) < ntr and att < ntr * 8:
        att += 1
        if rng.random() < 0.5:
            k = TRAIN_KS[rng.randrange(len(TRAIN_KS))]
            g = gen_far_pair(64, k, args.lam, rng, 1)
        else:
            k = [18, 20, 22, 24][rng.randrange(4)]
            g = gen_far_pair(64, k, args.lam, rng, 0)
        if g: train64.append(g)
        if att % 2000 == 0:
            print(f"    gen n=64 {len(train64)}/{ntr}", flush=True)
    os.makedirs(args.data_dir, exist_ok=True)
    with open(p64, "wb") as f:
        pickle.dump(train64, f)
    print(f"[data] built n=64 train={len(train64)}")
    return train64

def get_eval_graphs(args, n, pos_ks, neg_ks, tag, pairs):
    path = os.path.join(args.outdir, f"v5eval_{tag}_n{n}_p{pairs}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            ep, en = pickle.load(f)
        print(f"[data] loaded eval cache n={n} ({tag})")
        return ep, en
    rng = random.Random(1000 * n + len(tag) * 131 + args.seed)
    ep = {}
    for k in pos_ks:
        out, att = [], 0
        while len(out) < pairs and att < pairs * 40:
            att += 1
            g = gen_far_pair(n, k, args.lam, rng, 1)
            if g: out.append(g)
        ep[k] = out
        print(f"    gen n={n} pos k={k}: {len(out)}", flush=True)
    en = {}
    for k in neg_ks:
        out, att = [], 0
        while len(out) < pairs and att < pairs * 60:
            att += 1
            g = gen_far_pair(n, k, args.lam, rng, 0)
            if g: out.append(g)
        en[k] = out
        print(f"    gen n={n} neg L={k}: {len(out)}", flush=True)
    os.makedirs(args.outdir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((ep, en), f)
    print(f"[data] built eval n={n} ({tag})")
    return ep, en

def collate(items, n, device):
    B = len(items)
    e_max = max(4, max(len(it[0]) for it in items))
    src = torch.zeros(B, e_max, dtype=torch.long)
    dst = torch.zeros(B, e_max, dtype=torch.long)
    msk = torch.zeros(B, e_max)
    s_idx = torch.zeros(B, dtype=torch.long); t_idx = torch.zeros(B, dtype=torch.long)
    y = torch.zeros(B, dtype=torch.long)
    ks = torch.zeros(B, dtype=torch.long)
    for b, (edges, s, t, lab, k) in enumerate(items):
        for j, (u, v) in enumerate(edges):
            src[b, j], dst[b, j], msk[b, j] = u, v, 1.0
        s_idx[b], t_idx[b], y[b], ks[b] = s, t, lab, k
    adj = torch.zeros(B, n * n)
    adj.scatter_add_(1, src * n + dst, msk)
    adj = adj.view(B, n, n) + torch.eye(n).unsqueeze(0)
    adj = (adj > 0).float()
    return {"src": src.to(device), "dst": dst.to(device), "msk": msk.to(device),
            "adj": adj.to(device), "s_idx": s_idx.to(device),
            "t_idx": t_idx.to(device), "y": y.to(device),
            "ks": ks.to(device)}

# -------------------- holographic channel v5.3 -------------------- #
def cnorm_vec(v, eps=1e-6):
    return v / v.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(eps)

def cnorm_mat(C, eps=1e-6):
    return C / C.abs().pow(2).sum((-2, -1), keepdim=True).sqrt().clamp_min(eps)

def fourier_codebook(n, D):
    assert D >= n, "Fourier codebook needs D >= n"
    ph = 2 * math.pi * torch.outer(torch.arange(n).float(),
                                   torch.arange(D).float()) / D
    return torch.polar(torch.full_like(ph, 1.0 / math.sqrt(D)), ph)

class HoloChannel(nn.Module):
    """Anchored doubling O_t = B^(2^t). Returns A TUPLE:
         (scores_equalized, scores_raw)     each (B, T, n).
    eq view: per-atom gain normalized -> scale-free, support exact at
             every n (transferable). raw view: graded decay (learnable).
    codebook: 'fourier' (frozen) | 'random' (frozen)
              | 'learned' (random init, trainable)
              | 'learned4' (fourier init, trainable)."""

    def __init__(self, n_nodes, d_vsa, codebook="fourier", cleanup=None,
                 equalize=True):
        super().__init__()
        self.n, self.cleanup, self.kind = n_nodes, cleanup, codebook
        self.equalize = equalize
        if codebook == "fourier":
            theta = torch.angle(fourier_codebook(n_nodes, d_vsa))
            self.register_buffer("theta", theta)
        elif codebook == "random":
            theta = torch.empty(n_nodes, d_vsa).uniform_(0, 2 * math.pi)
            self.register_buffer("theta", theta)
        elif codebook == "learned":
            theta = torch.empty(n_nodes, d_vsa).uniform_(0, 2 * math.pi)
            self.theta = nn.Parameter(theta)
        elif codebook == "learned4":
            theta = torch.angle(fourier_codebook(n_nodes, d_vsa))
            self.theta = nn.Parameter(theta)
        else:
            raise ValueError(codebook)

    def codebook(self):
        return torch.polar(torch.ones_like(self.theta), self.theta)

    def forward(self, onehot, dst, s_idx, T):
        V = self.codebook()
        bundles = torch.matmul(onehot.transpose(1, 2).to(V.dtype), V[dst])
        bundles = cnorm_vec(bundles + V.unsqueeze(0))          # self-loops
        O = cnorm_mat(torch.einsum("bui,uj->bij", bundles, V.conj()))
        seed = cnorm_vec(V[s_idx])
        outs_eq, outs_raw = [], []
        for _ in range(T):
            O = cnorm_mat(torch.matmul(O, O))                  # ONE squaring
            F_raw = torch.matmul(O, seed.unsqueeze(-1)).squeeze(-1)
            if self.equalize:
                Ob = torch.einsum("xd,bde,ye->bxy", V.conj(), O, V)
                Ob = Ob / Ob.abs().pow(2).sum(1, keepdim=True).sqrt().clamp_min(1e-9)
                Oeq = torch.einsum("xd,bxy,ye->bde", V, Ob, V.conj())
                F_eq = torch.matmul(Oeq, seed.unsqueeze(-1)).squeeze(-1)
            else:
                F_eq = F_raw
            a_eq = torch.einsum("bd,xd->bx", F_eq, V.conj()).real
            a_raw = torch.einsum("bd,xd->bx", F_raw, V.conj()).real
            outs_eq.append(a_eq); outs_raw.append(a_raw)
        return torch.stack(outs_eq, dim=1), torch.stack(outs_raw, dim=1)

@torch.no_grad()
def coherence(ch):
    """Normalized Gram coherence of the codebook (max, mean off-diagonal)."""
    V = ch.codebook()
    Vn = V / V.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-9)
    G = (Vn @ Vn.conj().transpose(0, 1)).abs()
    n = V.shape[0]
    off = G[~torch.eye(n, dtype=torch.bool)]
    return off.max().item(), off.mean().item()

# ------------------------- self-test ------------------------------ #
def self_test(device):
    n, D = 10, 64
    edges = [(i, i + 1) for i in range(7)] + [(3, 8), (9, 0)]
    E = len(edges)
    onehot = torch.zeros(1, E, n); dst = torch.zeros(1, E, dtype=torch.long)
    for j, (u, v) in enumerate(edges):
        onehot[0, j, u] = 1.0; dst[0, j] = v
    ch = HoloChannel(n, D, "fourier", equalize=True).to(device)
    with torch.no_grad():
        out_eq, out_raw = ch(onehot.to(device), dst.to(device),
                             torch.zeros(1, dtype=torch.long, device=device), 3)
        sc = out_eq[0]
        sc_r = out_raw[0]
    ok = True; msgs = []
    def chk(name, val, lo, hi=None):
        nonlocal ok
        good = (val >= lo) if hi is None else (lo <= val <= hi)
        if not good: ok = False
        msgs.append(f"    {name}: {val:.5f} {'OK' if good else 'FAIL'}")
    chk("T=1 node1 in-ball (eq) ", sc[0, 1].item(), 0.05)
    chk("T=1 node3 OUT      (eq) ", sc[0, 3].item(), 0.0, 0.02)
    chk("T=2 node3 in-ball  (eq) ", sc[1, 3].item(), 0.05)
    chk("T=2 node5 OUT      (eq) ", sc[1, 5].item(), 0.0, 0.02)
    chk("T=3 node7 in-ball  (eq) ", sc[2, 7].item(), 0.03)
    chk("T=3 node9 unreach  (eq) ", sc[2, 9].item(), 0.0, 0.02)
    ratio = (sc[2, 7] / sc[2, 1].clamp_min(1e-9)).item()
    chk("T=3 eq ratio node7/node1", ratio, 0.25, 4.0)
    chk("T=3 node7 in-ball (raw)", sc_r[2, 7].item(), 0.03)
    chk("T=3 node9 unreach (raw)", sc_r[2, 9].item(), 0.0, 0.02)
    print("[SELF-TEST] anchored operator, eq+raw views:")
    print("\n".join(msgs))
    if not ok:
        print("[SELF-TEST] FAILED -- paste back; do not train."); sys.exit(1)
    print("[SELF-TEST] PASSED.\n")

# ----------------------- recurrent core --------------------------- #
class Block(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ls1 = nn.Parameter(torch.full((d,), 1e-4))
        self.ls2 = nn.Parameter(torch.full((d,), 1e-4))
    def forward(self, x, adj):
        B, N, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, N, self.h, self.dh).transpose(1, 2)
        k = k.view(B, N, self.h, self.dh).transpose(1, 2)
        v = v.view(B, N, self.h, self.dh).transpose(1, 2)
        att = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        att = att + (1.0 - adj).unsqueeze(1) * (-1e9)
        att = att.softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.ls1 * self.proj(o)
        return x + self.ls2 * self.ffn(self.ln2(x))

class Reasoner(nn.Module):
    """v5.3: identity-free core; hybrid or eq-only channel features."""
    def __init__(self, n_nodes, d_model, d_vsa, channel_mode="none",
                 max_steps=8, clamp_steps=3, codebook="fourier",
                 equalize=True, feat_mode="both"):
        super().__init__()
        assert channel_mode in ("none", "holo", "shuffle")
        assert feat_mode in ("both", "eq")
        self.n = n_nodes
        self.channel_mode = channel_mode
        self.clamp = clamp_steps
        self.feat_mode = feat_mode
        self.n_feat = 3 if feat_mode == "eq" else 6
        self.flag = nn.Linear(2, d_model)
        self.step_emb = nn.Parameter(torch.randn(max_steps, d_model) * 0.02)
        self.inj = nn.Linear(self.n_feat, d_model)
        self.ls_inj = nn.Parameter(torch.ones(d_model))
        nn.init.constant_(self.ls_inj, 0.5)
        if channel_mode != "none":
            self.holo = HoloChannel(n_nodes, d_vsa, codebook, equalize=equalize)
        self.block = Block(d_model)
        self.gate = nn.Linear(2 * d_model, d_model)
        nn.init.constant_(self.gate.bias, -2.0)
        self.readout = nn.Sequential(nn.Linear(3 * d_model, d_model),
                                     nn.GELU(), nn.Linear(d_model, 2))

    def forward(self, src, dst, msk, adj, s_idx, t_idx, T):
        B, N = adj.shape[0], adj.shape[-1]
        d = self.inj.out_features
        assert T <= self.step_emb.shape[0]
        scores_eq = scores_raw = None
        if self.channel_mode != "none":
            if self.channel_mode == "shuffle":
                s_use = (torch.roll(s_idx, 1, dims=0) if B > 1
                         else (s_idx + 1) % N)
            else:
                s_use = s_idx
            onehot = Fn.one_hot(src, N).float() * msk.unsqueeze(-1)
            scores_eq, scores_raw = self.holo(onehot, dst, s_use, T)
        flags = torch.zeros(B, N, 2, device=adj.device)
        flags[torch.arange(B, device=adj.device), s_idx, 0] = 1.0
        flags[torch.arange(B, device=adj.device), t_idx, 1] = 1.0
        H = self.flag(flags)
        for t in range(T):
            if self.channel_mode != "none":
                a_eq = scores_eq[:, t]
                def feats(a):
                    am = a.amax(-1, keepdim=True).clamp_min(1e-12)
                    rel = a / am
                    lg = torch.where(a > 1e-12, torch.log10(a.clamp_min(1e-16)),
                                     torch.full_like(a, -16.0))
                    return rel, lg, (rel > 1e-4).float()
                rel_e, lg_e, sup_e = feats(a_eq)
                if self.feat_mode == "eq":
                    feat = torch.stack([rel_e, lg_e, sup_e], dim=-1)
                else:
                    a_raw = scores_raw[:, t]
                    rel_r, lg_r, sup_r = feats(a_raw)
                    feat = torch.stack([rel_e, lg_e, sup_e,
                                        rel_r, lg_r, sup_r], dim=-1)
            else:
                feat = torch.zeros(B, N, self.n_feat, device=H.device)
            H = H + self.step_emb[min(t, self.clamp - 1)] \
                  + self.ls_inj * self.inj(feat)
            Hc = self.block(H, adj)
            z = torch.sigmoid(self.gate(torch.cat([Hc, H], dim=-1)))
            H = z * Hc + (1 - z) * H
        Ht = H.gather(1, t_idx.view(B, 1, 1).expand(B, 1, d)).squeeze(1)
        Hs = H.gather(1, s_idx.view(B, 1, 1).expand(B, 1, d)).squeeze(1)
        return self.readout(torch.cat([Ht, Hs, H.mean(dim=1)], dim=-1)).clamp(-30, 30)

def extend_model(model, n_new, d_vsa, device):
    """Swap in the constructive Fourier codebook for a larger graph."""
    if getattr(model, "holo", None) is not None:
        assert model.holo.kind == "fourier", "only frozen fourier extends"
        eq = getattr(model.holo, "equalize", True)
        model.holo = HoloChannel(n_new, d_vsa, "fourier", equalize=eq).to(device)
    model.n = n_new
    return model

# -------------------------- training ------------------------------ #
def train_model(model, name, train_set, cfg, rng, label_mode="reach"):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total, warm = cfg["iters"], 50
    def lr_at(s):
        if s < warm: return s / max(1, warm)
        p = (s - warm) / max(1, total - warm)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    t0, rl, rc, rn = time.time(), 0.0, 0, 0
    for it in range(1, total + 1):
        idx = rng.sample(range(len(train_set)), cfg["batch"])
        bt = collate([train_set[i] for i in idx], cfg["n"], cfg["device"])
        T = rng.randint(1, cfg["T_train_max"])
        logits = model(bt["src"], bt["dst"], bt["msk"], bt["adj"],
                       bt["s_idx"], bt["t_idx"], T)
        if label_mode == "ball":
            lab = (bt["ks"] <= 2 ** T).long()      # native channel semantics
        else:
            lab = bt["y"]
        loss = Fn.cross_entropy(logits, lab)
        if not torch.isfinite(loss): continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        rl += loss.item(); rn += 1
        rc += (logits.argmax(-1) == lab).sum().item()
        if it % 200 == 0 or it == total:
            print(f"    [{name}] iter {it:5d}/{total}  loss {rl/max(1,rn):.4f}  "
                  f"acc {rc/max(1,rn*cfg['batch']):.3f}  ({time.time()-t0:.0f}s)")
            rl, rc, rn = 0.0, 0, 0
    return model

def get_or_train(name, key, builder, train_set, cfg, train_seed, outdir, force,
                 label_mode="reach"):
    path = os.path.join(outdir, f"model_{key}.pt")
    torch.manual_seed(train_seed - 11)
    model = builder()
    if not force and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=cfg["device"]))
        print(f"    [{name}] loaded cached weights")
        return model
    model = train_model(model, name, train_set, cfg, random.Random(train_seed),
                        label_mode=label_mode)
    torch.save(model.state_dict(), path)
    return model

# ----------------------------- eval ------------------------------- #
@torch.no_grad()
def eval_acc(model, items, T, cfg):
    model.eval(); cp = pp = cokn = pn = 0
    for chn in chunks(items, 200):
        bt = collate(chn, cfg["n"], cfg["device"])
        logits = model(bt["src"], bt["dst"], bt["msk"], bt["adj"],
                       bt["s_idx"], bt["t_idx"], T)
        pred = logits.argmax(-1); ispos = bt["y"] == 1
        cp += ((pred == 1) & ispos).sum().item(); pp += ispos.sum().item()
        cokn += ((pred == 0) & ~ispos).sum().item(); pn += (~ispos).sum().item()
    return cp / max(1, pp), cokn / max(1, pn)

@torch.no_grad()
def eval_ball_acc(model, items, T, cfg):
    """Ball accuracy: pred must equal [true_dist <= 2^T]."""
    model.eval(); cok = tot = 0
    for chn in chunks(items, 200):
        bt = collate(chn, cfg["n"], cfg["device"])
        logits = model(bt["src"], bt["dst"], bt["msk"], bt["adj"],
                       bt["s_idx"], bt["t_idx"], T)
        pred = logits.argmax(-1)
        lab = (bt["ks"] <= 2 ** T).long()
        cok += (pred == lab).sum().item(); tot += len(chn)
    return cok / max(1, tot)

@torch.no_grad()
def eval_ball_neg(model, items, T, cfg):
    """On negatives only: (ball-accuracy, yes-rate). Under ball semantics
    a 'yes' on a dist>L negative is CORRECT once 2^T >= L."""
    model.eval(); cok = tot = ny = 0
    for chn in chunks(items, 200):
        bt = collate(chn, cfg["n"], cfg["device"])
        logits = model(bt["src"], bt["dst"], bt["msk"], bt["adj"],
                       bt["s_idx"], bt["t_idx"], T)
        pred = logits.argmax(-1)
        lab = (bt["ks"] <= 2 ** T).long()
        cok += (pred == lab).sum().item(); tot += len(chn)
        ny += (pred == 1).sum().item()
    return cok / max(1, tot), ny / max(1, tot)

@torch.no_grad()
def eval_heatmap(model, ks, eval_pos, neg_list, cfg, T_max, label=""):
    mat = []
    for T in range(1, T_max + 1):
        row = [eval_acc(model, eval_pos[k], T, cfg)[0] for k in ks]
        fp = 1 - eval_acc(model, neg_list, T, cfg)[1]
        mat.append(row)
        print(f"  {label} T={T}: " + " ".join(f"{a:.2f}" for a in row)
              + f" | FP {fp:.2f}")
    return mat

@torch.no_grad()
def eval_ball_heatmap(model, ks, eval_pos, neg_list, cfg, T_max, label=""):
    mat = []
    for T in range(1, T_max + 1):
        row = [eval_ball_acc(model, eval_pos[k], T, cfg) for k in ks]
        nacc, ny = eval_ball_neg(model, neg_list, T, cfg)
        mat.append(row)
        print(f"  {label} T={T}: " + " ".join(f"{a:.2f}" for a in row)
              + f" | neg-acc {nacc:.2f} (yes-rate {ny:.2f})")
    return mat

@torch.no_grad()
def vsa_eval_rel(channel, items, T, cfg, rel_thr=1e-4):
    ap = aok = tp = tn = 0
    for chn in chunks(items, 100):
        bt = collate(chn, cfg["n"], cfg["device"])
        onehot = Fn.one_hot(bt["src"], cfg["n"]).float() * bt["msk"].unsqueeze(-1)
        sc_eq, _sc_raw = channel(onehot, bt["dst"], bt["s_idx"], T)
        sc = sc_eq[:, T - 1]
        rel = sc / sc.amax(-1, keepdim=True).clamp_min(1e-12)
        st = rel.gather(1, bt["t_idx"].view(-1, 1)).squeeze(1)
        pred = (st > rel_thr).long(); pos = bt["y"] == 1
        ap += (pred[pos] == 1).sum().item(); tp += pos.sum().item()
        aok += (pred[~pos] == 0).sum().item(); tn += (~pos).sum().item()
    return ap / max(1, tp), aok / max(1, tn)

# -------------------------- reporting ----------------------------- #
def heat_sym(acc, syms):
    for cut, s in zip((0.95, 0.85, 0.70, 0.55), syms):
        if acc >= cut: return s
    return syms[4]

def print_heatmap(name, mat, ks, syms):
    h = len(ks) // 2
    print(f"\n  HEATMAP - {name}")
    print("      T\\k " + "".join(f"{k:>4}" for k in ks[:h]) + "   |" +
          "".join(f"{k:>4}" for k in ks[h:]))
    for i in range(len(mat)):
        c1 = "".join(f"   {heat_sym(mat[i][j], syms)} " for j in range(h))
        c2 = "".join(f"  {heat_sym(mat[i][j], syms)} " for j in range(h, len(ks)))
        print(f"    T={i+1:<2d} {c1} |{c2}")

def frontier(mat, ks, thr=0.9):
    return {k: next((i + 1 for i, a in enumerate([mat[i][j] for i in range(len(mat))])
                     if a >= thr), None) for j, k in enumerate(ks)}

def predict_log(k):
    return max(1, math.ceil(math.log2(k)))

def ood_mean(mat, T, ks, group):
    row = mat[T - 1]
    return sum(row[ks.index(k)] for k in group) / len(group)

def save_json(outdir, name, obj):
    with open(os.path.join(outdir, f"{name}.json"), "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[saved] {outdir}/{name}.json")

# -------------------------- experiments --------------------------- #
def exp_sets(args, cfg, outdir):
    t0 = time.time()
    print("""
[PREREGISTERED -- BROADCAST & DISTANCE (zero training)]
  B1: Jaccard >= 0.95, n=32 AND n=64
  D1: T* == ceil(log2 k) for ~all pairs
""")
    train, ep32, en32 = get_train_data(args, args.seed)
    items32 = [it for k in (3, 5, 7, 9, 11) for it in ep32[k]][:200]
    ch32 = HoloChannel(32, args.d_vsa, "fourier")
    print("[B1 n=32] set decode Jaccard (eq view):")
    for T in (1, 2, 3, 4):
        js = []
        with torch.no_grad():
            for it in items32:
                edges, s, t, lab, k = it
                onehot = torch.zeros(1, len(edges), 32)
                dst = torch.zeros(1, len(edges), dtype=torch.long)
                for j, (u, v) in enumerate(edges):
                    onehot[0, j, u] = 1.0; dst[0, j] = v
                sc_eq, _ = ch32(onehot, dst, torch.tensor([s]), T)
                sc = sc_eq[0, T - 1]
                rel = sc / sc.abs().max().clamp_min(1e-12)
                pred = set(i for i in range(32) if rel[i] > 1e-4)
                truth = bfs_ball(edges, s, 2 ** T)
                js.append(len(pred & truth) / max(1, len(pred | truth)))
        print(f"  T={T} (radius {2**T:2d}): mean Jaccard {sum(js)/len(js):.4f}")
    print("[D1 n=32] distance bisection (eq view):")
    ok = tot = 0
    with torch.no_grad():
        for it in items32:
            edges, s, t, lab, k = it
            onehot = torch.zeros(1, len(edges), 32)
            dst = torch.zeros(1, len(edges), dtype=torch.long)
            for j, (u, v) in enumerate(edges):
                onehot[0, j, u] = 1.0; dst[0, j] = v
            sc_eq, _ = ch32(onehot, dst, torch.tensor([s]), 5)
            rel = sc_eq[0] / sc_eq[0].amax(-1, keepdim=True).clamp_min(1e-12)
            tstar = next((T for T in range(1, 6) if rel[T - 1][t].item() > 1e-4), None)
            if tstar is not None:
                tot += 1; ok += int(tstar == predict_log(k))
    print(f"  T* == ceil(log2 k): {ok}/{tot} = {ok/max(1,tot):.3f}")
    print("[B1 n=64] set decode Jaccard (extended codebook):")
    ep64, _ = get_eval_graphs(args, 64, list(range(1, 17)),
                              list(range(18, 29)), "a", 25)
    items64 = [it for k in (5, 9, 13, 16) for it in ep64[k]][:120]
    ch64 = HoloChannel(64, args.d_vsa, "fourier")
    for T in (3, 4, 5):
        js = []
        with torch.no_grad():
            for it in items64:
                edges, s, t, lab, k = it
                onehot = torch.zeros(1, len(edges), 64)
                dst = torch.zeros(1, len(edges), dtype=torch.long)
                for j, (u, v) in enumerate(edges):
                    onehot[0, j, u] = 1.0; dst[0, j] = v
                sc_eq, _ = ch64(onehot, dst, torch.tensor([s]), T)
                sc = sc_eq[0, T - 1]
                rel = sc / sc.abs().max().clamp_min(1e-12)
                pred = set(i for i in range(64) if rel[i] > 1e-4)
                truth = bfs_ball(edges, s, 2 ** T)
                js.append(len(pred & truth) / max(1, len(pred | truth)))
        print(f"  T={T} (radius {2**T:2d}): mean Jaccard {sum(js)/len(js):.4f}")
    save_json(outdir, "sets", dict(note="see console"))
    print(f"\n[stage 'sets' finished in {time.time()-t0:.0f}s]")

def exp_ball(args, cfg, outdir, force):
    """THE closing experiment: train n=64 with BALL labels + eq features,
    then zero-shot n=128 with the full moving-edge sweep."""
    t0 = time.time()
    print("""
[PREREGISTERED -- BALL v5.3: train n=64 (T<=5, negs 18-24, ball labels, eq feats)]
  B1 in-dist n=64   : staircase ball-acc; k<=14 ~1.0 by T=5;
                      k=16 @T=5 >= 0.85 (eq fixes the boundary shell);
                      neg-acc 1.0 at T<=4, ~1.0 at T=5 (swallow learned)
  B2 zero-shot n=128: k<=16 ball-acc >= 0.75 @T=5;
                      neg(34-40): yes-rate 0.00 at T<=5 (32<34),
                      swallow-acc HIGH at T=6 (64>=34)  <- moving edge at 4x
  B3 baseline       : ball-acc on k>=9 < 0.6 at ANY T<=8 (frontier trap)
  B4 shuffle        : ~chance
""")
    syms = pick_symbols()
    dm, dv = args.d_model, args.d_vsa
    train64 = get_train_data_n64(args)
    cfg64 = dict(cfg, n=64, T_train_max=5)

    baseB = get_or_train("baseB", f"v53_baseB_i{cfg['iters']}_T6_s{args.seed}",
                         lambda: Reasoner(64, dm, dv, "none", 8, 6),
                         train64, dict(cfg64, T_train_max=6), args.seed + 11,
                         outdir, force, label_mode="ball")
    holoB = get_or_train("holoB", f"v53_holoB_v{dv}_i{cfg['iters']}_T5_s{args.seed}",
                         lambda: Reasoner(64, dm, dv, "holo", 8, 5, "fourier",
                                          feat_mode="eq"),
                         train64, cfg64, args.seed + 11,
                         outdir, force, label_mode="ball")
    shufB = get_or_train("shufB", f"v53_shufB_v{dv}_i{cfg['iters']}_T5_s{args.seed}",
                         lambda: Reasoner(64, dm, dv, "shuffle", 8, 5, "fourier",
                                          feat_mode="eq"),
                         train64, cfg64, args.seed + 11,
                         outdir, force, label_mode="ball")

    ep64, en64 = get_eval_graphs(args, 64, list(range(1, 17)),
                                 list(range(18, 29)), "a", 25)
    en64_list = [it for k in range(18, 29) for it in en64[k]]
    ks64 = list(range(1, 17))

    print("\n[B1 in-dist n=64] holoB ball-acc T<=6")
    mhb = eval_ball_heatmap(holoB, ks64, ep64, en64_list, cfg64, 6, "[holoB64]")
    print("[B3 in-dist n=64] baseB ball-acc T<=8")
    mbb = eval_ball_heatmap(baseB, ks64, ep64, en64_list, dict(cfg64, T_train_max=6),
                            8, "[baseB64]")
    print("[B4 in-dist n=64] shufB ball-acc T<=6")
    msb = eval_ball_heatmap(shufB, ks64, ep64, en64_list, cfg64, 6, "[shufB64]")
    print_heatmap("holoB n=64 (ball-acc)", mhb, ks64, syms)
    print_heatmap("baseB n=64 (ball-acc)", mbb, ks64, syms)
    k16_64 = mhb[4][ks64.index(16)] if len(mhb) >= 5 else 0.0
    negacc_T4 = eval_ball_neg(holoB, en64_list, 4, cfg64)[0]
    negacc_T5 = eval_ball_neg(holoB, en64_list, 5, cfg64)[0]
    b_k9plus = max(max(mbb[T - 1][ks64.index(k)] for T in range(1, 9))
                   for k in (9, 12, 16))
    ok_b1 = (k16_64 >= 0.85) and (negacc_T4 >= 0.95) and (negacc_T5 >= 0.95)
    print(f"\n  holoB64: k=16@T=5 ball-acc={k16_64:.2f}  "
          f"neg-acc@T4={negacc_T4:.2f} @T5={negacc_T5:.2f}")
    print(f"[VERDICT B1 - in-dist ball staircase] -> {'YES' if ok_b1 else 'NO'}")

    print("\n[B2 ZERO-SHOT n=128: same weights, extended codebook]")
    ep128, en128 = get_eval_graphs(args, 128, list(range(1, 17)),
                                   [34, 37, 40], "b", 10)
    en128_list = [it for k in (34, 37, 40) for it in en128[k]]
    cfg128 = dict(cfg, n=128)
    holo128 = extend_model(holoB, 128, dv, cfg["device"])
    mh128 = []
    for T in range(1, 7):
        row = [eval_ball_acc(holo128, ep128[k], T, cfg128) for k in ks64]
        nacc, ny = eval_ball_neg(holo128, en128_list, T, cfg128)
        mh128.append(row)
        print(f"  [holoB128] T={T}: " + " ".join(f"{a:.2f}" for a in row)
              + f" | neg(34-40) acc {nacc:.2f} (yes-rate {ny:.2f})")
    print_heatmap("holoB n=128 zero-shot (ball-acc)", mh128, ks64, syms)
    k16b = mh128[4][ks64.index(16)] if len(mh128) >= 5 else 0.0
    ny5 = eval_ball_neg(holo128, en128_list, 5, cfg128)[1]
    swallow6 = eval_ball_neg(holo128, en128_list, 6, cfg128)[0]
    ok_b2 = (k16b >= 0.75) and (ny5 <= 0.05) and (swallow6 >= 0.6)
    print(f"  k=16@T=5={k16b:.2f}  yes-rate(34-40)@T5={ny5:.2f} (want 0: 32<34)  "
          f"swallow-acc@T6={swallow6:.2f} (want HIGH: 64>=34)")
    print(f"[VERDICT B2 - zero-shot n=128 + moving edge] -> {'YES' if ok_b2 else 'NO'}")
    ok_b3 = b_k9plus <= 0.6
    print(f"[VERDICT B3 - baseline trapped] best ball-acc on k=9/12/16, any T<=8: "
          f"{b_k9plus:.2f} -> {'YES' if ok_b3 else 'NO'}")
    save_json(outdir, "ball", dict(k16_64=k16_64, k16_128=k16b,
                                   neg64_T4=negacc_T4, neg64_T5=negacc_T5,
                                   ny128_T5=ny5, swallow128_T6=swallow6,
                                   base_k9plus=b_k9plus,
                                   B1=ok_b1, B2=ok_b2, B3=ok_b3))
    print(f"\n[stage 'ball' finished in {time.time()-t0:.0f}s]")

def exp_scale(args, cfg, outdir, force):
    t0 = time.time()
    print("""
[PREREGISTERED -- SCALE v5.2 (hybrid eq+raw features)]
  training stability / holo32 sanity OOD(7,8)@T3 >= 0.85 /
  holo64+128 zero-shot staircases / baseline trapped / shuffle ~chance
""")
    syms = pick_symbols()
    dm, dv = args.d_model, args.d_vsa
    train, ep32, en32 = get_train_data(args, args.seed)
    en32_list = [it for k in NEG_KS for it in en32[k]]
    cfg32 = dict(cfg, n=32)

    base = get_or_train("base5", f"v52_base_i{cfg['iters']}_T6_s{args.seed}",
                        lambda: Reasoner(32, dm, dv, "none", 8, 6),
                        train, dict(cfg32, T_train_max=6), args.seed + 11,
                        outdir, force)
    holo = get_or_train("holo5", f"v52_holo_v{dv}_i{cfg['iters']}_T3_s{args.seed}",
                        lambda: Reasoner(32, dm, dv, "holo", 8, 3, "fourier"),
                        train, dict(cfg32, T_train_max=3), args.seed + 11,
                        outdir, force)
    shuf = get_or_train("shuf5", f"v52_shuf_v{dv}_i{cfg['iters']}_T3_s{args.seed}",
                        lambda: Reasoner(32, dm, dv, "shuffle", 8, 3, "fourier"),
                        train, dict(cfg32, T_train_max=3), args.seed + 11,
                        outdir, force)

    print("\n[sanity n=32, in-distribution] holo T<=5")
    mh32 = eval_heatmap(holo, EVAL_KS, ep32, en32_list, cfg32, 5, "[holo32]")
    tm32 = frontier(mh32, EVAL_KS)
    h78_32 = ood_mean(mh32, 3, EVAL_KS, OOD78)
    print(f"  holo32 T_min={tm32}  OOD(7,8)@T3={h78_32:.2f}")

    print("\n[SCALE-UP n=64: same weights, extended codebook]")
    ep64, en64 = get_eval_graphs(args, 64, list(range(1, 17)),
                                 list(range(18, 29)), "a", 25)
    en64_list = [it for k in range(18, 29) for it in en64[k]]
    cfg64 = dict(cfg, n=64)
    holo64 = extend_model(holo, 64, dv, cfg["device"])
    shuf64 = extend_model(shuf, 64, dv, cfg["device"])
    base64 = extend_model(base, 64, dv, cfg["device"])
    ks64 = list(range(1, 17))
    print("[baseline64] T<=8")
    mb64 = eval_heatmap(base64, ks64, ep64, en64_list, cfg64, 8, "[base64]")
    print("[holo64] T<=6")
    mh64 = eval_heatmap(holo64, ks64, ep64, en64_list, cfg64, 6, "[holo64]")
    print("[shuffle64] T<=6")
    ms64 = eval_heatmap(shuf64, ks64, ep64, en64_list, cfg64, 6, "[shuf64]")
    print("[untrained channel64] T<=5 (theorem sanity, eq view)")
    ch64 = HoloChannel(64, dv, "fourier")
    for T in range(1, 6):
        row = [vsa_eval_rel(ch64, ep64[k], T, cfg64)[0] for k in ks64]
        fp = 1 - vsa_eval_rel(ch64, en64_list, T, cfg64)[1]
        print(f"  [chan64] T={T}: " + " ".join(f"{a:.2f}" for a in row)
              + f" | FP {fp:.2f}")
    print_heatmap("baseline n=64", mb64, ks64, syms)
    print_heatmap("holo n=64", mh64, ks64, syms)
    print_heatmap("shuffle n=64", ms64, ks64, syms)
    tmh64, tmb64 = frontier(mh64, ks64), frontier(mb64, ks64)
    fpH4 = 1 - eval_acc(holo64, en64_list, 4, cfg64)[1]
    fpH5 = 1 - eval_acc(holo64, en64_list, 5, cfg64)[1]
    fpS4 = 1 - eval_acc(shuf64, en64_list, 4, cfg64)[1]
    b_k9plus = max(max(mb64[T - 1][ks64.index(k)] for T in range(1, 9))
                   for k in (9, 12, 16))
    h16 = mh64[4][ks64.index(16)] if len(mh64) >= 5 else 0.0
    h14 = mh64[5][ks64.index(14)] if len(mh64) >= 6 else 0.0
    s1 = (max(h16, h14) >= 0.80) and (fpH4 <= 0.15) and (b_k9plus <= 0.65)
    print(f"\n  holo64 T_min={tmh64}")
    print(f"  baseline64 T_min={tmb64}  best-on k=9/12/16: {b_k9plus:.2f}")
    print(f"  fingerprint64: holo FP@T4(18-28)={fpH4:.2f}, FP@T5={fpH5:.2f}, "
          f"shuffle FP@T4={fpS4:.2f}")
    print(f"[VERDICT S1 - 2x size transfer] 14-16 hops best={max(h14, h16):.2f} "
          f"-> {'YES' if s1 else 'NO'}")

    print("\n[SCALE-UP n=128: 4x training size]")
    ep128, en128 = get_eval_graphs(args, 128, list(range(1, 17)),
                                   [34, 37, 40], "b", 10)
    en128_list = [it for k in (34, 37, 40) for it in en128[k]]
    cfg128 = dict(cfg, n=128)
    holo128 = extend_model(holo, 128, dv, cfg["device"])
    ks128 = list(range(1, 17))
    print("[holo128] T<=6")
    mh128 = eval_heatmap(holo128, ks128, ep128, en128_list, cfg128, 6, "[holo128]")
    print("[untrained channel128] T<=5 (eq view)")
    ch128 = HoloChannel(128, dv, "fourier")
    for T in range(1, 6):
        row = [vsa_eval_rel(ch128, ep128[k], T, cfg128)[0] for k in ks128]
        fp = 1 - vsa_eval_rel(ch128, en128_list, T, cfg128)[1]
        print(f"  [chan128] T={T}: " + " ".join(f"{a:.2f}" for a in row)
              + f" | FP {fp:.2f}")
    print_heatmap("holo n=128", mh128, ks128, syms)
    tmh128 = frontier(mh128, ks128)
    fpH5b = 1 - eval_acc(holo128, en128_list, 5, cfg128)[1]
    best12 = max(mh128[T - 1][ks128.index(12)] for T in range(1, 7))
    s2 = (best12 >= 0.75) and (fpH5b <= 0.15)
    print(f"  holo128 T_min={tmh128}")
    print(f"  fingerprint128: holo FP@T5(34-40)={fpH5b:.2f}")
    print(f"[VERDICT S2 - 4x size transfer] best k=12 across T: {best12:.2f} "
          f"-> {'YES' if s2 else 'NO'}")

    print("\n[robustness: identity-free holo, 2 extra seeds]")
    margins = []
    for sd in (11, 47):
        cfg_s = dict(cfg32, iters=min(600, cfg["iters"]), T_train_max=3)
        torch.manual_seed(sd)
        h_s = train_model(Reasoner(32, dm, dv, "holo", 8, 3, "fourier"),
                          f"holo5-s{sd}", train, cfg_s, random.Random(sd + 11))
        s_s = train_model(Reasoner(32, dm, dv, "shuffle", 8, 3, "fourier"),
                          f"shuf5-s{sd}", train, cfg_s, random.Random(sd + 11))
        m_h = eval_heatmap(h_s, EVAL_KS, ep32, en32_list, cfg_s, 3, f"[h-s{sd}]")
        m_s = eval_heatmap(s_s, EVAL_KS, ep32, en32_list, cfg_s, 3, f"[s-s{sd}]")
        mg = ood_mean(m_h, 3, EVAL_KS, OOD78) - ood_mean(m_s, 3, EVAL_KS, OOD78)
        margins.append(mg)
        print(f"    seed {sd}: margin={mg:+.2f}")
    s3 = all(m >= 0.05 for m in margins)
    print(f"[VERDICT S3 - identity-free robustness] "
          f"margins={['%+.2f' % m for m in margins]} -> {'YES' if s3 else 'NO'}")
    save_json(outdir, "scale", dict(holo32_tmin=tm32, holo64_tmin=tmh64,
                                    base64_tmin=tmb64, holo128_tmin=tmh128,
                                    fp64=dict(h4=fpH4, h5=fpH5, s4=fpS4),
                                    fp128=dict(h5=fpH5b),
                                    S1=s1, S2=s2, S3=s3, margins=margins))
    print(f"\n[stage 'scale' finished in {time.time()-t0:.0f}s]")

def exp_origin(args, cfg, outdir, force):
    t0 = time.time()
    print("""
[PREREGISTERED -- ORIGIN: can exactness be learned, given, or kept?]
  learned  (random init)   : not emergent (L1)
  learned4 (fourier, free) : erodes at the boundary shell (L2)
  learned4frozen           : the A/B control (L2-ref)
""")
    dm, dv = args.d_model, args.d_vsa
    train, ep32, en32 = get_train_data(args, args.seed)
    en32_list = [it for k in NEG_KS for it in en32[k]]
    cfg32 = dict(cfg, n=32, T_train_max=3)
    torch.manual_seed(args.seed + 5)
    ch_ref = HoloChannel(32, dv, "random")
    mu0r, mo0r = coherence(ch_ref)
    ch_ref4 = HoloChannel(32, dv, "fourier")
    mu0f, mo0f = coherence(ch_ref4)
    print(f"  init coherence (normalized): random mu={mu0r:.3f}/{mo0r:.4f}   "
          f"fourier mu={mu0f:.6f}/{mo0f:.6f}")

    lk = f"v52_learned_v{dv}_i{cfg['iters']}_s{args.seed}"
    l4k = f"v52_learned4_v{dv}_i{cfg['iters']}_s{args.seed}"
    learned = get_or_train("learned", lk,
                           lambda: Reasoner(32, dm, dv, "holo", 8, 3, "learned"),
                           train, cfg32, args.seed + 11, outdir, force)
    learned4 = get_or_train("learned4", l4k,
                            lambda: Reasoner(32, dm, dv, "holo", 8, 3, "learned4"),
                            train, cfg32, args.seed + 11, outdir, force)
    mu1r, mo1r = coherence(learned.holo)
    mu1f, mo1f = coherence(learned4.holo)
    print(f"  trained coherence: learned mu={mu1r:.3f}/{mo1r:.4f}   "
          f"learned4 mu={mu1f:.6f}/{mo1f:.6f}")
    results = {}
    for nm, mdl in (("learned", learned), ("learned4", learned4)):
        print(f"[{nm}] eval T<=4")
        m = eval_heatmap(mdl, EVAL_KS, ep32, en32_list, cfg32, 4, f"[{nm}]")
        tmin = frontier(m, EVAL_KS)
        h78 = ood_mean(m, 3, EVAL_KS, OOD78)
        print(f"  {nm} T_min={tmin}  OOD(7,8)@T3={h78:.2f}")
        results[nm] = dict(tmin=tmin, ood78=h78)
    if mu1r <= 0.5 * mu0r and results["learned"]["ood78"] >= 0.75:
        print("[VERDICT L1 - emergent exactness] YES")
    else:
        print(f"[VERDICT L1 - emergent exactness] NO: mu {mu0r:.3f}->{mu1r:.3f}; "
              f"exactness was not learned -- it must be given")
    if mu1f <= max(0.05, 3 * mu0f) and results["learned4"]["ood78"] >= 0.75:
        print("[VERDICT L2 - exactness stable under free gradients] YES")
    else:
        print(f"[VERDICT L2 - exactness stable under free gradients] NO "
              f"(mu={mu1f:.6f} vs init {mu0f:.6f}; boundary-shell erosion)")
    save_json(outdir, "origin", dict(mu_random=(mu0r, mu1r),
                                     mu_fourier=(mu0f, mu1f), results=results))
    print(f"\n[stage 'origin' finished in {time.time()-t0:.0f}s]")

def exp_final(args, cfg, outdir, force):
    t0 = time.time()
    print("""
[FINAL: train n=64 (T<=4, negs 18-24, reach labels) -> n=128]
""")
    dm, dv = args.d_model, args.d_vsa
    train64 = get_train_data_n64(args)
    cfg64 = dict(cfg, n=64, T_train_max=4)
    baseF = get_or_train("baseF", f"v52F_base_i{cfg['iters']}_T6_s{args.seed}",
                         lambda: Reasoner(64, dm, dv, "none", 8, 6),
                         train64, dict(cfg64, T_train_max=6), args.seed + 11,
                         outdir, force)
    holoF = get_or_train("holoF", f"v52F_holo_v{dv}_i{cfg['iters']}_T4_s{args.seed}",
                         lambda: Reasoner(64, dm, dv, "holo", 8, 4, "fourier"),
                         train64, cfg64, args.seed + 11, outdir, force)
    shufF = get_or_train("shufF", f"v52F_shuf_v{dv}_i{cfg['iters']}_T4_s{args.seed}",
                         lambda: Reasoner(64, dm, dv, "shuffle", 8, 4, "fourier"),
                         train64, cfg64, args.seed + 11, outdir, force)
    ep64, en64 = get_eval_graphs(args, 64, list(range(1, 17)),
                                 list(range(18, 29)), "a", 25)
    en64_list = [it for k in range(18, 29) for it in en64[k]]
    ks64 = list(range(1, 17)); syms = pick_symbols()
    print("\n[in-dist n=64] holo T<=6")
    mh = eval_heatmap(holoF, ks64, ep64, en64_list, cfg64, 6, "[holoF64]")
    print("[in-dist n=64] baseline T<=8")
    mb = eval_heatmap(baseF, ks64, ep64, en64_list, dict(cfg64, T_train_max=6),
                      8, "[baseF64]")
    print("[in-dist n=64] shuffle T<=6")
    ms = eval_heatmap(shufF, ks64, ep64, en64_list, cfg64, 6, "[shufF64]")
    print_heatmap("holo n=64 (trained here)", mh, ks64, syms)
    print_heatmap("baseline n=64", mb, ks64, syms)
    tmh, tmb = frontier(mh, ks64), frontier(mb, ks64)
    k14 = mh[4][ks64.index(14)] if len(mh) >= 5 else 0.0
    fpH = {T: 1 - eval_acc(holoF, en64_list, T, cfg64)[1] for T in (3, 4, 5)}
    ok_f1 = k14 >= 0.85 and all(fpH[T] <= 0.15 for T in (3, 4))
    print(f"\n  holoF64 T_min={tmh}\n  baseline T_min={tmb}")
    print(f"  k=14 @T=5: {k14:.2f}   holo FP@T3/4/5(18-28): "
          + " ".join(f"{fpH[T]:.2f}" for T in (3, 4, 5)))
    print(f"[VERDICT F1 - in-dist n=64 staircase] -> {'YES' if ok_f1 else 'NO'}")
    print("\n[ZERO-SHOT n=128]")
    ep128, en128 = get_eval_graphs(args, 128, list(range(1, 17)),
                                   [34, 37, 40], "b", 10)
    en128_list = [it for k in (34, 37, 40) for it in en128[k]]
    cfg128 = dict(cfg, n=128)
    holo128 = extend_model(holoF, 128, dv, cfg["device"])
    mh128 = []
    for T in range(1, 7):
        row = [eval_acc(holo128, ep128[k], T, cfg128)[0] for k in ks64]
        fp = 1 - eval_acc(holo128, en128_list, T, cfg128)[1]
        mh128.append(row)
        print(f"  [holoF128] T={T}: " + " ".join(f"{a:.2f}" for a in row)
              + f" | FP(34-40) {fp:.2f}")
    print_heatmap("holo n=128 (zero-shot)", mh128, ks64, syms)
    best12 = max(mh128[T - 1][ks64.index(12)] for T in range(1, 7))
    fp5 = 1 - eval_acc(holo128, en128_list, 5, cfg128)[1]
    ok_f2 = best12 >= 0.75 and fp5 <= 0.15
    b_k9plus = max(max(mb[T - 1][ks64.index(k)] for T in range(1, 9))
                   for k in (10, 12, 16))
    print(f"[VERDICT F2 - zero-shot n=128] best k=12: {best12:.2f} "
          f"-> {'YES' if ok_f2 else 'NO'}")
    print(f"[VERDICT F4 - baseline trapped on k>=10] {b_k9plus:.2f} "
          f"-> {'YES' if b_k9plus <= 0.5 else 'NO'}")
    save_json(outdir, "final", dict(holo64_tmin=tmh, base64_tmin=tmb,
                                    k14_64=k14, fp128_T5=fp5, best12=best12,
                                    b_k10plus=b_k9plus,
                                    F1=ok_f1, F2=ok_f2))
    print(f"\n[stage 'final' finished in {time.time()-t0:.0f}s]")

# ----------------------------- main ------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp",
                    choices=["sets", "ball", "scale", "origin", "final", "all"],
                    default="sets")
    ap.add_argument("--n-nodes", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--d-vsa", type=int, default=256)
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--train-n", type=int, default=12000)
    ap.add_argument("--train-n-final", type=int, default=8000)
    ap.add_argument("--eval-pairs", type=int, default=40)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--data-dir", default="v3_results")
    ap.add_argument("--outdir", default="v5_results")
    args = ap.parse_args()
    if args.quick:
        args.iters, args.train_n, args.eval_pairs = 250, 4000, 12
    torch.manual_seed(args.seed); random.seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    cfg = dict(n=32, device=torch.device("cpu"),
               T_train_max=3, iters=args.iters, batch=args.batch, lr=3e-4)
    os.makedirs(args.outdir, exist_ok=True)
    print("=" * 68)
    print(" CAMPAIGN v5.3 -- BALL SEMANTICS + full campaign")
    print("=" * 68)
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | "
          f"exp={args.exp}{' | QUICK' if args.quick else ''}")
    self_test(cfg["device"])
    if args.exp in ("sets", "all"):   exp_sets(args, cfg, args.outdir)
    if args.exp in ("ball", "all"):   exp_ball(args, cfg, args.outdir, args.force)
    if args.exp in ("scale", "all"):  exp_scale(args, cfg, args.outdir, args.force)
    if args.exp in ("origin", "all"): exp_origin(args, cfg, args.outdir, args.force)
    if args.exp in ("final", "all"):  exp_final(args, cfg, args.outdir, args.force)
    print("\nDone. Paste the full console output back.")

if __name__ == "__main__":
    main()
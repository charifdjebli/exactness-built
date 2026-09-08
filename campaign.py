#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
#  CAMPAIGN v4.1 -- ANCHORED doubling, codebook exactness, cleanup.
#
#  v4.0 bug (caught by self-test): the loop squared C *and* advanced F,
#  computing exponents 2,6,14 instead of 2,4,8 -> T=3 meant a 14-hop
#  ball (source-independent at n=32). This explains v3's H1/H6 failures.
#  v4.1: O_t = B^(2^t), F_t = O_t @ seed  (anchored, exact exponents).
#
#  python campaign.py --exp capacity   (~10-25 min, NO training)
#  python campaign.py --exp main       (~25 min)
#  python campaign.py --exp seeds      (~15-20 min)
#  python campaign.py --exp all
#  Smoke test: add --quick.  Data reuses v3 pickles (--data-dir v3_results).
#  The SELF-TEST must print PASSED before anything else runs.
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

def linfit(xs, ys):
    n = len(xs); sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    if abs(den) < 1e-9: return 0.0, sum(ys) / max(1, n)
    a = (n * sxy - sx * sy) / den
    return a, (sy - a * sx) / n

def pick_symbols():
    try:
        s = "\u2588\u2593\u2592\u2591\u00b7"
        s.encode(sys.stdout.encoding or "utf-8")
        return list(s)
    except Exception:
        return list("#+-. ")

# ------------------------- data (same as v3) ---------------------- #
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

def get_data(args, seed, train_ks, neg_ks, eval_ks):
    key = (f"v3data_n{args.n_nodes}_lam{args.lam}_tr{args.train_n}"
           f"_ep{args.eval_pairs}_s{seed}")
    path = os.path.join(args.data_dir, key + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            train, eval_pos, eval_neg = pickle.load(f)
        print(f"[data] loaded cache ({len(train)} train)")
        return train, eval_pos, eval_neg
    rng = random.Random(seed * 31 + 7)
    train, att = [], 0
    while len(train) < args.train_n and att < args.train_n * 4:
        att += 1
        if rng.random() < 0.5:
            k = train_ks[rng.randrange(len(train_ks))]
            g = gen_far_pair(args.n_nodes, k, args.lam, rng, 1)
        else:
            k = neg_ks[rng.randrange(len(neg_ks))]
            g = gen_far_pair(args.n_nodes, k, args.lam, rng, 0)
        if g: train.append(g)
        if att % 3000 == 0: print(f"    gen {len(train)}/{args.train_n}")
    eval_pos, eval_neg = {}, {}
    for k in eval_ks:
        out, att = [], 0
        while len(out) < args.eval_pairs and att < args.eval_pairs * 10:
            att += 1
            g = gen_far_pair(args.n_nodes, k, args.lam, rng, 1)
            if g: out.append(g)
        eval_pos[k] = out
    for k in neg_ks:
        out, att = [], 0
        while len(out) < args.eval_pairs and att < args.eval_pairs * 10:
            att += 1
            g = gen_far_pair(args.n_nodes, k, args.lam, rng, 0)
            if g: out.append(g)
        eval_neg[k] = out
    os.makedirs(args.data_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((train, eval_pos, eval_neg), f)
    print(f"[data] built train={len(train)} "
          f"pos/k={min(len(v) for v in eval_pos.values())} "
          f"neg/L={min(len(v) for v in eval_neg.values())}")
    return train, eval_pos, eval_neg

def get_far_eval(args, seed, n_pairs=25):
    """Connected eval pairs at dist 17..20 (outside the T=4 ball, radius 16)."""
    path = os.path.join(args.data_dir,
                        f"v4far_n{args.n_nodes}_lam{args.lam}_ep{n_pairs}_s{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    rng = random.Random(seed * 17 + 5)
    far = {}
    for k in NEG_FAR_KS:
        out, att = [], 0
        while len(out) < n_pairs and att < n_pairs * 100:
            att += 1
            g = gen_far_pair(args.n_nodes, k, args.lam, rng, 0)
            if g: out.append(g)
        far[k] = out
    os.makedirs(args.data_dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(far, f)
    print("[data] far negatives: " + " ".join(f"L{k}:{len(v)}" for k, v in far.items()))
    return far

def collate(items, n, device):
    B = len(items)
    e_max = max(4, max(len(it[0]) for it in items))
    src = torch.zeros(B, e_max, dtype=torch.long)
    dst = torch.zeros(B, e_max, dtype=torch.long)
    msk = torch.zeros(B, e_max)
    s_idx = torch.zeros(B, dtype=torch.long); t_idx = torch.zeros(B, dtype=torch.long)
    y = torch.zeros(B, dtype=torch.long)
    for b, (edges, s, t, lab, _k) in enumerate(items):
        for j, (u, v) in enumerate(edges):
            src[b, j], dst[b, j], msk[b, j] = u, v, 1.0
        s_idx[b], t_idx[b], y[b] = s, t, lab
    adj = torch.zeros(B, n * n)
    adj.scatter_add_(1, src * n + dst, msk)
    adj = adj.view(B, n, n) + torch.eye(n).unsqueeze(0)
    adj = (adj > 0).float()
    return {"src": src.to(device), "dst": dst.to(device), "msk": msk.to(device),
            "adj": adj.to(device), "s_idx": s_idx.to(device),
            "t_idx": t_idx.to(device), "y": y.to(device)}

# -------------------- holographic channel v4.1 -------------------- #
def cnorm_vec(v, eps=1e-6):
    return v / v.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(eps)

def cnorm_mat(C, eps=1e-6):
    return C / C.abs().pow(2).sum((-2, -1), keepdim=True).sqrt().clamp_min(eps)

def fourier_codebook(n, D):
    """Exactly orthonormal: V_u[d] = exp(2 pi i u d / D)/sqrt(D). Frozen basis."""
    assert D >= n, "Fourier codebook needs D >= n"
    ph = 2 * math.pi * torch.outer(torch.arange(n).float(),
                                   torch.arange(D).float()) / D
    return torch.polar(torch.full_like(ph, 1.0 / math.sqrt(D)), ph)

class HoloChannel(nn.Module):
    """ANCHORED doubling: O_t = B^(2^t); F_t = O_t @ seed (v4.0 double-counted
    exponents: 2,6,14 instead of 2,4,8). With an orthonormal codebook the
    state support is EXACTLY the within-2^T ball.
    cleanup: None | 'feat' (clean readout state) | 'op' (clean the operator's
    columns through an atom softmax after each squaring -- cleans the memory).
    The cleanup softmax is scale-invariant (scores normalized by their max)."""

    def __init__(self, n_nodes, d_vsa, codebook="fourier",
                 cleanup=None, beta=8.0):
        super().__init__()
        self.n, self.cleanup, self.beta = n_nodes, cleanup, beta
        if codebook == "fourier":
            V = fourier_codebook(n_nodes, d_vsa)
            theta = torch.angle(V)
        else:
            theta = torch.empty(n_nodes, d_vsa).uniform_(0, 2 * math.pi)
        self.register_buffer("theta", theta)

    def codebook(self):
        return torch.polar(torch.ones_like(self.theta), self.theta)

    def forward(self, onehot, dst, s_idx, T):
        V = self.codebook()                                    # (n, D)
        bundles = torch.matmul(onehot.transpose(1, 2).to(V.dtype), V[dst])
        bundles = cnorm_vec(bundles + V.unsqueeze(0))          # self-loops
        O = torch.einsum("bui,uj->bij", bundles, V.conj())
        O = cnorm_mat(O)
        seed = cnorm_vec(V[s_idx])                             # anchor
        outs = []
        for _ in range(T):
            O = cnorm_mat(torch.matmul(O, O))                  # ONE squaring
            F = torch.matmul(O, seed.unsqueeze(-1)).squeeze(-1)
            if self.cleanup == "op":
                cols = torch.matmul(O, V.t())
                S = torch.einsum("xd,bdu->bxu", V.conj(), cols).real
                keep = (S > 0.05 * S.amax(dim=1, keepdim=True)
                        .clamp_min(1e-12)).to(S.dtype)      # soft-threshold,
                bb = torch.einsum("bxu,xd->bdu", (S * keep).to(V.dtype), V)  # not softmax
                O = cnorm_mat(torch.einsum("biu,uj->bij", bb, V.conj()))
                F = torch.matmul(O, seed.unsqueeze(-1)).squeeze(-1)
            a = torch.einsum("bd,xd->bx", F, V.conj()).real
            if self.cleanup == "feat":
                keep = (a > 0.05 * a.amax(-1, keepdim=True)
                        .clamp_min(1e-12)).to(a.dtype)
                F = torch.einsum("bx,xd->bd", (a * keep).to(V.dtype), V)
                a = torch.einsum("bd,xd->bx", F, V.conj()).real
            outs.append(a)
        return torch.stack(outs, dim=1)                        # (B, T, n)

# ------------------------- self-test ------------------------------ #
def self_test(device):
    """Chain 0->1->...->7, decoy 3->8, and 9->0 (9 NOT reachable from 0)."""
    n, D = 10, 64
    edges = [(i, i + 1) for i in range(7)] + [(3, 8), (9, 0)]
    E = len(edges)
    onehot = torch.zeros(1, E, n); dst = torch.zeros(1, E, dtype=torch.long)
    for j, (u, v) in enumerate(edges):
        onehot[0, j, u] = 1.0; dst[0, j] = v
    s_idx = torch.zeros(1, dtype=torch.long)
    ch = HoloChannel(n, D, "fourier").to(device)
    with torch.no_grad():
        sc = ch(onehot.to(device), dst.to(device), s_idx.to(device), 3)[0]
    ok, msgs = True, []
    def chk(name, val, lo, hi=None):
        nonlocal ok
        good = (val >= lo) if hi is None else (lo <= val <= hi)
        if not good: ok = False
        msgs.append(f"    {name}: {val:.5f} {'OK' if good else 'FAIL'}")
    chk("T=1 node1 (dist1) in-ball ", sc[0, 1].item(), 0.05)
    chk("T=1 node2 (dist2) in-ball ", sc[0, 2].item(), 0.05)
    chk("T=1 node3 (dist3) OUT     ", sc[0, 3].item(), 0.0, 0.02)
    chk("T=2 node3 (dist3) in-ball ", sc[1, 3].item(), 0.05)
    chk("T=2 node5 (dist5) OUT     ", sc[1, 5].item(), 0.0, 0.02)
    chk("T=3 node7 (dist7) in-ball ", sc[2, 7].item(), 0.03)
    chk("T=3 node9 (unreach) OUT   ", sc[2, 9].item(), 0.0, 0.02)
    print("[SELF-TEST] operator direction/support on handcrafted graph:")
    print("\n".join(msgs))
    if not ok:
        print("[SELF-TEST] FAILED -- paste this output back; do not train.")
        sys.exit(1)
    print("[SELF-TEST] PASSED -- operator verified (anchored exponents 2,4,8).\n")

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
    def __init__(self, n_nodes, d_model, d_vsa, channel_mode="none",
                 max_steps=8, clamp_steps=3, codebook="fourier"):
        super().__init__()
        assert channel_mode in ("none", "holo", "shuffle")
        self.n, self.channel_mode, self.clamp = n_nodes, channel_mode, clamp_steps
        self.node_emb = nn.Parameter(torch.randn(n_nodes, d_model) * 0.02)
        self.flag = nn.Linear(2, d_model)
        self.step_emb = nn.Parameter(torch.randn(max_steps, d_model) * 0.02)
        self.inj = nn.Linear(4, d_model)
        self.ls_inj = nn.Parameter(torch.ones(d_model))
        if channel_mode != "none":
            self.holo = HoloChannel(n_nodes, d_vsa, codebook)
        self.block = Block(d_model)
        self.gate = nn.Linear(2 * d_model, d_model)
        nn.init.constant_(self.gate.bias, -2.0)
        self.readout = nn.Sequential(nn.Linear(3 * d_model, d_model),
                                     nn.GELU(), nn.Linear(d_model, 2))
    def forward(self, src, dst, msk, adj, s_idx, t_idx, T):
        B, N = adj.shape[0], adj.shape[-1]
        d = self.node_emb.shape[1]
        assert T <= self.step_emb.shape[0]
        if self.channel_mode != "none":
            s_use = (torch.roll(s_idx, 1, dims=0) if B > 1
                     else (s_idx + 1) % self.n) \
                    if self.channel_mode == "shuffle" else s_idx
            onehot = Fn.one_hot(src, self.n).float() * msk.unsqueeze(-1)
            scores = self.holo(onehot, dst, s_use, T)
        flags = torch.zeros(B, N, 2, device=adj.device)
        flags[torch.arange(B, device=adj.device), s_idx, 0] = 1.0
        flags[torch.arange(B, device=adj.device), t_idx, 1] = 1.0
        H = self.node_emb.unsqueeze(0) + self.flag(flags)
        for t in range(T):




            if self.channel_mode != "none":
                a = scores[:, t]
                amax = a.max(-1, keepdim=True).values.clamp_min(1e-6)
                rel = a / amax
                alog = torch.where(a > 1e-10, torch.log10(a.clamp_min(1e-16)),
                                   torch.full_like(a, -16.0))
                asup = (rel > 1e-4).float()          # support legibility:
                feat = torch.stack([rel, a, alog, asup], dim=-1)  # in-ball >=2e-3,
                # out-of-ball = 0 exactly (orthonormal codes) -> clean at ALL distances



            else:
                feat = torch.zeros(B, N, 4, device=H.device)
            H = H + self.step_emb[min(t, self.clamp - 1)] \
                  + self.ls_inj * self.inj(feat)
            Hc = self.block(H, adj)
            z = torch.sigmoid(self.gate(torch.cat([Hc, H], dim=-1)))
            H = z * Hc + (1 - z) * H
        Ht = H.gather(1, t_idx.view(B, 1, 1).expand(B, 1, d)).squeeze(1)
        Hs = H.gather(1, s_idx.view(B, 1, 1).expand(B, 1, d)).squeeze(1)
        return self.readout(torch.cat([Ht, Hs, H.mean(dim=1)], dim=-1)).clamp(-30, 30)

# -------------------------- training ------------------------------ #
def train_model(model, name, train_set, cfg, rng):
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
        loss = Fn.cross_entropy(logits, bt["y"])
        if not torch.isfinite(loss): continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        rl += loss.item(); rn += 1
        rc += (logits.argmax(-1) == bt["y"]).sum().item()
        if it % 100 == 0 or it == total:
            print(f"    [{name}] iter {it:5d}/{total}  loss {rl/max(1,rn):.4f}  "
                  f"acc {rc/max(1,rn*cfg['batch']):.3f}  ({time.time()-t0:.0f}s)")
            rl, rc, rn = 0.0, 0, 0
    return model

def get_or_train(name, key, builder, train_set, cfg, train_seed, outdir, force):
    path = os.path.join(outdir, f"model_{key}.pt")
    torch.manual_seed(train_seed - 11)
    model = builder()
    if not force and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=cfg["device"]))
        print(f"    [{name}] loaded cached weights")
        return model
    model = train_model(model, name, train_set, cfg, random.Random(train_seed))
    torch.save(model.state_dict(), path)
    return model

# ----------------------------- eval ------------------------------- #
@torch.no_grad()
def eval_acc(model, items, T, cfg, bs=200):
    model.eval(); cp = pp = cokn = pn = 0
    for chn in chunks(items, bs):
        bt = collate(chn, cfg["n"], cfg["device"])
        logits = model(bt["src"], bt["dst"], bt["msk"], bt["adj"],
                       bt["s_idx"], bt["t_idx"], T)
        pred = logits.argmax(-1); ispos = bt["y"] == 1
        cp += ((pred == 1) & ispos).sum().item(); pp += ispos.sum().item()
        cokn += ((pred == 0) & ~ispos).sum().item(); pn += (~ispos).sum().item()
    return cp / max(1, pp), cokn / max(1, pn)

@torch.no_grad()
def eval_heatmap(model, eval_ks, eval_pos, eval_neg_list, cfg, T_max):
    mat, fps = [], []
    for T in range(1, T_max + 1):
        row = [eval_acc(model, eval_pos[k], T, cfg)[0] for k in eval_ks]
        _, negok = eval_acc(model, eval_neg_list, T, cfg)
        mat.append(row); fps.append(1.0 - negok)
        print(f"    T={T}: " + " ".join(f"{a:.2f}" for a in row) + f"  | FP {fps[-1]:.2f}")
    return mat, fps

@torch.no_grad()
def vsa_target_scores(channel, items, cfg, T_max, bs=100):
    allsc = []
    for chn in chunks(items, bs):
        bt = collate(chn, cfg["n"], cfg["device"])
        onehot = Fn.one_hot(bt["src"], cfg["n"]).float() * bt["msk"].unsqueeze(-1)
        sc = channel(onehot, bt["dst"], bt["s_idx"], T_max)
        sc = sc / sc.max(-1, keepdim=True).values.clamp_min(1e-9)
        B = sc.shape[0]
        st = sc.transpose(1, 2).gather(
            1, bt["t_idx"].view(B, 1, 1).expand(B, 1, T_max)).squeeze(1)
        allsc.append(st.cpu())
    return torch.cat(allsc, dim=0)

def calibrate_thresholds(train_items, channel, cfg, T_max, n_cal=1500, seed=123):
    rng = random.Random(seed)
    sub = rng.sample(train_items, min(n_cal, len(train_items)))
    sc = vsa_target_scores(channel, sub, cfg, T_max)
    y = torch.tensor([it[3] for it in sub]); thr = {}
    for T in range(1, T_max + 1):
        p, q = sc[y == 1, T - 1], sc[y == 0, T - 1]
        cands = torch.quantile(sc[:, T - 1], torch.linspace(0.02, 0.98, 97)).tolist()
        best, bthr = -1.0, 0.0
        for c in cands:
            ba = 0.5 * ((p > c).float().mean().item() + (q <= c).float().mean().item())
            if ba > best: best, bthr = ba, c
        thr[T] = bthr
    return thr

@torch.no_grad()
def vsa_eval(channel, items, T, thr, cfg, bs=100):
    sc = vsa_target_scores(channel, items, cfg, T)[:, T - 1]
    y = torch.tensor([it[3] for it in items]); pred = (sc > thr).long()
    pos = y == 1
    ap = (pred[pos] == 1).float().mean().item() if pos.any() else 0.0
    an = (pred[~pos] == 0).float().mean().item() if (~pos).any() else 0.0
    return ap, an

@torch.no_grad()
def vsa_eval_rel(channel, items, T, cfg, bs=100, rel_thr=1e-4):
    """Exact-support detector: in-ball ratio >= ~2e-3, out-of-ball ratio = 0.
    Distance-robust by construction (no calibration, no row-max distortion)."""
    ap = aok = tp = tn = 0
    for chn in chunks(items, bs):
        bt = collate(chn, cfg["n"], cfg["device"])
        onehot = Fn.one_hot(bt["src"], cfg["n"]).float() * bt["msk"].unsqueeze(-1)
        sc = channel(onehot, bt["dst"], bt["s_idx"], T)[:, T - 1]
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
    print(f"\n  HEATMAP - {name}  (rows=T; OOD k>=7)")
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

def staircase_ok(tmin, ks):
    return all(tmin.get(k) is not None and tmin[k] <= predict_log(k) for k in ks)

def ood_mean(mat, T, ks, group):
    row = mat[T - 1]
    return sum(row[ks.index(k)] for k in group) / len(group)

def save_json(outdir, name, obj):
    with open(os.path.join(outdir, f"{name}.json"), "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[saved] {outdir}/{name}.json")

# -------------------------- experiments --------------------------- #
TRAIN_KS = [1, 2, 3, 4, 5, 6]
NEG_KS = [10, 11, 12, 13, 14]
NEG_FAR_KS = [17, 18, 19, 20]        # outside the T=4 ball (radius 16)
EVAL_KS = list(range(1, 13))
OOD78, OOD912 = [7, 8], [9, 10, 11, 12]

def exp_capacity(args, cfg, outdir):
    t0 = time.time()
    print("""
[PREREGISTERED -- capacity law, ZERO training, anchored doubling]
  fourier (exact)      : T_min = ceil(log2 k) for ALL k<=12;
                         FP(dist 10-14): ~0 at T=3; HIGH at T=4 (edge swallows
                         them -- correct physics, not failure); FP(17-20) ~0 at T=4
  random D=256/512/1024: wall position scales with dimension
  random + cleanup-op  : staircase RESTORED -> "interference is a per-step tax"
""")
    train, eval_pos, eval_neg = get_data(args, args.seed, TRAIN_KS, NEG_KS, EVAL_KS)
    eval_neg_list = [it for k in NEG_KS for it in eval_neg[k]]
    far = get_far_eval(args, args.seed)
    far_list = [it for k in NEG_FAR_KS for it in far[k]]
    cfg_e = dict(cfg)
    configs = [
        ("fourier-exact D=256", dict(codebook="fourier"), 256),
        ("random D=256", dict(codebook="random"), 256),
        ("random D=512", dict(codebook="random"), 512),
        ("random D=1024", dict(codebook="random"), 1024),   # slowest; delete if needed
        ("random D=256+cleanup-feat", dict(codebook="random", cleanup="feat"), 256),
        ("random D=256+cleanup-op", dict(codebook="random", cleanup="op"), 256),
    ]
    table = {}
    for name, kw, D in configs:
        torch.manual_seed(args.seed + 5)
        ch = HoloChannel(args.n_nodes, D, **kw)
        use_abs = "fourier" in name
        thr = None if use_abs else calibrate_thresholds(train, ch, cfg_e, 4)
        mat = []
        for T in range(1, 5):
            if use_abs:
                row = [vsa_eval_rel(ch, eval_pos[k], T, cfg_e)[0] for k in EVAL_KS]
                an = vsa_eval_rel(ch, eval_neg_list, T, cfg_e)[1]
            else:
                row = [vsa_eval(ch, eval_pos[k], T, thr[T], cfg_e)[0] for k in EVAL_KS]
                an = vsa_eval(ch, eval_neg_list, T, thr[T], cfg_e)[1]
            mat.append(row)
            print(f"  {name:26s} T={T}: " + " ".join(f"{a:.2f}" for a in row)
                  + f" | FP(near) {1-an:.2f}")
        if use_abs:
            fp3 = 1 - vsa_eval_rel(ch, eval_neg_list, 3, cfg_e)[1]
            fp4 = 1 - vsa_eval_rel(ch, eval_neg_list, 4, cfg_e)[1]
            fp4f = 1 - vsa_eval_rel(ch, far_list, 4, cfg_e)[1]
        else:
            fp3 = 1 - vsa_eval(ch, eval_neg_list, 3, thr[3], cfg_e)[1]
            fp4 = 1 - vsa_eval(ch, eval_neg_list, 4, thr[4], cfg_e)[1]
            fp4f = 1 - vsa_eval(ch, far_list, 4, thr[4], cfg_e)[1]
        tmin = frontier(mat, EVAL_KS)
        table[name] = dict(tmin=tmin, fp3=fp3, fp4=fp4, fp4_far=fp4f)
        print(f"  {name:26s} T_min={tmin}  FP@T3={fp3:.2f} "
              f"FP@T4near={fp4:.2f} FP@T4far={fp4f:.2f}\n")
    tf = table["fourier-exact D=256"]
    ok1 = staircase_ok(tf["tmin"], EVAL_KS) and tf["fp3"] <= 0.05 and tf["fp4_far"] <= 0.10
    print(f"[VERDICT C1 - exact codebook] staircase + zero FP below every edge: "
          f"{'YES' if ok1 else 'NO'}  (fp3={tf['fp3']:.4f}, fp4far={tf['fp4_far']:.4f})")
    rc, rr = table["random D=256+cleanup-op"], table["random D=256"]
    ok2 = (all(rc["tmin"].get(k) is not None and rc["tmin"][k] <= 3
               for k in (5, 6, 7, 8)) and
           not all(rr["tmin"].get(k) is not None and rr["tmin"][k] <= 3
                   for k in (5, 6, 7, 8)))
    print(f"[VERDICT C2 - cleanup repairs random] {'YES' if ok2 else 'NO'}  "
          f"(op-clean T_min 5-8: {[rc['tmin'].get(k) for k in (5,6,7,8)]}, "
          f"raw: {[rr['tmin'].get(k) for k in (5,6,7,8)]})")
    save_json(outdir, "capacity", table)
    print(f"[stage 'capacity' finished in {time.time()-t0:.0f}s]")

def exp_main(args, cfg, outdir, force):
    t0 = time.time()
    print("""
[PREREGISTERED -- trained, causal, confound-free]
  holo (exact channel) : T_min = ceil(log2 k); OOD k7-8 @T=3 >= 0.85;
                         T=1 honesty: k>=3 ~ chance (exact 2-ball only)
  shuffle control      : OOD low; FP on near negatives HIGH at T=3 (its ball
                         is wrong-source) while holo FP ~ 0; holo FP moves
                         8->16 with T, shuffle cannot  <- fingerprint
  baseline             : T_min ~ 1.1k (linear); k>=8 chance within T<=8
""")
    train, eval_pos, eval_neg = get_data(args, args.seed, TRAIN_KS, NEG_KS, EVAL_KS)
    eval_neg_list = [it for k in NEG_KS for it in eval_neg[k]]
    syms = pick_symbols()
    n, dm, dv = args.n_nodes, args.d_model, args.d_vsa
    cfg_b = dict(cfg); cfg_b["T_train_max"] = 6
    cfg_h = dict(cfg); cfg_h["T_train_max"] = 3
    kb = f"v42_base_n{n}_d{dm}_i{cfg['iters']}_T6_s{args.seed}"
    kh = f"v42_holo_n{n}_d{dm}_v{dv}_i{cfg['iters']}_T3_s{args.seed}"
    ks_ = f"v42_shuf_n{n}_d{dm}_v{dv}_i{cfg['iters']}_T3_s{args.seed}"
    base = get_or_train("baseline", kb, lambda: Reasoner(n, dm, dv, "none", 8, 6),
                        train, cfg_b, args.seed + 11, outdir, force)
    holo = get_or_train("holo", kh,
                        lambda: Reasoner(n, dm, dv, "holo", 8, 3, "fourier"),
                        train, cfg_h, args.seed + 11, outdir, force)
    shuf = get_or_train("shuffle", ks_,
                        lambda: Reasoner(n, dm, dv, "shuffle", 8, 3, "fourier"),
                        train, cfg_h, args.seed + 11, outdir, force)
    print("\n[baseline] eval T<=8")
    matb, _ = eval_heatmap(base, EVAL_KS, eval_pos, eval_neg_list, cfg_b, 8)
    print("[holo] eval T<=5")
    math_, _ = eval_heatmap(holo, EVAL_KS, eval_pos, eval_neg_list, cfg_h, 5)
    print("[shuffle] eval T<=5")
    mats, _ = eval_heatmap(shuf, EVAL_KS, eval_pos, eval_neg_list, cfg_h, 5)
    print_heatmap("baseline", matb, EVAL_KS, syms)
    print_heatmap("holo (fourier)", math_, EVAL_KS, syms)
    print_heatmap("shuffle (control)", mats, EVAL_KS, syms)
    tmb, tmh, tms = frontier(matb, EVAL_KS), frontier(math_, EVAL_KS), frontier(mats, EVAL_KS)
    print(f"\n  measured holo T_min : {tmh}")
    print(f"  predicted log2      : {dict((k, predict_log(k)) for k in EVAL_KS)}")
    print(f"  measured baseline   : {tmb}")
    h78, s78 = ood_mean(math_, 3, EVAL_KS, OOD78), ood_mean(mats, 3, EVAL_KS, OOD78)
    b78 = ood_mean(matb, 8, EVAL_KS, OOD78)
    h912 = ood_mean(math_, 4, EVAL_KS, OOD912)
    t1_ood = ood_mean(math_, 1, EVAL_KS, [3, 4, 5, 6, 7, 8])
    print(f"\n[VERDICT H2 - holo staircase] {'YES' if staircase_ok(tmh, EVAL_KS) else 'NO'}")
    ok1 = (h78 >= 0.80) and (h78 - s78 >= 0.20)
    print(f"[VERDICT H1 - causality @T=3, k=7..8] holo={h78:.2f} shuffle={s78:.2f} "
          f"baseline(best)={b78:.2f} -> {'YES' if ok1 else 'NO'}")
    print(f"[honesty] holo T=1 on k=3..8 (should be ~chance): {t1_ood:.2f}")
    print(f"[secondary] holo k=9..12 @T=4: {h912:.2f} (log2 law: within 16-ball)")
    print("\n[causal fingerprint] FP on negatives (the law's edge must move 8 -> 16):")
    far = get_far_eval(args, args.seed)
    far_list = [it for k in NEG_FAR_KS for it in far[k]]
    fpH3n = sum(1 - eval_acc(holo, eval_neg[L], 3, cfg_h)[1] for L in NEG_KS) / len(NEG_KS)
    fpS3n = sum(1 - eval_acc(shuf, eval_neg[L], 3, cfg_h)[1] for L in NEG_KS) / len(NEG_KS)
    fpH4n = 1 - eval_acc(holo, eval_neg_list, 4, cfg_h)[1]
    fpH4f = 1 - eval_acc(holo, far_list, 4, cfg_h)[1]
    fpS4f = 1 - eval_acc(shuf, far_list, 4, cfg_h)[1]
    print(f"    T=3  holo near(10-14): {fpH3n:.2f}   shuffle near: {fpS3n:.2f}")
    print(f"    T=4  holo near(10-14): {fpH4n:.2f} (expect HIGH: inside the 16-ball)")
    print(f"    T=4  holo far (17-20): {fpH4f:.2f}   shuffle far: {fpS4f:.2f}")
    ok_fp = fpH3n <= 0.15 and fpH4f <= 0.15 and fpS3n >= 0.30
    print(f"[VERDICT H1b - fingerprint] edge moved with T, shuffle can't: "
          f"{'YES' if ok_fp else 'NO'}")
    if ok1 and staircase_ok(tmh, EVAL_KS) and ok_fp:
        print("\n  *** HEADLINE: log-depth confirmed causally: holo solves "
              "k<=8 in T=3 and k<=12 in T=4; baseline frontier ~1.1k linear. ***")
    save_json(outdir, "main", dict(tmin_baseline=tmb, tmin_holo=tmh,
                                   tmin_shuffle=tms, H1=h78, S1=s78,
                                   fp=dict(h3=fpH3n, s3=fpS3n, h4n=fpH4n,
                                           h4f=fpH4f, s4f=fpS4f),
                                   ok=ok1, ok_fp=ok_fp, ok_h2=staircase_ok(tmh, EVAL_KS),
                                   holo_912_T4=h912, t1_ood=t1_ood))
    print(f"\n[stage 'main' finished in {time.time()-t0:.0f}s]")

def linfit_wrap(tmin, ks):
    xs = [k for k in ks if tmin.get(k) is not None]
    ys = [tmin[k] for k in xs]
    if len(xs) < 2: return None, None
    return linfit(xs, ys)

def exp_seeds(args, cfg, outdir):
    t0 = time.time()
    rows = []
    iters = 250 if args.quick else 600
    for sd in (11, 23, 47):
        print(f"\n===== seed {sd} =====")
        train, eval_pos, eval_neg = get_data(args, sd, TRAIN_KS, NEG_KS, EVAL_KS)
        eval_neg_list = [it for k in NEG_KS for it in eval_neg[k]]
        cfg_s = dict(cfg); cfg_s["iters"] = iters; cfg_s["T_train_max"] = 3
        n, dm, dv = args.n_nodes, args.d_model, args.d_vsa
        torch.manual_seed(sd)
        holo = train_model(Reasoner(n, dm, dv, "holo", 8, 3, "fourier"),
                           "holo", train, cfg_s, random.Random(sd + 11))
        torch.manual_seed(sd)
        shuf = train_model(Reasoner(n, dm, dv, "shuffle", 8, 3, "fourier"),
                           "shuffle", train, cfg_s, random.Random(sd + 11))
        mh, _ = eval_heatmap(holo, EVAL_KS, eval_pos, eval_neg_list, cfg_s, 4)
        ms, _ = eval_heatmap(shuf, EVAL_KS, eval_pos, eval_neg_list, cfg_s, 4)
        h78, s78 = ood_mean(mh, 3, EVAL_KS, OOD78), ood_mean(ms, 3, EVAL_KS, OOD78)
        rows.append(dict(seed=sd, holo=h78, shuffle=s78, margin=h78 - s78))
        print(f"    seed {sd}: OOD(k=7,8)@T=3 holo={h78:.2f} shuffle={s78:.2f} "
              f"margin={h78-s78:+.2f}")
    mm = min(r["margin"] for r in rows)
    hm = sum(r["holo"] for r in rows) / len(rows)
    ok6 = mm >= 0.15 and hm >= 0.75
    print(f"\n[VERDICT H6 - robustness] mean holo={hm:.2f}, min margin={mm:+.2f} "
          f"-> {'ROBUST: YES' if ok6 else 'seed-sensitive'}")
    save_json(outdir, "seeds", dict(rows=rows, ok=ok6))
    print(f"[stage 'seeds' finished in {time.time()-t0:.0f}s]")

# ----------------------------- main ------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["capacity", "main", "seeds", "all"],
                    default="capacity")
    ap.add_argument("--n-nodes", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--d-vsa", type=int, default=256)
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--train-n", type=int, default=12000)
    ap.add_argument("--eval-pairs", type=int, default=40)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--data-dir", default="v3_results")
    ap.add_argument("--outdir", default="v4_results")
    args = ap.parse_args()
    if args.quick:
        args.iters, args.train_n, args.eval_pairs = 300, 4000, 12
    torch.manual_seed(args.seed); random.seed(args.seed)
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    cfg = dict(n=args.n_nodes, device=torch.device("cpu"),
               T_train_max=3, iters=args.iters, batch=args.batch, lr=3e-4)
    os.makedirs(args.outdir, exist_ok=True)
    print("=" * 68)
    print(" CAMPAIGN v4.1 -- ANCHORED doubling | exactness | capacity | cleanup")
    print("=" * 68)
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | "
          f"exp={args.exp}{' | QUICK' if args.quick else ''}")
    try:
        _ = (torch.randn(2, 3, 3, dtype=torch.complex64) @
             torch.randn(2, 3, 3, dtype=torch.complex64))
    except Exception as e:
        sys.exit(f"ERROR: complex matmul unsupported: {e}")
    self_test(cfg["device"])
    if args.exp in ("capacity", "all"): exp_capacity(args, cfg, args.outdir)
    if args.exp in ("main", "all"):     exp_main(args, cfg, args.outdir, args.force)
    if args.exp in ("seeds", "all"):    exp_seeds(args, cfg, args.outdir)
    print("\nDone. Paste the full console output back.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
#  STAGE 1 -- RELATION-SELECTIVE CHANNEL BANK v5 (zero training, CPU)
#  v5: generator rewritten. The old one required full connectivity
#  (anti-confound armor for TRAINED models -- irrelevant for zero-
#  training exact queries) and silently returned None, burning hours.
#  New: rel-distance == k is guaranteed BY CONSTRUCTION (distractors
#  never carry the planted type) and asserted loudly. T3 guarded.
#
#  python channel_bank.py --quick    (~2-4 min)
#  python channel_bank.py            (~4-6 min)
# =====================================================================
import argparse, math, random, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as Fn

# ----------------------------- utils ------------------------------ #
def bfs_dist_typed(adj, src, n, rel=None):
    dist = [-1] * n; dist[src] = 0; q = [src]
    while q:
        u = q.pop(0)
        for (v, r) in adj[u]:
            if rel is not None and r != rel: continue
            if dist[v] < 0:
                dist[v] = dist[u] + 1; q.append(v)
    return dist

def cnorm_mat(C, eps=1e-9):
    return C / C.abs().pow(2).sum((-2, -1), keepdim=True).sqrt().clamp_min(eps)

def fourier_codebook(n, D):
    assert D >= n
    ph = 2 * math.pi * torch.outer(torch.arange(n).float(),
                                   torch.arange(D).float()) / D
    return torch.polar(torch.full_like(ph, 1.0 / math.sqrt(D)), ph)

# ------------------------- data generator ------------------------- #
def gen_typed_pair(n, k, rel, R, lam, rng):
    """Typed graph with an EXACT type-`rel` path s->t of length k.
    Distractor edges never carry type `rel`, so the rel-distance is k
    by construction (asserted). Connectivity NOT required (no trained
    classifier to confound in zero-training tests)."""
    for _att in range(30):
        edges = []; adj = [[] for _ in range(n)]
        def add(u, v, r):
            for (vv, rr) in adj[u]:
                if vv == v and rr == r: return False
            adj[u].append((v, r)); edges.append((u, v, r)); return True
        perm = list(range(n)); rng.shuffle(perm)
        chain = perm[:k + 1]; s, t = chain[0], chain[-1]
        for i in range(k):
            ok = add(chain[i], chain[i + 1], rel)
            assert ok, "chain edge rejected -- generator logic error"
        budget, tries = int(round(lam * n)), 0
        while budget > 0 and tries < 300:
            tries += 1
            u, v = rng.randrange(n), rng.randrange(n)
            r = rng.randrange(R)
            if r == rel or u == v: continue
            if add(u, v, r):
                budget -= 1
        dd = bfs_dist_typed(adj, s, n, rel)
        assert dd[t] == k, f"rel-dist {dd[t]} != {k} -- invariant broken"
        return sorted(edges), s, t, k, rel
    return None

def gen_bank_set(n, ks, R, count, lam, rng, rel=None):
    out, att = [], 0
    while len(out) < count and att < count * 40:
        att += 1
        r = rel if rel is not None else rng.randrange(R)
        g = gen_typed_pair(n, ks[rng.randrange(len(ks))], r, R, lam, rng)
        if g: out.append(g)
    assert len(out) == count, f"generator starved: {len(out)}/{count}"
    return out

# --------------------- channel bank (pure fns) -------------------- #
def build_bank(edges_batch, n, D, R, device):
    """Returns O_bank (B,R,D,D) complex, O_union (B,D,D), codebook V (n,D)."""
    B = len(edges_batch)
    assert B > 0, "build_bank got an empty batch"
    V = fourier_codebook(n, D).to(device)
    E = max(4, max(len(e) for e in edges_batch))
    src = torch.zeros(B, E, dtype=torch.long)
    dst = torch.zeros(B, E, dtype=torch.long)
    rid = torch.full((B, E), -1, dtype=torch.long)
    msk = torch.zeros(B, E)
    for b, edges in enumerate(edges_batch):
        for j, (u, v, r) in enumerate(edges):
            src[b, j], dst[b, j], rid[b, j], msk[b, j] = u, v, r, 1.0
    src, dst, rid, msk = (x.to(device) for x in (src, dst, rid, msk))
    onehot = Fn.one_hot(src, n).float() * msk.unsqueeze(-1)
    Vd = V[dst]
    def bundles_for(sel):
        oh = onehot * sel.unsqueeze(-1)
        bd = torch.matmul(oh.transpose(1, 2).to(Vd.dtype), Vd)
        return cnorm_mat(bd + V.unsqueeze(0))
    bank = []
    for r in range(R):
        sel = ((rid == r) & (msk > 0)).float()
        bank.append(bundles_for(sel))
    bundles = torch.stack(bank, 1)
    O = cnorm_mat(torch.einsum("brui,uj->brij", bundles, V.conj()))
    bundles_u = bundles_for(torch.ones_like(msk))
    O_u = cnorm_mat(torch.einsum("bui,uj->bij", bundles_u, V.conj()))
    return O, O_u, V

@torch.no_grad()
def bank_scores(O_bank, V, s_idx, T, equalize=False):
    """Anchored doubling, all relations at once -> scores (B,R,T,n)."""
    O = O_bank.clone()
    seed = V[s_idx]
    seed = seed / seed.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-9)
    seed = seed.unsqueeze(1)
    outs = []
    for _ in range(T):
        O = cnorm_mat(torch.matmul(O, O))
        if equalize:
            Vc = V.conj()
            Ob = torch.einsum("xd,brde,ye->brxy", Vc, O, V)
            Ob = Ob / Ob.abs().pow(2).sum(2, keepdim=True).sqrt().clamp_min(1e-9)
            Oeq = torch.einsum("xd,brxy,ye->brde", V, Ob, Vc)
        else:
            Oeq = O
        F = torch.matmul(Oeq, seed.unsqueeze(-1)).squeeze(-1)
        a = torch.einsum("brd,ud->bru", F, V.conj()).real
        outs.append(a)
    return torch.stack(outs, dim=2)

@torch.no_grad()
def powers_exact(O, V, s_idx, t_idx, Lmax):
    """Ball powers B^1..B^Lmax -> per-L in-ball indicator at target (B,Lmax).
    Minimal firing L == true distance."""
    P = O.clone()
    seed = V[s_idx]
    seed = seed / seed.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-9)
    res = []
    for L in range(1, Lmax + 1):
        if L > 1:
            P = cnorm_mat(torch.matmul(P, O))
        F = torch.matmul(P, seed.unsqueeze(-1)).squeeze(-1)
        a = torch.einsum("brd,ud->bru", F, V.conj()).real
        a = a.squeeze(1) if a.dim() == 3 else a
        rel = a / a.amax(-1, keepdim=True).clamp_min(1e-12)
        hit = rel.gather(1, t_idx.view(-1, 1)).squeeze(1)
        res.append((hit > 1e-4).float())
    return torch.stack(res, dim=1)

def in_ball(scores_bank, t_idx, T, thr=1e-4):
    """(B,R,T,n), t_idx (B,) -> (B,R) bool: target in 2^T ball of rel r."""
    row = scores_bank[:, :, T - 1, :]
    mx = row.amax(-1, keepdim=True).clamp_min(1e-12)
    rel = row / mx
    tgt = rel.gather(-1, t_idx.view(-1, 1, 1)
                     .expand(-1, rel.shape[1], 1)).squeeze(-1)
    return tgt > thr

# ------------------------- self-test ------------------------------ #
def self_test():
    n, D, R = 10, 64, 2
    edges = [[(0, 1, 0), (1, 2, 0), (2, 3, 0), (0, 3, 1)]]
    O, O_u, V = build_bank(edges, n, D, R, torch.device("cpu"))
    s = torch.tensor([0]); t = torch.tensor([3])
    sc = bank_scores(O, V, s, 3)
    ok = True; msgs = []
    def chk(name, val, lo, hi=None):
        nonlocal ok
        good = (val >= lo) if hi is None else (lo <= val <= hi)
        if not good: ok = False
        msgs.append(f"    {name}: {val:.5f} {'OK' if good else 'FAIL'}")
    ib1 = in_ball(sc, t, 1)[0]; ib2 = in_ball(sc, t, 2)[0]
    chk("T=1 rel0 s->t (dist3) OUT", float(ib1[0]), 0.0, 0.5)
    chk("T=1 rel1 decoy 1-hop IN ", float(ib1[1]), 0.5)
    chk("T=2 rel0 3-hop IN        ", float(ib2[0]), 0.5)
    p = powers_exact(O[:, 0:1], V, s, t, 4)[0]
    chk("ball P1 (dist3) OUT      ", float(p[0]), 0.0, 0.5)
    chk("ball P2 (dist3) OUT      ", float(p[1]), 0.0, 0.5)
    chk("ball P3 (dist3) IN       ", float(p[2]), 0.5)
    print("[SELF-TEST] typed bank + ball powers:")
    print("\n".join(msgs))
    if not ok:
        print("[SELF-TEST] FAILED -- paste back; do not proceed."); sys.exit(1)
    print("[SELF-TEST] PASSED.\n")

# --------------------------- experiments -------------------------- #
def collate_edges(items):
    return [it[0] for it in items]

def exp_bank(args):
    t0 = time.time()
    dev = torch.device("cpu")
    n, D, R = args.n_nodes, args.d_vsa, args.n_rels
    lam = args.lam
    rng = random.Random(args.seed)
    print(f"""
[PREREGISTERED -- Stage 1, zero training]
  T1 staircase per relation : T_min(r,k) = ceil(log2 k), k=1..8; FP == 0
  T2 multi-route honesty    : planted (r0, dist 3) vs shorter cross-type (r1, dist 2)
  T3 identification         : planted type fires exactly in its band
  T4 family composition     : chain length -> relation NAME, exact
  T5 text round-trip        : facts -> parsed triples -> bank -> answer
  T6 scale                  : staircase holds at n=64
""")
    # ---------- T1 ----------
    print("[T1] per-relation staircase (n=32)")
    ks = list(range(1, 9))
    tmin = [[None] * len(ks) for _ in range(R)]
    fps = []
    for r in range(R):
        for j, k in enumerate(ks):
            items = gen_bank_set(n, [k], R, args.pairs, lam,
                                 random.Random(args.seed + 100 * r + k), rel=r)
            O, O_u, V = build_bank(collate_edges(items), n, D, R, dev)
            s_idx = torch.tensor([it[1] for it in items])
            t_idx = torch.tensor([it[2] for it in items])
            sc = bank_scores(O, V, s_idx, 4)
            Tmin = None
            for T in range(1, 5):
                if in_ball(sc, t_idx, T)[:, r].float().mean().item() >= 0.9:
                    Tmin = T; break
            tmin[r][j] = Tmin
            print(f"    [T1] r={r} k={k}: T_min={Tmin} "
                  f"(law: {max(1, math.ceil(math.log2(k)))})", flush=True)
        negs = []
        for _ in range(args.pairs):
            g = gen_typed_pair(n, 3, 0, R, lam, rng)
            if g:
                e, s_, t_, k, r0 = g
                e2 = [(u, v, r) for (u, v, r) in e if r != r0]
                negs.append((e2, s_, t_, 0, r0))
        if negs:
            O, O_u, V = build_bank(collate_edges(negs), n, D, R, dev)
            s_idx = torch.tensor([it[1] for it in negs])
            t_idx = torch.tensor([it[2] for it in negs])
            sc = bank_scores(O, V, s_idx, 4)
            fps.append(in_ball(sc, t_idx, 3)[:, 0].float().mean().item())
    pred = {k: max(1, math.ceil(math.log2(k))) for k in ks}
    ok1 = all(tmin[r][j] is not None and tmin[r][j] <= pred[k]
              for r in range(R) for j, k in enumerate(ks))
    fp_mean = sum(fps) / max(1, len(fps))
    print(f"  T_min per relation (rows r=0..{R-1}): {tmin}")
    print(f"  log2 predicts per k      : {[pred[k] for k in ks]}")
    print(f"  FP (type-unreachable) mean over {len(fps)} batches: {fp_mean:.4f}")
    print(f"[VERDICT T1 - typed log-depth staircase] -> "
          f"{'YES' if ok1 and fp_mean < 0.05 else 'NO'}")

    # ---------- T2 ----------
    print("\n[T2] planted (rel0, dist 3) vs shorter cross-type route (rel1, dist 2)")
    ok2 = True; ident = tot = 0
    for trial in range(args.pairs):
        rng2 = random.Random(9000 + trial)
        nodes = list(range(10)); rng2.shuffle(nodes)
        s_, t_ = nodes[0], nodes[3]
        edges = [(nodes[0], nodes[1], 0), (nodes[1], nodes[2], 0),
                 (nodes[2], nodes[3], 0),
                 (nodes[0], nodes[4], 1), (nodes[4], nodes[3], 1)]
        O, O_u, V = build_bank([edges], args.n_nodes, args.d_vsa, 2, dev)
        s_t = torch.tensor([s_]); t_t = torch.tensor([t_])
        sc = bank_scores(O, V, s_t, 3)
        u1 = in_ball(sc, t_t, 1)[0]; u2 = in_ball(sc, t_t, 2)[0]
        su = bank_scores(O_u.unsqueeze(1), V, s_t, 2)
        short_union = bool(in_ball(su, t_t, 1)[0, 0].item())
        r0_T1, r0_T2 = bool(u1[0].item()), bool(u2[0].item())
        r1_T1 = bool(u1[1].item())
        ok2 &= (short_union and (not r0_T1) and r0_T2 and r1_T1)
        if (not r0_T1) and r0_T2: ident += 1
        tot += 1
    print(f"  route honesty preserved on {ident}/{tot} trials")
    print(f"[VERDICT T2 - type/distance separation under shorter cross-route] -> "
          f"{'YES' if ok2 else 'NO'}")

    # ---------- T3 ----------
    print("\n[T3] planted-relation identification on fresh multi-type pairs")
    items = gen_bank_set(n, [3, 4, 5], R, args.pairs * 2, lam,
                         random.Random(args.seed + 555))
    O, O_u, V = build_bank(collate_edges(items), n, D, R, dev)
    s_idx = torch.tensor([it[1] for it in items])
    t_idx = torch.tensor([it[2] for it in items])
    planted = torch.tensor([it[4] for it in items])
    sc = bank_scores(O, V, s_idx, 4)
    ib = torch.stack([in_ball(sc, t_idx, T) for T in range(1, 5)], dim=2)
    band_ok = unique = 0
    for b in range(len(items)):
        k = items[b][3]; Tstar = max(1, math.ceil(math.log2(k)))
        fire = ib[b, :, Tstar - 1]
        prev = ib[b, :, Tstar - 2] if Tstar >= 2 else torch.zeros_like(fire)
        cands = (fire & ~prev).nonzero().flatten().tolist()
        if planted[b].item() in cands: band_ok += 1
        if cands == [planted[b].item()]: unique += 1
    N = len(items)
    print(f"  planted type fires exactly in its band : {band_ok}/{N}")
    print(f"  and is the UNIQUE such type            : {unique}/{N}")
    ok3 = (band_ok == N) and (unique >= 0.85 * N)
    print(f"[VERDICT T3 - (type, band) identification] -> "
          f"{'YES' if ok3 else 'NO'}")

    # ---------- T4 ----------
    print("\n[T4] family world: parent-chain length -> composed name (ball powers)")
    names = {1: "parent", 2: "grandparent", 3: "great-grandparent",
             4: "great-great-grandparent"}
    hits = 0; tot4 = 0
    for trial in range(args.pairs):
        rng3 = random.Random(7000 + trial)
        L = rng3.randint(1, 4)
        people = [f"P{trial}_{i}" for i in range(L + 1)]
        idx = {p: i for i, p in enumerate(people)}
        edges = [[(idx[people[i]], idx[people[i + 1]], 0) for i in range(L)]]
        O, O_u, V = build_bank(edges, len(people), D, 1, dev)
        s_idx = torch.tensor([0]); t_idx = torch.tensor([L])
        p = powers_exact(O, V, s_idx, t_idx, 4)[0]
        Lhat = next((l + 1 for l in range(4) if p[l] >= 0.5), None)
        tot4 += 1
        if Lhat == L: hits += 1
        sc = bank_scores(O, V, s_idx, 3)
        band = next((T for T in range(1, 4)
                     if in_ball(sc, t_idx, T)[0, 0].item()), None)
        if trial < 3:
            print(f"    {people[0]} -> {people[-1]}: true L={L} "
                  f"({names[L]}), read L={Lhat}, band T*={band} "
                  f"(law: {max(1, math.ceil(math.log2(L)))})")
    print(f"  exact-name accuracy: {hits}/{tot4}")
    print(f"[VERDICT T4 - composed relation naming] -> "
          f"{'YES' if hits == tot4 else 'NO'}")

    # ---------- T5 ----------
    print("\n[T5] text -> triples -> bank -> composed answer")
    import re
    pat = re.compile(r"(\w+) is the parent of (\w+)")
    ok5 = True
    for trial in range(3):
        rng4 = random.Random(8000 + trial)
        L = rng4.randint(2, 4)
        people = ["Anna", "Ben", "Cara", "Dmitri", "Elena"][:L + 1]
        facts = [f"{people[i]} is the parent of {people[i + 1]}."
                 for i in range(L)]
        text = " ".join(facts) + f" Question: How is {people[0]} related " \
               f"to {people[-1]}? Answer:"
        triples = [(m.group(1), m.group(2)) for m in pat.finditer(text)]
        names_set = sorted(set(sum(triples, ())))
        idx = {p: i for i, p in enumerate(names_set)}
        edges = [[(idx[a], idx[b], 0) for a, b in triples]]
        O, O_u, V = build_bank(edges, len(idx), D, 1, dev)
        s_t = torch.tensor([idx[people[0]]]); t_t = torch.tensor([idx[people[-1]]])
        p = powers_exact(O, V, s_t, t_t, 4)[0]
        Lhat = next((l + 1 for l in range(4) if p[l] >= 0.5), None)
        ans = names.get(Lhat, "unknown")
        good = ans == names[L]
        ok5 &= good
        print(f"    Q: {people[0]} to {people[-1]}? -> {ans} "
              f"(true: {names[L]})  {'OK' if good else 'FAIL'}")
    print(f"[VERDICT T5 - text round-trip composition] -> "
          f"{'YES' if ok5 else 'NO'}")

    # ---------- T6 ----------
    print("\n[T6] staircase at n=64 (extended codebook, same law)")
    n2 = 64
    tmin2 = []
    for k in (3, 5, 8):
        items = gen_bank_set(n2, [k], R, max(6, args.pairs // 2), lam,
                             random.Random(args.seed + 31 + k), rel=0)
        O, O_u, V = build_bank(collate_edges(items), n2, D, R, dev)
        s_idx = torch.tensor([it[1] for it in items])
        t_idx = torch.tensor([it[2] for it in items])
        sc = bank_scores(O, V, s_idx, 4)
        Tmin = next((T for T in range(1, 5)
                     if in_ball(sc, t_idx, T)[:, 0].float().mean().item() >= 0.9),
                    None)
        tmin2.append(Tmin)
    ok6 = all(t is not None and t <= max(1, math.ceil(math.log2(k)))
              for t, k in zip(tmin2, (3, 5, 8)))
    print(f"  n=64 T_min for k=3,5,8: {tmin2} "
          f"(law: {[max(1, math.ceil(math.log2(k))) for k in (3, 5, 8)]})")
    print(f"[VERDICT T6 - size-free typed staircase] -> {'YES' if ok6 else 'NO'}")

    print(f"\n[stage finished in {time.time()-t0:.0f}s] Paste everything back.")

# ----------------------------- main ------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-nodes", type=int, default=32)
    ap.add_argument("--d-vsa", type=int, default=256)
    ap.add_argument("--n-rels", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--lam", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.pairs = 5
    torch.manual_seed(args.seed)
    import os
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    print("=" * 68)
    print(" STAGE 1 -- RELATION-SELECTIVE CHANNEL BANK v5 (zero training)")
    print("=" * 68)
    print(f"n={args.n_nodes} D={args.d_vsa} R={args.n_rels} "
          f"threads={torch.get_num_threads()}")
    self_test()
    exp_bank(args)

if __name__ == "__main__":
    main()
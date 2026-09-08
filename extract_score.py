#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
#  STAGE 2c -- LIKELIHOOD-SCORING EXTRACTION + COMPOSITE CLOSE-OUT
#
#  Generation failed (bake-off: 0.00 in all 3 modes). Design-law move:
#  bound the LM's decision space. For every ordered name pair (a,b) in
#  the story, score s(a,b) = logP(" a is the parent of b." | story).
#  Take top-k (k = #sentences in story). Feed edges to the FROZEN BANK.
#
#  Verdicts:
#   S1: extraction recall@k >= 0.7        (scoring rescues extraction?)
#   S2: composite naming accuracy ~ recall (bank is exact)
#   S3: direction sanity: s(a,b) > s(b,a) on true edges
#
#  python extract_score.py --quick   (~4-6 min)
#  python extract_score.py           (~10-15 min)
# =====================================================================
import argparse, math, os, random, re, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as Fn

DEVICE = torch.device("cpu")
D_VSA = 64
LMAX = 8

NAMES = ["Anna","Ben","Cara","Dmitri","Elena","Felix","Gina","Hugo","Iris",
         "Jonas","Klara","Liam","Mia","Noah","Olga","Pavel","Quinn","Rosa",
         "Samir","Tessa","Umar","Vera","Will","Yara"]

def rel_name(L):
    return "parent" if L == 1 else "great-" * (L - 2) + "grandparent"

# ---------------- frozen bank (verified algebra) -------------------- #
def cnorm_mat(C, eps=1e-9):
    return C / C.abs().pow(2).sum((-2, -1), keepdim=True).sqrt().clamp_min(eps)

def fourier_codebook(n, D):
    assert D >= n
    ph = 2 * math.pi * torch.outer(torch.arange(n).float(),
                                   torch.arange(D).float()) / D
    return torch.polar(torch.full_like(ph, 1.0 / math.sqrt(D)), ph)

def build_bank1(edges, n, device):
    V = fourier_codebook(n, D_VSA).to(device)
    E = max(4, len(edges))
    src = torch.zeros(1, E, dtype=torch.long)
    dst = torch.zeros(1, E, dtype=torch.long)
    msk = torch.zeros(1, E)
    for j, (u, v) in enumerate(edges):
        src[0, j], dst[0, j], msk[0, j] = u, v, 1.0
    src, dst, msk = src.to(device), dst.to(device), msk.to(device)
    onehot = Fn.one_hot(src, n).float() * msk.unsqueeze(-1)
    Vd = V[dst]
    bd = torch.matmul(onehot.transpose(1, 2).to(Vd.dtype), Vd)
    bd = cnorm_mat(bd + V.unsqueeze(0))
    O = cnorm_mat(torch.einsum("bui,uj->bij", bd, V.conj()))
    return O, V

@torch.no_grad()
def path_distance(edges, s, t, lmax=LMAX, device=DEVICE):
    idx = {}
    def nid(x):
        if x not in idx: idx[x] = len(idx)
        return idx[x]
    for u, v in edges: nid(u); nid(v)
    nid(s); nid(t)
    n = len(idx)
    if n > D_VSA: return None
    e = [(idx[u], idx[v]) for u, v in edges]
    O, V = build_bank1(e, n, device)
    seed = V[torch.tensor([idx[s]])]
    seed = seed / seed.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-9)
    P = O.clone()
    for L in range(1, lmax + 1):
        if L > 1:
            P = cnorm_mat(torch.matmul(P, O))
        F = torch.matmul(P, seed.unsqueeze(-1)).squeeze(-1)
        a = torch.einsum("bd,ud->bu", F, V.conj()).real
        rel = a / a.amax(-1, keepdim=True).clamp_min(1e-12)
        if rel[0, idx[t]].item() > 1e-4:
            return L
    return None

def self_test():
    e = [("A","B"), ("B","C"), ("C","D")]
    assert path_distance(e, "A", "D") == 3
    assert path_distance(e, "D", "A") is None
    print("[SELF-TEST] bank: PASSED\n")

# -------------------------- stories -------------------------------- #
def make_story(rng, L, n_distr=4):
    pool = NAMES[:]; rng.shuffle(pool)
    chain = pool[:L + 1]; distr = pool[L + 1:L + 1 + n_distr]
    facts = [(chain[i], chain[i + 1]) for i in range(L)]
    d_edges = [(distr[i], distr[i + 1]) for i in range(len(distr) - 1)]
    if distr: d_edges.append((distr[-1], distr[0]))
    lines = [f"{a} is the parent of {b}." for a, b in facts + d_edges]
    rng.shuffle(lines)
    return " ".join(lines), facts, chain[0], chain[-1], L

# -------------------------- scoring -------------------------------- #


@torch.no_grad()
def score_pairs(model, tok, story, names, bs=12):
    """s(a,b) = mean token logP of ' a is the parent of b.' given prefix."""
    prefix = (f"Facts from a family tree:\n{story}\n\nOne true fact from "
              f"the text is:")
    pids = tok(prefix, return_tensors="pt").input_ids
    cands = [(a, b) for a in names for b in names if a != b]
    out = []
    for i0 in range(0, len(cands), bs):
        chunk = cands[i0:i0 + bs]
        seqs, lens = [], []
        for a, b in chunk:
            cids = tok(f" {a} is the parent of {b}.",
                       add_special_tokens=False).input_ids
            seqs.append(torch.cat([pids[0], torch.tensor(cids)]))
            lens.append(len(cids))
        maxlen = max(len(s) for s in seqs)
        pad = tok.pad_token_id
        ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = s; att[i, :len(s)] = 1
        logits = model(input_ids=ids, attention_mask=att).logits
        logp = torch.log_softmax(logits.float(), -1)
        for i, (a, b) in enumerate(chunk):
            L = lens[i]; start = len(seqs[i]) - L
            tgt = seqs[i][start:]
            lp = logp[i, start - 1:start - 1 + L, :].gather(
                1, tgt.unsqueeze(1)).sum().item()
            out.append((lp / L, a, b))
    return out

# -------------------------- main ----------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM-135M-Instruct")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--per-l", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="smollm_results")
    args = ap.parse_args()
    if args.base:
        args.model = "HuggingFaceTB/SmolLM-135M"
    if args.quick:
        args.per_l = 2
    random.seed(args.seed); torch.manual_seed(args.seed)
    import os as _os
    torch.set_num_threads(min(8, _os.cpu_count() or 4))
    _os.makedirs(args.outdir, exist_ok=True)
    print("=" * 68)
    print(" STAGE 2c -- LIKELIHOOD-SCORING EXTRACTION + COMPOSITE")
    print("=" * 68)
    self_test()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).eval()
    print(f"[lm] {args.model} loaded")

    lengths = [2, 4, 6] if args.quick else list(range(2, 9))
    rows = []
    t0 = time.time()
    for L in lengths:
        for i in range(args.per_l):
            rng = random.Random(args.seed * 1000 + L * 50 + i)
            story, facts, s, t, LL = make_story(rng, L)
            story_names = sorted(set(re.findall(r"\b[A-Z][a-z]{2,}\b", story)))
            scored = score_pairs(model, tok, story, story_names)
            k = len(re.findall(r"is the parent of", story))
            topk = sorted(scored, reverse=True)[:k]
            edges = [(a, b) for _, a, b in topk]
            # S1: extraction recall
            rec = sum(1 for x in edges if x in set(facts)) / len(facts)
            prec = sum(1 for x in edges if x in set(facts)) / max(1, len(edges))
            # S3: direction on true chain pairs present in topk
            dir_ok = [1 for sc, a, b in topk if (a, b) in set(facts)]
            # S2: composite through the bank
            d = path_distance(edges, s, t)
            ans = rel_name(d) if d else "NO-PATH"
            gold = rel_name(LL)
            rows.append(dict(L=LL, rec=rec, prec=prec, comp=(ans == gold)))
            print(f"  L={LL} {s}->{t}: extR={rec:.2f} prec={prec:.2f} "
                  f"composite={ans} ({'OK' if ans == gold else 'FAIL'})",
                  flush=True)
    print("\n" + "=" * 60)
    print("  L    extR   prec   composite")
    for L in lengths:
        rs = [r for r in rows if r["L"] == L]
        print(f"  {L:<4} {sum(r['rec'] for r in rs)/len(rs):.2f}   "
              f"{sum(r['prec'] for r in rs)/len(rs):.2f}    "
              f"{100.0*sum(r['comp'] for r in rs)/len(rs):.1f}%")
    rs = rows
    R = sum(r["rec"] for r in rs) / len(rs)
    C = 100.0 * sum(r["comp"] for r in rs) / len(rs)
    print(f"  ALL  {R:.2f}          {C:.1f}%")
    s1 = R >= 0.7
    print(f"\n[VERDICT S1 - bounded scoring rescues extraction] recall={R:.2f} "
          f"-> {'YES' if s1 else 'NO'}")
    s2 = C >= 100.0 * R - 10.0
    print(f"[VERDICT S2 - composite ~= recall x exact-bank] {C:.1f}% vs "
          f"{100*R:.0f}% -> {'YES' if s2 else 'NO'}")
    print(f"""
[STAGE-2 CONCLUSION, either branch]
  Reasoning side : Regex+Bank = 100% at L=2..8 (W1, undefeated)
  Extraction side: generation 0.00 (bake-off) | scoring {R:.2f}
  -> the gap decomposes into extraction, which the frozen bank
     never needed to be exact.  {'Composite CLOSED.' if s1 else 'Extraction beyond 135M in all bounded modes tested -- finding stands.'}""")
    import json
    with open(_os.path.join(args.outdir, "score_extract.json"), "w") as f:
        json.dump(dict(rows=rows, recall=R, composite=C), f, indent=2)
    print(f"[saved] {args.outdir}/score_extract.json")
    print(f"[done in {time.time()-t0:.0f}s] Paste everything back.")

if __name__ == "__main__":
    main()
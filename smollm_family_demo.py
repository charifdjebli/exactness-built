#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
#  STAGE 2 v2 -- SMOLLM x FROZEN RELATION BANK (the money table)
#
#  v1 findings: Regex+Bank 100% at every L (W1 banked). Extraction
#  failed (extR=0.00) blind -- v2 adds: RAW extraction output printed
#  (instrument before diagnosing), a permissive extractor (capitalized
#  name pairs, with -> arrow AND prose fallbacks), the ALL-row fix.
#
#  python smollm_family_demo.py --quick
# =====================================================================
import argparse, hashlib, json, math, os, random, re, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as Fn

DEVICE = torch.device("cpu")
D_VSA = 64
LMAX = 8

NAMES = ["Anna","Ben","Cara","Dmitri","Elena","Felix","Gina","Hugo","Iris",
         "Jonas","Klara","Liam","Mia","Noah","Olga","Pavel","Quinn","Rosa",
         "Samir","Tessa","Umar","Vera","Will","Yara"]
EX_NAMES = ["Nina","Omar","Pia","Rita","Sven","Timo"]

def rel_name(L):
    if L == 1: return "parent"
    return "great-" * (L - 2) + "grandparent"

REL_VOCAB = ["parent", "grandparent"] + \
            [f"great-{'great-' * i}grandparent" for i in range(0, 7)]

# -------------------------- bank (verified v5 algebra) ------------- #
def cnorm_mat(C, eps=1e-9):
    return C / C.abs().pow(2).sum((-2, -1), keepdim=True).sqrt().clamp_min(eps)

def fourier_codebook(n, D):
    assert D >= n
    ph = 2 * math.pi * torch.outer(torch.arange(n).float(),
                                   torch.arange(D).float()) / D
    return torch.polar(torch.full_like(ph, 1.0 / math.sqrt(D)), ph)

def build_bank1(edges, n, device):
    B = 1
    V = fourier_codebook(n, D_VSA).to(device)
    E = max(4, len(edges))
    src = torch.zeros(B, E, dtype=torch.long)
    dst = torch.zeros(B, E, dtype=torch.long)
    msk = torch.zeros(B, E)
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
        if x not in idx:
            idx[x] = len(idx)
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

def bfs_dist(edges, s, t):
    adj = {}
    for u, v in edges: adj.setdefault(u, []).append(v)
    seen = {s}; q = [s]; d = 0
    while q:
        nxt = []
        for u in q:
            if u == t: return d
            for v in adj.get(u, []):
                if v not in seen:
                    seen.add(v); nxt.append(v)
        q = nxt; d += 1
    return None

def self_test():
    e = [("A","B"), ("B","C"), ("C","D")]
    assert bfs_dist(e, "A", "D") == 3
    assert path_distance(e, "A", "D") == 3
    assert path_distance(e, "A", "C") == 2
    assert path_distance(e, "D", "A") is None
    print("[SELF-TEST] bank path-distance: PASSED\n")

# -------------------------- story generation ----------------------- #
def make_story(rng, L, n_distr=4):
    pool = NAMES[:]; rng.shuffle(pool)
    chain = pool[:L + 1]; rest = pool[L + 1:]
    distr = rest[:n_distr]
    facts = [(chain[i], chain[i + 1]) for i in range(L)]
    distr_edges = []
    for i in range(len(distr) - 1):
        distr_edges.append((distr[i], distr[i + 1]))
    if distr:
        distr_edges.append((distr[-1], distr[0]))
    all_true = facts + distr_edges
    assert bfs_dist(all_true, chain[0], chain[-1]) == L
    lines = [f"{a} is the parent of {b}." for a, b in all_true]
    rng.shuffle(lines)
    return dict(story=" ".join(lines), true_edges=all_true,
                chain_len=L, s=chain[0], t=chain[-1],
                chain_facts=facts, n_distr_edges=len(distr_edges))

# -------------------------- LM wrapper ----------------------------- #
class LM:
    def __init__(self, model_id, outdir):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.instruct = "instruct" in model_id.lower()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.eval()
        self.cache_path = os.path.join(outdir, "gen_cache.json")
        self.cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path) as f: self.cache = json.load(f)
        print(f"[lm] {model_id} loaded")
    def _wrap(self, prompt):
        if self.instruct:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        return prompt
    @torch.no_grad()
    def gen(self, prompt, max_new=48):
        key = hashlib.sha1((prompt + f"#{max_new}").encode()).hexdigest()[:16]
        if key in self.cache:
            return self.cache[key]
        enc = self.tok(self._wrap(prompt), return_tensors="pt")
        out = self.model.generate(**enc, max_new_tokens=max_new,
                                  do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        txt = self.tok.decode(out[0][enc["input_ids"].shape[1]:],
                              skip_special_tokens=True)
        self.cache[key] = txt
        return txt
    def save_cache(self):
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)
    @torch.no_grad()
    def p_yes(self, prompt):
        enc = self.tok(self._wrap(prompt), return_tensors="pt")
        out = self.model(**enc, use_cache=False)
        logits = out.logits[0, -1]
        yid = self.tok(" Yes", add_special_tokens=False).input_ids[0]
        nid = self.tok(" No", add_special_tokens=False).input_ids[0]
        return torch.sigmoid(logits[yid] - logits[nid]).item()

# -------------------------- prompts / parsing ---------------------- #
def extract_prompt(story):
    ex = (f"Text: Nina is the parent of Omar. Pia is the parent of Nina.\n"
          f"Pairs:\nNina -> Omar\nPia -> Nina")
    return ("Extract all \"X is the parent of Y\" relationships from the "
            "text. Write one per line in the format: X -> Y\n\n"
            f"Example:\n{ex}\n\nText: {story}\nPairs:")

def direct_prompt(story, s, t, shots=None):
    base = (f"{story}\nQuestion: How is {s} related to {t}? Answer with "
            f"exactly one relation: parent, grandparent, great-grandparent, "
            f"great-great-grandparent, ...\nAnswer:")
    if not shots:
        return base
    demo = [(EX_NAMES[0], EX_NAMES[1], 1), (EX_NAMES[1], EX_NAMES[2], 2),
            (EX_NAMES[2], EX_NAMES[4], 3)]
    ex = [f"Text: {a} is the parent of {b}.\nQuestion: How is {a} related "
          f"to {b}?\nAnswer: {rel_name(L)}" for a, b, L in demo]
    return "\n\n".join(ex) + f"\n\n{base}"

def reach_prompt(story, s, t):
    return (f"{story}\nQuestion: Starting from {s} and following \"is the "
            f"parent of\" links, can you reach {t}? Answer Yes or No.\nAnswer:")

def parse_name(text):
    t = (text or "").lower()
    for r in sorted(REL_VOCAB, key=len, reverse=True):
        if r in t: return r
    return None

NAME_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
ARROW_RE = re.compile(r"([A-Z][a-z]+)\s*(?:->|—|-->|=>|to)\s*([A-Z][a-z]+)")

def parse_triples(text, valid_names):
    """Permissive: arrow patterns first, then prose fallback
    'X is the parent of Y'. Validated against the story's name pool."""
    vs = set(valid_names)
    out = []
    for a, b in ARROW_RE.findall(text):
        if a in vs and b in vs and a != b:
            out.append((a, b))
    if not out:
        for a, b in re.findall(r"(\w+) is the parent of (\w+)", text):
            if a in vs and b in vs and a != b:
                out.append((a, b))
    # dedup, keep order
    seen = set(); res = []
    for x in out:
        if x not in seen:
            seen.add(x); res.append(x)
    return res

# -------------------------- main ----------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM-135M-Instruct")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--per-l", type=int, default=4)
    ap.add_argument("--skip-lm", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="smollm_results")
    args = ap.parse_args()
    if args.base:
        args.model = "HuggingFaceTB/SmolLM-135M"
    if args.quick:
        args.per_l = 2
    torch.manual_seed(args.seed); random.seed(args.seed)
    import os as _os
    torch.set_num_threads(min(8, _os.cpu_count() or 4))
    _os.makedirs(args.outdir, exist_ok=True)
    print("=" * 68)
    print(" STAGE 2 v2 -- SMOLLM x FROZEN RELATION BANK (money table)")
    print("=" * 68)
    self_test()

    lengths = [2, 4, 6] if args.quick else list(range(2, 9))
    stories = []
    for L in lengths:
        for i in range(args.per_l):
            rng = random.Random(args.seed * 1000 + L * 50 + i)
            st = make_story(rng, L)
            st["key"] = f"L{L}_{i}"
            stories.append(st)
    print(f"[data] {len(stories)} stories, L in {lengths}\n")

    lm = None
    if not args.skip_lm:
        try:
            import transformers  # noqa
        except ImportError:
            sys.exit("pip install transformers  (or --skip-lm)")
        lm = LM(args.model, args.outdir)

    rows = []
    t0 = time.time()
    for si, st in enumerate(stories):
        s, t, L = st["s"], st["t"], st["chain_len"]
        gold = rel_name(L)
        rex = re.findall(r"(\w+) is the parent of (\w+)", st["story"])
        dR = path_distance(rex, s, t)
        a_regex = rel_name(dR) if dR else "NO-PATH"
        row = dict(key=st["key"], L=L, gold=gold,
                   regex_ok=(a_regex == gold))
        if lm is not None:
            raw = lm.gen(extract_prompt(st["story"]), max_new=80)
            lm.save_cache()
            if si < 3:                                   # INSTRUMENT: raw output
                print(f"    [raw-extract L={L}] {raw[:220]!r}")
            valid = set(NAMES) | set(EX_NAMES)
            triples = parse_triples(raw, valid)
            true_set = set(st["true_edges"]); chain_set = set(st["chain_facts"])
            prec = (sum(1 for x in triples if x in true_set)
                    / len(triples)) if triples else 0.0
            rec = (sum(1 for x in triples if x in chain_set)
                   / len(st["chain_facts"]))
            dF = path_distance(triples, s, t)
            a_full = rel_name(dF) if dF else "NO-PATH"
            d0 = parse_name(lm.gen(direct_prompt(st["story"], s, t),
                                   max_new=16) or "")
            df = parse_name(lm.gen(direct_prompt(st["story"], s, t, shots=True),
                                   max_new=16) or "")
            py = lm.p_yes(reach_prompt(st["story"], s, t))
            row.update(direct_ok=(d0 == gold), few_ok=(df == gold),
                       reach_p=py, reach_ok=((py > 0.5) == (L <= LMAX)),
                       ext_recall=rec, ext_prec=prec,
                       n_ext=len(triples), full_ok=(a_full == gold),
                       full_ans=a_full, direct_ans=d0 or "?")
        rows.append(row)
        msg = (f"  [{si+1}/{len(stories)}] L={L} {s}->{t}: gold={gold} "
               f"regex={'OK' if row['regex_ok'] else 'FAIL'}")
        if lm is not None:
            msg += (f" direct={row['direct_ans']} full={row['full_ans']}"
                    f" extR={row['ext_recall']:.2f}")
        print(msg, flush=True)

    def agg(filt, field):
        v = [r[field] for r in rows if filt(r)]
        return 100.0 * sum(v) / max(1, len(v)), len(v)
    def agg_f(filt, field):
        v = [r[field] for r in rows if filt(r)]
        return sum(v) / max(1, len(v))

    print("\n" + "=" * 86)
    print("  THE MONEY TABLE  (name exact-match %; extR = chain-fact recall)")
    print("=" * 86)
    print(f"  {'L':>3} {'n':>3} {'extR':>6} {'Direct':>8} {'FewShot':>8} "
          f"{'ReachAcc':>9} {'Regex+Bank':>11} {'Full':>7}")
    for L in lengths:
        f = lambda r, L=L: r["L"] == L
        nL = sum(1 for r in rows if f(r))
        if lm is not None:
            print(f"  {L:>3} {nL:>3} {agg_f(f,'ext_recall'):>6.2f} "
                  f"{agg(f,'direct_ok')[0]:>8.1f} {agg(f,'few_ok')[0]:>8.1f} "
                  f"{agg_f(f,'reach_ok')*100:>9.1f} "
                  f"{agg(f,'regex_ok')[0]:>11.1f} {agg(f,'full_ok')[0]:>7.1f}")
        else:
            print(f"  {L:>3} {nL:>3} {'-':>6} {'-':>8} {'-':>8} {'-':>9} "
                  f"{agg(f,'regex_ok')[0]:>11.1f} {'-':>7}")
    f_all = lambda r: True
    nALL = len(rows)
    print("  " + "-" * 84)
    if lm is not None:
        print(f"  ALL{nALL:>3} {agg_f(f_all,'ext_recall'):>6.2f} "
              f"{agg(f_all,'direct_ok')[0]:>8.1f} {agg(f_all,'few_ok')[0]:>8.1f} "
              f"{agg_f(f_all,'reach_ok')*100:>9.1f} "
              f"{agg(f_all,'regex_ok')[0]:>11.1f} "
              f"{agg(f_all,'full_ok')[0]:>7.1f}")
    else:
        print(f"  ALL{nALL:>3} {'-':>6} {'-':>8} {'-':>8} {'-':>9} "
              f"{agg(f_all,'regex_ok')[0]:>11.1f} {'-':>7}")

    print()
    w1 = agg(f_all, "regex_ok")[0] == 100.0
    print(f"[VERDICT W1 - reasoning exact through oracle extractor] "
          f"Regex+Bank = {agg(f_all,'regex_ok')[0]:.1f}% -> "
          f"{'YES' if w1 else 'NO'}")
    if lm is not None:
        deep = [r for r in rows if r["L"] >= 4]
        fd = 100.0 * sum(r["few_ok"] for r in deep) / max(1, len(deep))
        ff = 100.0 * sum(r["full_ok"] for r in deep) / max(1, len(deep))
        w2 = fd < 60.0
        print(f"[VERDICT W2 - LM degrades on deep chains] few-shot at L>=4 = "
              f"{fd:.1f}% -> {'YES' if w2 else 'NO (LM stronger)'}")
        w3 = (ff >= 85.0) and (fd < 60.0)
        print(f"[VERDICT W3 - flat vs degrading] Full at L>=4 = {ff:.1f}% "
              f"vs LM {fd:.1f}% -> {'YES' if w3 else 'NO'}")
        er = agg_f(f_all, "ext_recall")
        fr = agg(f_all, "full_ok")[0] / 100.0
        print(f"[DECOMPOSITION] extraction recall={er:.2f}, full={fr:.2f}, "
              f"regex+bank=1.00 -> errors are "
              f"{'extraction-side' if fr >= 0.95*er - 0.05 else 'mixed'}")
        with open(os.path.join(args.outdir, "family_demo.json"), "w") as f:
            json.dump(dict(rows=rows, W1=w1, W2=w2, W3=w3, ext_recall=er),
                      f, indent=2, default=str)
        print(f"[saved] {args.outdir}/family_demo.json")
    print(f"\n[done in {time.time()-t0:.0f}s] Paste the full output back.")

if __name__ == "__main__":
    main()
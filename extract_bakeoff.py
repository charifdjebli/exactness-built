#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =====================================================================
#  EXTRACTOR BAKE-OFF -- who can pull "X is the parent of Y" pairs
#  out of a family story at 135M scale?
#    E1: Instruct + strict no-code prompt
#    E2: Instruct + few-shot completion
#    E3: Base    + few-shot completion (pattern continuation -- the bet)
#  Prints RAW outputs for the first story of each strategy and recall
#  for all.  Winner feeds the money table.
#  python extract_bakeoff.py        (~3-4 min, downloads base weights once)
# =====================================================================
import argparse, hashlib, json, os, random, re, sys, time
import torch

NAMES = ["Anna","Ben","Cara","Dmitri","Elena","Felix","Gina","Hugo","Iris",
         "Jonas","Klara","Liam","Mia","Noah","Olga","Pavel","Quinn","Rosa",
         "Samir","Tessa","Umar","Vera","Will","Yara"]
EX = [("Nina","Omar"),("Omar","Pia"),("Pia","Rita")]

def make_story(rng, L, n_distr=4):
    pool = NAMES[:]; rng.shuffle(pool)
    chain = pool[:L+1]; distr = pool[L+1:L+1+n_distr]
    facts = [(chain[i], chain[i+1]) for i in range(L)]
    d_edges = [(distr[i], distr[i+1]) for i in range(len(distr)-1)]
    if distr: d_edges.append((distr[-1], distr[0]))
    lines = [f"{a} is the parent of {b}." for a, b in facts + d_edges]
    rng.shuffle(lines)
    return " ".join(lines), facts

class LM:
    def __init__(self, model_id):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.instruct = "instruct" in model_id.lower()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id).eval()
        print(f"[lm] {model_id}")
    def wrap(self, p):
        if self.instruct:
            return self.tok.apply_chat_template(
                [{"role":"user","content":p}], tokenize=False,
                add_generation_prompt=True)
        return p
    @torch.no_grad()
    def gen(self, p, mx=90):
        enc = self.tok(self.wrap(p), return_tensors="pt")
        out = self.model.generate(**enc, max_new_tokens=mx, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)

def shots_block(pairs_src=None):
    lines = []
    for a, b in EX:
        lines.append(f"{a} -> {b}")
    return "\n".join(lines)

def p_e1(story):     # Instruct, strict no-code
    return ("Extract the parent pairs from the text below. Output ONLY "
            "pairs in the format 'Name -> Name', one per line. Do NOT "
            "write code. Do NOT explain.\n\n"
            f"Example:\nNina is the parent of Omar. Pia is the parent of "
            f"Nina.\nNina -> Omar\nPia -> Nina\n\n"
            f"Text: {story}\n\nPairs:")

def p_e2(story):     # Instruct, few-shot completion style
    return ("Text: Nina is the parent of Omar. Pia is the parent of Nina.\n"
            "Nina -> Omar\nPia -> Nina\n\n"
            "Text: Omar is the parent of Rita. Sven is the parent of Tina.\n"
            "Omar -> Rita\nSven -> Tina\n\n"
            f"Text: {story}\n")

def p_e3(story):     # Base, pure pattern continuation
    return ("X is the parent of Y.\nX -> Y\n\n"
            "Nina is the parent of Omar. Pia is the parent of Nina.\n"
            "Nina -> Omar\nPia -> Nina\n\n"
            "Omar is the parent of Rita. Sven is the parent of Tina.\n"
            "Omar -> Rita\nSven -> Tina\n\n"
            f"{story}\n")

NAME_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
ARROW = re.compile(r"([A-Z][a-z]+)\s*(?:->|=>|-->|to)\s*([A-Z][a-z]+)")

def parse_triples(text, pool):
    vs = set(pool); out = []
    for a, b in ARROW.findall(text or ""):
        if a in vs and b in vs and a != b: out.append((a, b))
    seen = set(); res = []
    for x in out:
        if x not in seen: seen.add(x); res.append(x)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="smollm_results")
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    import os as _os
    torch.set_num_threads(min(8, _os.cpu_count() or 4))
    _os.makedirs(args.outdir, exist_ok=True)
    print("=" * 68)
    print(" EXTRACTOR BAKE-OFF -- 135M family-pair extraction")
    print("=" * 68)
    stories = []
    for L in (2, 4, 6, 8):
        for i in range(2):
            rng = random.Random(args.seed * 31 + L * 7 + i)
            s, facts = make_story(rng, L)
            stories.append((L, s, facts))
    ins = LM("HuggingFaceTB/SmolLM-135M-Instruct")
    base = LM("HuggingFaceTB/SmolLM-135M")
    pool = NAMES
    strategies = [("E1 instruct/no-code", ins, p_e1),
                  ("E2 instruct/fewshot", ins, p_e2),
                  ("E3 base/completion", base, p_e3)]
    results = {}
    for name, lm, pf in strategies:
        print(f"\n--- {name} ---")
        recs = []; shown = 0
        for L, story, facts in stories:
            raw = lm.gen(pf(story))
            if shown < 2:
                print(f"  [raw L={L}] {raw[:200]!r}"); shown += 1
            tr = parse_triples(raw, pool)
            hits = sum(1 for x in tr if x in set(facts))
            recs.append(hits / len(facts))
        m = sum(recs) / len(recs)
        results[name] = m
        print(f"  chain-fact recall: {m:.2f}  ({sum(1 for r in recs if r>=0.99)}/{len(recs)} stories perfect)")
    best = max(results, key=results.get)
    print(f"\n[WINNER] {best}  recall={results[best]:.2f}")
    with open(_os.path.join(args.outdir, "bakeoff.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("[saved] smollm_results/bakeoff.json")

if __name__ == "__main__":
    main()
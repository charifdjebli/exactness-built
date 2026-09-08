
# Exactness Must Be Built

> Code and artifacts for *"Exactness Must Be Built: Frozen Orthonormal Superposition Channels Give Depth-Recurrent Transformers Log-Depth, Size-Free Latent Reasoning"*

---

## ⚡ Quick Start

Install dependencies:
```bash
pip install torch matplotlib
```

Run baseline experiments:
```bash
# Untrained staircase (~15 min)
python campaign.py --exp capacity

# Relation bank (~3 min)
python channel_bank.py --quick
```

---

## 📜 Scripts & Reproduction

| Script | What it reproduces |
| :--- | :--- |
| `campaign.py --exp capacity/main/seeds` | Prop. 1 staircase, causal absorption (Tab. 2), dichotomy (a,c) |
| `campaign_v5.py --exp sets/scale/ball` | Broadcast memory, zero-shot 4x transfer (Tab. 3), moving edge (Fig. 3) |
| `smollm_holo_test.py` | V1–V4, LM interface (Tab. 4 rows V1, B0, C0) |
| `smollm_anchored.py` | Bounded vs unbounded anchoring (Tab. 4 rows D1, D1v2) |
| `channel_bank.py` | Relation bank T1–T6 (Tab. 5) |
| `smollm_family_demo.py` | Money table, regex+bank 100% (Fig. 6 left) |
| `extract_ladder.py` | Extraction scaling curve (Fig. 6 right) |
| `make_figures.py` | All paper figures from JSONs |

> **Note:** Every script runs a `self_test()` that verifies the operator algebra and aborts on failure. Random seeds are fixed; all numbers reported in the paper will regenerate deterministically.
```

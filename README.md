Exactness Must Be Built
Code and artifacts for "Exactness Must Be Built: Frozen OrthonormalSuperposition Channels Give Depth-Recurrent Transformers Log-Depth,Size-Free Latent Reasoning" 

Quick start
pip install torch matplotlibpython campaign.py --exp capacity # the untrained staircase (~15 min)python channel_bank.py --quick # the relation bank (~3 min)

Scripts
Script	What it reproduces
campaign.py --exp capacity/main/seeds	Prop. 1 staircase, causal absorption (Tab. 2), dichotomy (a,c)
campaign_v5.py --exp sets/scale/ball	broadcast memory, zero-shot 4x transfer (Tab. 3), moving edge (Fig. 3)
smollm_holo_test.py	V1-V4, LM interface (Tab. 4 rows V1,B0,C0)
smollm_anchored.py	bounded vs unbounded anchoring (Tab. 4 rows D1,D1v2)
channel_bank.py	relation bank T1-T6 (Tab. 5)
smollm_family_demo.py	money table, regex+bank 100% (Fig. 6 left)
extract_ladder.py	extraction scaling curve (Fig. 6 right)
make_figures.py	all paper figures from JSONs
Every script runs a self_test() that verifies the operator algebra andaborts on failure. Seeds are fixed; all numbers in the paper regenerate.

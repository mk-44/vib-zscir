#!/usr/bin/env python
"""Minimal driver: run negative-structure analysis on JUST the baseline and
the beta=0.1 VIB, to verify Section E's table. Reuses neg_analysis.py functions.
Run from codes/ with PYTHONPATH=/notebooks/experiment/codes
"""
import sys, json
sys.path.insert(0, "/notebooks/experiment/codes")
from evaluation.neg_analysis import CheckpointSpec, analyze_checkpoint

BASE = "/notebooks/experiment"
VAL_SHARD = f"{BASE}/train_data/cc12m-train-0793.tar"

specs = [
    CheckpointSpec(
        name="baseline",
        path=f"{BASE}/4M_checkpoints/base_hd256_tok1_lslr0.2_lsmax116/epoch_02_step_03000.pt",
        group="ref", hidden_dim=256, num_tokens=1, mapper_type="base",
        description="deterministic baseline beta=0",
    ),
    CheckpointSpec(
        name="vib_b0.1",
        path=f"{BASE}/4M_checkpoints/vib_hd256_tok1_lslr0.2_lsmax116_vib0.0001/epoch_01_step_01500.pt",
        group="ref", hidden_dim=256, num_tokens=1, mapper_type="vib",
        description="VIB beta=0.1 headline",
    ),
]

print(f"VAL_SHARD = {VAL_SHARD}")
print("(NOTE: this is a CC12M shard, not CIRCO)\n")

rows = []
for spec in specs:
    print(f"\n{'='*60}\nAnalyzing {spec.name}: {spec.path}\n{'='*60}")
    r = analyze_checkpoint(spec, val_shard_path=VAL_SHARD, device="cuda",
                           max_batches=5, batch_size=1024)
    rows.append((spec.name, r))

print("\n\n" + "="*78)
print("SECTION E VERIFICATION TABLE")
print("="*78)
keys = [
    ("effective_negs_mean",     "Eff negs (w/ temp)"),
    ("effective_negs_raw_mean", "Eff negs (raw)"),
    ("sim_gt_mean",             "Positive similarity"),   # may be named differently
    ("gt_minus_hardest_mean",   "Positive - hardest"),
    ("neg_std_mean",            "Raw negative std"),
]
print(f"{'metric':24s} | {'baseline':>12s} | {'VIB b=0.1':>12s}")
print("-"*56)
for k, label in keys:
    b = rows[0][1].get(k, "N/A")
    v = rows[1][1].get(k, "N/A")
    bs = f"{b:.4f}" if isinstance(b, float) else str(b)
    vs = f"{v:.4f}" if isinstance(v, float) else str(v)
    print(f"{label:24s} | {bs:>12s} | {vs:>12s}")

# also dump ALL keys in case sim_gt is named differently
print("\n--- ALL keys available (baseline) for reference ---")
for k in sorted(rows[0][1].keys()):
    if k.endswith("_mean"):
        print(f"  {k}: {rows[0][1][k]}")
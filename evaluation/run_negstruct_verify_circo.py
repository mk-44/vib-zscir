"""
run_negstruct_verify_circo.py
Verify the Section E negative-structure table on CIRCO (NOT a CC12M shard).

The earlier run used /notebooks/experiment/train_data/cc12m-train-0793.tar, which is
the in-domain training shard -> P(gt) ~ 1.0, eff_negs ~ 5, and the base->VIB direction
flips. Section E is measured on the CIRCO eval embeddings, the open-domain gallery.

This script reuses the EXACT load_dataset / load_model / compute_metrics from
neg_analysis_all.py, so the numbers reproduce that pipeline by construction. It loads
the CIRCO embeddings once (seed-42 shuffle, 5*1024, batches of 1024) and runs both
checkpoints on the SAME batches, then prints the Section E rows and the expected values.

Run:
    cd /notebooks/experiment/codes
    export PYTHONPATH=/notebooks/experiment/codes
    python evaluation/run_negstruct_verify_circo.py 2>&1 | tee /notebooks/experiment/negstruct_verify_circo.txt
"""
import sys

sys.path.insert(0, "/notebooks/experiment/codes")
sys.path.insert(0, "/notebooks/experiment/codes/evaluation")

# Reuse the original pipeline verbatim — same data loading, same metric math.
from neg_analysis_all import load_dataset, load_model, compute_metrics, DATASETS

# Headline checkpoints (the same two used for Section E columns #2 and #5).
CKPTS = {
    "baseline": "/notebooks/experiment/4M_checkpoints/"
                "base_hd256_tok1_lslr0.2_lsmax116/epoch_02_step_03000.pt",
    "VIB b=0.1": "/notebooks/experiment/4M_checkpoints/"
                 "vib_hd256_tok1_lslr0.2_lsmax116_vib0.0001/epoch_01_step_01500.pt",
}

# Section E rows -> the keys compute_metrics() actually returns.
ROWS = [
    ("Eff negs (w/ temp)",  "effective_negs"),
    ("Eff negs (raw)",      "effective_negs_raw"),
    ("Positive similarity", "sim_p_vraw"),       # query's cosine to its own source image
    ("Positive - hardest",  "gt_minus_hardest"),
    ("Raw negative std",    "neg_std"),
    ("P(gt)",               "prob_on_gt"),        # printed for context (not a table row)
]

EXPECTED = {  # Section E table (CIRCO), baseline / VIB
    "effective_negs":      (132.0, 67.7),
    "effective_negs_raw":  (1022.8, 1022.6),      # the row we could not confirm from the workbook
    "sim_p_vraw":          (0.167, 0.210),
    "gt_minus_hardest":    (-0.050, -0.021),
    "neg_std":             (0.0207, 0.0263),
}


def run_on_batches(path, batches):
    model, ls, meta = load_model(path)
    if model is None:
        raise RuntimeError(f"checkpoint skipped by load_model: {meta}")
    acc, n = {}, 0
    for b in batches:
        out = compute_metrics(model, b, ls)         # per-query lists
        for k, v in out.items():
            acc.setdefault(k, []).extend(v)
        n += b.shape[0]
    means = {k: sum(v) / len(v) for k, v in acc.items()}   # flat mean over all queries
    means["_n"], means["_ls"] = n, ls
    import torch
    del model
    torch.cuda.empty_cache()
    return means


def main():
    circo_dir = DATASETS["circo"]                  # /notebooks/experiment/eval_data/circo
    print(f"CIRCO source: {circo_dir}")
    batches = load_dataset(circo_dir, "circo")     # load ONCE -> identical batches for both
    print(f"Loaded {len(batches)} batches x {batches[0].shape[0]} "
          f"(~{sum(b.shape[0] for b in batches)} CIRCO images)\n")

    res = {name: run_on_batches(p, batches) for name, p in CKPTS.items()}
    b, v = res["baseline"], res["VIB b=0.1"]

    print("=" * 72)
    print(f"SECTION E VERIFICATION  (CIRCO, n={b['_n']}, "
          f"LS base={b['_ls']:.1f} / VIB={v['_ls']:.1f})")
    print("=" * 72)
    print(f"{'metric':<22}|{'baseline':>11} |{'VIB b=0.1':>11} |  {'expected (base / VIB)':<22}")
    print("-" * 72)
    for label, key in ROWS:
        exp = EXPECTED.get(key)
        exp_s = f"{exp[0]} / {exp[1]}" if exp else ""
        print(f"{label:<22}|{b[key]:>11.4f} |{v[key]:>11.4f} |  {exp_s:<22}")
    print("-" * 72)
    print("If 'baseline' and 'VIB b=0.1' match the expected column, the Section E table\n"
          "is reproduced, and 'Eff negs (raw)' gives the value missing from the workbook.")


if __name__ == "__main__":
    main()
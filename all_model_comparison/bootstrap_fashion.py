"""
Paired bootstrap confidence intervals for Fashion-IQ metrics.

WHAT THIS SCRIPT DOES
---------------------
Given checkpoints A, B, C, ... and an optional --reference baseline R, it
computes 95% confidence intervals for:

  Single-checkpoint mode (for each ckpt):
    "BaseMLP R@10 = 25.84, 95% CI [25.21, 26.47]"

  Pair mode (each non-reference ckpt vs the reference):
    "VIB - BaseMLP diff in R@10 = -3.66, 95% CI [-4.49, -2.85]"
    If CI excludes 0, the difference is statistically significant at 5%.

WHY THIS MATTERS
----------------
Fashion-IQ val has ~2000 queries per category. With finite samples, our
mean estimates have uncertainty. A claimed "+1 R@10 improvement" might be
real, or it might be sampling noise. Bootstrap gives us a principled
range of plausible values.

THE BOOTSTRAP MATH
------------------
Given per-query metric values v[1..n]:
  - Resample n indices WITH REPLACEMENT (so some queries are picked
    multiple times, some are omitted).
  - Compute the mean on this resampled subset.
  - Repeat B = 5000 times → distribution of B means.
  - The 2.5th and 97.5th percentiles of this distribution = 95% CI.

For PAIRED bootstrap (comparing two checkpoints):
  - Same resampled indices used for BOTH checkpoints each iteration.
  - This means the comparison is fair: both methods are tested on the
    same hard/easy queries each iter.
  - Without pairing, you could conclude wrong things due to sample bias.

PER-CATEGORY HANDLING
---------------------
Fashion has 3 categories (dress/shirt/toptee). We bootstrap each category
separately, then average per-iteration to get the averaged CI:

  avg_b = (cat_dress_b + cat_shirt_b + cat_toptee_b) / 3

This is correct because the original "averaged" metric is just the mean
of per-category metrics — we mimic that under resampling.

Usage:
    PYTHONPATH=/notebooks/experiment/codes python all_model_comparison/bootstrap_fashion.py \
        --checkpoints A.pt B.pt C.pt \
        --reference baseline.pt \
        --n_bootstrap 5000
"""


import argparse
import json
from pathlib import Path
import numpy as np
from metrics.paths import METRICS_ROOT

CATEGORIES = ['dress', 'shirt', 'toptee']
ALL_FASHION_JSON = Path('/notebooks/experiment/codes/all_model_comparison/all_fashion_metrics.json')
OUTPUT_JSON = METRICS_ROOT / 'bootstrap' / 'bootstrap_fashion_results.json'


# ---------------------------------------------------------------------------
# Compute aggregate metrics from per-query rank arrays.
# This is the function whose mean we resample to get the bootstrap distribution.
# 
# Math:
#   R@K     = mean over queries of (rank_target <= K)
#   R10_ref = mean over queries of (rank_ref <= 10)
#   The "<=" returns 0/1 per query, mean gives the rate (a probability).
# ---------------------------------------------------------------------------


def _metrics_from_ranks(rank_target, rank_ref):
    """Given rank arrays for n queries, return scalar metric dict."""
    rt = np.asarray(rank_target)
    rr = np.asarray(rank_ref)
    return {
        # Recall@K — target appears in top-K of predictions
        'r1':           float((rt <=  1).mean()),
        'r5':           float((rt <=  5).mean()),
        'r10':          float((rt <= 10).mean()),
        'r50':          float((rt <= 50).mean()),
        # Reference stickiness — how often ref ranks high (NOT excluded from gallery)
        'r1_ref':       float((rr <=  1).mean()),
        'r5_ref':       float((rr <=  5).mean()),
        'r10_ref':      float((rr <= 10).mean()),
        'r50_ref':      float((rr <= 50).mean()),
        # Rank distribution stats
        'mean_rank_ref':    float(rr.mean()),
        'median_rank_ref':  float(np.median(rr)),
        'mean_rank_target': float(rt.mean()),
    }
    
# ---------------------------------------------------------------------------
# Single-checkpoint bootstrap for one Fashion category.
#
# Algorithm:
#   For b = 1..B:
#     idx_b = sample n indices with replacement from [0..n)
#     compute metrics on rank_target[idx_b], rank_ref[idx_b]
#     store each metric value
#   Return dict of metric_name -> array of B values (= the bootstrap distribution)
#
# We don't compute CIs here — caller does that. We just return raw samples
# so the caller can either compute CIs OR (in paired mode) combine with another
# checkpoint's samples for difference CIs.
# ---------------------------------------------------------------------------

def _bootstrap_per_category(rank_target, rank_ref, n_bootstrap, rng):
    n = len(rank_target)
    rt = np.asarray(rank_target)
    rr = np.asarray(rank_ref)
    # Initialize empty list per metric
    metric_samples = {k: [] for k in _metrics_from_ranks(rt, rr).keys()}

    for _ in range(n_bootstrap):
        # Sample n indices WITH replacement (some duplicated, some missed) →
        # this is the heart of bootstrap. On average ~63% unique queries.
        idx = rng.integers(0, n, n)
        sample = _metrics_from_ranks(rt[idx], rr[idx])
        for k, v in sample.items():
            metric_samples[k].append(v)

    return {k: np.array(v) for k, v in metric_samples.items()}


# ---------------------------------------------------------------------------
# PAIRED bootstrap for one Fashion category.
#
# Key idea: same resampled indices used for both checkpoints each iteration.
# This is critical — using different indices gives MUCH wider CIs because
# we'd be conflating "the methods differ" with "the queries differ".
#
# Returns: dict of metric_name -> array of B values of (metric_A - metric_B)
# ---------------------------------------------------------------------------

def _paired_bootstrap_per_category(rank_target_A, rank_ref_A,
                                    rank_target_B, rank_ref_B,
                                    n_bootstrap, rng):
    n = len(rank_target_A)
    assert len(rank_target_B) == n, f'Checkpoints must have same per-category query count'
    rta = np.asarray(rank_target_A); rra = np.asarray(rank_ref_A)
    rtb = np.asarray(rank_target_B); rrb = np.asarray(rank_ref_B)

    metric_names = list(_metrics_from_ranks(rta, rra).keys())
    diff_samples = {k: [] for k in metric_names}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        # IDENTICAL indices applied to both — that's what makes this 'paired'
        sa = _metrics_from_ranks(rta[idx], rra[idx])
        sb = _metrics_from_ranks(rtb[idx], rrb[idx])
        for k in metric_names:
            diff_samples[k].append(sa[k] - sb[k])

    return {k: np.array(v) for k, v in diff_samples.items()}


# ---------------------------------------------------------------------------
# Compute CI summary from a bootstrap distribution (array of B samples).
# 
# 95% CI = [2.5th percentile, 97.5th percentile] of the bootstrap distribution.
# Standard convention; B = 5000 gives ~125 samples in each tail for stable estimates.
# ---------------------------------------------------------------------------
def _ci_summary(samples, ci=95):
    lo_q = (100 - ci) / 2          # e.g. 2.5 for 95% CI
    hi_q = 100 - lo_q              # e.g. 97.5
    return {
        'mean':   float(np.mean(samples)),
        'std':    float(np.std(samples)),
        'ci_lo':  float(np.percentile(samples, lo_q)),
        'ci_hi':  float(np.percentile(samples, hi_q)),
        'median': float(np.median(samples)),
    }


# ---------------------------------------------------------------------------
# CI summary for paired differences, with additional "is the sign reliable"
# diagnostics:
#   - frac_sign_flip: fraction of bootstrap iterations where the difference
#     has opposite sign from the mean. If this is < 2.5%, the sign is reliable
#     (equivalent to the CI not crossing 0).
# ---------------------------------------------------------------------------
def _ci_diff_summary(diff_samples):
    s = _ci_summary(diff_samples)
    mean_sign = 1 if s['mean'] >= 0 else -1
    # frac_sign_flip = how often did a bootstrap iteration disagree about sign?
    s['frac_sign_flip'] = float((diff_samples * mean_sign < 0).mean())
    return s


# ---------------------------------------------------------------------------
# Combine per-category bootstrap distributions into "averaged" distributions.
# 
# Important: we average AT EACH BOOTSTRAP ITERATION (not at the end), giving:
#   avg_b = (dress_b + shirt_b + toptee_b) / 3   for each b in 1..B
# 
# This preserves the joint sampling variation correctly.
# ---------------------------------------------------------------------------
def _averaged_from_per_cat(per_cat_samples):
    """Given dict[cat] -> dict[metric] -> B-array, return dict[metric] -> B-array (avg)."""
    metric_names = list(per_cat_samples[CATEGORIES[0]].keys())
    avg_samples = {}
    for m in metric_names:
        # Stack the 3 category arrays into (3, B) and average across axis 0 → (B,)
        stacked = np.stack([per_cat_samples[c][m] for c in CATEGORIES], axis=0)
        avg_samples[m] = stacked.mean(axis=0)
    return avg_samples


# ---------------------------------------------------------------------------
# Single-checkpoint analysis: CIs for every metric, per category + averaged.
# Stores raw bootstrap samples (in '_samples_*') so that downstream pairwise
# code can use them — but caller will strip these before saving the JSON
# (they're huge: ~5000 * 11 metrics * 3 categories of floats).
# ---------------------------------------------------------------------------
def analyze_checkpoint(ckpt_path, all_metrics, n_bootstrap, rng):
    if ckpt_path not in all_metrics:
        raise KeyError(f'{ckpt_path} not found in {ALL_FASHION_JSON}')
    entry = all_metrics[ckpt_path]
    if entry.get('eval_status') != 'success':
        raise ValueError(f'{ckpt_path} eval_status != success')

    fashion = entry['fashion']
    per_cat_results = {}
    for cat in CATEGORIES:
        pq = fashion[cat]['per_query']
        # Point estimate: metrics on the original (un-resampled) data
        point = _metrics_from_ranks(pq['rank_target'], pq['rank_ref'])
        # Bootstrap distribution
        bs_samples = _bootstrap_per_category(pq['rank_target'], pq['rank_ref'],
                                              n_bootstrap, rng)
        # CIs from bootstrap, decorated with the point estimate
        cis = {m: _ci_summary(bs) for m, bs in bs_samples.items()}
        for m in cis:
            cis[m]['point'] = point[m]
        per_cat_results[cat] = {
            'cis': cis,
            'n_queries': len(pq['rank_target']),
            '_samples': bs_samples,
        }

    # Averaged CIs across the 3 categories
    per_cat_samples = {c: per_cat_results[c]['_samples'] for c in CATEGORIES}
    avg_samples = _averaged_from_per_cat(per_cat_samples)
    avg_cis = {m: _ci_summary(bs) for m, bs in avg_samples.items()}
    # Add point estimates from the existing 'averaged' block in source JSON
    avg_block = fashion['averaged']
    avg_point = {
        'r1':           avg_block['recall_atk']['@1'],
        'r5':           avg_block['recall_atk']['@5'],
        'r10':          avg_block['recall_atk']['@10'],
        'r50':          avg_block['recall_atk']['@50'],
        'r1_ref':       avg_block['stickiness']['r1_ref'],
        'r5_ref':       avg_block['stickiness']['r5_ref'],
        'r10_ref':      avg_block['stickiness']['r10_ref'],
        'r50_ref':      avg_block['stickiness']['r50_ref'],
        'mean_rank_ref':    avg_block['stickiness']['mean_rank_ref'],
        'median_rank_ref':  avg_block['stickiness']['median_rank_ref'],
        'mean_rank_target': avg_block['stickiness']['mean_rank_target'],
    }
    for m in avg_cis:
        if m in avg_point:
            avg_cis[m]['point'] = avg_point[m]

    return {
        'per_category': {c: {'cis': per_cat_results[c]['cis'],
                             'n_queries': per_cat_results[c]['n_queries']}
                          for c in CATEGORIES},
        'averaged': {'cis': avg_cis},
        # Internal — for pair computation, stripped before save
        '_samples_per_cat': per_cat_samples,
        '_samples_averaged': avg_samples,
    }


# ---------------------------------------------------------------------------
# Pair analysis: compute paired differences (A - B) and their CIs.
# Per-category paired bootstrap, then average per-iter for the averaged result.
# ---------------------------------------------------------------------------
def analyze_pair(ckpt_A, ckpt_B, all_metrics, n_bootstrap, rng):
    eA = all_metrics[ckpt_A]['fashion']
    eB = all_metrics[ckpt_B]['fashion']
    per_cat_diff = {}
    for cat in CATEGORIES:
        pa = eA[cat]['per_query']
        pb = eB[cat]['per_query']
        assert len(pa['rank_target']) == len(pb['rank_target']), \
            f'{cat}: query count mismatch'
        diff_samples = _paired_bootstrap_per_category(
            pa['rank_target'], pa['rank_ref'],
            pb['rank_target'], pb['rank_ref'],
            n_bootstrap, rng,
        )
        per_cat_diff[cat] = diff_samples

    # Averaged differences across categories
    avg_diff_samples = _averaged_from_per_cat(per_cat_diff)

    return {
        'per_category': {
            cat: {m: _ci_diff_summary(s) for m, s in per_cat_diff[cat].items()}
            for cat in CATEGORIES
        },
        'averaged': {m: _ci_diff_summary(s) for m, s in avg_diff_samples.items()},
    }

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', nargs='+', required=True,
                   help='List of checkpoint paths to analyze')
    p.add_argument('--reference', type=str, default=None,
                   help='Reference checkpoint; all others paired against it')
    p.add_argument('--n_bootstrap', type=int, default=5000)
    p.add_argument('--seed', type=int, default=42,
                   help='RNG seed for reproducible bootstrap samples')
    p.add_argument('--output_json', type=Path, default=OUTPUT_JSON)
    return p.parse_args()


def main():
    args = parse_args()
    # Seeded RNG — exact same bootstrap distribution every run (reproducibility)
    rng = np.random.default_rng(args.seed)

    print(f'Loading {ALL_FASHION_JSON}...')
    with open(ALL_FASHION_JSON) as f:
        all_metrics = json.load(f)
    print(f'  {len(all_metrics)} checkpoints in JSON')
    
    # Diagnostic: print per-category sizes to make the averaging strategy explicit
    sample_entry = next(iter(all_metrics.values()))['fashion']
    print(f'\nFashion-IQ per-category query counts (Fashion-IQ convention: simple average):')
    for cat in CATEGORIES:
        print(f'  {cat:<8s} n = {sample_entry[cat]["n_queries"]}')
    print(f'  total    = {sum(sample_entry[c]["n_queries"] for c in CATEGORIES)}')


    # If reference given, include it in the target list (for single-ckpt CIs)
    targets = list(args.checkpoints)
    if args.reference and args.reference not in targets:
        targets = [args.reference] + targets

    print(f'\nAnalyzing {len(targets)} checkpoint(s) with {args.n_bootstrap} bootstrap iters...')
    if args.reference:
        print(f'  Reference: {args.reference}')

    # --- Single-checkpoint CIs ---
    single_results = {}
    for ckpt in targets:
        print(f'\n  Bootstrapping single: {ckpt}')
        res = analyze_checkpoint(ckpt, all_metrics, args.n_bootstrap, rng)
        # Strip the internal _samples_* keys before saving — they're big and
        # serve no purpose downstream
        res.pop('_samples_per_cat', None)
        res.pop('_samples_averaged', None)
        single_results[ckpt] = res

    # --- Pairwise (vs reference) CIs ---
    pair_results = {}
    if args.reference:
        for ckpt in args.checkpoints:
            if ckpt == args.reference:
                continue
            print(f'\n  Bootstrapping pair: {ckpt} vs reference')
            pair_results[f'{ckpt}__vs__{args.reference}'] = analyze_pair(
                ckpt, args.reference, all_metrics, args.n_bootstrap, rng
            )

    # --- Save to JSON ---
    output = {
        'meta': {
            'n_bootstrap': args.n_bootstrap,
            'seed': args.seed,
            'reference': args.reference,
            'checkpoints': args.checkpoints,
        },
        'single': single_results,
        'paired': pair_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved bootstrap results to {args.output_json}')

    # --- Print a compact summary of key pairs ---
    # '*' next to a metric = CI excludes zero → statistically significant at 5%
    print()
    print('=' * 100)
    print('PAIRED DIFFERENCES (averaged across categories), key metrics:')
    print('=' * 100)
    if pair_results:
        for pair_key, pair_res in pair_results.items():
            label = pair_key.split('__vs__')[0].split('/')[-2:]
            print(f'\n  {"/".join(label)}  vs  reference:')
            for m in ['r10', 'r50', 'r10_ref', 'mean_rank_ref']:
                d = pair_res['averaged'][m]
                sig = '*' if (d['ci_lo'] > 0 or d['ci_hi'] < 0) else ' '
                print(f'    {m:<20s} Δ = {d["mean"]:+.4f}  '
                      f'95% CI [{d["ci_lo"]:+.4f}, {d["ci_hi"]:+.4f}]  {sig}')


if __name__ == '__main__':
    main()
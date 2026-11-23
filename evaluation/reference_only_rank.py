"""
evaluation/reference_only_rank.py

How buried is the TARGET when you rank the whole gallery by similarity to the
REFERENCE ALONE (ignoring the modifier)?  This is the retrieval-native version
of "how many distractors sit as close to the reference as the target", and it is
the right way to test the intro claim that a reference-sticky query cannot
identify the target on open-domain CIRCO.

MODEL-INDEPENDENT: uses only cached gallery image features + each benchmark's val
annotations. No checkpoint, no mapper, no modifier text. Pure cos(image, ref).

Per query (ref, target[s]) it computes, over that benchmark's actual eval
gallery (CIRCO = full ~123K, CIRR ~2.2K, Fashion-IQ per-category):

  reference-only rank  = 1 + #{ distractor d : cos(d, ref) > cos(target, ref) }
                         (the reference itself is excluded; for CIRCO multi-GT we
                          take the BEST = most reference-similar GT, i.e. the
                          most favorable case for a reference-sticky query)
  normalized rank      = rank / (#distractors + 1)   in (0, 1]; scale-free, so
                         CIRCO's 123K does not trivially dominate
  confusers_ge_target  = rank - 1  = your "# distractors as close to the ref as
                                     the target", done target-relative not at a
                                     fixed 0.8
  threshold counts      = #{ d : cos(d, ref) > t } for t in --thresholds, plus
                          the fraction count/N (the scale-comparable form)

WHY rank, not a fixed cos>0.8 count: a raw count above 0.8 is (a) confounded by
gallery size -- 123K vs 2K means CIRCO wins by construction -- and (b) not
comparable across galleries whose random-pair floors differ (Fashion 0.70 vs
CIRCO 0.50), so 0.8 means different things in each. The target-relative rank
removes both confounds; the fixed-threshold counts are reported too, for context.

Run (CPU is fine; nothing is encoded):
    cd /notebooks/experiment/codes
    PYTHONPATH=. python evaluation/reference_only_rank.py
Outputs: console tables, a box plot PNG, and a per-query npz for paper replotting,
all under metrics/.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CLIP_TAG_DEFAULT = 'ViT-L/14'
FASHION_CATS = ['dress', 'shirt', 'toptee']


def normalize(embs):
    return F.normalize(embs.float(), dim=-1)


# --------------------------------------------------------------------------
# core per-shard diagnostic  (one gallery, a list of (ref_idx, [gt_idx...]))
# --------------------------------------------------------------------------
def shard_diagnostics(embs, queries, thresholds, device='cpu', skipped=0):
    """Return per-query arrays for one (gallery, queries) shard."""
    embs = embs.to(device)
    N = embs.shape[0]
    rank_abs, rank_norm, conf, tgt_sim = [], [], [], []
    thr_counts = {t: [] for t in thresholds}

    for ref_idx, gt_idxs in queries:
        sims = embs @ embs[ref_idx]                      # [N] cos to reference
        gt_sims = sims[gt_idxs]
        best = float(gt_sims.max())                      # most ref-similar GT
        n_distractors = N - 1 - len(gt_idxs)             # exclude ref + all GTs

        # images strictly more ref-similar than the best GT, minus the
        # reference (sim 1.0). All GTs are <= best by definition, so they are
        # already excluded from a strict ">" comparison.
        n_ahead = int((sims > best).sum()) - 1
        n_ahead = max(n_ahead, 0)

        rank_abs.append(n_ahead + 1)
        rank_norm.append((n_ahead + 1) / (n_distractors + 1))
        conf.append(n_ahead)
        tgt_sim.append(best)
        for t in thresholds:
            # distractors above t = (all above t) - reference - GTs above t
            above = int((sims > t).sum())
            gts_above = int((gt_sims > t).sum())
            ref_above = 1 if 1.0 > t else 0
            thr_counts[t].append(max(above - ref_above - gts_above, 0))

    return {
        'rank_abs': np.asarray(rank_abs, dtype=np.int64),
        'rank_norm': np.asarray(rank_norm, dtype=np.float64),
        'confusers_ge_target': np.asarray(conf, dtype=np.int64),
        'target_ref_sim': np.asarray(tgt_sim, dtype=np.float64),
        'thr_counts': {t: np.asarray(v, dtype=np.int64) for t, v in thr_counts.items()},
        'gallery_size': N,
        'skipped': skipped,
    }


def merge_shards(shards):
    """Pool per-query arrays across shards (Fashion-IQ = 3 category shards)."""
    keys = ['rank_abs', 'rank_norm', 'confusers_ge_target', 'target_ref_sim']
    out = {k: np.concatenate([s[k] for s in shards]) for k in keys}
    thr = {}
    for t in shards[0]['thr_counts']:
        thr[t] = np.concatenate([s['thr_counts'][t] for s in shards])
    out['thr_counts'] = thr
    out['gallery_size'] = shards[0]['gallery_size'] if len(shards) == 1 \
        else [s['gallery_size'] for s in shards]            # list if per-category
    out['skipped'] = sum(s['skipped'] for s in shards)
    return out


def report(label, d, thresholds):
    ra, rn, cf = d['rank_abs'], d['rank_norm'], d['confusers_ge_target']
    gs = d['gallery_size']
    gs_str = f'{gs}' if not isinstance(gs, list) else 'per-cat ' + '/'.join(map(str, gs))
    print('\n' + '=' * 74)
    print(f'{label}   —   reference-only target rank   (gallery N = {gs_str})')
    print('=' * 74)
    if d['skipped']:
        print(f'  (skipped {d["skipped"]} queries: ref or all GTs absent from gallery)')
    print(f'  queries={len(ra)}')
    print(f'  reference-only RANK of target (lower = ref-sticky query finds it sooner):')
    print(f'     median={np.median(ra):.0f}   mean={ra.mean():.1f}'
          f'   p25={np.percentile(ra,25):.0f}   p75={np.percentile(ra,75):.0f}'
          f'   p90={np.percentile(ra,90):.0f}   max={ra.max()}')
    print(f'  NORMALIZED rank (rank / #distractors; scale-free):')
    print(f'     median={np.median(rn):.4f}   mean={rn.mean():.4f}'
          f'   p90={np.percentile(rn,90):.4f}')
    print(f'  confusers as ref-close as target (= rank-1): '
          f'median={np.median(cf):.0f}   mean={cf.mean():.1f}')
    print(f'  fixed-threshold distractor counts (raw count / fraction of gallery):')
    for t in thresholds:
        c = d['thr_counts'][t]
        frac = c / (gs if not isinstance(gs, list) else np.mean(gs))
        print(f'     cos>{t:.2f}:  mean_count={c.mean():8.1f}   median_count={np.median(c):7.0f}'
              f'   mean_fraction={frac.mean():.4f}')
    return {
        'label': label, 'queries': int(len(ra)), 'gallery_size': gs,
        'rank_median': float(np.median(ra)), 'rank_mean': float(ra.mean()),
        'rank_p90': float(np.percentile(ra, 90)), 'rank_max': int(ra.max()),
        'norm_rank_median': float(np.median(rn)), 'norm_rank_mean': float(rn.mean()),
        'confusers_median': float(np.median(cf)), 'confusers_mean': float(cf.mean()),
        'thr': {f'{t}': {'mean_count': float(d['thr_counts'][t].mean()),
                         'median_count': float(np.median(d['thr_counts'][t]))}
                for t in thresholds},
    }


# --------------------------------------------------------------------------
# loaders  (mirror evaluation/dataset_ref_target_similarity.py exactly)
# --------------------------------------------------------------------------
def load_cirr(args, tag):
    gal = torch.load(args.cirr_images_root / f'cirr_gallery_val_{tag}.pt', map_location='cpu')
    embs = normalize(gal['embeddings'])
    n2i = gal['name_to_idx']
    cap = json.load(open(args.cirr_ann_root / 'captions' / 'cap.rc2.val.json'))
    if isinstance(cap, dict):
        cap = list(cap.values())
    queries, skipped = [], 0
    for q in cap:
        ri, ti = n2i.get(q['reference']), n2i.get(q['target_hard'])
        if ri is None or ti is None:
            skipped += 1
            continue
        queries.append((ri, [ti]))
    return [shard_diagnostics(embs, queries, args.thresholds, args.device, skipped)]


def load_fashion(args, tag):
    shards = []
    for cat in FASHION_CATS:
        names = json.load(open(args.fashion_data_path / 'image_splits' / f'split.{cat}.val.json'))
        n2i = {n: i for i, n in enumerate(names)}
        embs = normalize(torch.load(
            args.fashion_data_path / f'gallery_cache_{cat}_{tag}.pt', map_location='cpu'))
        ann = json.load(open(args.fashion_data_path / 'captions' / f'cap.{cat}.val.json'))
        queries, skipped = [], 0
        for a in ann:
            ri, ti = n2i.get(a['candidate']), n2i.get(a['target'])
            if ri is None or ti is None:
                skipped += 1
                continue
            queries.append((ri, [ti]))
        shards.append(shard_diagnostics(embs, queries, args.thresholds, args.device, skipped))
    return shards


def load_circo(args, tag):
    gal = torch.load(args.circo_data_path / f'gallery_cache_{tag}.pt', map_location='cpu')
    embs = normalize(gal['embeddings'])
    id2i = gal['img_id_to_idx']
    repo_src = args.circo_data_path / 'circo_repo' / 'src'
    spec = importlib.util.spec_from_file_location('_circo_ds', repo_src / 'dataset.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        ds = mod.CIRCODataset(data_path=str(args.circo_data_path), split='val',
                              mode='relative', preprocess=(lambda x: x))
    except Exception:
        import clip
        _, preprocess = clip.load(args.clip_model, device='cpu')
        ds = mod.CIRCODataset(data_path=str(args.circo_data_path), split='val',
                              mode='relative', preprocess=preprocess)
    queries, skipped = [], 0
    for ann in ds.annotations:
        ri = id2i.get(str(ann['reference_img_id']))
        gts = [id2i.get(str(g)) for g in ann['gt_img_ids']]
        gts = [g for g in gts if g is not None]
        if ri is None or not gts:
            skipped += 1
            continue
        queries.append((ri, gts))
    return [shard_diagnostics(embs, queries, args.thresholds, args.device, skipped)]


# --------------------------------------------------------------------------
def make_boxplot(merged, out_png):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'\n[!] matplotlib unavailable ({e}); skipping box plot, npz still saved.')
        return
    labels = list(merged.keys())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].boxplot([merged[l]['rank_abs'] for l in labels], labels=labels, showfliers=False)
    ax[0].set_yscale('log')
    ax[0].set_ylabel('reference-only target rank (log)')
    ax[0].set_title('Absolute rank (raw difficulty; gallery-size sensitive)')
    ax[1].boxplot([merged[l]['rank_norm'] for l in labels], labels=labels, showfliers=False)
    ax[1].set_ylabel('normalized rank  (rank / #distractors)')
    ax[1].set_title('Normalized rank (scale-free)')
    for a in ax:
        a.grid(True, axis='y', alpha=0.3)
    fig.suptitle('How buried is the target under reference-similarity alone?')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f'\nbox plot  -> {out_png}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip_model', type=str, default=CLIP_TAG_DEFAULT)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--thresholds', type=float, nargs='+', default=[0.7, 0.8, 0.9])
    ap.add_argument('--cirr_ann_root',     type=Path, default=Path('/notebooks/experiment/CIRR/annotations'))
    ap.add_argument('--cirr_images_root',  type=Path, default=Path('/notebooks/experiment/CIRR'))
    ap.add_argument('--circo_data_path',   type=Path, default=Path('/notebooks/experiment/eval_data/circo'))
    ap.add_argument('--fashion_data_path', type=Path, default=Path('/notebooks/experiment/eval_data/fashion_iq'))
    ap.add_argument('--benchmarks', nargs='+', default=['cirr', 'circo', 'fashion'])
    ap.add_argument('--out_dir', type=Path, default=Path('/notebooks/experiment/codes/metrics'))
    args = ap.parse_args()
    tag = args.clip_model.replace('/', '-')

    runners = {'cirr': (load_cirr, 'CIRR (val)'),
               'circo': (load_circo, 'CIRCO (val, best-GT)'),
               'fashion': (load_fashion, 'Fashion-IQ (val, pooled)')}

    merged, summaries = {}, []
    for b in args.benchmarks:
        loader, label = runners[b]
        try:
            shards = loader(args, tag)
            d = merge_shards(shards)
            merged[label] = d
            summaries.append(report(label, d, args.thresholds))
        except Exception as e:
            print(f'\n[!] {b} failed: {type(e).__name__}: {e}')

    if summaries:
        print('\n' + '=' * 74)
        print('CROSS-BENCHMARK SUMMARY  —  reference-only target rank')
        print('=' * 74)
        print(f'{"benchmark":<26} {"N_gal":>8} {"rank_med":>9} {"rank_p90":>9} '
              f'{"norm_med":>9} {"conf_med":>9}')
        for s in summaries:
            gs = s['gallery_size']
            gs = gs if not isinstance(gs, list) else int(np.mean(gs))
            print(f'{s["label"]:<26} {gs:>8} {s["rank_median"]:>9.0f} {s["rank_p90"]:>9.0f} '
                  f'{s["norm_rank_median"]:>9.4f} {s["confusers_median"]:>9.0f}')
        print('\nrank_med = median # gallery images more reference-similar than the target.')
        print('Large absolute rank with a modest normalized rank => the gallery SCALE,')
        print('not poor target separability, is what buries the target for a ref-sticky query.')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(summaries, open(args.out_dir / 'reference_only_rank.json', 'w'), indent=2)
    np.savez(args.out_dir / 'reference_only_rank_perquery.npz',
             **{f'{l}__rank_abs': merged[l]['rank_abs'] for l in merged},
             **{f'{l}__rank_norm': merged[l]['rank_norm'] for l in merged})
    print(f'\nsaved -> {args.out_dir / "reference_only_rank.json"}')
    print(f'saved -> {args.out_dir / "reference_only_rank_perquery.npz"}')
    make_boxplot(merged, args.out_dir / 'reference_only_rank_boxplot.png')


if __name__ == '__main__':
    main()
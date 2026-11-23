"""
evaluation/circo_gap_mrr.py

Computes the full Table-4 diagnostic block on CIRCO *val*, for two checkpoints
(deterministic baseline + VIB), in one run. All quantities use the composed
(captioned) query, the full 123K gallery, and RAW cosine (no logit scale -- ranking
is scale-invariant and the eval never applies it).

Two NEW retrieval-native metrics (replacing the old in-batch gt-hard and eff-negs):

  gap_mean = mean_{t in GT} cos(c, u_t)  -  max_{g not in {ref} U GT} cos(c, u_g)
  gap_best =  max_{t in GT} cos(c, u_t)  -  max_{g not in {ref} U GT} cos(c, u_g)
      (>0 = a target out-ranks the whole field; <0 = target buried)

  MRR      = mean_q 1 / rank_best_GT(q)        [reference masked]
      rank_best_GT = 1 + #{distractors scoring above the best GT}

Plus the THREE kept metrics, reproduced here as a cross-check against eval_circo:

  query_reference_cos = mean_q cos(c, u_ref)                 (should match 0.329 / 0.290)
  query_target_cos    = mean_q mean_{t in GT} cos(c, u_t)    (should match 0.263 / 0.250)
  ref_target_margin   = query_reference_cos - query_target_cos  (should match +0.066 / +0.040)

Usage (run once with both checkpoints):
    cd /notebooks/experiment/codes
    PYTHONPATH=/notebooks/experiment/codes python evaluation/circo_gap_mrr.py \
        --checkpoints /path/to/deterministic_best.pt /path/to/vib_best.pt \
        --labels Baseline VIB
"""

import argparse
import json
import time
from pathlib import Path

import clip
import numpy as np
import PIL.Image
import torch

# Reuse the EXACT model + CIRCO loaders from the real eval so VIB inference (z = mu),
# config, gallery cache, and the official CIRCO dataset are handled identically.
from evaluation.eval_circo import load_model, _load_circo_official


@torch.no_grad()
def compute(model, clip_model_name, device, circo_data_path, gallery_cache_path):
    cache         = torch.load(gallery_cache_path, map_location='cpu')
    gallery       = cache['embeddings'].to(device)          # (N, 768) normalized fp32
    img_id_to_idx = cache['img_id_to_idx']
    assert cache['clip_model'] == clip_model_name, \
        f"Cache is for {cache['clip_model']}, model uses {clip_model_name}"

    CIRCODataset, _ = _load_circo_official(circo_data_path)
    _, preprocess   = clip.load(clip_model_name, device='cpu')
    ds = CIRCODataset(data_path=str(circo_data_path), split='val',
                      mode='relative', preprocess=preprocess)

    rows = []
    t0 = time.time()
    for q in range(len(ds)):
        ann        = ds.annotations[q]
        ref_img_id = ann['reference_img_id']
        gt_img_ids = ann['gt_img_ids']
        caption    = ann['relative_caption']

        ref_path = ds.img_paths[ds.img_ids_indexes_map[str(ref_img_id)]]
        ref_img  = preprocess(PIL.Image.open(ref_path).convert('RGB')).unsqueeze(0).to(device)
        txt_tok  = clip.tokenize([caption], truncate=True).to(device)

        c      = model.encode_query(ref_img, txt_tok)       # (1, 768) normalized; VIB -> mu
        scores = (c @ gallery.T).squeeze(0)                 # (N,) raw cosine

        ref_idx    = img_id_to_idx[str(ref_img_id)]
        gt_indices = [img_id_to_idx[str(g)] for g in gt_img_ids if str(g) in img_id_to_idx]
        if not gt_indices:
            continue

        # ---- cosines ----
        sim_ref   = scores[ref_idx].item()
        gt_scores = scores[gt_indices]
        best_gt   = gt_scores.max().item()
        mean_gt   = gt_scores.mean().item()

        # ---- hardest distractor: mask reference AND every GT, take the max ----
        masked_gap             = scores.clone()
        masked_gap[ref_idx]    = float('-inf')
        masked_gap[gt_indices] = float('-inf')
        hardest_distractor     = masked_gap.max().item()

        # ---- rank of best GT (reference masked, GTs kept in the ranking) ----
        masked_ref          = scores.clone()
        masked_ref[ref_idx] = float('-inf')
        rank_best_gt        = int((masked_ref > best_gt).sum().item()) + 1   # ties -> not counted

        rows.append({
            'sim_ref'            : sim_ref,
            'best_gt'            : best_gt,
            'mean_gt'            : mean_gt,
            'hardest_distractor' : hardest_distractor,
            'gap_best'           : best_gt - hardest_distractor,
            'gap_mean'           : mean_gt - hardest_distractor,
            'rank_best_gt'       : rank_best_gt,
            'recip_rank'         : 1.0 / rank_best_gt,
            'gt_top1'            : rank_best_gt == 1,
        })

        if (q + 1) % 50 == 0:
            print(f'  [{q + 1}/{len(ds)}]  {(q + 1) / (time.time() - t0):.1f} q/s')

    def mean(k): return float(np.mean([r[k] for r in rows]))
    def sem(k):
        v = np.array([r[k] for r in rows], dtype=float)
        return float(v.std(ddof=1) / np.sqrt(len(v)))

    query_reference_cos = mean('sim_ref')
    query_target_cos    = mean('mean_gt')

    return {
        'n_queries'             : len(rows),
        # --- two NEW headline metrics ---
        'gap_mean'              : mean('gap_mean'),     'gap_mean_sem': sem('gap_mean'),
        'mrr'                   : mean('recip_rank'),
        # --- supporting rank stats ---
        'gap_best'              : mean('gap_best'),     'gap_best_sem': sem('gap_best'),
        'median_best_gt_rank'   : float(np.median([r['rank_best_gt'] for r in rows])),
        'mean_best_gt_rank'     : mean('rank_best_gt'),
        'gt_top1_rate'          : float(np.mean([r['gt_top1'] for r in rows])),
        'hardest_distractor_cos': mean('hardest_distractor'),
        # --- three KEPT metrics, reproduced as a cross-check ---
        'query_reference_cos'   : query_reference_cos,
        'query_target_cos'      : query_target_cos,
        'ref_target_margin'     : query_reference_cos - query_target_cos,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', type=Path, nargs='+', required=True,
                   help='Two checkpoints: deterministic_best then vib_best.')
    p.add_argument('--labels', type=str, nargs='*', default=None)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--circo_data_path', type=Path,
                   default=Path('/notebooks/experiment/eval_data/circo'))
    p.add_argument('--gallery_cache_path', type=Path, default=None,
                   help='Default: <circo_data_path>/gallery_cache_<clip_tag>.pt')
    p.add_argument('--save', type=Path, default=Path('circo_gap_mrr.json'))
    return p.parse_args()


def main():
    args   = parse_args()
    labels = args.labels or [c.parent.name or c.stem for c in args.checkpoints]
    assert len(labels) == len(args.checkpoints), 'labels must match #checkpoints'

    results = {}
    for label, ckpt in zip(labels, args.checkpoints):
        print('=' * 70)
        print(f'{label}: {ckpt}')
        print('=' * 70)
        model, clip_model_name = load_model(ckpt, args.device)
        print(f'  logit_scale.exp() = {model.clip.logit_scale.exp().item():.2f}  '
              f'(sanity only -- does NOT affect ranking)')

        gcache = args.gallery_cache_path
        if gcache is None:
            gcache = args.circo_data_path / f"gallery_cache_{clip_model_name.replace('/', '-')}.pt"

        results[label] = compute(model, clip_model_name, args.device,
                                 args.circo_data_path, gcache)
        del model
        torch.cuda.empty_cache()

    # ---- comparison table ----
    print('\n' + '=' * 86)
    print('CIRCO val diagnostics (composed query, full 123K gallery, raw cosine)')
    print('=' * 86)
    table = [
        ('gap_mean',              'Target - hardest distractor (mean-GT)   [NEW]'),
        ('mrr',                   'MRR (1 / rank of best GT, ref masked)    [NEW]'),
        ('query_reference_cos',   'Query-reference cosine'),
        ('query_target_cos',      'Query-target cosine'),
        ('ref_target_margin',     'Reference-target margin (ref - target)'),
        ('---',                   ''),
        ('gap_best',              '  (supporting) gap best-GT - hardest'),
        ('median_best_gt_rank',   '  (supporting) median best-GT rank'),
        ('mean_best_gt_rank',     '  (supporting) mean best-GT rank'),
        ('gt_top1_rate',          '  (supporting) frac. a GT beats the field'),
        ('hardest_distractor_cos','  (supporting) cos(c, hardest distractor)'),
    ]
    hdr = f'{"metric":46s}' + ''.join(f'{l:>13s}' for l in labels)
    print(hdr)
    print('-' * len(hdr))
    for k, name in table:
        if k == '---':
            print('-' * len(hdr)); continue
        vals = ''.join(f'{results[l][k]:>13.4f}' for l in labels)
        print(f'{name:46s}{vals}')
    print(f'\n  n_queries (val) = {results[labels[0]]["n_queries"]}')
    print('  gap_mean SEM:    ' + '  '.join(f'{l}={results[l]["gap_mean_sem"]:.4f}' for l in labels))

    with open(args.save, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved -> {args.save}')


if __name__ == '__main__':
    main()
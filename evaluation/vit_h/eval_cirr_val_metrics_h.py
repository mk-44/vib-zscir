"""
evaluation/eval_cirr_val_metrics.py

Comprehensive CIRR *validation* metrics on the FULL gallery (reference masked,
exactly as official CIRR Recall is computed). NO subset metric here.

Standalone analysis script: it IMPORTS the machinery from eval_cirr.py
(model loading, the shared val gallery cache, annotation parsing) and does NOT
modify it, so the submission path stays intact and these numbers come from a
byte-identical model + gallery to the ones you submitted.

Run (the val gallery cache is already built and is checkpoint-independent, so
all checkpoints reuse one encode). Pass as many checkpoints as you like:

    cd /notebooks/experiment/codes
    PYTHONPATH=. python evaluation/eval_cirr_val_metrics.py \
        --checkpoint \
        /notebooks/experiment/4M_checkpoints/base_hd256_tok1_lslr0.2_lsmax116/epoch_01/checkpoint.pt \
        /notebooks/experiment/4M_checkpoints/base_hd256_tok1_lslr0.2_lsmax116/epoch_02_step_03000.pt \
        /notebooks/experiment/4M_checkpoints/vib_hd256_tok1_lslr0.2_lsmax116_vib0.0001/epoch_01_step_01500.pt \
        /notebooks/experiment/4M_checkpoints/vib_hd256_tok1_lslr0.2_lsmax116_vib0.0002/epoch_02_step_03000.pt \
        /notebooks/experiment/4M_checkpoints/vib_hd256_tok1_initbase_lslr0.2_lsmax116_vib0.0001_binit3e-06_lvbias-6.0_vwarm0.5_klvw0.1_lvclip-3.0/epoch_00_step_02000.pt \
        /notebooks/experiment/4M_checkpoints/vib_hd256_tok1_initbase_lslr0.2_lsmax116_vib0.0001_binit3e-06_lvbias-6.0_vwarm0.5_klvw0.1_lvclip-3.0/epoch_01_step_01000.pt

Per-checkpoint metrics (full gallery of N imgs, query's reference removed):
    Recall@{1,2,3,5,10,20,50,100}
    MRR                  mean(1 / rank_gt)
    mAP@{5,10,25,50}     single hard GT  =>  AP@k = 1/rank_gt if rank_gt<=k else 0
                         (=> mAP_full == MRR; per-k reported only for CIRCO parity)
    mean_rank_gt, median_rank_gt
Stickiness (reference-copying; reference NOT masked, so we see where it lands):
    R@{1,5,10}_ref       fraction of queries with the reference itself in top-k
    mean_rank_ref, median_rank_ref
    sim_ref, sim_target, sim_delta = sim_ref - sim_target   (+ve => sticky)
    frac_closer_to_ref   fraction of queries with sim_ref > sim_target
Dataset characterization (checkpoint-INDEPENDENT, frozen CLIP image features):
    cos(target_img, reference_img): mean / median / std / percentiles / histogram
    -> how visually similar CIRR targets are to their references (the axis the
       distribution-mismatch argument rests on; run the same on CIRCO to contrast)
"""

import argparse
import json
import time
from pathlib import Path

import clip
import numpy as np
import PIL.Image
import torch

from metrics.paths import METRICS_ROOT, derive_run_id_and_ckpt_tag

try:  # run as `python evaluation/eval_cirr_val_metrics.py` with PYTHONPATH=.
    from evaluation.vit_h.eval_cirr_h import (
        load_model, load_cirr_annotations, resolve_path, build_or_load_gallery,
    )
except ImportError:  # fall back to sibling import (script dir is on sys.path[0])
    from eval_cirr_h import (
        load_model, load_cirr_annotations, resolve_path, build_or_load_gallery,
    )

RECALL_K = (1, 2, 3, 5, 10, 20, 50, 100)
MAP_K    = (5, 10, 25, 50)
REF_K    = (1, 5, 10)


# ---------------------------------------------------------------------------
def short_label(run_id: str, ckpt_tag: str) -> str:
    """Compact, disambiguating tag for the combined table."""
    if 'binit' in run_id or 'lvbias' in run_id:
        fam = 'vib-elab'
    elif 'vib0.0002' in run_id:
        fam = 'vib2e-4'
    elif 'vib0.0001' in run_id:
        fam = 'vib1e-4'
    elif 'base' in run_id:
        fam = 'base'
    else:
        fam = run_id[:10]
    ep = (ckpt_tag.replace('epoch_', 'e')
                  .replace('_step_', 's')
                  .replace('checkpoint', 'e01'))
    return f'{fam}/{ep}'


def dataset_reference_similarity(queries, gallery):
    """cos(target_img, reference_img) for every val query (model-independent)."""
    embs = gallery['embeddings']                 # (N, 768) fp32, L2-normalized, cpu
    n2i  = gallery['name_to_idx']
    sims = []
    for q in queries:
        ti = n2i.get(q['target_hard'])
        ri = n2i.get(q['reference'])
        if ti is None or ri is None:
            continue
        sims.append(float(torch.dot(embs[ti], embs[ri])))
    return np.asarray(sims, dtype=np.float64)


def text_histogram(x, lo, hi, bins, width=44):
    counts, edges = np.histogram(x, bins=bins, range=(lo, hi))
    top = counts.max() if counts.max() > 0 else 1
    lines = []
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
        bar = '#' * int(round(width * c / top))
        lines.append(f'   [{e0:.2f}, {e1:.2f})  {c:5d}  {bar}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_metrics(model, queries, split_map, gallery, images_root, preprocess, device):
    gallery_embs = gallery['embeddings'].to(device)        # (N, 768) normalized
    n2i = gallery['name_to_idx']

    rank_gt, rank_ref, sim_ref, sim_tgt = [], [], [], []

    t0 = time.time()
    for qi, q in enumerate(queries):
        ref, cap, tgt = q['reference'], q['caption'], q['target_hard']

        ref_path = resolve_path(images_root, split_map[ref])
        ref_img  = preprocess(PIL.Image.open(ref_path).convert('RGB')).unsqueeze(0).to(device)
        txt      = model.clip.tokenize([cap]).to(device)
        p        = model.encode_query(ref_img, txt)         # (1, 768) normalized
        scores   = (p @ gallery_embs.T).squeeze(0)          # (N,)

        ri, ti = n2i[ref], n2i[tgt]
        sim_ref.append(scores[ri].item())
        sim_tgt.append(scores[ti].item())

        masked = scores.clone()
        masked[ri] = float('-inf')                          # CIRR masks the reference
        rg = int((masked > masked[ti]).sum().item()) + 1    # rank of target (ref removed)
        rr = int((scores  > scores[ri]).sum().item()) + 1   # rank of reference (unmasked)
        rank_gt.append(rg)
        rank_ref.append(rr)

        if (qi + 1) % 500 == 0:
            print(f'    [{qi+1:4d}/{len(queries)}] {(qi+1)/(time.time()-t0):.1f} q/s')

    rank_gt  = np.asarray(rank_gt,  dtype=np.float64)
    rank_ref = np.asarray(rank_ref, dtype=np.float64)
    sim_ref  = np.asarray(sim_ref,  dtype=np.float64)
    sim_tgt  = np.asarray(sim_tgt,  dtype=np.float64)

    return {
        'recall'         : {f'@{k}': float(np.mean(rank_gt <= k)) for k in RECALL_K},
        'mrr'            : float(np.mean(1.0 / rank_gt)),
        'map'            : {f'@{k}': float(np.mean(np.where(rank_gt <= k, 1.0 / rank_gt, 0.0)))
                            for k in MAP_K},
        'mean_rank_gt'   : float(np.mean(rank_gt)),
        'median_rank_gt' : float(np.median(rank_gt)),
        'stickiness'     : {
            **{f'r{k}_ref': float(np.mean(rank_ref <= k)) for k in REF_K},
            'mean_rank_ref'      : float(np.mean(rank_ref)),
            'median_rank_ref'    : float(np.median(rank_ref)),
            'sim_ref_mean'       : float(np.mean(sim_ref)),
            'sim_target_mean'    : float(np.mean(sim_tgt)),
            'sim_delta'          : float(np.mean(sim_ref) - np.mean(sim_tgt)),
            'frac_closer_to_ref' : float(np.mean(sim_ref > sim_tgt)),
        },
        'n_queries' : len(queries),
        'n_gallery' : int(gallery_embs.shape[0]),
    }


def print_block(tag, m):
    print('\n' + '-' * 72)
    print(f'  {tag}')
    print('-' * 72)
    r = m['recall']
    print('  Recall@k (full gallery, reference masked):')
    print('    ' + '   '.join(f'@{k}={r[f"@{k}"]*100:5.2f}' for k in RECALL_K))
    print(f'  MRR = {m["mrr"]*100:.2f}   (== mAP_full for single hard GT)')
    print('  mAP@k:  ' + '   '.join(f'@{k}={m["map"][f"@{k}"]*100:5.2f}' for k in MAP_K))
    print(f'  rank_gt:  mean={m["mean_rank_gt"]:.1f}   median={m["median_rank_gt"]:.0f}')
    s = m['stickiness']
    print('  Stickiness (reference-copying):')
    print('    ' + '   '.join(f'R{k}_ref={s[f"r{k}_ref"]*100:5.2f}' for k in REF_K))
    print(f'    rank_ref: mean={s["mean_rank_ref"]:.1f}   median={s["median_rank_ref"]:.0f}')
    print(f'    sim_ref={s["sim_ref_mean"]:.4f}   sim_target={s["sim_target_mean"]:.4f}'
          f'   simD(ref-tgt)={s["sim_delta"]:+.4f}'
          f'   closer_to_ref={s["frac_closer_to_ref"]*100:.1f}%')


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, nargs='+', required=True)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--ann_root',    type=Path, default=Path('/notebooks/experiment/CIRR/annotations'))
    ap.add_argument('--images_root', type=Path, default=Path('/notebooks/experiment/CIRR'))
    ap.add_argument('--gallery_cache_path', type=Path, default=None)
    args = ap.parse_args()

    split = 'val'
    queries, split_map = load_cirr_annotations(args.ann_root, split)
    print('=' * 72)
    print(f'CIRR {split} — comprehensive full-gallery metrics')
    print('=' * 72)
    print(f'  queries = {len(queries):,}   gallery images = {len(split_map):,}')

    # gallery is checkpoint-independent; load the first ckpt only to learn the CLIP name
    model0, clip_model_name = load_model(args.checkpoint[0], args.device)
    preprocess = model0.clip.preprocess   # open_clip ViT-H preprocess
    del model0
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    # preprocess obtained from model0 above (open_clip ViT-H)
    clip_tag = clip_model_name.replace('/', '-')
    cache_path = args.gallery_cache_path or (
        args.images_root / f'cirr_gallery_{split}_{clip_tag}.pt'
    )
    gallery = build_or_load_gallery(split, split_map, args.images_root,
                                    clip_model_name, args.device, cache_path)

    # ---- dataset characterization (model-independent) ----
    gtref = dataset_reference_similarity(queries, gallery)
    print('\n' + '=' * 72)
    print('DATASET CHARACTERIZATION  —  cos(target_img, reference_img), frozen CLIP')
    print('  (how visually similar CIRR targets are to their references)')
    print('=' * 72)
    pct = {p: float(np.percentile(gtref, p)) for p in (10, 25, 50, 75, 90)}
    print(f'  n={len(gtref)}   mean={gtref.mean():.4f}   median={np.median(gtref):.4f}'
          f'   std={gtref.std():.4f}   min={gtref.min():.4f}   max={gtref.max():.4f}')
    print(f'  p10={pct[10]:.3f}   p25={pct[25]:.3f}   p50={pct[50]:.3f}'
          f'   p75={pct[75]:.3f}   p90={pct[90]:.3f}')
    lo = float(np.floor(gtref.min() * 20) / 20)
    hi = float(np.ceil(gtref.max() * 20) / 20)
    nb = max(8, int(round((hi - lo) / 0.05)))
    print('  histogram (bin width 0.05):')
    print(text_histogram(gtref, lo, hi, nb))

    # ---- per-checkpoint metrics ----
    results = []
    for ckpt in args.checkpoint:
        run_id, ckpt_tag = derive_run_id_and_ckpt_tag(ckpt)
        tag = f'{run_id} :: {ckpt_tag}'
        print('\n' + '#' * 72)
        print(f'# {tag}')
        print('#' * 72)
        model, _ = load_model(ckpt, args.device)
        print(f'  logit_scale.exp() = {model.clip.logit_scale.exp().item():.2f}')
        m = compute_metrics(model, queries, split_map, gallery,
                            args.images_root, preprocess, args.device)
        print_block(tag, m)
        out_dir = METRICS_ROOT / 'cirr_val_full'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{run_id}__{ckpt_tag}.json'
        json.dump(m, open(out_path, 'w'), indent=2)
        print(f'  saved -> {out_path}')
        results.append((short_label(run_id, ckpt_tag), m))
        del model
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()

    # ---- combined compact table ----
    print('\n' + '=' * 72)
    print('COMBINED  (CIRR val, full gallery)')
    print('=' * 72)
    print(f'{"checkpoint":<22} {"R@1":>6} {"R@5":>6} {"R@10":>6} {"R@50":>6} '
          f'{"MRR":>6} {"mAP@10":>7} {"medGT":>6}  {"R10ref":>7} {"medRef":>7} '
          f'{"simD":>8} {"clRef%":>7}')
    for lbl, m in results:
        r, s = m['recall'], m['stickiness']
        print(f'{lbl:<22} {r["@1"]*100:6.2f} {r["@5"]*100:6.2f} {r["@10"]*100:6.2f} '
              f'{r["@50"]*100:6.2f} {m["mrr"]*100:6.2f} {m["map"]["@10"]*100:7.2f} '
              f'{m["median_rank_gt"]:6.0f}  {s["r10_ref"]*100:7.2f} '
              f'{s["median_rank_ref"]:7.0f} {s["sim_delta"]:+8.4f} '
              f'{s["frac_closer_to_ref"]*100:7.1f}')
    print('\nNote: one hard GT per query -> mAP_full == MRR; mAP@k is the rank-capped')
    print('reciprocal rank, shown per-k only for parity with the CIRCO mAP columns.')
    print('simD = sim(query,ref) - sim(query,target);  +ve = query sits closer to the')
    print('reference than to the target (sticky).  clRef% = share of queries where that holds.')


if __name__ == '__main__':
    main()
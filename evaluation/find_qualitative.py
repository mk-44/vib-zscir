"""
evaluation/find_qualitative.py

Identify the best CIRCO-val query for the qualitative figure (single-column:
reference + caption on top, baseline top-5 and VIB top-5 stacked below).

Selection filter (the honest cherry-pick: pick by rule, then by legibility):
  (a) baseline buries the target : baseline best-GT rank  > --base_gt_min   (default 10)
  (b) VIB surfaces the target    : vib      best-GT rank <= --vib_gt_max    (default 5)
  (c) baseline is reference-sticky: reference rank (UNMASKED) <= --ref_max  (default 5)
      -> baseline's composed query sits on top of the reference, so its top-5
         are reference-lookalikes and the target is buried. THIS is the picture
         of "re-encodes the reference"; VIB demotes it and the target surfaces.

Ranks use the official CIRCO retrieval convention: the reference image is masked
out of the gallery before ranking the target (you can't retrieve the reference as
the answer). The reference rank in (c) is computed UNMASKED on purpose -- that is
the stickiness signal.

Candidates are sorted to favour cardinality / spatial captions (unmistakable at
thumbnail size) and a large baseline->VIB rank improvement. For each candidate it
prints caption, both GT ranks, the reference rank, whether any reference-lookalike
sits in baseline's top-5, and the top-5 gallery ids for both models. A JSON is saved
for the render script to consume.

RUN (own process, PYTHONPATH=codes):
  python evaluation/find_qualitative.py \
    --baseline_ckpt /notebooks/experiment/4M_checkpoints/base_hd256_tok1_lslr0.2_lsmax116/epoch_02_step_03000.pt \
    --vib_ckpt      /notebooks/experiment/4M_checkpoints/vib_hd256_tok1_lslr0.2_lsmax116_vib0.0001/epoch_01_step_01500.pt
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.stickiness_compare import queries_ours, load_gallery

# caption-category keywords (legible-at-thumbnail edits float to the top)
CARD = [r'\btwo\b', r'\bthree\b', r'\bfour\b', r'\bfive\b', r'\bsix\b', r'\bseveral\b',
        r'\bmultiple\b', r'\bmore\b', r'\bfewer\b', r'\bless\b', r'\bsingle\b',
        r'\bpair\b', r'\bone\b', r'\bonly\b', r'\d+']
SPAT = [r'\bleft\b', r'\bright\b', r'\bbehind\b', r'\bfront\b', r'\bnext to\b',
        r'\bbeside\b', r'\babove\b', r'\bbelow\b', r'\bon top\b', r'\bunder\b',
        r'\bbetween\b', r'\bunderneath\b']
NEG = [r'\bwithout\b', r'\bno\b', r'\bnot\b', r'\bremove\b']


def caption_tags(cap):
    c = cap.lower()
    tags = []
    if any(re.search(p, c) for p in CARD):
        tags.append('CARD')
    if any(re.search(p, c) for p in SPAT):
        tags.append('SPAT')
    if any(re.search(p, c) for p in NEG):
        tags.append('NEG')
    return tags


@torch.no_grad()
def per_query_ranks(Q, gallery, img_ids, id2idx, anns, topk=5):
    """For each query return target ranks (reference MASKED), reference rank
    (UNMASKED), and masked top-k gallery ids."""
    out = []
    for q, a in enumerate(anns):
        p = Q[q].to(gallery.device)
        scores = (p @ gallery.T).squeeze(0)
        ref_idx = id2idx[str(a['reference_img_id'])]
        ref_rank = int((scores > scores[ref_idx]).sum().item()) + 1   # UNMASKED

        sm = scores.clone()
        sm[ref_idx] = float('-inf')                                   # official mask
        gt_idx = [id2idx[str(g)] for g in a['gt_img_ids'] if str(g) in id2idx]
        gt_ranks = [int((sm > sm[gi]).sum().item()) + 1 for gi in gt_idx]
        best_gt = min(gt_ranks) if gt_ranks else None
        top_ids = [int(img_ids[i]) for i in sm.topk(topk).indices.cpu().tolist()]
        # how reference-similar is the top-k (mean cosine of top-k to the reference img)
        ref_vec = gallery[ref_idx]
        topk_sim_to_ref = float((gallery[sm.topk(topk).indices] @ ref_vec).mean().item())
        out.append(dict(best_gt=best_gt, ref_rank=ref_rank, top_ids=top_ids,
                        gt_ids=[int(g) for g in a['gt_img_ids']],
                        topk_sim_to_ref=topk_sim_to_ref))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline_ckpt', type=Path, required=True)
    ap.add_argument('--vib_ckpt', type=Path, required=True)
    ap.add_argument('--circo_data_path', type=Path,
                    default=Path('/notebooks/experiment/eval_data/circo'))
    ap.add_argument('--openai_gallery', type=Path, default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--base_gt_min', type=int, default=10, help='baseline GT rank must exceed this')
    ap.add_argument('--vib_gt_max', type=int, default=5, help='VIB GT rank must be <= this')
    ap.add_argument('--ref_max', type=int, default=5, help='baseline reference rank must be <= this')
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--show', type=int, default=15, help='how many candidates to print')
    ap.add_argument('--out', type=Path,
                    default=Path('/notebooks/experiment/codes/metrics/qualitative_candidates.json'))
    args = ap.parse_args()
    dev = args.circo_data_path
    cdp = args.circo_data_path
    if args.openai_gallery is None:
        args.openai_gallery = cdp / 'gallery_cache_ViT-L-14.pt'

    print('encoding baseline composed queries (val)...')
    Qb, anns, _, _ = queries_ours(args.baseline_ckpt, cdp, args.device, 'val')
    print('encoding vib composed queries (val)...')
    Qv, anns_v, _, _ = queries_ours(args.vib_ckpt, cdp, args.device, 'val')
    assert [a['query_id'] for a in anns] == [a['query_id'] for a in anns_v], \
        'val ordering mismatch between the two runs'

    gallery, img_ids, id2idx = load_gallery(args.openai_gallery, args.device)
    Qb = F.normalize(Qb.float(), dim=-1)
    Qv = F.normalize(Qv.float(), dim=-1)

    rb = per_query_ranks(Qb, gallery, img_ids, id2idx, anns, args.topk)
    rv = per_query_ranks(Qv, gallery, img_ids, id2idx, anns, args.topk)

    cands = []
    for i, a in enumerate(anns):
        b, v = rb[i], rv[i]
        if b['best_gt'] is None or v['best_gt'] is None:
            continue
        if (b['best_gt'] > args.base_gt_min and v['best_gt'] <= args.vib_gt_max
                and b['ref_rank'] <= args.ref_max):
            tags = caption_tags(a['relative_caption'])
            cands.append(dict(
                query_id=a['query_id'],
                reference_img_id=int(a['reference_img_id']),
                caption=a['relative_caption'],
                tags=tags,
                baseline_gt_rank=b['best_gt'],
                vib_gt_rank=v['best_gt'],
                ref_rank_baseline=b['ref_rank'],
                rank_gain=b['best_gt'] - v['best_gt'],
                baseline_top5_sim_to_ref=round(b['topk_sim_to_ref'], 3),
                vib_top5_sim_to_ref=round(v['topk_sim_to_ref'], 3),
                baseline_top5=b['top_ids'],
                vib_top5=v['top_ids'],
                gt_ids=b['gt_ids'],
            ))

    # sort: legible captions first (CARD/SPAT), then biggest rank improvement,
    # then baseline top-5 most reference-similar (cleanest "re-encodes ref" picture)
    def key(c):
        legible = 0 if ('CARD' in c['tags'] or 'SPAT' in c['tags']) else 1
        return (legible, -c['rank_gain'], -c['baseline_top5_sim_to_ref'])
    cands.sort(key=key)

    print(f'\n{len(cands)} candidates pass the filter '
          f'(baseline GT>{args.base_gt_min}, VIB GT<={args.vib_gt_max}, ref<= {args.ref_max})\n')
    hdr = f"{'query_id':>9s} {'tags':>9s} {'base_gt':>7s} {'vib_gt':>6s} {'ref':>4s} {'gain':>4s} {'b.sim':>6s}  caption"
    print(hdr); print('-' * len(hdr))
    for c in cands[:args.show]:
        print(f"{str(c['query_id']):>9s} {','.join(c['tags']) or '-':>9s} "
              f"{c['baseline_gt_rank']:7d} {c['vib_gt_rank']:6d} {c['ref_rank_baseline']:4d} "
              f"{c['rank_gain']:4d} {c['baseline_top5_sim_to_ref']:6.3f}  {c['caption'][:60]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cands, open(args.out, 'w'), indent=2)
    print(f'\nsaved {len(cands)} candidates -> {args.out}')
    print('Pick a query_id from the top of the list (prefer CARD/SPAT, high gain, '
          'high b.sim), eyeball its images for legibility, and tell me which one to render.')


if __name__ == '__main__':
    main()
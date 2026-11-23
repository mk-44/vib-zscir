"""
evaluation/dataset_ref_target_similarity.py

Quantify how visually similar TARGETS are to REFERENCES in each ZS-CIR benchmark,
in frozen-CLIP space. MODEL-INDEPENDENT: uses only the cached gallery image
features + each benchmark's val annotations. No checkpoint, no model forward.

For each benchmark it reports:
  cos(target_img, reference_img):   mean / median / std / p10..p90 / histogram
  random-pair floor:                mean cosine between two random gallery images
                                    (exact closed form) + a sampled std
  gap = mean(cos_gt_ref) - floor:   how much MORE similar than chance targets are
                                    to references  --  the anisotropy-controlled,
                                    cross-benchmark-comparable number

WHY the floor matters: CLIP image-image cosine has a high, benchmark-specific
baseline (embedding anisotropy + how homogeneous a gallery is). "0.69 is high"
only means something relative to that floor and to the other benchmarks, so the
GAP is the number to compare, not the raw cosine.

Run (CPU is fine; galleries are cached, nothing is encoded):
    cd /notebooks/experiment/codes
    PYTHONPATH=. python evaluation/dataset_ref_target_similarity.py

Each benchmark runs independently; a missing dataset/cache is reported and
skipped while the others still produce numbers.
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


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def normalize(embs):
    return F.normalize(embs.float(), dim=-1)


def random_pair_floor(embs, n_pairs=100_000, seed=0):
    """Exact mean pairwise cosine of a normalized gallery, plus a sampled std.

    For L2-normalized rows g_i, sum_{i!=j} g_i.g_j = (sum_i g_i).(sum_j g_j) - N,
    over N(N-1) ordered pairs -> exact mean with no O(N^2) blowup.
    """
    N = embs.shape[0]
    s = embs.sum(0).to(torch.float64)
    mean = (float(s @ s) - N) / (N * (N - 1))
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(0, N, (n_pairs,), generator=g)
    j = torch.randint(0, N, (n_pairs,), generator=g)
    keep = i != j
    cij = (embs[i[keep]] * embs[j[keep]]).sum(1)
    return mean, float(cij.std())


def text_histogram(x, lo, hi, bins, width=44):
    counts, edges = np.histogram(x, bins=bins, range=(lo, hi))
    top = counts.max() if counts.max() > 0 else 1
    out = []
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
        out.append(f'   [{e0:.2f}, {e1:.2f})  {c:6d}  {"#" * int(round(width * c / top))}')
    return '\n'.join(out)


def report(label, cos, floor_mean, floor_std, n_skipped=0):
    cos = np.asarray(cos, dtype=np.float64)
    pct = {p: float(np.percentile(cos, p)) for p in (10, 25, 50, 75, 90)}
    print('\n' + '=' * 72)
    print(f'{label}   —   cos(target_img, reference_img)')
    print('=' * 72)
    extra = f'   (skipped {n_skipped}: img id absent from gallery)' if n_skipped else ''
    print(f'  pairs={len(cos)}{extra}')
    print(f'  mean={cos.mean():.4f}   median={np.median(cos):.4f}   std={cos.std():.4f}'
          f'   min={cos.min():.4f}   max={cos.max():.4f}')
    print(f'  p10={pct[10]:.3f}  p25={pct[25]:.3f}  p50={pct[50]:.3f}'
          f'  p75={pct[75]:.3f}  p90={pct[90]:.3f}')
    fs = 'nan' if floor_std != floor_std else f'{floor_std:.4f}'   # nan-safe
    print(f'  random-pair floor: mean={floor_mean:.4f}  std={fs}')
    print(f'  GAP (mean - floor): {cos.mean() - floor_mean:+.4f}')
    lo = float(np.floor(min(cos.min(), floor_mean) * 20) / 20)
    hi = float(np.ceil(cos.max() * 20) / 20)
    nb = max(8, int(round((hi - lo) / 0.05)))
    print('  histogram (bin 0.05):')
    print(text_histogram(cos, lo, hi, nb))
    return {
        'label': label, 'n': len(cos), 'mean': float(cos.mean()),
        'median': float(np.median(cos)), 'std': float(cos.std()),
        'floor_mean': float(floor_mean), 'gap': float(cos.mean() - floor_mean),
    }


# --------------------------------------------------------------------------
# CIRR  (val gallery cache keyed by name; ref + target both in gallery)
# --------------------------------------------------------------------------
def do_cirr(args, clip_tag):
    gal = torch.load(args.cirr_images_root / f'cirr_gallery_val_{clip_tag}.pt',
                     map_location='cpu')
    embs = normalize(gal['embeddings'])
    n2i = gal['name_to_idx']
    cap = json.load(open(args.cirr_ann_root / 'captions' / 'cap.rc2.val.json'))
    if isinstance(cap, dict):
        cap = list(cap.values())
    cos = []
    for q in cap:
        ti, ri = n2i.get(q['target_hard']), n2i.get(q['reference'])
        if ti is None or ri is None:
            continue
        cos.append(float(torch.dot(embs[ti], embs[ri])))
    fm, fs = random_pair_floor(embs, args.n_pairs)
    return report('CIRR (val)', cos, fm, fs)


# --------------------------------------------------------------------------
# Fashion-IQ  (per-category gallery tensor parallel to split; candidate/target)
# --------------------------------------------------------------------------
def do_fashion(args, clip_tag):
    all_cos, floor_accum, ntot = [], [], 0
    for cat in FASHION_CATS:
        names = json.load(open(args.fashion_data_path / 'image_splits' / f'split.{cat}.val.json'))
        n2i = {n: i for i, n in enumerate(names)}
        embs = normalize(torch.load(
            args.fashion_data_path / f'gallery_cache_{cat}_{clip_tag}.pt', map_location='cpu'))
        ann = json.load(open(args.fashion_data_path / 'captions' / f'cap.{cat}.val.json'))
        cos = []
        for a in ann:
            ri, ti = n2i.get(a['candidate']), n2i.get(a['target'])
            if ri is None or ti is None:
                continue
            cos.append(float(torch.dot(embs[ti], embs[ri])))
        fm, fs = random_pair_floor(embs, args.n_pairs)
        report(f'Fashion-IQ / {cat} (val)', cos, fm, fs)
        all_cos.extend(cos)
        floor_accum.append((fm, len(cos)))
        ntot += len(cos)
    pooled_floor = sum(fm * n for fm, n in floor_accum) / max(ntot, 1)
    print('\n' + '-' * 72)
    print('Fashion-IQ pooled (3 categories); floor = query-weighted mean of category floors')
    return report('Fashion-IQ (val, pooled)', all_cos, pooled_floor, float('nan'))


# --------------------------------------------------------------------------
# CIRCO  (123k gallery keyed by img_id; multi-GT via official dataset class)
# --------------------------------------------------------------------------
def do_circo(args, clip_tag):
    gal = torch.load(args.circo_data_path / f'gallery_cache_{clip_tag}.pt',
                     map_location='cpu')
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

    cos, skipped = [], 0
    for ann in ds.annotations:
        ri = id2i.get(str(ann['reference_img_id']))
        if ri is None:
            skipped += 1
            continue
        gts = [id2i.get(str(g)) for g in ann['gt_img_ids']]
        gts = [g for g in gts if g is not None]
        if not gts:
            skipped += 1
            continue
        for gi in gts:                                   # pool all (ref, gt) pairs
            cos.append(float(torch.dot(embs[gi], embs[ri])))
    fm, fs = random_pair_floor(embs, args.n_pairs)
    return report('CIRCO (val, pooled over multi-GT)', cos, fm, fs, skipped)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip_model', type=str, default=CLIP_TAG_DEFAULT)
    ap.add_argument('--n_pairs', type=int, default=100_000)
    ap.add_argument('--cirr_ann_root',     type=Path, default=Path('/notebooks/experiment/CIRR/annotations'))
    ap.add_argument('--cirr_images_root',  type=Path, default=Path('/notebooks/experiment/CIRR'))
    ap.add_argument('--circo_data_path',   type=Path, default=Path('/notebooks/experiment/eval_data/circo'))
    ap.add_argument('--fashion_data_path', type=Path, default=Path('/notebooks/experiment/eval_data/fashion_iq'))
    ap.add_argument('--benchmarks', nargs='+', default=['cirr', 'circo', 'fashion'])
    args = ap.parse_args()
    clip_tag = args.clip_model.replace('/', '-')

    runners = {'cirr': do_cirr, 'circo': do_circo, 'fashion': do_fashion}
    summaries = []
    for b in args.benchmarks:
        try:
            summaries.append(runners[b](args, clip_tag))
        except Exception as e:
            print(f'\n[!] {b} failed: {type(e).__name__}: {e}')

    if summaries:
        print('\n' + '=' * 72)
        print('CROSS-BENCHMARK SUMMARY  —  reference/target visual similarity (frozen CLIP)')
        print('=' * 72)
        print(f'{"benchmark":<32} {"pairs":>8} {"mean":>7} {"median":>7} {"floor":>7} {"gap":>8}')
        for s in summaries:
            print(f'{s["label"]:<32} {s["n"]:>8} {s["mean"]:>7.4f} {s["median"]:>7.4f} '
                  f'{s["floor_mean"]:>7.4f} {s["gap"]:>+8.4f}')
        print('\ngap = mean cos(target,ref) - the gallery\'s random-pair floor.')
        print('Higher gap => targets sit closer to references than chance => a more')
        print('reference-similar benchmark, where a reference-sticky mapper is rewarded.')

    out = Path('/notebooks/experiment/codes/metrics/dataset_ref_target_similarity.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summaries, open(out, 'w'), indent=2)
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
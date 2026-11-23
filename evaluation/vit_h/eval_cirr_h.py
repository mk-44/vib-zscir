"""
CIRR evaluation — validation (local Recall@K + Recall_Subset@K + stickiness)
and test1 (the two submission JSONs for the official ANU server).

Mirrors eval_circo_v2.py / eval_circo_test.py exactly where it can:
  * identical load_model (reads mapper_type + config from the checkpoint,
    restores the trainable logit_scale)
  * identical query encoding via model.encode_query(ref_img, caption_tokens)
  * gallery = frozen CLIP image features, fp32, L2-normalized, cached per split.
    The cache is checkpoint-INDEPENDENT, so the deterministic baseline and the
    VIB model share one gallery encode (built once, reused).

CIRR specifics handled here (different from CIRCO):
  * one relative caption per query (not two joined like Fashion-IQ)
  * global Recall@{1,5,10,50} with the query's reference MASKED from candidates
  * Recall_Subset@{1,2,3}: rank only within img_set.members, reference excluded
  * test1 has NO ground truth -> we emit TWO submission files
        recall  : {pairid: top-50 names}  + {'version':'rc2','metric':'recall'}
        subset  : {pairid: top-3  names}  + {'version':'rc2','metric':'recall_subset'}
    Upload each SEPARATELY at https://cirr.cecs.anu.edu.au/test_process/

Image resolution is by name:
    split.rc2[name] == './dev/<name>.png'  ->  <images_root>/dev/<name>.png
    (val uses the dev-* images, test1 uses the test1-* images)

Usage (from the codes root, same pattern as the CIRCO scripts):
    cd /notebooks/experiment/codes

    # validation — local metrics (run for BOTH baseline and VIB checkpoints):
    PYTHONPATH=. python evaluation/eval_cirr.py --checkpoint <ckpt.pt> --split val

    # test1 — generate the two submission JSONs:
    PYTHONPATH=. python evaluation/eval_cirr.py --checkpoint <ckpt.pt> --split test1
"""

import argparse
import json
import time
from pathlib import Path

import clip
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config.config_h import load_from_yaml
from model.model_h import Pic2Word
from metrics.paths import METRICS_ROOT, derive_run_id_and_ckpt_tag


# ---------------------------------------------------------------------------
# Model loading — byte-for-byte the same as eval_circo_v2.py / eval_circo_test.py
# so the CIRR numbers are produced by an identical model to the CIRCO numbers.
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: Path, device: str):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    ckpt_cfg = ckpt.get('config', {})

    cfg = load_from_yaml(
        '/notebooks/experiment/codes/config/config_h.yaml',
        clip_model          = ckpt_cfg.get('clip_model',          'ViT-L/14'),
        mapper_type         = ckpt['mapper_type'],
        hidden_dim          = ckpt_cfg.get('hidden_dim',          256),
        num_tokens          = ckpt_cfg.get('num_tokens',          1),
        num_heads           = ckpt_cfg.get('num_heads',           8),
        num_layers          = ckpt_cfg.get('num_layers',          1),
        dropout             = ckpt_cfg.get('dropout',             0.1),
        merge_tokens        = ckpt_cfg.get('merge_tokens',        None),
        merge_type          = ckpt_cfg.get('merge_type',          'learned'),
        max_token_len       = ckpt_cfg.get('max_token_len',       77),
        clip_prompt         = ckpt_cfg.get('clip_prompt',         'a photo of'),
        vib_logvar_bias     = ckpt_cfg.get('vib_logvar_bias',     -2.0),
        vib_logvar_clip_max = ckpt_cfg.get('vib_logvar_clip_max',  2.0),
        clip_backend        = ckpt_cfg.get('clip_backend',        'open_clip'),
        clip_pretrained     = ckpt_cfg.get('clip_pretrained',     None),
        device              = device,
    )

    model = Pic2Word(cfg).to(device)
    model.mapper.load_state_dict(ckpt['mapper_state'])
    if ckpt.get('merger_state') is not None and model.merger is not None:
        model.merger.load_state_dict(ckpt['merger_state'])
    if 'logit_scale' in ckpt:
        with torch.no_grad():
            model.clip.logit_scale.fill_(ckpt['logit_scale'])
    model.eval()
    return model, cfg.clip_model


# ---------------------------------------------------------------------------
# CIRR annotation + path helpers
# ---------------------------------------------------------------------------
def load_cirr_annotations(ann_root: Path, split: str):
    """Returns (queries: list[dict], split_map: dict[name -> './dev/<name>.png'])."""
    cap_path   = ann_root / 'captions'     / f'cap.rc2.{split}.json'
    split_path = ann_root / 'image_splits' / f'split.rc2.{split}.json'
    queries   = json.load(open(cap_path))
    split_map = json.load(open(split_path))
    if isinstance(queries, dict):                 # be tolerant; cap files are lists
        queries = list(queries.values())
    return queries, split_map


def resolve_path(images_root: Path, relpath: str) -> Path:
    """'./dev/dev-244-0-img0.png' -> <images_root>/dev/dev-244-0-img0.png"""
    return images_root / relpath.removeprefix('./')


# ---------------------------------------------------------------------------
# Gallery: frozen CLIP image features for every unique image in the split.
# Checkpoint-independent -> built once, shared by baseline and VIB.
# (Same encode as build_circo_gallery.py: encode_image -> float -> L2-normalize.)
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_or_load_gallery(split, split_map, images_root, clip_model_name,
                          device, cache_path, batch_size=64, force=False):
    if cache_path.exists() and not force:
        cache = torch.load(cache_path, map_location='cpu')
        if cache.get('clip_model') == clip_model_name and cache.get('split') == split:
            print(f'  gallery cache hit: {cache_path}  ({len(cache["names"]):,} imgs)')
            return cache
        print('  gallery cache present but stale (clip_model/split mismatch) -> rebuilding')

    names = list(split_map.keys())
    paths = [resolve_path(images_root, split_map[n]) for n in names]

    missing = [str(p) for p in paths if not p.exists()]
    assert not missing, (
        f'{len(missing)} gallery images missing on disk, e.g. {missing[:3]}.\n'
        f'Check --images_root (expected <root>/dev for val, <root>/test1 for test1).'
    )

    import open_clip
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained='laion2b_s32b_b79k', device=device)
    clip_model = clip_model.float()
    clip_model.eval()

    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc=f'  encode {split} gallery'):
        batch = paths[i:i + batch_size]
        imgs = torch.stack(
            [preprocess(PIL.Image.open(p).convert('RGB')) for p in batch]
        ).to(device)
        e = clip_model.encode_image(imgs)
        e = F.normalize(e.float(), dim=-1)
        embs.append(e.cpu())

    embeddings = torch.cat(embs, dim=0)
    assert embeddings.dtype == torch.float32
    norms = embeddings.norm(dim=-1)
    assert (norms - 1.0).abs().max() < 1e-3, 'gallery embeddings not normalized'

    cache = {
        'embeddings'  : embeddings,
        'names'       : names,
        'name_to_idx' : {n: i for i, n in enumerate(names)},
        'clip_model'  : clip_model_name,
        'split'       : split,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f'  gallery built & cached: {cache_path}  ({embeddings.shape[0]:,} imgs)')
    return cache


# ---------------------------------------------------------------------------
# Core: encode every query, score against the gallery, build ranked lists.
# Global recall masks the reference; subset recall ranks within img_set.members
# (reference removed). On val (target_hard present) we also compute metrics and
# stickiness; on test1 we only build the submission lists.
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_queries(model, queries, split_map, gallery, images_root, preprocess, device):
    gallery_embs = gallery['embeddings'].to(device)        # (N, 768)
    name_to_idx  = gallery['name_to_idx']
    names        = gallery['names']

    GLOBAL_K = (1, 5, 10, 50)
    SUBSET_K = (1, 2, 3)
    have_gt  = all('target_hard' in q for q in queries)    # val: True, test1: False

    recall_submission = {}                                 # pairid -> top-50 names
    subset_submission = {}                                 # pairid -> top-3 names

    rec_hits = {k: [] for k in GLOBAL_K}
    sub_hits = {k: [] for k in SUBSET_K}
    sim_refs, rank_refs, sim_tgts = [], [], []

    t0 = time.time()
    for qi, q in enumerate(queries):
        pairid  = q['pairid']
        ref     = q['reference']
        caption = q['caption']
        members = q['img_set']['members']

        ref_path = resolve_path(images_root, split_map[ref])
        ref_img  = preprocess(PIL.Image.open(ref_path).convert('RGB')).unsqueeze(0).to(device)
        txt_tok  = model.clip.tokenize([caption]).to(device)
        p        = model.encode_query(ref_img, txt_tok)    # (1, 768) normalized

        scores = (p @ gallery_embs.T).squeeze(0)           # (N,)

        ref_idx = name_to_idx[ref]

        # ----- global ranking (reference masked, per CIRR spec) -----
        masked = scores.clone()
        masked[ref_idx] = float('-inf')
        top50_idx   = masked.topk(50).indices.cpu().tolist()
        top50_names = [names[i] for i in top50_idx]
        assert ref not in top50_names and len(set(top50_names)) == 50
        recall_submission[str(pairid)] = top50_names

        # ----- subset ranking (members minus reference) -----
        member_cands  = [m for m in members if m != ref]
        member_idx    = [name_to_idx[m] for m in member_cands]
        member_scores = scores[member_idx]
        order         = member_scores.argsort(descending=True).cpu().tolist()
        subset_ranked = [member_cands[j] for j in order]
        subset_submission[str(pairid)] = subset_ranked[:3]

        # ----- val-only metrics + stickiness diagnostics -----
        if have_gt:
            target = q['target_hard']
            for k in GLOBAL_K:
                rec_hits[k].append(float(target in top50_names[:k]))
            for k in SUBSET_K:
                sub_hits[k].append(float(target in subset_ranked[:k]))

            sim_ref  = scores[ref_idx].item()
            rank_ref = int((scores > scores[ref_idx]).sum().item()) + 1   # 1-indexed
            sim_tgt  = scores[name_to_idx[target]].item()
            sim_refs.append(sim_ref); rank_refs.append(rank_ref); sim_tgts.append(sim_tgt)

        if (qi + 1) % 200 == 0:
            print(f'  [{qi+1:4d}/{len(queries)}] {(qi+1)/(time.time()-t0):.1f} q/s')

    elapsed = time.time() - t0
    print(f'  queries done in {elapsed:.1f}s ({len(queries)/elapsed:.1f} q/s)')

    metrics = None
    if have_gt:
        metrics = {
            'recall_atk'        : {f'@{k}': float(np.mean(rec_hits[k])) for k in GLOBAL_K},
            'recall_subset_atk' : {f'@{k}': float(np.mean(sub_hits[k])) for k in SUBSET_K},
            'stickiness'        : {
                'sim_ref_mean'    : float(np.mean(sim_refs)),
                'sim_target_mean' : float(np.mean(sim_tgts)),
                'sim_delta'       : float(np.mean(sim_refs) - np.mean(sim_tgts)),  # +ve = sticky
                'r10_ref'         : float(np.mean([r <= 10 for r in rank_refs])),
                'mean_rank_ref'   : float(np.mean(rank_refs)),
                'median_rank_ref' : int(np.median(rank_refs)),
            },
            'n_queries' : len(queries),
            'n_gallery' : int(gallery_embs.shape[0]),
        }

    return recall_submission, subset_submission, metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',         type=Path, required=True)
    p.add_argument('--split',              type=str,  choices=['val', 'test1'], default='val')
    p.add_argument('--device',             type=str,  default='cuda')
    p.add_argument('--ann_root',           type=Path,
                   default=Path('/notebooks/experiment/CIRR/annotations'))
    p.add_argument('--images_root',        type=Path,
                   default=Path('/notebooks/experiment/CIRR'))
    p.add_argument('--gallery_cache_path', type=Path, default=None)
    p.add_argument('--rebuild_gallery',    action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    print('=' * 70)
    print(f'CIRR evaluation — split = {args.split}')
    print('=' * 70)
    print(f'  checkpoint : {args.checkpoint}')

    model, clip_model_name = load_model(args.checkpoint, args.device)
    print(f'  model loaded. logit_scale.exp() = {model.clip.logit_scale.exp().item():.2f}')

    # preprocess transform for the reference images (matches the gallery encode)
    preprocess = model.clip.preprocess   # open_clip ViT-H preprocess

    queries, split_map = load_cirr_annotations(args.ann_root, args.split)
    print(f'  queries = {len(queries):,}   gallery images = {len(split_map):,}')

    clip_tag = clip_model_name.replace('/', '-')
    cache_path = args.gallery_cache_path or (
        args.images_root / f'cirr_gallery_{args.split}_{clip_tag}.pt'
    )
    gallery = build_or_load_gallery(
        args.split, split_map, args.images_root, clip_model_name,
        args.device, cache_path, force=args.rebuild_gallery,
    )

    recall_sub, subset_sub, metrics = run_queries(
        model, queries, split_map, gallery, args.images_root, preprocess, args.device,
    )

    run_id, ckpt_tag = derive_run_id_and_ckpt_tag(args.checkpoint)

    if args.split == 'val':
        print('\n' + '=' * 70)
        print('CIRR val metrics')
        print('=' * 70)
        for k, v in metrics['recall_atk'].items():
            print(f'  R{k:<4}        : {v * 100:.2f}')
        for k, v in metrics['recall_subset_atk'].items():
            print(f'  R_subset{k:<4}: {v * 100:.2f}')
        s = metrics['stickiness']
        print(f'  R10_ref      : {s["r10_ref"] * 100:.1f}%')
        print(f'  simΔ(ref-tgt): {s["sim_delta"]:+.4f}   median_rank_ref: {s["median_rank_ref"]}')

        out_dir = METRICS_ROOT / 'cirr'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{run_id}__{ckpt_tag}.json'
        json.dump(metrics, open(out_path, 'w'), indent=2)
        print(f'\n  saved metrics -> {out_path}')

    else:  # test1 — write the two submission files
        assert len(recall_sub) == len(queries), 'recall submission count mismatch'
        assert len(subset_sub) == len(queries), 'subset submission count mismatch'

        recall_sub['version'] = 'rc2'; recall_sub['metric'] = 'recall'
        subset_sub['version'] = 'rc2'; subset_sub['metric'] = 'recall_subset'

        sub_dir = METRICS_ROOT / 'cirr_test_submissions'
        sub_dir.mkdir(parents=True, exist_ok=True)
        rpath = sub_dir / f'{run_id}__{ckpt_tag}__recall.json'
        spath = sub_dir / f'{run_id}__{ckpt_tag}__subset.json'
        json.dump(recall_sub, open(rpath, 'w'), separators=(',', ':'))
        json.dump(subset_sub, open(spath, 'w'), separators=(',', ':'))

        print('\n' + '=' * 70)
        print('CIRR test1 submissions written')
        print('=' * 70)
        print(f'  recall  -> {rpath}')
        print(f'  subset  -> {spath}')
        print('  Upload each SEPARATELY at https://cirr.cecs.anu.edu.au/test_process/')
        print('  (Recall and Recall_Subset are validated on different files.)')


if __name__ == '__main__':
    main()
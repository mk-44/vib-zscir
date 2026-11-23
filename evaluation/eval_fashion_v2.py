"""
Fashion-IQ evaluation with stickiness diagnostics.

FAST PATH: uses cached unnormalized CLIP visual features (built by
build_fashion_query_cache.py) for ref images. Skips CLIP image forward
per checkpoint — only mapper + text transform per ref image.

SLOW FALLBACK: for mappers that need intermediate CLS layers (Weighted,
MultiLayer, FusedWeighted), falls back to the standard extract_features
path that runs the full CLIP image forward.

Per category (dress / shirt / toptee):
  - Standard recall@K (target = ground truth)
  - Reference stickiness (rank of ref image, R@K_ref)
  - Target rank diagnostics
  - Similarity-geometry diagnostics
  - Score-distribution sanity

Plus an 'averaged' block (mean across 3 categories).
Per-query ranks stored for downstream bootstrap CIs.

Usage (single checkpoint):
    cd /notebooks/experiment/codes
    PYTHONPATH=/notebooks/experiment/codes python evaluation/eval_fashion_v2.py \
        --checkpoint /path/to/checkpoint.pt
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config.config import load_from_yaml
from model.model import Pic2Word, BaseMLP, VIBMapper
from evaluation.retrieval import extract_features, recall_at_k


CATEGORIES = ['dress', 'shirt', 'toptee']
DEFAULT_K_VALUES = [1, 5, 10, 50]


# ---------------------------------------------------------------------------
# Dataset (only used in slow fallback)
# ---------------------------------------------------------------------------
class _FashionIQImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224))
        return (self.transform(img),)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _load_annotations(category, split, eval_dir):
    with open(eval_dir / 'captions' / f'cap.{category}.{split}.json') as f:
        return json.load(f)


def _load_split(category, split, eval_dir):
    with open(eval_dir / 'image_splits' / f'split.{category}.{split}.json') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    ckpt_cfg = ckpt.get('config', {})

    cfg = load_from_yaml(
        '/notebooks/experiment/codes/config/config.yaml',
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
        # Forward-pass edit z <- rho*z/||z|| of unit-norm runs must also apply at test; absent/0.0 = off.
        # (Activates Step 4 of _encode_queries_from_cache below, which reads model.pseudo_norm.)
        pseudo_norm         = ckpt_cfg.get('pseudo_norm',          0.0),
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
# FAST PATH: encode queries from cached v_raw
# Replicates Pic2Word.encode_query() but skips the CLIP image forward.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _encode_queries_from_cache(model, v_raw_cached, modifiers, device,
                                batch_size=256):
    """
    v_raw_cached: (Q, 768) tensor on CPU, UNNORMALIZED CLIP visual features.
    modifiers   : list[str], length Q.

    Returns: (Q, 768) normalized fp32 on CPU — the query features.
    """
    Q = v_raw_cached.shape[0]
    assert len(modifiers) == Q, f'Q mismatch: {Q} vs {len(modifiers)}'

    all_feats = []
    for i in range(0, Q, batch_size):
        v_raw_batch = v_raw_cached[i:i+batch_size].to(device)            # (B, 768)
        text_batch  = modifiers[i:i+batch_size]
        text_tokens = clip.tokenize(text_batch, truncate=True).to(device)

        # --- Mirror Pic2Word.encode_query() exactly, just skip step 1 ---
        # Step 1 SKIPPED (use cached v_raw)
        # Step 2: mapper
        pseudo_tokens = model.mapper(v_raw_batch)
        # Step 3: optional merger
        if model.merger is not None:
            pseudo_tokens = model.merger(pseudo_tokens)
        # Step 4: optional pseudo_norm
        if model.pseudo_norm > 0:
            pseudo_tokens = F.normalize(pseudo_tokens, dim=-1) * model.pseudo_norm
        # Step 5: build token sequence + text transform
        tok_emb, eot_pos = model._build_token_sequence(pseudo_tokens, text_tokens)
        feats = model.clip.forward_text_transform(tok_emb, eot_pos)
        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0)


# ---------------------------------------------------------------------------
# Per-category eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def _eval_category(model, category, fashion_data_path, clip_model_name,
                   transform, device, k_values=DEFAULT_K_VALUES,
                   batch_size=256, num_workers=4):

    images_dir = fashion_data_path / 'images' / 'images'
    clip_tag   = clip_model_name.replace('/', '-')

    # 1. Load gallery cache (normalized, fp32)
    gallery_cache_path = fashion_data_path / f'gallery_cache_{category}_{clip_tag}.pt'
    gallery_feats = torch.load(gallery_cache_path, map_location='cpu')

    # 2. Build name → gallery-index map
    image_names = _load_split(category, 'val', fashion_data_path)
    name_to_idx = {name: i for i, name in enumerate(image_names)}
    assert len(image_names) == gallery_feats.shape[0], \
        f'Gallery cache count ({gallery_feats.shape[0]}) != split count ({len(image_names)})'

    # 3. Build queries
    annotations = _load_annotations(category, 'val', fashion_data_path)
    ref_paths, modifiers = [], []
    ref_indices, target_indices = [], []
    ref_names = []
    skipped = 0

    for ann in annotations:
        ref_name    = ann['candidate']
        target_name = ann['target']
        if ref_name not in name_to_idx or target_name not in name_to_idx:
            skipped += 1
            continue
        ref_path    = images_dir / f'{ref_name}.png'
        target_path = images_dir / f'{target_name}.png'
        if not ref_path.exists() or not target_path.exists():
            skipped += 1
            continue
        ref_paths.append(ref_path)
        modifiers.append(ann['captions'][0] + ' and ' + ann['captions'][1])
        ref_indices.append(name_to_idx[ref_name])
        target_indices.append(name_to_idx[target_name])
        ref_names.append(ref_name)

    Q = len(modifiers)
    if Q == 0:
        raise RuntimeError(f'[{category}] No valid queries')

    ref_indices_t    = torch.tensor(ref_indices,    dtype=torch.long)
    target_indices_t = torch.tensor(target_indices, dtype=torch.long)

    # 4. Compute query features
    # Decide path based on mapper type
    use_fast_path = isinstance(model.mapper, (BaseMLP, VIBMapper))

    if use_fast_path:
        # FAST: use cached v_raw
        query_cache_path = fashion_data_path / f'query_cache_{category}_{clip_tag}.pt'
        if not query_cache_path.exists():
            raise FileNotFoundError(
                f'Query cache not found: {query_cache_path}. '
                f'Run evaluation/build_fashion_query_cache.py first.'
            )
        qcache = torch.load(query_cache_path, map_location='cpu')
        # Look up cached v_raw for each ref name
        cache_n2i = qcache['name_to_idx']
        v_raw_cached = qcache['embeddings'][[cache_n2i[n] for n in ref_names]]
        # (Q, 768) UNNORMALIZED on CPU

        query_feats = _encode_queries_from_cache(
            model, v_raw_cached, modifiers, device, batch_size=batch_size
        )
    else:
        # SLOW FALLBACK: full CLIP image forward (multilayer mappers)
        query_dataset = _FashionIQImageDataset(ref_paths, transform)
        query_loader = DataLoader(
            query_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        query_feats = extract_features(model, query_loader, device, text_list=modifiers)

    assert query_feats.shape == (Q, 768)

    # 5. Compute similarity matrix on GPU
    qf = query_feats.to(device)
    gf = gallery_feats.to(device)
    sims = qf @ gf.T   # (Q, N)

    # 6. Standard recall@K
    recalls = recall_at_k(sims.cpu(), target_indices_t, k_values=k_values)

    # 7. Stickiness diagnostics
    ref_idx_gpu = ref_indices_t.to(device)
    tgt_idx_gpu = target_indices_t.to(device)

    sim_ref = sims[torch.arange(Q, device=device), ref_idx_gpu]
    sim_gt  = sims[torch.arange(Q, device=device), tgt_idx_gpu]
    rank_ref    = (sims > sim_ref.unsqueeze(1)).sum(dim=1) + 1
    rank_target = (sims > sim_gt.unsqueeze(1)).sum(dim=1) + 1
    top1_scores  = sims.topk(1,  dim=1).values.squeeze(1)
    top10_scores = sims.topk(10, dim=1).values.mean(dim=1)

    rank_ref_cpu    = rank_ref.cpu()
    rank_target_cpu = rank_target.cpu()
    sim_ref_cpu     = sim_ref.cpu()
    sim_gt_cpu      = sim_gt.cpu()
    top1_cpu        = top1_scores.cpu()
    top10_cpu       = top10_scores.cpu()

    stickiness = {
        'r1_ref':  float((rank_ref_cpu <=  1).float().mean()),
        'r5_ref':  float((rank_ref_cpu <=  5).float().mean()),
        'r10_ref': float((rank_ref_cpu <= 10).float().mean()),
        'r50_ref': float((rank_ref_cpu <= 50).float().mean()),
        'r1_target':  float((rank_target_cpu <=  1).float().mean()),
        'r5_target':  float((rank_target_cpu <=  5).float().mean()),
        'r10_target': float((rank_target_cpu <= 10).float().mean()),
        'r50_target': float((rank_target_cpu <= 50).float().mean()),
        'mean_rank_ref':   float(rank_ref_cpu.float().mean()),
        'median_rank_ref': int(rank_ref_cpu.float().median()),
        'mean_rank_target':   float(rank_target_cpu.float().mean()),
        'median_rank_target': int(rank_target_cpu.float().median()),
        'sim_ref_mean': float(sim_ref_cpu.mean()),
        'sim_ref_std':  float(sim_ref_cpu.std()),
        'sim_gt_mean':  float(sim_gt_cpu.mean()),
        'sim_gt_std':   float(sim_gt_cpu.std()),
        'sim_delta':                float(sim_ref_cpu.mean() - sim_gt_cpu.mean()),
        'sim_delta_per_query_mean': float((sim_ref_cpu - sim_gt_cpu).mean()),
        'frac_ref_above_target': float((sim_ref_cpu > sim_gt_cpu).float().mean()),
        'top1_minus_top10_mean': float((top1_cpu - top10_cpu).mean()),
        'score_top1_mean':       float(top1_cpu.mean()),
    }

    per_query = {
        'rank_ref':    rank_ref_cpu.tolist(),
        'rank_target': rank_target_cpu.tolist(),
    }

    return {
        'recall_atk': {f'@{k}': v for k, v in recalls.items()},
        'stickiness': stickiness,
        'per_query':  per_query,
        'n_queries':  Q,
        'n_gallery':  gallery_feats.shape[0],
        'n_skipped':  skipped,
        'used_fast_path': use_fast_path,
    }


def _averaged(per_cat_results):
    cats = [per_cat_results[c] for c in CATEGORIES]
    avg_recall = {k: sum(c['recall_atk'][k] for c in cats) / len(cats)
                  for k in cats[0]['recall_atk']}
    avg_stick = {k: sum(c['stickiness'][k] for c in cats) / len(cats)
                 for k in cats[0]['stickiness']}
    return {'recall_atk': avg_recall, 'stickiness': avg_stick}


# ---------------------------------------------------------------------------
# Top-level eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_fashion(model, clip_model_name, device, fashion_data_path,
                 k_values=DEFAULT_K_VALUES, batch_size=256, num_workers=4):
    _, transform = clip.load(clip_model_name, device='cpu')

    results = {}
    for category in CATEGORIES:
        t0 = time.time()
        print(f'  [{category}] starting...')
        results[category] = _eval_category(
            model, category, fashion_data_path, clip_model_name,
            transform, device, k_values, batch_size, num_workers
        )
        elapsed = time.time() - t0
        r = results[category]['recall_atk']
        s = results[category]['stickiness']
        path_tag = 'fast' if results[category]['used_fast_path'] else 'slow'
        print(f'  [{category}] done in {elapsed:.1f}s ({path_tag}) '
              f' R@10={r["@10"]*100:.2f}  R10_ref={s["r10_ref"]*100:.1f}%  '
              f'simΔ={s["sim_delta"]:+.4f}')

    results['averaged'] = _averaged(results)
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',         type=Path, required=True)
    p.add_argument('--device',             type=str,  default='cuda')
    p.add_argument('--fashion_data_path',  type=Path,
                   default=Path('/notebooks/experiment/eval_data/fashion_iq'))
    p.add_argument('--batch_size',         type=int,  default=256)
    p.add_argument('--num_workers',        type=int,  default=4)
    p.add_argument('--save',               action='store_true', default=True)
    return p.parse_args()


def main():
    args = parse_args()

    print('=' * 70)
    print('Fashion-IQ evaluation (v2 — cached fast path + stickiness)')
    print('=' * 70)
    print(f'  Checkpoint: {args.checkpoint}')

    model, clip_model_name = load_model(args.checkpoint, args.device)
    print(f'  Model loaded. logit_scale.exp() = {model.clip.logit_scale.exp().item():.2f}')
    print(f'  Mapper type   : {type(model.mapper).__name__}')
    print()

    results = eval_fashion(
        model              = model,
        clip_model_name    = clip_model_name,
        device             = args.device,
        fashion_data_path  = args.fashion_data_path,
        batch_size         = args.batch_size,
        num_workers        = args.num_workers,
    )

    print()
    print('=' * 70)
    print('Summary')
    print('=' * 70)
    for cat in CATEGORIES + ['averaged']:
        r = results[cat]['recall_atk']
        s = results[cat]['stickiness']
        line = (f'  {cat:<10s}  R@10={r["@10"]*100:5.2f}  R@50={r["@50"]*100:5.2f}  '
                f'R10_ref={s["r10_ref"]*100:5.1f}%  '
                f'med_rank_ref={s["median_rank_ref"]:>5.1f}  '
                f'simΔ={s["sim_delta"]:+.4f}')
        print(line)
    print('=' * 70)

    if args.save:
        out_path = args.checkpoint.parent / 'fashion_eval_v2.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nSaved results to {out_path}')


if __name__ == '__main__':
    main()
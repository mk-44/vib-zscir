"""
CIRCO evaluation for VIB / BaseMLP checkpoints.

For each query (220 val):
  1. Encode (ref_img, relative_caption) → query feat p
  2. Score against all 123k gallery embeddings (cached, frozen CLIP)
  3. Mask ref image from candidates
  4. Top-50 → predictions list (img_ids as ints)
  5. Build predictions_dict for CIRCO's official compute_metrics()

Also computes stickiness diagnostics (parallel to Fashion-IQ analysis):
  - R10_ref / med_ref_rank / sim_ref / mean_rank_ref  (reference stickiness
                                                       — what fraction of queries
                                                       rank the reference image
                                                       in top-10 if NOT excluded?)
  - sim_gt / mean_rank_gt / best_gt_rank              (target/GT signals)
  - sim_Δ = sim_ref - mean_sim_gt                    (stickiness signal)

Usage (single checkpoint):
    cd /notebooks/experiment/codes
    PYTHONPATH=/notebooks/experiment/codes python evaluation/eval_circo.py \
        --checkpoint /path/to/checkpoint.pt

Outputs:
    - Prints metrics to stdout
    - Saves <checkpoint_dir>/circo_eval.json
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import clip
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from config.config import load_from_yaml
from model.model import Pic2Word


# ---------------------------------------------------------------------------
# Load CIRCO's official files by path (avoid name collision with project's dataset/ package)
# ---------------------------------------------------------------------------
def _load_module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_circo_official(circo_data_path):
    """
    Load CIRCODataset and compute_metrics from the cloned CIRCO repo.

    Subtle: CIRCO's evaluation.py contains `from dataset import CIRCODataset`,
    which resolves to OUR project's dataset/ package and fails. We pre-inject
    the CIRCO dataset module into sys.modules under the name 'dataset' so
    that import line resolves correctly. We restore the original sys.modules
    state afterward to avoid breaking anything else in the process.
    """
    repo_src = Path(circo_data_path) / 'circo_repo' / 'src'

    # 1. Load CIRCO's dataset.py under a unique private name (always works)
    ds_module = _load_module_from_file('_circo_dataset_internal', repo_src / 'dataset.py')

    # 2. Temporarily make it available as 'dataset' in sys.modules so that
    #    evaluation.py's `from dataset import CIRCODataset` resolves to it.
    saved_dataset = sys.modules.get('dataset')
    sys.modules['dataset'] = ds_module
    try:
        eval_module = _load_module_from_file('_circo_evaluation_internal',
                                             repo_src / 'evaluation.py')
    finally:
        # Restore whatever was there before (your project's dataset/ package)
        if saved_dataset is not None:
            sys.modules['dataset'] = saved_dataset
        else:
            del sys.modules['dataset']

    return ds_module.CIRCODataset, eval_module.compute_metrics

# ---------------------------------------------------------------------------
# Model loading (mirrors evaluation/analyze.py logic)
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: Path, device: str):
    ckpt     = torch.load(checkpoint_path, map_location='cpu')
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
# CIRCO eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_circo(model, clip_model_name: str, device: str,
               circo_data_path: Path, gallery_cache_path: Path,
               ranks=(5, 10, 25, 50)):
    # 1. Load gallery cache
    print(f'Loading gallery cache from {gallery_cache_path}...')
    cache = torch.load(gallery_cache_path, map_location='cpu')
    gallery_embs   = cache['embeddings'].to(device)              # (N, 768) normalized fp32
    img_ids        = cache['img_ids']                            # list[str], len N
    img_id_to_idx  = cache['img_id_to_idx']                      # dict[str, int]
    assert cache['clip_model'] == clip_model_name, f'Cache is for {cache["clip_model"]}, model uses {clip_model_name}'
    print(f'  Gallery: {gallery_embs.shape[0]:,} images, {gallery_embs.shape[1]}-dim')

    # 2. Load CIRCO dataset (val)
    CIRCODataset, compute_metrics = _load_circo_official(circo_data_path)
    _, preprocess = clip.load(clip_model_name, device='cpu')
    ds = CIRCODataset(
        data_path=str(circo_data_path),
        split='val',
        mode='relative',
        preprocess=preprocess,
    )
    print(f'  Val queries: {len(ds)}')

    # 3. Per-query inference
    predictions_dict = {}
    diagnostics      = []   # for stickiness analysis

    t0 = time.time()
    for q_idx in range(len(ds)):
        ann = ds.annotations[q_idx]
        query_id     = ann['id']
        ref_img_id   = ann['reference_img_id']
        target_id    = ann['target_img_id']
        gt_img_ids   = ann['gt_img_ids']
        caption      = ann['relative_caption']

        # Load + preprocess ref image (RGB conversion important — same fix as gallery)
        ref_path = ds.img_paths[ds.img_ids_indexes_map[str(ref_img_id)]]
        ref_img  = preprocess(PIL.Image.open(ref_path).convert('RGB')).unsqueeze(0).to(device)
        txt_tok  = clip.tokenize([caption], truncate=True).to(device)

        # Query feature
        p = model.encode_query(ref_img, txt_tok)                 # (1, 768) normalized

        # Score against full gallery
        scores = (p @ gallery_embs.T).squeeze(0)                 # (N,)

        # ----- DIAGNOSTICS (computed BEFORE masking ref) -----
        ref_idx = img_id_to_idx[str(ref_img_id)]
        sim_ref = scores[ref_idx].item()
        # Rank of ref image (1-indexed) when ref is NOT excluded
        rank_ref = (scores > scores[ref_idx]).sum().item() + 1

        # GT scores (multi-GT)
        gt_indices  = [img_id_to_idx[str(g)] for g in gt_img_ids if str(g) in img_id_to_idx]
        sim_gt      = scores[gt_indices].cpu().tolist() if gt_indices else []
        # Best (lowest = closest to top) GT rank, when ref NOT excluded
        gt_ranks    = [(scores > scores[gi]).sum().item() + 1 for gi in gt_indices]
        best_gt_rank = min(gt_ranks) if gt_ranks else None

        # ----- METRIC COMPUTATION (with ref excluded, per official spec) -----
        scores_masked = scores.clone()
        scores_masked[ref_idx] = float('-inf')
        topk_idx = scores_masked.topk(50).indices.cpu().tolist()
        top50_img_ids = [int(img_ids[i]) for i in topk_idx]
        # Sanity: must be unique and not include ref
        assert len(set(top50_img_ids)) == 50, f'Query {query_id}: duplicate predictions'
        assert ref_img_id not in top50_img_ids, f'Query {query_id}: ref in predictions'

        # P(gt): full-gallery, reference-masked softmax mass on target(s)
        _tau = model.clip.logit_scale.exp().item()
        _probs = torch.softmax(_tau * scores_masked, dim=0)
        _ti = img_id_to_idx.get(str(target_id))
        p_target = _probs[_ti].item() if _ti is not None else 0.0
        p_gtset  = _probs[gt_indices].sum().item() if gt_indices else 0.0

        predictions_dict[str(query_id)] = top50_img_ids

        # Save diagnostics
        diagnostics.append({
            'query_id'    : query_id,
            'ref_img_id'  : ref_img_id,
            'target_img_id': target_id,
            'sim_ref'     : sim_ref,
            'rank_ref'    : rank_ref,
            'sim_gt_mean' : float(np.mean(sim_gt)) if sim_gt else None,
            'best_gt_rank': best_gt_rank,
            'gt_ranks'    : gt_ranks,
            'p_target'    : p_target,
            'p_gtset'     : p_gtset,
        })

        if (q_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate    = (q_idx + 1) / elapsed
            print(f'  [{q_idx+1:3d}/{len(ds)}] {rate:.1f} q/s')

    elapsed = time.time() - t0
    print(f'\nQueries done in {elapsed:.1f}s ({len(ds)/elapsed:.1f} q/s)')

    # 4. Run official mAP@K + R@K computation
    print('\nComputing mAP@K + Recall@K via official CIRCO evaluation...')
    map_atk, recall_atk, semantic_map_at10 = compute_metrics(
        Path(circo_data_path), predictions_dict, list(ranks)
    )

    # 5. Aggregate stickiness signals
    sim_refs       = [d['sim_ref']      for d in diagnostics]
    sim_gt_means   = [d['sim_gt_mean']  for d in diagnostics if d['sim_gt_mean'] is not None]
    rank_refs      = [d['rank_ref']     for d in diagnostics]
    best_gt_ranks  = [d['best_gt_rank'] for d in diagnostics if d['best_gt_rank'] is not None]
    r10_ref        = sum(1 for r in rank_refs if r <= 10) / len(rank_refs)
    p_targets      = [d['p_target'] for d in diagnostics]
    p_gtsets       = [d['p_gtset']  for d in diagnostics]

    stickiness = {
        'sim_ref_mean'       : float(np.mean(sim_refs)),
        'sim_gt_mean'        : float(np.mean(sim_gt_means)),
        'sim_delta'          : float(np.mean(sim_refs) - np.mean(sim_gt_means)),
        'r10_ref'            : float(r10_ref),
        'mean_rank_ref'      : float(np.mean(rank_refs)),
        'median_rank_ref'    : int(np.median(rank_refs)),
        'best_gt_rank_mean'  : float(np.mean(best_gt_ranks)),
        'best_gt_rank_median': int(np.median(best_gt_ranks)),
        'p_target_mean'      : float(np.mean(p_targets)),
        'p_gtset_mean'       : float(np.mean(p_gtsets)),
    }

    return {
        'map_atk'           : {f'@{k}': v for k, v in map_atk.items()},
        'recall_atk'        : {f'@{k}': v for k, v in recall_atk.items()},
        'semantic_map_at10' : semantic_map_at10,
        'stickiness'        : stickiness,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',         type=Path, required=True)
    p.add_argument('--device',             type=str,  default='cuda')
    p.add_argument('--circo_data_path',    type=Path,
                   default=Path('/notebooks/experiment/eval_data/circo'))
    p.add_argument('--gallery_cache_path', type=Path, default=None,
                   help='Default: <circo_data_path>/gallery_cache_<clip_tag>.pt')
    p.add_argument('--save',               action='store_true', default=True,
                   help='Save circo_eval.json next to the checkpoint')
    return p.parse_args()


def main():
    args = parse_args()

    print('=' * 70)
    print(f'CIRCO evaluation')
    print('=' * 70)
    print(f'  Checkpoint: {args.checkpoint}')

    # Load model
    model, clip_model_name = load_model(args.checkpoint, args.device)
    print(f'  Model loaded. logit_scale.exp() = {model.clip.logit_scale.exp().item():.2f}')

    # Default gallery cache
    if args.gallery_cache_path is None:
        clip_tag = clip_model_name.replace('/', '-')
        args.gallery_cache_path = args.circo_data_path / f'gallery_cache_{clip_tag}.pt'

    # Eval
    results = eval_circo(
        model               = model,
        clip_model_name     = clip_model_name,
        device              = args.device,
        circo_data_path     = args.circo_data_path,
        gallery_cache_path  = args.gallery_cache_path,
    )

    # Print summary
    print()
    print('=' * 70)
    print('Results')
    print('=' * 70)
    print('mAP@K:')
    for k, v in results['map_atk'].items():
        print(f'  {k}: {v*100:.2f}')
    print('Recall@K:')
    for k, v in results['recall_atk'].items():
        print(f'  {k}: {v*100:.2f}')
    print('\nStickiness diagnostics:')
    s = results['stickiness']
    print(f'  R10_ref           : {s["r10_ref"]*100:.2f}%   (lower = less sticky)')
    print(f'  median_rank_ref   : {s["median_rank_ref"]}')
    print(f'  mean_rank_ref     : {s["mean_rank_ref"]:.1f}')
    print(f'  sim_ref           : {s["sim_ref_mean"]:.4f}')
    print(f'  sim_gt            : {s["sim_gt_mean"]:.4f}')
    print(f'  sim_delta         : {s["sim_delta"]:+.4f}   (lower = less sticky)')
    print(f'  best_gt_rank med  : {s["best_gt_rank_median"]}')
    print(f'  best_gt_rank mean : {s["best_gt_rank_mean"]:.1f}')
    print(f'  P(gt) target      : {s["p_target_mean"]:.4f}   (mass on single target)')
    print(f'  P(gt) gt-set       : {s["p_gtset_mean"]:.4f}   (mass over all GTs)')
    print('\nSemantic mAP@10 by aspect:')
    for aspect, v in results['semantic_map_at10'].items():
        print(f'  {aspect:35s}: {v*100:.2f}')

    # Save
    if args.save:
        out_path = args.checkpoint.parent / 'circo_eval.json'
        # If checkpoint is .../epoch_NN/checkpoint.pt, save next to it.
        # If checkpoint is .../epoch_NN_step_SSSSS.pt, save next to it.
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nSaved results to {out_path}')


if __name__ == '__main__':
    main()
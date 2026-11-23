"""
CIRCO evaluation v2 — same evaluation as eval_circo.py but also saves per-query data needed for downstream bootstrap confidence intervals.

WHAT'S NEW vs eval_circo.py
---------------------------
1. We compute AP@K and Recall@K PER QUERY (not the aggregate mAP/recall).
   The aggregate mAP@K is just the mean of per-query AP@K values, 
   so storing per-query values lets us later resample queries and recompute means 
   (= bootstrap), giving us confidence intervals for free.

2. We store a 'per_query' block in the output JSON containing:
       rank_ref, best_gt_rank, sim_ref, sim_gt_mean    (stickiness signals)
       ap_at_k[@5,@10,@25,@50]                          (per-query AP for mAP bootstrap)
       recall_at_k[@5,@10,@25,@50]                      (per-query 0/1 for recall bootstrap)
   The old eval_circo.py only saved aggregates — bootstrap was impossiblemfrom those alone.

3. Cross-check: we run CIRCO's official compute_metrics() and verify our
   own per-query averages match. If they don't, we print a WARN — protects
   us from silent computation bugs in our per-query AP formula.

Usage (single checkpoint):
    cd /notebooks/experiment/codes
    PYTHONPATH=/notebooks/experiment/codes python evaluation/eval_circo_v2.py \
        --checkpoint /path/to/checkpoint.pt
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

from config.config import load_from_yaml
from model.model import Pic2Word
from metrics.paths import metric_file_path, ensure_metrics_dirs



# ---------------------------------------------------------------------------
# Load CIRCO official files by path (avoid name collision with project's
# 'dataset/' package — same workaround as eval_circo.py)
#
# CIRCO's evaluation.py does `from dataset import CIRCODataset` at module
# level. Without intervention, Python resolves 'dataset' to our project's
# dataset/ package and the import fails. We temporarily inject CIRCO's
# dataset module into sys.modules under the name 'dataset' just long
# enough to import their evaluation.py, then restore the original state.
# ---------------------------------------------------------------------------

def _load_module_from_file(name, path):
    """Load a Python file as a module via importlib (sidesteps sys.path)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _load_circo_official(circo_data_path):
    """Returns (CIRCODataset class, compute_metrics function) from CIRCO repo."""
    repo_src = Path(circo_data_path) / 'circo_repo' / 'src'

    # Step 1: load CIRCO's dataset.py under a unique private name (always works)
    ds_module = _load_module_from_file('_circo_dataset_internal', repo_src / 'dataset.py')

    # Step 2: temporarily make it available as 'dataset' so evaluation.py imports it
    saved_dataset = sys.modules.get('dataset')
    sys.modules['dataset'] = ds_module
    try:
        eval_module = _load_module_from_file('_circo_evaluation_internal', repo_src / 'evaluation.py')
    finally:
        # Always restore — even if loading evaluation.py raised
        if saved_dataset is not None:
            sys.modules['dataset'] = saved_dataset
        else:
            del sys.modules['dataset']

    return ds_module.CIRCODataset, eval_module.compute_metrics


# ---------------------------------------------------------------------------
# Model loading
# Identical to eval_circo.py. Reads mapper_type + config from checkpoint,
# constructs matching Pic2Word, loads weights, sets logit_scale, returns
# model + the CLIP backbone name (needed downstream for matching gallery
# cache).
# ---------------------------------------------------------------------------


def load_model(checkpoint_path : Path, device : str):
    ckpt = torch.load(checkpoint_path, map_location = 'cpu')
    ckpt_cfg = ckpt.get('config', {})

    # Build a config object using checkpoint values where present, defaults otherwise.
    # This makes us robust to old checkpoints that don't have every key in 'config'.
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
        device              = device,
    )

    model = Pic2Word(cfg).to(device)
    model.mapper.load_state_dict(ckpt['mapper_state'])
    # Some checkpoints have a token merger (multi-token configs)
    if ckpt.get('merger_state') is not None and model.merger is not None:
        model.merger.load_state_dict(ckpt['merger_state'])
    # logit_scale is the only CLIP-side parameter we restore (it's trainable)
    if 'logit_scale' in ckpt:
        with torch.no_grad():
            model.clip.logit_scale.fill_(ckpt['logit_scale'])
    model.eval()
    return model, cfg.clip_model


# ---------------------------------------------------------------------------
# Per-query Average Precision at K  (the math)
#
# Plain explanation:
#   For one query, you ranked 50 candidates. Some of them are "correct"
#   (in the GT set), most aren't. AP@K measures how high up the correct
#   answers ranked, with strong reward for putting correct ones near the top.
#
# Math:
#   AP@K = (1 / min(|GT|, K)) * sum_{i=1..K} precision_at_i * relevance_i
#
#   where:
#     relevance_i  = 1 if position i is a correct answer, else 0
#     precision_at_i = (number of correct answers seen in top-i) / i
#     |GT|         = total number of correct answers for this query
#
#   Concretely: if GT = {A, B, C}, K = 10, predictions = [X, A, Y, B, Z, ...]
#       pos 1: X wrong, no contribution
#       pos 2: A correct, precision = 1/2 = 0.5,  contrib = 0.5
#       pos 3: Y wrong, no contribution
#       pos 4: B correct, precision = 2/4 = 0.5,  contrib = 0.5
#       ... rest don't include C, so:
#       AP@10 = (0.5 + 0.5) / min(3, 10) = 1.0 / 3 = 0.333
#
# mAP@K is just mean(AP@K) over all queries. So storing per-query AP@K
# lets us bootstrap mAP@K trivially.
# ---------------------------------------------------------------------------

def _compute_query_ap_at_k(predicted_ids, gt_img_ids, k):
    """AP@K for a single query (mirrors CIRCO's official metric)."""
    gt_set = set(gt_img_ids)
    if not gt_set:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, pid in enumerate(predicted_ids[:k]):
        if pid in gt_set:
            hits += 1
            sum_prec += hits / (i + 1)   # precision at this position
    # Normalize by min(|GT|, K), matching CIRCO official spec
    return sum_prec / min(len(gt_set), k)

def _compute_query_recall_at_k(predicted_ids, gt_img_ids, k):
    """1 if any GT in top-K, else 0. R@K = mean of these over queries."""
    gt_set = set(gt_img_ids)
    if not gt_set:
        return 0.0
    return float(any(pid in gt_set for pid in predicted_ids[: k]))


# ---------------------------------------------------------------------------
# CIRCO eval with per-query saving
#
# For each of the 220 val queries:
#   1. Encode (ref_img, modifier_caption) through the model -> query feature p
#   2. Score p against all 123k gallery embeddings (cosine sim via dot product since both are normalized)
#   3. Compute "before-masking" diagnostics:
#         sim_ref = sim(p, ref_image_embed)
#         rank_ref = position of ref image in the 123k ranking
#         These tell us how "sticky" the model is to the reference.
#   4. Mask the reference (set its score to -infinity) per CIRCO eval spec
#   5. Take top-50 predictions, compute per-query AP@K and recall@K
#   6. Store everything per-query for later bootstrap analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_circo(model, clip_model_name, device, circo_data_path, gallery_cache_path, ranks=(5, 10, 25, 50)):
    # --- Load cached gallery embeddings ---
    # The gallery is the same ~123k COCO unlabeled images for every checkpoint,
    # so we encode it once (build_circo_gallery.py) and reuse forever.
    print(f'Loading gallery cache from {gallery_cache_path}...')
    cache = torch.load(gallery_cache_path, map_location = 'cpu')
    gallery_embs   = cache['embeddings'].to(device)   # (N, 768) normalized fp32
    img_ids        = cache['img_ids']                  # list[str], len N, parallel to embeddings
    img_id_to_idx  = cache['img_id_to_idx']            # dict[str, int] for O(1) lookup
    assert cache['clip_model'] == clip_model_name, f'Cache is for {cache["clip_model"]}, model uses {clip_model_name}'
    print(f'  Gallery: {gallery_embs.shape[0]:,} images')
    
    # --- Load CIRCO val dataset (220 queries with multi-GT annotations) ---
    CIRCODataset, compute_metrics = _load_circo_official(circo_data_path)
    _, preprocess = clip.load(clip_model_name, device='cpu')
    ds = CIRCODataset(
        data_path=str(circo_data_path),
        split='val',
        mode='relative',     # 'relative' mode = query-side iteration
        preprocess=preprocess,
    )
    print(f'  Val queries: {len(ds)}')

    # --- Containers for per-query results ---
    predictions_dict = {}                              # query_id → list of top-50 img_ids
    diagnostics      = []                              # one dict per query (stickiness etc.)
    per_query_ap     = {f'@{k}': [] for k in ranks}    # for bootstrap of mAP@K
    per_query_recall = {f'@{k}': [] for k in ranks}    # for bootstrap of recall@K

    t0 = time.time()
    for q_idx in range(len(ds)):
        ann          = ds.annotations[q_idx]
        query_id     = ann['id']
        ref_img_id   = ann['reference_img_id']
        target_id    = ann['target_img_id']
        gt_img_ids   = ann['gt_img_ids']               # multi-GT list (3-5 imgs typical)
        caption      = ann['relative_caption']

        # Encode ref image (preprocess + CLIP visual + map to pseudo-token + text-transform)
        ref_path = ds.img_paths[ds.img_ids_indexes_map[str(ref_img_id)]]
        ref_img  = preprocess(PIL.Image.open(ref_path).convert('RGB')).unsqueeze(0).to(device)
        txt_tok  = clip.tokenize([caption], truncate=True).to(device)
        p        = model.encode_query(ref_img, txt_tok)               # (1, 768) normalized

        # Cosine similarity against entire gallery (cheap matmul on cached features)
        scores = (p @ gallery_embs.T).squeeze(0)                      # (N,)
        
        # --- Stickiness diagnostics (BEFORE masking ref from candidates) ---
        # We want to know if the model would have ranked the ref image high
        # if it were a valid candidate. High rank = sticky.
        ref_idx = img_id_to_idx[str(ref_img_id)]
        sim_ref = scores[ref_idx].item()
        # Rank is 1-indexed: count items strictly greater than ref's score, +1
        rank_ref = (scores > scores[ref_idx]).sum().item() + 1
    
        # Multi-GT similarity stats (mean over the 3-5 GT images)
        gt_indices = [img_id_to_idx[str(g)] for g in gt_img_ids if str(g) in img_id_to_idx]
        sim_gt_list = scores[gt_indices].cpu().tolist() if gt_indices else []
        # Best (= lowest) rank among the multi-GTs
        gt_ranks = [(scores > scores[gi]).sum().item() + 1 for gi in gt_indices]
        best_gt_rank = min(gt_ranks) if gt_ranks else None
        
        # --- Generate predictions (WITH ref masked out per official spec) ---
        scores_masked = scores.clone()
        scores_masked[ref_idx] = float('-inf')
        topk_idx = scores_masked.topk(50).indices.cpu().tolist()
        top50_img_ids = [int(img_ids[i]) for i in topk_idx]
        predictions_dict[str(query_id)] = top50_img_ids
        
        for k in ranks:
            per_query_ap[f'@{k}'].append(_compute_query_ap_at_k(top50_img_ids, gt_img_ids, k))
            per_query_recall[f'@{k}'].append(_compute_query_recall_at_k(top50_img_ids, gt_img_ids, k))

        diagnostics.append({
            'query_id'     : query_id,
            'ref_img_id'   : ref_img_id,
            'target_img_id': target_id,
            'sim_ref'      : sim_ref,
            'rank_ref'     : rank_ref,
            'sim_gt_mean'  : float(np.mean(sim_gt_list)) if sim_gt_list else None,
            'best_gt_rank' : best_gt_rank,
            'gt_ranks'     : gt_ranks,
        })

        if (q_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f'  [{q_idx+1:3d}/{len(ds)}] {(q_idx+1)/elapsed:.1f} q/s')
    
    elapsed = time.time() - t0
    print(f'\nQueries done in {elapsed:.1f}s ({len(ds)/elapsed:.1f} q/s)')

    # --- Run official CIRCO metrics for sanity check ---
    # Our per-query AP averages should match official mAP exactly.
    # If they don't match, our per-query AP formula has a bug.
    print('\nComputing official metrics for cross-check...')
    map_atk_official, recall_atk_official, semantic_map_at10 = compute_metrics(
        Path(circo_data_path), predictions_dict, list(ranks)
    )
    
    for k in ranks:
        our_map = float(np.mean(per_query_ap[f'@{k}']))
        their_map = map_atk_official[k]
        # Tolerance 1e-4: allows tiny float diffs but flags real bugs
        if abs(our_map - their_map) > 1e-4:
            print(f'  WARN: mAP@{k} mismatch: ours={our_map:.4f} theirs={their_map:.4f}')
        else:
            print(f'  mAP@{k} match: {our_map:.4f} (per-query average == official)')

    # --- Aggregate stickiness signals ---
    sim_refs       = [d['sim_ref']      for d in diagnostics]
    sim_gt_means   = [d['sim_gt_mean']  for d in diagnostics if d['sim_gt_mean'] is not None]
    rank_refs      = [d['rank_ref']     for d in diagnostics]
    best_gt_ranks  = [d['best_gt_rank'] for d in diagnostics if d['best_gt_rank'] is not None]
    r10_ref        = sum(1 for r in rank_refs if r <= 10) / len(rank_refs)

    stickiness = {
        'sim_ref_mean'       : float(np.mean(sim_refs)),
        'sim_gt_mean'        : float(np.mean(sim_gt_means)),
        'sim_delta'          : float(np.mean(sim_refs) - np.mean(sim_gt_means)),  # ← positive = sticky
        'r10_ref'            : float(r10_ref),
        'mean_rank_ref'      : float(np.mean(rank_refs)),
        'median_rank_ref'    : int(np.median(rank_refs)),
        'best_gt_rank_mean'  : float(np.mean(best_gt_ranks)),
        'best_gt_rank_median': int(np.median(best_gt_ranks)),
    }

    # --- Per-query block (THE KEY ADDITION vs eval_circo.py) ---
    # Everything we need for downstream bootstrap is here.
    # Order is parallel: per_query[i] corresponds to query_id[i].
    per_query = {
        'query_id'    : [d['query_id']     for d in diagnostics],
        'rank_ref'    : [d['rank_ref']     for d in diagnostics],
        'best_gt_rank': [d['best_gt_rank'] for d in diagnostics],
        'sim_ref'     : [d['sim_ref']      for d in diagnostics],
        'sim_gt_mean' : [d['sim_gt_mean']  for d in diagnostics],
        'ap_at_k'     : per_query_ap,         # dict of '@K' -> list of 220 floats
        'recall_at_k' : per_query_recall,     # dict of '@K' -> list of 220 0/1s
    }

    return {
        'map_atk'           : {f'@{k}': v for k, v in map_atk_official.items()},
        'recall_atk'        : {f'@{k}': v for k, v in recall_atk_official.items()},
        'semantic_map_at10' : semantic_map_at10,
        'stickiness'        : stickiness,
        'per_query'         : per_query,
        'n_queries'         : len(ds),
        'n_gallery'         : gallery_embs.shape[0],
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',         type=Path, required=True)
    p.add_argument('--device',             type=str,  default='cuda')
    p.add_argument('--circo_data_path',    type=Path,
                   default=Path('/notebooks/experiment/eval_data/circo'))
    p.add_argument('--gallery_cache_path', type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    print('=' * 70)
    print('CIRCO evaluation v2 (with per-query data)')
    print('=' * 70)
    print(f'  Checkpoint: {args.checkpoint}')

    model, clip_model_name = load_model(args.checkpoint, args.device)
    print(f'  Model loaded. logit_scale.exp() = {model.clip.logit_scale.exp().item():.2f}')

    # Default gallery cache path: <circo_data>/gallery_cache_<clip_tag>.pt
    if args.gallery_cache_path is None:
        clip_tag = clip_model_name.replace('/', '-')
        args.gallery_cache_path = args.circo_data_path / f'gallery_cache_{clip_tag}.pt'

    results = eval_circo(
        model               = model,
        clip_model_name     = clip_model_name,
        device              = args.device,
        circo_data_path     = args.circo_data_path,
        gallery_cache_path  = args.gallery_cache_path,
    )

    # --- Print summary to stdout ---
    print()
    print('=' * 70)
    print('Aggregate metrics:')
    print('=' * 70)
    for k, v in results['map_atk'].items():
        print(f'  mAP{k}: {v*100:.2f}')
    for k, v in results['recall_atk'].items():
        print(f'  R{k}:   {v*100:.2f}')
    s = results['stickiness']
    print(f'  R10_ref: {s["r10_ref"]*100:.1f}%')
    print(f'  simΔ:    {s["sim_delta"]:+.4f}')

    # --- Save next to checkpoint as circo_eval_v2.json ---
    # bootstrap_circo.py reads from this filename, so don't rename.
    ensure_metrics_dirs()
    out_path = metric_file_path(args.checkpoint, benchmark='circo')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved results (with per-query) to {out_path}')


if __name__ == '__main__':
    main()
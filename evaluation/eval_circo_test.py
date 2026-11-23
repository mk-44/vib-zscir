"""
CIRCO TEST split evaluation — generates submission JSON for the official
evaluation server.

WHY THIS EXISTS
---------------
CIRCO test has 800 queries (vs 220 in val). With 3.6× more queries, CIs
become ~1.9× tighter, which is the path to confirming/denying the
"+1 mAP@10 from VIB" claim that came out underpowered on val.

WHAT IT DOES (per checkpoint)
-----------------------------
1. Loads the model checkpoint
2. (Sanity) Prints existing Fashion-IQ val + CIRCO val metrics from our
   stored JSONs so we can confirm we're submitting the right checkpoint
3. Encodes all 800 test queries through the model
4. Scores each query against full 123K gallery
5. Masks the reference image per CIRCO spec
6. Takes top-50 predictions, writes JSON in server submission format:
       { "0": [img_id, img_id, ...], "1": [...], ..., "799": [...] }

SAFETY CHECKS (added per friend's review)
-----------------------------------------
- Per-query: top-50 contains no duplicates, ref image not in top-50
- Submission-level: all 800 query IDs present, no missing, no extras
- Between checkpoints: GPU memory explicitly freed (CLIP is ~1.7 GB)

OUTPUT
------
For each checkpoint, saves submission JSON to:
    metrics/circo_test_submissions/<run_id>__<ckpt_tag>__submission.json

You then manually upload each JSON to https://circo.micc.unifi.it/evaluation
and get results emailed back.

Usage (single checkpoint):
    cd /notebooks/experiment/codes
    PYTHONPATH=/notebooks/experiment/codes python evaluation/eval_circo_test.py \
        --checkpoint /path/to/checkpoint.pt

Usage (multiple checkpoints sequentially):
    PYTHONPATH=/notebooks/experiment/codes python evaluation/eval_circo_test.py \
        --checkpoints ckpt_a.pt ckpt_b.pt ckpt_c.pt
"""
import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import clip
import PIL.Image
import torch

from config.config import load_from_yaml
from model.model import Pic2Word
from metrics.paths import METRICS_ROOT, derive_run_id_and_ckpt_tag, ensure_metrics_dirs


# ---------------------------------------------------------------------------
# Where to save submission JSONs.
# Reusing METRICS_ROOT pattern from our other scripts for consistency.
# ---------------------------------------------------------------------------
SUBMISSION_DIR = METRICS_ROOT / 'circo_test_submissions'
ALL_FASHION_JSON = Path('/notebooks/experiment/codes/all_model_comparison/all_fashion_metrics.json')


# ---------------------------------------------------------------------------
# CIRCO module loader — same name-collision workaround as eval_circo_v2.py
# CIRCO's evaluation.py does `from dataset import CIRCODataset`, which collides
# with our project's dataset/ package. We inject CIRCO's dataset module
# temporarily under the name 'dataset', restore on exit.
# ---------------------------------------------------------------------------
def _load_module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_circo_dataset(circo_data_path):
    """Returns CIRCODataset class from CIRCO repo."""
    repo_src = Path(circo_data_path) / 'circo_repo' / 'src'
    ds_module = _load_module_from_file('_circo_dataset_internal', repo_src / 'dataset.py')
    return ds_module.CIRCODataset


# ---------------------------------------------------------------------------
# Model loading — identical to eval_circo_v2.py
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: Path, device: str):
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
# Print existing val metrics for sanity — confirms we're submitting the
# checkpoint we think we're submitting.
# Reads from:
#   - all_fashion_metrics.json (Fashion-IQ averaged R@10 etc.)
#   - metrics/circo/<run_id>__<ckpt_tag>.json (CIRCO val mAP etc., if exists)
# ---------------------------------------------------------------------------
def _print_existing_val_metrics(checkpoint_path):
    print('=' * 70)
    print('SANITY CHECK — existing val metrics for this checkpoint:')
    print('=' * 70)

    ckpt_str = str(checkpoint_path)

    # Fashion-IQ from bulk JSON
    try:
        with open(ALL_FASHION_JSON) as f:
            fdata = json.load(f)
        if ckpt_str in fdata and fdata[ckpt_str].get('eval_status') == 'success':
            avg = fdata[ckpt_str]['fashion']['averaged']
            r = avg['recall_atk']
            s = avg['stickiness']
            print(f'  Fashion-IQ (val, averaged):')
            print(f'    R@10     = {r["@10"]*100:5.2f}')
            print(f'    R@50     = {r["@50"]*100:5.2f}')
            print(f'    R10_ref  = {s["r10_ref"]*100:5.1f}%')
            print(f'    simΔ     = {s["sim_delta"]:+.4f}')
        else:
            print(f'  Fashion-IQ: not found in {ALL_FASHION_JSON.name} (or not success)')
    except Exception as e:
        print(f'  Fashion-IQ: error reading bulk JSON: {e}')

    # CIRCO val from per-checkpoint JSON
    run_id, ckpt_tag = derive_run_id_and_ckpt_tag(checkpoint_path)
    circo_v2 = METRICS_ROOT / 'circo' / f'{run_id}__{ckpt_tag}.json'
    try:
        if circo_v2.exists():
            with open(circo_v2) as f:
                cdata = json.load(f)
            print(f'  CIRCO (val, 220 queries):')
            for k, v in cdata['map_atk'].items():
                print(f'    mAP{k:<4s} = {v*100:5.2f}')
            s = cdata['stickiness']
            print(f'    R10_ref  = {s["r10_ref"]*100:5.1f}%')
            print(f'    simΔ     = {s["sim_delta"]:+.4f}')
        else:
            print(f'  CIRCO: {circo_v2.name} not found')
            print(f'    (Run eval_circo_v2.py first if you want val sanity check)')
    except Exception as e:
        print(f'  CIRCO: error reading {circo_v2}: {e}')

    print('=' * 70)


# ---------------------------------------------------------------------------
# Main: generate submission JSON for one checkpoint.
#
# Algorithm (mirrors eval_circo_v2.py but on test split, no GT, no eval):
#   For each of 800 test queries:
#     1. Load ref image, encode through model with modifier text
#     2. Score against full gallery (123K images, cached)
#     3. Mask reference (set its score to -inf)
#     4. Take top-50 indices → convert to img_ids
#     5. Per-query safety checks (no dups, ref masked)
#     6. Store under string query_id
#   After loop:
#     - Submission-level validation (800 IDs, no missing/extra)
#   Write JSON in CIRCO server format.
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_test_submission(checkpoint_path, device, circo_data_path,
                              gallery_cache_path):
    print()
    print('=' * 70)
    print(f'CIRCO TEST submission generation')
    print('=' * 70)
    print(f'  Checkpoint: {checkpoint_path}')

    # --- Print existing val metrics ---
    _print_existing_val_metrics(checkpoint_path)

    # --- Load model ---
    model, clip_model_name = load_model(checkpoint_path, device)
    print(f'\n  Model loaded. logit_scale.exp() = {model.clip.logit_scale.exp().item():.2f}')
    print(f'  Mapper type   : {type(model.mapper).__name__}')

    # --- Load gallery cache (same one used for val) ---
    cache = torch.load(gallery_cache_path, map_location='cpu')
    gallery_embs   = cache['embeddings'].to(device)     # (N, 768) normalized fp32
    img_ids        = cache['img_ids']                    # list[str|int]
    img_id_to_idx  = cache['img_id_to_idx']
    assert cache['clip_model'] == clip_model_name, \
        f'Gallery cache CLIP ({cache["clip_model"]}) != model CLIP ({clip_model_name})'
    print(f'  Gallery: {gallery_embs.shape[0]:,} images (cached)')

    # --- Load TEST split dataset ---
    CIRCODataset = _load_circo_dataset(circo_data_path)
    _, preprocess = clip.load(clip_model_name, device='cpu')
    ds = CIRCODataset(
        data_path=str(circo_data_path),
        split='test',                   # ← KEY DIFFERENCE from val
        mode='relative',
        preprocess=preprocess,
    )
    print(f'  Test queries: {len(ds)}')

    # --- Generate predictions ---
    submission = {}                     # {query_id_str: [top50_img_ids_int]}
    t0 = time.time()

    for q_idx in range(len(ds)):
        ann          = ds.annotations[q_idx]
        query_id     = ann['id']
        ref_img_id   = ann['reference_img_id']
        caption      = ann['relative_caption']
        # NOTE: test split has NO target_img_id, NO gt_img_ids — we don't use them.
        # NOTE: We do NOT concatenate shared_concept to caption — standard CIRCO
        #       protocol uses only relative_caption as the modifier text.

        # Encode query
        ref_path = ds.img_paths[ds.img_ids_indexes_map[str(ref_img_id)]]
        ref_img  = preprocess(PIL.Image.open(ref_path).convert('RGB')).unsqueeze(0).to(device)
        txt_tok  = clip.tokenize([caption], truncate=True).to(device)
        p        = model.encode_query(ref_img, txt_tok)

        # Score against gallery
        scores = (p @ gallery_embs.T).squeeze(0)            # (N,)

        # Mask reference per official spec
        ref_idx = img_id_to_idx[str(ref_img_id)]
        scores[ref_idx] = float('-inf')

        # Top-50
        topk_idx = scores.topk(50).indices.cpu().tolist()
        top50_img_ids = [int(img_ids[i]) for i in topk_idx]

        # Per-query safety checks
        # (a) No duplicates in top-50 (defensive: shouldn't happen, but protects against index issues)
        assert len(set(top50_img_ids)) == 50, \
            f'Query {query_id}: duplicate predictions in top-50'
        # (b) Reference image was correctly masked out (defensive: shouldn't be in top-50)
        assert int(ref_img_id) not in top50_img_ids, \
            f'Query {query_id}: reference image (id={ref_img_id}) appears in top-50 — masking failed'

        # Save under string key (server expects string keys)
        submission[str(query_id)] = top50_img_ids

        if (q_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f'  [{q_idx+1:3d}/{len(ds)}] {(q_idx+1)/elapsed:.1f} q/s')

    elapsed = time.time() - t0
    print(f'\n  Test queries done in {elapsed:.1f}s ({len(ds)/elapsed:.1f} q/s)')

    # --- Validate submission format ---
    assert len(submission) == len(ds), \
        f'Submission has {len(submission)} entries, expected {len(ds)}'
    # All keys must be strings
    assert all(isinstance(k, str) for k in submission.keys()), 'Keys must be strings'
    # All values must be lists of 50 ints
    for k, v in submission.items():
        assert isinstance(v, list), f'Query {k}: value is not a list'
        assert len(v) == 50, f'Query {k}: list has {len(v)} entries, expected 50'
        assert all(isinstance(x, int) for x in v), f'Query {k}: non-int img_id'

    # Query-id completeness: every test query_id must appear exactly once.
    # Catches the case where, e.g., a duplicate query_id in annotations
    # would silently overwrite another in the submission dict.
    expected_ids = {str(ds.annotations[i]['id']) for i in range(len(ds))}
    actual_ids = set(submission.keys())
    missing = expected_ids - actual_ids
    extra   = actual_ids - expected_ids
    assert not missing and not extra, \
        f'Query ID mismatch. Missing={sorted(missing)[:5]}... Extra={sorted(extra)[:5]}...'

    print('  Format validation: PASSED')
    print(f'    {len(submission)} queries, all IDs match test split, no duplicates within top-50')

    # --- Save ---
    ensure_metrics_dirs()
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    run_id, ckpt_tag = derive_run_id_and_ckpt_tag(checkpoint_path)
    out_path = SUBMISSION_DIR / f'{run_id}__{ckpt_tag}__submission.json'
    with open(out_path, 'w') as f:
        json.dump(submission, f, separators=(',', ':'))     # compact format, smaller file
    file_size_kb = out_path.stat().st_size / 1024
    print(f'  Submission saved: {out_path}')
    print(f'  File size      : {file_size_kb:.1f} KB (limit usually <5 MB)')
    print('=' * 70)

    return out_path


def parse_args():
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--checkpoint',  type=Path,
                     help='Single checkpoint to process')
    grp.add_argument('--checkpoints', type=Path, nargs='+',
                     help='Multiple checkpoints (sequential)')
    p.add_argument('--device',             type=str,  default='cuda')
    p.add_argument('--circo_data_path',    type=Path,
                   default=Path('/notebooks/experiment/eval_data/circo'))
    p.add_argument('--gallery_cache_path', type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    ckpts = [args.checkpoint] if args.checkpoint else args.checkpoints

    # Default gallery cache path
    if args.gallery_cache_path is None:
        # Probe one checkpoint for its clip_model to build cache path
        sample = torch.load(ckpts[0], map_location='cpu')
        clip_model_name = sample.get('config', {}).get('clip_model', 'ViT-L/14')
        clip_tag = clip_model_name.replace('/', '-')
        args.gallery_cache_path = args.circo_data_path / f'gallery_cache_{clip_tag}.pt'
        del sample
    assert args.gallery_cache_path.exists(), f'Gallery cache not found: {args.gallery_cache_path}'

    submissions_made = []
    for i, ckpt in enumerate(ckpts):
        print(f'\n\n[{i+1}/{len(ckpts)}] Processing: {ckpt}')
        try:
            out_path = generate_test_submission(
                ckpt, args.device, args.circo_data_path, args.gallery_cache_path
            )
            submissions_made.append(out_path)
        except Exception as e:
            print(f'  FAILED: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            continue
        finally:
            # Free GPU memory between checkpoints — model holds CLIP (~1.7 GB)
            # plus mapper. Without explicit cleanup, sequential checkpoints can
            # accumulate and OOM after a few iterations.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final summary
    print()
    print('=' * 70)
    print('ALL SUBMISSIONS')
    print('=' * 70)
    for p in submissions_made:
        print(f'  {p}')
    print(f'\nTotal: {len(submissions_made)}/{len(ckpts)} successful')
    print()
    print('NEXT STEPS:')
    print('  1. Upload each JSON manually to https://circo.micc.unifi.it/evaluation')
    print('  2. Provide an email address — results will be sent there')
    print('  3. Wait 1-2 days for results')


if __name__ == '__main__':
    main()
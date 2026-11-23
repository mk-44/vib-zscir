"""
all_model_comparison/comparison.py

Run CIRCO eval on every .pt checkpoint found anywhere under
/notebooks/experiment/ (recursive). Saves results to a single JSON,
written incrementally so a crash never loses progress.

Skips test artifacts in /temp_test_outputs/, gallery caches in /eval_data/,
and CIRCO repo files in /circo_repo/.

Handles 3 checkpoint naming conventions:
  - <run>/epoch_NN/checkpoint.pt           (new, nested)
  - <run>/epoch_NN_step_SSSSS.pt           (mid-epoch, flat)
  - <run>/epoch_NN.pt                      (old, flat end-of-epoch)
Plus catch-all for final.pt etc.

Always tries to evaluate. If config extraction fails, stores
"failed to load" in config_summary but continues with eval. Only when
the model itself fails to load do we mark eval_status=failed.

Usage:
    cd /notebooks/experiment/codes
    PYTHONPATH=/notebooks/experiment/codes python all_model_comparison/comparison.py
"""
import contextlib
import gc
import io
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, '/notebooks/experiment/codes')
from evaluation.eval_circo import load_model, eval_circo


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEARCH_ROOT  = Path('/notebooks/experiment/')
CIRCO_DATA   = Path('/notebooks/experiment/eval_data/circo')
GALLERY_PATH = CIRCO_DATA / 'gallery_cache_ViT-L-14.pt'
OUTPUT_JSON  = Path('/notebooks/experiment/codes/all_model_comparison/all_circo_metrics.json')
DEVICE       = 'cuda'

# Skip non-checkpoint .pt files
SKIP_PATH_SUBSTRINGS = [
    '/eval_data/',           # gallery caches
    '/circo_repo/',          # CIRCO repo files
    '/temp_test_outputs/',   # unit-test artifacts
]


# ---------------------------------------------------------------------------
# Discover all .pt files recursively
# ---------------------------------------------------------------------------
def discover_checkpoints(root):
    found = []
    for pt_path in root.rglob('*.pt'):
        path_str = str(pt_path)
        if any(sub in path_str for sub in SKIP_PATH_SUBSTRINGS):
            continue
        found.append(pt_path)
    return sorted(found)


# ---------------------------------------------------------------------------
# Parse run info from path
# ---------------------------------------------------------------------------
def parse_run_info(checkpoint_path):
    parent      = checkpoint_path.parent
    parent_name = parent.name
    stem        = checkpoint_path.stem

    # Pattern 1 (new): <run>/epoch_NN/checkpoint.pt   (end-of-epoch in subfolder)
    if parent_name.startswith('epoch_') and stem == 'checkpoint':
        try:
            epoch = int(parent_name.split('_')[-1])
        except ValueError:
            epoch = None
        return {
            'run_name'    : parent.parent.name,
            'epoch'       : epoch,
            'step'        : None,
            'is_mid_epoch': False,
        }

    # Pattern 2 (new): <run>/epoch_NN_step_SSSSS.pt   (mid-epoch flat)
    if stem.startswith('epoch_') and 'step' in stem:
        parts = stem.split('_')
        try:
            epoch = int(parts[1])
            step  = int(parts[3])
        except (IndexError, ValueError):
            epoch = step = None
        return {
            'run_name'    : parent_name,
            'epoch'       : epoch,
            'step'        : step,
            'is_mid_epoch': True,
        }

    # Pattern 3 (old): <run>/epoch_NN.pt   (end-of-epoch flat)
    if stem.startswith('epoch_') and 'step' not in stem:
        try:
            epoch = int(stem.split('_')[-1])
        except ValueError:
            epoch = None
        return {
            'run_name'    : parent_name,
            'epoch'       : epoch,
            'step'        : None,
            'is_mid_epoch': False,
        }

    # Catch-all (final.pt, latest.pt, anything weird)
    return {
        'run_name'    : parent_name,
        'epoch'       : None,
        'step'        : None,
        'is_mid_epoch': False,
    }


# ---------------------------------------------------------------------------
# Try to extract config summary, never fail
# ---------------------------------------------------------------------------
CONFIG_KEYS_OF_INTEREST = [
    'mapper_type', 'hidden_dim', 'num_tokens', 'num_heads', 'num_layers', 'dropout',
    'lr', 'batch_size',
    'ls_lr_scale', 'ls_max',
    'vib_beta', 'vib_beta_initial', 'vib_beta_warmup',
    'vib_logvar_bias', 'vib_kl_mu_weight', 'vib_kl_var_weight', 'vib_logvar_clip_max',
    'merge_tokens', 'merge_type',
    'repulsion_weight', 'decorr_weight',
    'noise_sigma', 'pseudo_norm', 'margin_weight',
    'init_from',
]

def extract_config_summary(checkpoint_path):
    try:
        ckpt     = torch.load(checkpoint_path, map_location='cpu')
        ckpt_cfg = ckpt.get('config', {})
        summary  = {k: ckpt_cfg.get(k) for k in CONFIG_KEYS_OF_INTEREST if k in ckpt_cfg}
        if 'mapper_type' in ckpt and 'mapper_type' not in summary:
            summary['mapper_type'] = ckpt['mapper_type']
        del ckpt
        return summary
    except Exception:
        return 'failed to load'


# ---------------------------------------------------------------------------
# JSON I/O (atomic via tmp + rename)
# ---------------------------------------------------------------------------
def load_existing_results(path):
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f'WARNING: {path} is corrupted, starting fresh')
    return {}

def save_results(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Eval one checkpoint
# ---------------------------------------------------------------------------
def eval_one(checkpoint_path, results):
    key = str(checkpoint_path)

    # Skip already-successful
    if key in results and results[key].get('eval_status') == 'success':
        return 'already_done'

    info = parse_run_info(checkpoint_path)
    entry = {
        'checkpoint_path': key,
        'run_name'       : info['run_name'],
        'epoch'          : info['epoch'],
        'step'           : info['step'],
        'is_mid_epoch'   : info['is_mid_epoch'],
        'evaluated_at'   : datetime.now().isoformat(timespec='seconds'),
    }
    entry['config_summary'] = extract_config_summary(checkpoint_path)

    t0 = time.time()
    model = None
    try:
        model, clip_model_name = load_model(checkpoint_path, DEVICE)
        entry['logit_scale'] = float(model.clip.logit_scale.exp().item())
        entry['mapper_type'] = type(model.mapper).__name__

        circo_results = eval_circo(
            model              = model,
            clip_model_name    = clip_model_name,
            device             = DEVICE,
            circo_data_path    = CIRCO_DATA,
            gallery_cache_path = GALLERY_PATH,
        )
        entry['circo']       = circo_results
        entry['eval_status'] = 'success'

    except Exception as e:
        entry['eval_status'] = 'failed'
        entry['error_type']  = type(e).__name__
        entry['error_msg']   = str(e)[:500]
        entry['traceback']   = traceback.format_exc()[-1500:]

    finally:
        entry['eval_time_s'] = round(time.time() - t0, 2)
        if model is not None:
            del model
        torch.cuda.empty_cache()
        gc.collect()
        results[key] = entry

    return entry['eval_status']


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f'Recursively searching for .pt files under {SEARCH_ROOT}...')
    checkpoints = discover_checkpoints(SEARCH_ROOT)
    print(f'  Found {len(checkpoints)} candidate .pt files\n')

    results = load_existing_results(OUTPUT_JSON)
    if results:
        n_done   = sum(1 for v in results.values() if v.get('eval_status') == 'success')
        n_failed = sum(1 for v in results.values() if v.get('eval_status') == 'failed')
        print(f'Existing results: {n_done} success, {n_failed} failed\n')

    to_eval = [
        cp for cp in checkpoints
        if str(cp) not in results or results[str(cp)].get('eval_status') != 'success'
    ]
    print(f'Will evaluate: {len(to_eval)} checkpoints '
          f'(skipping {len(checkpoints) - len(to_eval)} already done)\n')
    print('=' * 80)

    n_success = 0
    n_failed  = 0
    t_start   = time.time()

    for i, cp in enumerate(to_eval):
        rel = str(cp).replace('/notebooks/experiment/', '')
        print(f'[{i+1:>3}/{len(to_eval)}] {rel}', flush=True)

        with contextlib.redirect_stdout(io.StringIO()):
            status = eval_one(cp, results)

        entry = results[str(cp)]
        if status == 'success':
            n_success += 1
            r = entry['circo']
            print(f'        OK   mAP@10={r["map_atk"]["@10"]*100:5.2f}  '
                  f'R10_ref={r["stickiness"]["r10_ref"]*100:5.1f}%  '
                  f'({entry["eval_time_s"]:.1f}s)', flush=True)
        elif status == 'failed':
            n_failed += 1
            err_short = entry['error_msg'][:80]
            print(f'        FAIL [{entry["error_type"]}] {err_short}', flush=True)

        save_results(OUTPUT_JSON, results)

        if (i + 1) % 10 == 0:
            elapsed   = time.time() - t_start
            rate      = (i + 1) / elapsed
            remaining = (len(to_eval) - i - 1) / max(rate, 1e-6) / 60
            print(f'        --- {n_success} ok, {n_failed} failed, '
                  f'{rate:.2f} ckpt/sec, ETA {remaining:.1f} min ---', flush=True)

    print('=' * 80)
    print(f'\nFinished.')
    print(f'  Total time : {(time.time()-t_start)/60:.1f} min')
    print(f'  Success    : {n_success}')
    print(f'  Failed     : {n_failed}')
    print(f'  Output JSON: {OUTPUT_JSON}')


if __name__ == '__main__':
    main()
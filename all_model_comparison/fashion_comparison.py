"""
all_model_comparison/fashion_comparison.py

Run Fashion-IQ eval (3 categories + averaged) on every .pt checkpoint
under /notebooks/experiment/ (recursive). Saves results to a single JSON,
written incrementally so a crash never loses progress.

Mirrors comparison.py (CIRCO version) — same scan, same skip filters,
same atomic save, same resumability.

Usage:
    cd /notebooks/experiment/codes
    PYTHONPATH=/notebooks/experiment/codes python all_model_comparison/fashion_comparison.py
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

# Reuse helpers from the CIRCO bulk script
from all_model_comparison.comparison import (
    discover_checkpoints,
    parse_run_info,
    extract_config_summary,
    load_existing_results,
    save_results,
    SKIP_PATH_SUBSTRINGS,
)

# Fashion-IQ evaluator + model loader
from evaluation.eval_fashion_v2 import load_model, eval_fashion


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEARCH_ROOT       = Path('/notebooks/experiment/')
FASHION_DATA      = Path('/notebooks/experiment/eval_data/fashion_iq')
OUTPUT_JSON       = Path('/notebooks/experiment/codes/all_model_comparison/all_fashion_metrics.json')
DEVICE            = 'cuda'


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

        fashion_results = eval_fashion(
            model              = model,
            clip_model_name    = clip_model_name,
            device             = DEVICE,
            fashion_data_path  = FASHION_DATA,
        )
        entry['fashion']     = fashion_results
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

        # Suppress eval_fashion's verbose stdout
        with contextlib.redirect_stdout(io.StringIO()):
            status = eval_one(cp, results)

        entry = results[str(cp)]
        if status == 'success':
            n_success += 1
            avg_r = entry['fashion']['averaged']['recall_atk']
            avg_s = entry['fashion']['averaged']['stickiness']
            print(f'        OK   avgR@10={avg_r["@10"]*100:5.2f}  '
                  f'avgR10_ref={avg_s["r10_ref"]*100:5.1f}%  '
                  f'avgSimΔ={avg_s["sim_delta"]:+.4f}  '
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
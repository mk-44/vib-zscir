"""Re-evaluate the unit-norm run (pseudo_norm = 1.0) with its pseudo-token projection applied.

Background
    The unit-norm run trains the mapper with the fixed-norm projection z <- rho * z / ||z|| applied to
    the pseudo-token inside Pic2Word.forward and Pic2Word.encode_query. Until commit e2b91fa the
    evaluation loaders rebuilt the model config from a fixed list of checkpoint keys that left out
    `pseudo_norm`, so these checkpoints were evaluated without the projection they were trained with
    (paper: Table 1, Table S8, Sec. D.2). This script re-evaluates every checkpoint of the run through
    the fixed loaders, applies the paper's selection rule (argmax CIRCO-val mAP@10, Table S1) and writes
    a CIRCO test-server file for the selected checkpoint.

What it runs (the repo's own evaluation code, unchanged)
    CIRCO val    eval_circo_v2.load_model + eval_circo_v2.eval_circo   official mAP/recall, stickiness,
                                                                     per-query data for bootstrap CIs
                 eval_circo.eval_circo on the same model               adds P(gt); its mAP must equal v2's
    Fashion-IQ   eval_fashion.evaluate                                 the eval_category used in training
    CIRCO test   eval_circo_test.generate_test_submission              selected checkpoint only

Order (steps 1 and 2 stop the run on failure, so no test file comes from an unverified pipeline)
    0 preflight  packages, GPU, data, and every checkpoint: present, byte-identical to the archived
                 copy (sha256), pseudo_norm as expected
    1 control    the bounded baseline (no pseudo_norm) reproduces its stored CIRCO-val mAP@10 and its
                 training-time Fashion-IQ R@1/5/10/50: the evaluation pipeline itself is unchanged
    2 gate       the 9 unit-norm checkpoints with training-time Fashion-IQ evals (computed in-process,
                 projection on) reproduce them: the fixed loaders restore the trained forward pass
    3 sweep      the remaining unit-norm checkpoints (final.pt is byte-identical to
                 epoch_00/checkpoint.pt and reuses its results)
    4 select     argmax CIRCO-val mAP@10 over the run's checkpoints -> selection.json
    5 test file  CIRCO test submission for the selected checkpoint, with a receipt (sha256)

Persistence
    Every evaluation is written to its own JSON the moment it finishes (atomic write) and summary.csv is
    rebuilt after each checkpoint, so a stopped kernel loses at most the evaluation in progress. Rerun
    the same command to resume; finished work is skipped. Results go to --out_dir only: the legacy
    metrics/ files and earlier submissions are never touched. The only other writes are
    model-independent caches (CLIP weights in ~/.cache/clip; a Fashion-IQ gallery cache if missing).

Usage (Paperspace, repo cloned at /notebooks/experiment/codes)
    PYTHONPATH=/notebooks/experiment/codes python -u evaluation/reeval_unit_norm.py --preflight_only
    PYTHONPATH=/notebooks/experiment/codes python -u evaluation/reeval_unit_norm.py
"""

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The checkpoints and caches are this project's own files. PyTorch >= 2.6 defaults torch.load to
# weights_only=True, which can reject plain-Python objects stored in them; keep the loading semantics
# the original evaluations ran with. No effect on older PyTorch.
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')


# ---------------------------------------------------------------------------
# Fixed inputs
# ---------------------------------------------------------------------------
EXPERIMENT  = Path('/notebooks/experiment')
CODES_ROOT  = EXPERIMENT / 'codes'                    # every loader hardcodes this location
CONFIG_YAML = CODES_ROOT / 'config' / 'config.yaml'
RUN_DIR     = EXPERIMENT / '4M_checkpoints' / 'base_hd256_tok1_lslr0.2_lsmax116_pnorm1.0'
BASELINE    = EXPERIMENT / '4M_checkpoints' / 'base_hd256_tok1_lslr0.2_lsmax116' / 'epoch_02_step_03000.pt'
CIRCO_DIR   = EXPERIMENT / 'eval_data' / 'circo'
GALLERY     = CIRCO_DIR / 'gallery_cache_ViT-L-14.pt'
FIQ_DIR     = EXPERIMENT / 'eval_data' / 'fashion_iq'
FIQ_CATS    = ('dress', 'shirt', 'toptee')
KS          = ('1', '5', '10', '50')

# All 19 checkpoint files of the unit-norm run in training order, with the sha256 of the archived
# copy (keep_checkpoints.zip). final.pt is byte-identical to epoch_00/checkpoint.pt: train.py copies
# the best end-of-epoch checkpoint (by training-time Fashion-IQ R@10) to final.pt.
UNIT_NORM = {
    'epoch_00_step_00500.pt': '03525277df710c391887c4df0c2574d43b561d9907ca84559f90b39848e4c3c6',
    'epoch_00_step_01000.pt': '0b0cabb1a568d7f07c7b866ef2b9e334fd405b1f5e7a62af3722924fbc7b4a86',
    'epoch_00_step_01500.pt': 'f7fe1affec3bd6765f31d0dcc62ae95f3821e771aad7cd8d0dd6f1b048ade6ca',
    'epoch_00_step_02000.pt': '46c40708bef2debcb6dd9c5ad8e91c60d417a224e2ce1d74449a733090435671',
    'epoch_00_step_02500.pt': '2eb6691875a4a1343ccb4d65de8a1f90da80b067afba6c25f3825503cf0738e0',
    'epoch_00_step_03000.pt': '8e0589e89161bb3cb0f68174df8bdcc11bb16ea48096ef9412231fa6dc539905',
    'epoch_00_step_03500.pt': '0b610054ff831501e978af5ba6d576df4ed557bff44449c61536e59aaacc681e',
    'epoch_00/checkpoint.pt': '1efb0191dc42851a913d6c463255de8b1921ed93e91027026f4eb29ae1153b04',
    'epoch_01_step_00500.pt': 'f63ebe27eda8c90df6528603e1839f786eaec270eb1f7f6bec63e6f0ccd627de',
    'epoch_01_step_01000.pt': '58573d0b6a108f46c7c9233ff64c0c70c91e41f82c7e2f142a294b9cf4756c39',
    'epoch_01_step_01500.pt': '6964d66e6d8bca3d16537988a4110144c31c2a5c3c3a2317efa165f85c7e5e93',
    'epoch_01_step_02000.pt': '8e86392b0b5ae23fcf7853408f98144e2e35a10778b46b1a59ed88f63e839b3c',
    'epoch_01_step_02500.pt': 'eae47cd1104ad8a44cb4213f88ed6b86e83dbec2be22a0a026278a40deb47e65',
    'epoch_01_step_03000.pt': '5594e99bcc5f97e9983148395c8db2afb522b9bea014ad467665d9c3555cbf38',
    'epoch_01_step_03500.pt': '1ca9c0efeee2f52ed659d5ea936641fe699ea0e8ca58a8523d1b57e134d958d3',
    'epoch_01/checkpoint.pt': '4b32b5c6772c0d39c659c6fbe27e9c3d5c977a59797766805aa9712ba296679b',
    'epoch_02_step_00500.pt': '894dd38def523d2336d4cf99a41ed6a5923a7a6f2544f86951cf89a6e985a301',
    'epoch_02_step_01000.pt': 'b61eab2bece3d7f82f0af9e39448458611600430333d20348a26aeb69cff1933',
    'final.pt':               '1efb0191dc42851a913d6c463255de8b1921ed93e91027026f4eb29ae1153b04',
}
BASELINE_SHA256 = 'b6618d3e0da5f6aecb59e2ea08b272caaed87f2b8829af1de2321c28dcc39d4f'
PREVIOUS_SELECTION = 'epoch_01_step_01500.pt'   # behind the paper's unit-norm rows (no projection)

# Training-time Fashion-IQ evals of the unit-norm run: average over dress/shirt/toptee of R@1, R@5,
# R@10, R@50 (train.py rounds each category to 6 dp, then the average to 6 dp). They were computed
# in-process by eval_category with the projection on, right after the checkpoint was saved.
GATE = {
    'epoch_00_step_01000.pt': ((0.075447, 0.169288, 0.225653, 0.412471), 'step_evals/step_01000_epoch_00.json'),
    'epoch_00_step_02000.pt': ((0.074229, 0.173050, 0.232053, 0.422242), 'step_evals/step_02000_epoch_00.json'),
    'epoch_00_step_03000.pt': ((0.072534, 0.172744, 0.232412, 0.429698), 'step_evals/step_03000_epoch_00.json'),
    'epoch_00/checkpoint.pt': ((0.076210, 0.176018, 0.232744, 0.432546), 'epoch_00/fashion_eval.json'),
    'epoch_01_step_01000.pt': ((0.072693, 0.172903, 0.235748, 0.429921), 'step_evals/step_01000_epoch_01.json'),
    'epoch_01_step_02000.pt': ((0.075354, 0.168270, 0.228612, 0.417628), 'step_evals/step_02000_epoch_01.json'),
    'epoch_01_step_03000.pt': ((0.067713, 0.166399, 0.230100, 0.414903), 'step_evals/step_03000_epoch_01.json'),
    'epoch_01/checkpoint.pt': ((0.070378, 0.168510, 0.228302, 0.419689), 'epoch_01/fashion_eval.json'),
    'epoch_02_step_01000.pt': ((0.069013, 0.168865, 0.228676, 0.419400), 'step_evals/step_01000_epoch_02.json'),
}
# Control: the bounded baseline (no pseudo_norm key). Fashion-IQ from its training-time
# step_evals/step_03000_epoch_02.json; CIRCO-val mAP@10 from the stored eval_circo_v2 result
# metrics/circo/base_hd256_tok1_lslr0.2_lsmax116__epoch_02_step_03000.json.
BASELINE_FASHION     = (0.069583, 0.182814, 0.249871, 0.451093)
BASELINE_CIRCO_MAP10 = 0.12124054533429532

FASHION_TOL = 1e-3   # avg recall (fraction); one query flip moves a 3-category average by ~1.7e-4
CIRCO_TOL   = 2e-3   # baseline CIRCO-val mAP@10 (fraction); a broken pipeline moves it by >1e-2


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')


def log(msg=''):
    print(msg, flush=True)


def banner(title):
    log()
    log('=' * 78)
    log(title)
    log('=' * 78)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _json_default(o):
    if hasattr(o, 'item'):            # numpy / torch scalars
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f'not JSON serializable: {type(o).__name__}')


def write_json(path, obj):
    """Atomic write: a crash mid-write never leaves a truncated file that a resume would trust."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def key_for(path):
    """'<run_id>__<ckpt_tag>', the naming of metrics/circo/ (epoch_NN/checkpoint.pt -> epoch_NN)."""
    p = Path(path)
    if p.stem == 'checkpoint':
        return f'{p.parent.parent.name}__{p.parent.name}'
    return f'{p.parent.name}__{p.stem}'


def git_commit():
    """Commit of this clone, read from .git without running git."""
    git = REPO_ROOT / '.git'
    try:
        head = (git / 'HEAD').read_text().strip()
        if not head.startswith('ref:'):
            return head
        ref = head.split(' ', 1)[1]
        if (git / ref).exists():
            return (git / ref).read_text().strip()
        for line in (git / 'packed-refs').read_text().splitlines():
            if line.endswith(' ' + ref):
                return line.split(' ', 1)[0]
    except OSError:
        pass
    return None


@contextlib.contextmanager
def cudnn_settings(deterministic):
    """Each benchmark runs under the cuDNN settings that produced its reference numbers:
    Fashion-IQ references come from the in-training eval (train.py sets deterministic=True,
    benchmark=False); CIRCO references come from standalone eval_circo_v2.py (library defaults)."""
    import torch
    old = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = deterministic, False
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = old


def free_gpu():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def first_with_same_file(rel):
    """Earlier checkpoint of the run with identical bytes (final.pt -> epoch_00/checkpoint.pt)."""
    for other, sha in UNIT_NORM.items():
        if other == rel:
            return None
        if sha == UNIT_NORM[rel]:
            return other
    return None


# ---------------------------------------------------------------------------
# Step 0: preflight
# ---------------------------------------------------------------------------
def preflight():
    problems, warnings, ckpts = [], [], {}

    try:
        import torch
        if not torch.cuda.is_available():
            problems.append('No CUDA GPU is visible to PyTorch.')
    except Exception as e:                                   # noqa: BLE001
        torch = None
        problems.append(f'PyTorch is not importable: {e}')
    try:
        import clip
        if 'ViT-L/14' not in clip.available_models():
            problems.append('The installed clip package does not offer ViT-L/14.')
    except Exception as e:                                   # noqa: BLE001
        problems.append(f'OpenAI CLIP is not importable ({e}). Install it with: '
                        'pip install ftfy regex tqdm git+https://github.com/openai/CLIP.git')
    for mod in ('numpy', 'PIL', 'yaml'):
        try:
            __import__(mod)
        except Exception as e:                               # noqa: BLE001
            problems.append(f'Python package {mod} is not importable: {e}')

    if REPO_ROOT != CODES_ROOT.resolve():
        problems.append(f'This clone is at {REPO_ROOT}, but the loaders hardcode {CODES_ROOT}. '
                        'Clone the repo there.')
    if not CONFIG_YAML.exists():
        problems.append(f'Missing {CONFIG_YAML}')

    # Data the repo's own code opens directly.
    needed = [CIRCO_DIR / 'circo_repo' / 'src' / 'dataset.py',
              CIRCO_DIR / 'circo_repo' / 'src' / 'evaluation.py',
              FIQ_DIR / 'images' / 'images']
    needed += [FIQ_DIR / 'captions' / f'cap.{c}.val.json' for c in FIQ_CATS]
    needed += [FIQ_DIR / 'image_splits' / f'split.{c}.val.json' for c in FIQ_CATS]
    for p in needed:
        if not p.exists():
            problems.append(f'Missing data: {p}')
    if not GALLERY.exists():
        problems.append(f'Missing CIRCO gallery cache {GALLERY}. Build it once with: '
                        f'PYTHONPATH={CODES_ROOT} python evaluation/build_circo_gallery.py')
    # Layout CIRCO's own dataset.py expects; a real gap fails within minutes at step 1 anyway.
    for p in (CIRCO_DIR / 'annotations' / 'val.json', CIRCO_DIR / 'annotations' / 'test.json',
              CIRCO_DIR / 'COCO2017_unlabeled' / 'unlabeled2017'):
        if not p.exists():
            warnings.append(f'Not found (expected by CIRCO dataset.py): {p}')
    for c in FIQ_CATS:
        cache = FIQ_DIR / f'gallery_cache_{c}_ViT-L-14.pt'
        if not cache.exists():
            warnings.append(f'No Fashion-IQ gallery cache {cache.name}: it will be computed once.')

    # Every checkpoint: present, byte-identical to the archived copy, pseudo_norm as trained.
    expected = [('baseline', BASELINE, BASELINE_SHA256, 0.0)]
    expected += [(rel, RUN_DIR / rel, sha, 1.0) for rel, sha in UNIT_NORM.items()]
    for name, path, sha, pn_expected in expected:
        entry = {'path': str(path), 'sha256_expected': sha}
        ckpts[name] = entry
        if not path.exists():
            problems.append(f'Missing checkpoint {path}')
            continue
        entry['sha256'] = sha256_file(path)
        if entry['sha256'] != sha:
            problems.append(f'{path}: sha256 {entry["sha256"][:16]}... differs from the archived copy '
                            f'{sha[:16]}...; upload the archived file instead')
        if torch is None:
            continue
        try:
            ck = torch.load(path, map_location='cpu')
            ck_cfg = ck.get('config', {}) or {}
            pn = float(ck_cfg.get('pseudo_norm', 0.0) or 0.0)
            entry['pseudo_norm'] = pn
            entry['has_pseudo_norm_key'] = 'pseudo_norm' in ck_cfg
            entry['mapper_type'] = ck.get('mapper_type')
            if ck.get('logit_scale') is not None:
                entry['tau'] = math.exp(float(ck['logit_scale']))
            if pn != pn_expected:
                problems.append(f'{path}: checkpoint config pseudo_norm={pn}, expected {pn_expected}')
            for k in ('mapper_state', 'mapper_type'):
                if k not in ck:
                    problems.append(f'{path}: no "{k}" entry')
            del ck
        except Exception as e:                               # noqa: BLE001
            problems.append(f'{path}: torch.load failed: {e}')

    return problems, warnings, ckpts


# ---------------------------------------------------------------------------
# Steps 1-3: evaluate one checkpoint (each benchmark saved as soon as it finishes)
# ---------------------------------------------------------------------------
def fashion_eval_cfg(device):
    """The eval-side Config that eval_fashion.py's CLI builds (data dir, batch size, workers)."""
    from config.config import load_from_yaml
    cfg = load_from_yaml(str(CONFIG_YAML))
    cfg.device = device
    return cfg


def run_circo_val(ckpt, key, pn_expected, out_dir, device):
    path = out_dir / 'circo_val' / f'{key}.json'
    if path.exists():
        log(f'  CIRCO val  : already done ({path.name})')
        return read_json(path)
    from evaluation import eval_circo, eval_circo_v2
    t0 = time.time()
    with cudnn_settings(deterministic=False):
        model, clip_name = eval_circo_v2.load_model(ckpt, device)
        pn_model = float(model.pseudo_norm)
        tau = float(model.clip.logit_scale.exp().item())
        log(f'  model      : pseudo_norm={pn_model}  tau={tau:.2f}')
        if pn_model != pn_expected:
            raise RuntimeError(f'{key}: model.pseudo_norm={pn_model} but the checkpoint config says '
                               f'{pn_expected}; the loader did not forward pseudo_norm')
        v2 = eval_circo_v2.eval_circo(model, clip_name, device, CIRCO_DIR, GALLERY)
        v1 = eval_circo.eval_circo(model, clip_name, device, CIRCO_DIR, GALLERY)
    del model
    free_gpu()
    same = all(abs(float(v1['map_atk'][k]) - float(v2['map_atk'][k])) <= 1e-9 for k in v2['map_atk'])
    rec = {
        'checkpoint'          : str(ckpt),
        'key'                 : key,
        'pseudo_norm_in_model': pn_model,
        'tau'                 : tau,
        'evaluated_at'        : now(),
        'seconds'             : round(time.time() - t0, 1),
        'v1_v2_map_identical' : same,
        'v2'                  : v2,     # same schema as metrics/circo/*.json (incl. per-query data)
        'v1'                  : {'map_atk': v1['map_atk'], 'recall_atk': v1['recall_atk'],
                                 'stickiness': v1['stickiness']},   # stickiness adds P(gt)
    }
    write_json(path, rec)
    s, p = v2['stickiness'], v1['stickiness']
    m = v2['map_atk']
    log(f'[RESULT] CIRCO-val  {key}  mAP@5/10/25/50 = '
        f'{100*m["@5"]:.2f}/{100*m["@10"]:.2f}/{100*m["@25"]:.2f}/{100*m["@50"]:.2f}  '
        f'Ref@10 stick = {100*s["r10_ref"]:.1f}%  simDelta = {s["sim_delta"]:+.4f}  '
        f'P(gt) = {p["p_target_mean"]:.4f}')
    if not same:
        log('  WARNING: eval_circo and eval_circo_v2 report different mAP for the same model')
    return rec


def run_fashion(ckpt, key, out_dir, cfg):
    path = out_dir / 'fashion_val' / f'{key}.json'
    if path.exists():
        log(f'  Fashion-IQ : already done ({path.name})')
        return read_json(path)
    from evaluation import eval_fashion
    t0 = time.time()
    with cudnn_settings(deterministic=True):
        recalls = eval_fashion.evaluate(str(ckpt), cfg)
    free_gpu()
    per_cat = {c: {str(k): float(v) for k, v in recalls[c].items()} for c in eval_fashion.CATEGORIES}
    # train.py's averaging (each category rounded to 6 dp, then the average) so values compare exactly
    avg = {k: round(sum(round(per_cat[c][k], 6) for c in eval_fashion.CATEGORIES)
                    / len(eval_fashion.CATEGORIES), 6) for k in KS}
    rec = {
        'checkpoint'  : str(ckpt),
        'key'         : key,
        'evaluated_at': now(),
        'seconds'     : round(time.time() - t0, 1),
        'per_category': per_cat,
        'avg'         : avg,
    }
    write_json(path, rec)
    log(f'[RESULT] Fashion-IQ {key}  avg R@1/5/10/50 = '
        f'{100*avg["1"]:.2f}/{100*avg["5"]:.2f}/{100*avg["10"]:.2f}/{100*avg["50"]:.2f}')
    return rec


def evaluate_checkpoint(ckpt, pn_expected, out_dir, device, cfg):
    key = key_for(ckpt)
    log()
    log(f'--- {key}   [{now()}]')
    c = run_circo_val(ckpt, key, pn_expected, out_dir, device)
    f = run_fashion(ckpt, key, out_dir, cfg)
    write_summary(out_dir)
    return c, f


def link_duplicate(rel, src_rel, out_dir):
    """final.pt is byte-identical to an earlier file: reuse its results, recording the link."""
    src_key, key = key_for(RUN_DIR / src_rel), key_for(RUN_DIR / rel)
    for sub in ('circo_val', 'fashion_val'):
        dst = out_dir / sub / f'{key}.json'
        if not dst.exists():
            rec = dict(read_json(out_dir / sub / f'{src_key}.json'))
            rec.update(checkpoint=str(RUN_DIR / rel), key=key, same_file_as=src_key)
            write_json(dst, rec)
    log(f'[RESULT] {rel} is byte-identical to {src_rel} (same sha256): results reused')
    write_summary(out_dir)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def compare_fashion(rec, ref, tol):
    deltas = {k: rec['avg'][k] - r for k, r in zip(KS, ref)}
    ok = all(abs(d) <= tol for d in deltas.values())
    exact = all(d == 0 for d in deltas.values())
    return ok, exact, deltas


def fmt_fashion_row(name, rec, ref, deltas, status):
    cells = '  '.join(f'R@{k} {100*r:6.3f}->{100*rec["avg"][k]:6.3f} ({100*deltas[k]:+.3f})'
                      for k, r in zip(KS, ref))
    return f'  {name:24s} {cells}  {status}'


def check_control(c, f, args, out_dir):
    d_circo = float(c['v2']['map_atk']['@10']) - BASELINE_CIRCO_MAP10
    ok_c = abs(d_circo) <= args.circo_tol
    ok_f, exact_f, deltas = compare_fashion(f, BASELINE_FASHION, args.fashion_tol)
    ok_pn = c['pseudo_norm_in_model'] == 0.0
    log(f'  CIRCO-val mAP@10  stored {100*BASELINE_CIRCO_MAP10:.4f}  now '
        f'{100*float(c["v2"]["map_atk"]["@10"]):.4f}  delta {100*d_circo:+.4f} pts  '
        f'{"ok" if ok_c else "FAIL"} (tol {100*args.circo_tol:.2f} pts)')
    log(fmt_fashion_row('Fashion-IQ (training)', f, BASELINE_FASHION, deltas,
                        'exact' if exact_f else ('ok' if ok_f else 'FAIL')))
    log(f'  pseudo_norm in model = {c["pseudo_norm_in_model"]}  {"ok" if ok_pn else "FAIL"}')
    ok = ok_c and ok_f and ok_pn
    gate = read_json(out_dir / 'gate.json') if (out_dir / 'gate.json').exists() else {}
    gate['control'] = {'ok': ok, 'circo_map10_stored': BASELINE_CIRCO_MAP10,
                       'circo_map10_now': float(c['v2']['map_atk']['@10']), 'circo_delta': d_circo,
                       'fashion_reference': dict(zip(KS, BASELINE_FASHION)), 'fashion_now': f['avg'],
                       'fashion_deltas': deltas, 'fashion_exact': exact_f,
                       'tolerances': {'circo': args.circo_tol, 'fashion': args.fashion_tol}}
    write_json(out_dir / 'gate.json', gate)
    return ok


def check_gate(args, out_dir):
    rows, ok_all = [], True
    for rel, (ref, source) in GATE.items():
        key = key_for(RUN_DIR / rel)
        f = read_json(out_dir / 'fashion_val' / f'{key}.json')
        c = read_json(out_dir / 'circo_val' / f'{key}.json')
        ok, exact, deltas = compare_fashion(f, ref, args.fashion_tol)
        ok = ok and c['pseudo_norm_in_model'] == 1.0
        ok_all &= ok
        log(fmt_fashion_row(rel, f, ref, deltas, 'exact' if exact else ('ok' if ok else 'FAIL')))
        rows.append({'checkpoint': rel, 'reference_file': source, 'reference': dict(zip(KS, ref)),
                     'now': f['avg'], 'deltas': deltas, 'exact': exact, 'ok': ok,
                     'pseudo_norm_in_model': c['pseudo_norm_in_model']})
    gate = read_json(out_dir / 'gate.json') if (out_dir / 'gate.json').exists() else {}
    gate['unit_norm_gate'] = {'ok': ok_all, 'rows': rows, 'tolerance': args.fashion_tol}
    write_json(out_dir / 'gate.json', gate)
    return ok_all


# ---------------------------------------------------------------------------
# Step 4: selection (paper's rule, Table S1)
# ---------------------------------------------------------------------------
def select(out_dir):
    rows, identical = [], {}
    for rel in UNIT_NORM:
        src = first_with_same_file(rel)
        if src is not None:
            identical[rel] = src
            continue
        key = key_for(RUN_DIR / rel)
        c = read_json(out_dir / 'circo_val' / f'{key}.json')
        f = read_json(out_dir / 'fashion_val' / f'{key}.json')
        rows.append({'checkpoint': rel, 'key': key,
                     'circo_val_map10': float(c['v2']['map_atk']['@10']),
                     'ref10_stickiness': float(c['v2']['stickiness']['r10_ref']),
                     'sim_delta': float(c['v2']['stickiness']['sim_delta']),
                     'fiq_r10': f['avg']['10'], 'fiq_r50': f['avg']['50']})
    best = max(r['circo_val_map10'] for r in rows)
    ties = [r['checkpoint'] for r in rows if r['circo_val_map10'] == best]
    chosen = next(r for r in rows if r['checkpoint'] == ties[0])   # earliest in training order
    return {
        'rule'              : 'argmax CIRCO-val mAP@10 over the checkpoints of the run (paper, Table S1); '
                              'an exact tie between distinct files goes to the earliest in training order',
        'selected'          : chosen,
        'ties'              : ties,
        'identical_files'   : identical,
        'n_files'           : len(UNIT_NORM),
        'n_distinct'        : len(rows),
        'previous_selection': {'checkpoint': PREVIOUS_SELECTION,
                               'note': 'behind the paper unit-norm rows; evaluated without the projection'},
        'ranking'           : sorted(rows, key=lambda r: -r['circo_val_map10']),
        'selected_at'       : now(),
    }


# ---------------------------------------------------------------------------
# Step 5: CIRCO test submission for the selected checkpoint
# ---------------------------------------------------------------------------
def make_submission(rel, out_dir, device, selection):
    ckpt = RUN_DIR / rel
    key = key_for(ckpt)
    sub_dir = out_dir / 'circo_test_submission'
    final = sub_dir / f'{key}__with_projection__submission.json'
    receipt_path = sub_dir / 'receipt.json'
    if final.exists() and receipt_path.exists():
        log(f'  already done: {final.name}')
        return read_json(receipt_path)

    from evaluation import eval_circo_test
    seen = {}
    original_load = eval_circo_test.load_model

    def checked_load(path, dev):            # record what the test-side loader actually built
        model, name = original_load(path, dev)
        seen['pseudo_norm'] = float(model.pseudo_norm)
        seen['tau'] = float(model.clip.logit_scale.exp().item())
        return model, name

    eval_circo_test.load_model = checked_load
    # Its legacy val-metric printout reads files that predate the fix; keep it out of the log.
    eval_circo_test._print_existing_val_metrics = (
        lambda p: log('  (legacy val-metric lookup skipped: those files predate the pseudo_norm fix)'))
    # Write into a staging folder so metrics/circo_test_submissions/ (old submissions) is never touched.
    eval_circo_test.SUBMISSION_DIR = sub_dir / '_staging'
    try:
        with cudnn_settings(deterministic=False):
            produced = eval_circo_test.generate_test_submission(ckpt, device, CIRCO_DIR, GALLERY)
    finally:
        eval_circo_test.load_model = original_load
    free_gpu()
    if seen.get('pseudo_norm') != 1.0:
        raise RuntimeError(f'test-side loader built pseudo_norm={seen.get("pseudo_norm")}, expected 1.0')
    os.replace(produced, final)
    try:
        (sub_dir / '_staging').rmdir()
    except OSError:
        pass
    n_queries = len(read_json(final))
    f = read_json(out_dir / 'fashion_val' / f'{key}.json')
    receipt = {
        'submission_file'     : final.name,
        'submission_sha256'   : sha256_file(final),
        'n_queries'           : n_queries,
        'checkpoint'          : str(ckpt),
        'checkpoint_sha256'   : UNIT_NORM[rel],
        'pseudo_norm_in_model': seen['pseudo_norm'],
        'tau'                 : seen['tau'],
        'circo_val_map10'     : selection['selected']['circo_val_map10'],
        'fiq_r10'             : f['avg']['10'],
        'fiq_r50'             : f['avg']['50'],
        'selection_rule'      : selection['rule'],
        'code_commit'         : git_commit(),
        'created_at'          : now(),
        'server'              : 'https://circo.micc.unifi.it/evaluation',
        'server_result'       : None,   # fill in from the CIRCO server email (mAP@5/10/25/50)
    }
    write_json(receipt_path, receipt)
    log(f'  saved {final}  ({n_queries} queries, sha256 {receipt["submission_sha256"][:16]}...)')
    return receipt


# ---------------------------------------------------------------------------
# summary.csv (rebuilt after every checkpoint from the saved JSONs)
# ---------------------------------------------------------------------------
SUMMARY_COLUMNS = [
    'role', 'checkpoint', 'same_file_as', 'sha256', 'pseudo_norm', 'tau',
    'circo_val_map5', 'circo_val_map10', 'circo_val_map25', 'circo_val_map50',
    'circo_val_r5', 'circo_val_r10', 'circo_val_r25', 'circo_val_r50',
    'ref10_stickiness', 'sim_delta', 'sim_ref_mean', 'sim_gt_mean', 'median_rank_ref',
    'p_gt_target', 'p_gt_set',
    'fiq_r1', 'fiq_r5', 'fiq_r10', 'fiq_r50',
    'fiq_dress_r10', 'fiq_dress_r50', 'fiq_shirt_r10', 'fiq_shirt_r50', 'fiq_toptee_r10', 'fiq_toptee_r50',
    'train_time_fiq_r10', 'train_time_fiq_r50', 'selected',
]


def write_summary(out_dir):
    selected = None
    if (out_dir / 'selection.json').exists():
        selected = read_json(out_dir / 'selection.json')['selected']['checkpoint']
    entries = [('control', 'baseline ' + BASELINE.name, BASELINE, BASELINE_SHA256, BASELINE_FASHION)]
    for rel, sha in UNIT_NORM.items():
        role = 'gate' if rel in GATE else 'sweep'
        entries.append((role, rel, RUN_DIR / rel, sha, GATE[rel][0] if rel in GATE else None))
    rows = []
    for role, name, path, sha, ref in entries:
        key = key_for(path)
        row = dict.fromkeys(SUMMARY_COLUMNS, '')
        row.update(role=role, checkpoint=name, sha256=sha,
                   selected='yes' if name == selected else '')
        if ref is not None:
            row.update(train_time_fiq_r10=ref[2], train_time_fiq_r50=ref[3])
        cp, fp = out_dir / 'circo_val' / f'{key}.json', out_dir / 'fashion_val' / f'{key}.json'
        if cp.exists():
            c = read_json(cp)
            v2, st, pg = c['v2'], c['v2']['stickiness'], c['v1']['stickiness']
            row.update(same_file_as=c.get('same_file_as', ''), pseudo_norm=c['pseudo_norm_in_model'],
                       tau=round(c['tau'], 4), ref10_stickiness=st['r10_ref'], sim_delta=st['sim_delta'],
                       sim_ref_mean=st['sim_ref_mean'], sim_gt_mean=st['sim_gt_mean'],
                       median_rank_ref=st['median_rank_ref'],
                       p_gt_target=pg['p_target_mean'], p_gt_set=pg['p_gtset_mean'])
            for k in ('5', '10', '25', '50'):
                row[f'circo_val_map{k}'] = v2['map_atk'][f'@{k}']
                row[f'circo_val_r{k}'] = v2['recall_atk'][f'@{k}']
        if fp.exists():
            f = read_json(fp)
            for k in KS:
                row[f'fiq_r{k}'] = f['avg'][k]
            for cat in FIQ_CATS:
                row[f'fiq_{cat}_r10'] = f['per_category'][cat]['10']
                row[f'fiq_{cat}_r50'] = f['per_category'][cat]['50']
        rows.append(row)
    tmp = out_dir / 'summary.csv.tmp'
    with open(tmp, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out_dir / 'summary.csv')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def environment(args):
    info = {'started_at': now(), 'argv': sys.argv, 'code_commit': git_commit(),
            'python': platform.python_version(), 'device': args.device}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        import numpy
        info['numpy'] = numpy.__version__
    except Exception:                                        # noqa: BLE001
        pass
    return info


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out_dir', type=Path, default=REPO_ROOT / 'results' / 'unit_norm_reeval')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--preflight_only', action='store_true',
                   help='check packages, GPU, data and checkpoints, then exit')
    p.add_argument('--fashion_tol', type=float, default=FASHION_TOL)
    p.add_argument('--circo_tol', type=float, default=CIRCO_TOL)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    banner(f'Unit-norm re-evaluation   {now()}')
    log(f'  code   : {REPO_ROOT}  (commit {git_commit()})')
    log(f'  run    : {RUN_DIR}')
    log(f'  output : {out_dir}')

    # ---- Step 0 -------------------------------------------------------------
    banner('Step 0/5  Preflight')
    problems, warnings, ckpts = preflight()
    for name, e in ckpts.items():
        state = 'ok  ' if e.get('sha256') == e['sha256_expected'] else 'FAIL'
        tau = f'{e["tau"]:.2f}' if 'tau' in e else '-'
        log(f'  {state} {name:24s} sha256 {e.get("sha256", "missing")[:16]}  '
            f'pseudo_norm {e.get("pseudo_norm", "-")}  tau {tau}')
    for w in warnings:
        log(f'  note: {w}')
    info = environment(args)
    write_json(out_dir / 'preflight.json', {'checked_at': now(), 'environment': info,
                                            'problems': problems, 'warnings': warnings,
                                            'checkpoints': ckpts})
    history = read_json(out_dir / 'run_info.json') if (out_dir / 'run_info.json').exists() else []
    history.append(info)
    write_json(out_dir / 'run_info.json', history)
    if problems:
        log('\nPREFLIGHT FAILED:')
        for pr in problems:
            log(f'  - {pr}')
        sys.exit(1)
    log(f'\n  Preflight passed: torch {info.get("torch")}, CUDA {info.get("cuda")}, GPU {info.get("gpu")}')
    if args.preflight_only:
        return

    cfg = fashion_eval_cfg(args.device)

    # ---- Step 1 -------------------------------------------------------------
    banner('Step 1/5  Control: bounded baseline (no pseudo_norm) must reproduce its numbers')
    c, f = evaluate_checkpoint(BASELINE, 0.0, out_dir, args.device, cfg)
    log()
    if not check_control(c, f, args, out_dir):
        banner('STOPPED: the control did not reproduce. Nothing else was run. Send run_log.txt and gate.json.')
        sys.exit(2)

    # ---- Step 2 -------------------------------------------------------------
    banner(f'Step 2/5  Gate: {len(GATE)} unit-norm checkpoints vs their training-time Fashion-IQ evals')
    for rel in GATE:
        evaluate_checkpoint(RUN_DIR / rel, 1.0, out_dir, args.device, cfg)
    log()
    if not check_gate(args, out_dir):
        banner('STOPPED: the gate did not reproduce. No test file was made. Send run_log.txt and gate.json.')
        sys.exit(3)

    # ---- Step 3 -------------------------------------------------------------
    remaining = [rel for rel in UNIT_NORM if rel not in GATE]
    banner(f'Step 3/5  Sweep: the remaining {len(remaining)} checkpoint files')
    failures, durations = [], []
    for i, rel in enumerate(remaining, 1):
        src = first_with_same_file(rel)
        if src is not None:
            link_duplicate(rel, src, out_dir)
            continue
        t0 = time.time()
        try:
            evaluate_checkpoint(RUN_DIR / rel, 1.0, out_dir, args.device, cfg)
        except Exception:                                    # noqa: BLE001
            log(traceback.format_exc())
            log(f'  FAILED: {rel} (continuing with the others)')
            failures.append(rel)
            free_gpu()
            continue
        durations.append(time.time() - t0)
        left = len(remaining) - i
        if durations and left:
            eta = left * sum(durations) / len(durations) / 60
            log(f'  progress   : {i}/{len(remaining)} done, about {eta:.0f} min left in the sweep')
    if failures:
        banner(f'STOPPED: {len(failures)} checkpoint(s) failed: {failures}. Rerun to retry them.')
        sys.exit(4)

    # ---- Step 4 -------------------------------------------------------------
    banner('Step 4/5  Selection: argmax CIRCO-val mAP@10 (paper, Table S1)')
    selection = select(out_dir)
    write_json(out_dir / 'selection.json', selection)
    write_summary(out_dir)
    log(f'  {"checkpoint":24s} {"CIRCO mAP@10":>12s} {"Ref@10 stick":>12s} {"simDelta":>9s} '
        f'{"FIQ R@10":>9s} {"FIQ R@50":>9s}')
    for r in selection['ranking']:
        mark = '  <- selected' if r['checkpoint'] == selection['selected']['checkpoint'] else ''
        prev = '  (previous selection)' if r['checkpoint'] == PREVIOUS_SELECTION else ''
        log(f'  {r["checkpoint"]:24s} {100*r["circo_val_map10"]:12.2f} {100*r["ref10_stickiness"]:11.1f}% '
            f'{r["sim_delta"]:+9.4f} {100*r["fiq_r10"]:9.2f} {100*r["fiq_r50"]:9.2f}{mark}{prev}')
    if len(selection['ties']) > 1:
        log(f'  note: exact tie between {selection["ties"]}; the earliest was selected')

    # ---- Step 5 -------------------------------------------------------------
    banner(f'Step 5/5  CIRCO test submission for {selection["selected"]["checkpoint"]}')
    receipt = make_submission(selection['selected']['checkpoint'], out_dir, args.device, selection)
    selection['submission'] = receipt
    write_json(out_dir / 'selection.json', selection)

    banner(f'DONE in {(time.time() - t_start) / 60:.1f} min   {now()}')
    log(f'  results    : {out_dir}')
    log(f'  summary    : {out_dir / "summary.csv"}')
    log(f'  selection  : {out_dir / "selection.json"}')
    log(f'  submission : {out_dir / "circo_test_submission" / receipt["submission_file"]}')
    log('  Upload the submission at https://circo.micc.unifi.it/evaluation and keep the result email.')


if __name__ == '__main__':
    main()

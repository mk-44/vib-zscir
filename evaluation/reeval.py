"""Re-evaluate the paper's checkpoints on a freshly rebuilt evaluation setup.

Why
    1. The unit-norm run (pseudo_norm = 1.0) trains the mapper with the fixed-norm projection
       z <- rho * z / ||z|| inside Pic2Word.forward and Pic2Word.encode_query. Until commit e2b91fa the
       evaluation loaders dropped `pseudo_norm` from the checkpoint config, so the paper's unit-norm
       numbers (Table 1, Table S8, Sec. D.2) were measured without the projection the mapper was
       trained with. All 19 checkpoints of that run are re-evaluated, the paper's selection rule
       (argmax CIRCO-val mAP@10, Table S1) is applied, and a CIRCO test file is written.
    2. Four more checkpoints get the numbers the paper still needs: the second and third VIB runs and
       the margin and additive-noise remedies, each at its own CIRCO-val argmax.
    3. The evaluation data are rebuilt from backups and public sources (evaluation/prepare_eval_data.py),
       so two paper models with known outputs are evaluated first and must reproduce them.

What it runs (the repo's own evaluation code, unchanged)
    CIRCO val    eval_circo_v2.load_model + eval_circo_v2.eval_circo  official mAP/recall, stickiness and
                                                                    per-query data
                 eval_circo.eval_circo on the same model              adds P(gt); its mAP must equal v2's
    Fashion-IQ   eval_fashion.evaluate                                the eval_category used in training
    CIRCO test   eval_circo_test.generate_test_submission             test-server files

Steps (1 and 2 stop the run on failure, so nothing new comes from an unverified setup)
    0 preflight  packages, GPU, data, and every checkpoint: present, byte-identical to the archived copy
                 (sha256), pseudo_norm as trained
    1 known      bounded baseline and VIB headline reproduce their stored CIRCO-val results query by
                 query, their Fashion-IQ numbers, and their original CIRCO test files
    2 gate       the 9 unit-norm checkpoints with training-time Fashion-IQ evals (computed in-process with
                 the projection on) reproduce them: the fixed loaders restore the trained forward pass
    3 sweep      the other unit-norm checkpoints (final.pt is byte-identical to epoch_00/checkpoint.pt and
                 reuses its results)
    4 select     argmax CIRCO-val mAP@10 over the unit-norm run -> selection_unit_norm.json
    5 test file  CIRCO test file for the selected unit-norm checkpoint
    6 extras     CIRCO val, Fashion-IQ and CIRCO test for the four extra checkpoints; where the paper's
                 files exist they are reproduced and compared, otherwise new test files are written

Outputs (results/reeval_2026_11/)
    circo_val/<run>__<ckpt>.json      eval_circo_v2 record (+ P(gt) from eval_circo)
    fashion_val/<run>__<ckpt>.json    per-category and averaged Fashion-IQ recalls
    checks/                           known_models.json, unit_norm_gate.json, extras.json
    selection_unit_norm.json          the ranking and the selected checkpoint
    circo_test/to_upload/             NEW test files for the CIRCO server
    circo_test/rechecks/              regenerated copies of already-submitted files, compared with them
    circo_test/test_files.json        sha256, checkpoint and comparison for every test file
    summary.csv                       one row per checkpoint, rebuilt after every evaluation
    preflight.json, run_info.json     environment, inputs and invocation history

Persistence
    Every result is written the moment it is computed (atomic writes); rerunning the same command
    resumes and skips finished work, so a stopped kernel loses at most the evaluation in progress.
    Nothing outside the output folder is written apart from model-independent caches (CLIP weights in
    ~/.cache/clip); the stored metrics/ files and the original submissions are only read.

Usage (repo cloned at /notebooks/experiment/codes, data prepared with evaluation/prepare_eval_data.py)
    PYTHONPATH=/notebooks/experiment/codes python -u evaluation/reeval.py --preflight_only
    PYTHONPATH=/notebooks/experiment/codes python -u evaluation/reeval.py
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
CIRCO_DIR   = EXPERIMENT / 'eval_data' / 'circo'
GALLERY     = CIRCO_DIR / 'gallery_cache_ViT-L-14.pt'
FIQ_DIR     = EXPERIMENT / 'eval_data' / 'fashion_iq'
FIQ_CATS    = ('dress', 'shirt', 'toptee')
KS          = ('1', '5', '10', '50')
STORED_VAL  = REPO_ROOT / 'metrics' / 'circo'                     # stored eval_circo_v2 results
ORIGINAL_SUBMISSIONS = REPO_ROOT / 'metrics' / 'circo_test_submissions'
CK          = EXPERIMENT / '4M_checkpoints'
RUN_DIR     = CK / 'base_hd256_tok1_lslr0.2_lsmax116_pnorm1.0'

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

# Paper models whose original outputs are stored in this repo (metrics/circo/<key>.json with per-query
# data; metrics/circo_test_submissions/<key>__submission.json). They must reproduce them.
KNOWN = [
    {'label': 'baseline', 'sha256': 'b6618d3e0da5f6aecb59e2ea08b272caaed87f2b8829af1de2321c28dcc39d4f',
     'path': CK / 'base_hd256_tok1_lslr0.2_lsmax116' / 'epoch_02_step_03000.pt',
     'paper': 'bounded baseline, Tab. 2 / S8: F 25.0, CIRCO val 12.12, test 11.39',
     # training-time step_evals/step_03000_epoch_02.json of the baseline run (6 dp)
     'fashion_ref': (0.069583, 0.182814, 0.249871, 0.451093)},
    {'label': 'vib_headline', 'sha256': 'cfbab52c8b00dff9f5ee0c3e64c5ea43607cf36bf038ab93a622e2865fb7b784',
     'path': CK / 'vib_hd256_tok1_lslr0.2_lsmax116_vib0.0001' / 'epoch_01_step_01500.pt',
     'paper': 'VIB beta=0.1, sec/4_experiment.tex: F 22.15 / 40.76, CIRCO val 12.38, test 13.22',
     # No exact record survives, only printed values, and each must print the same now: the paper's
     # two-decimal avg R@10 / R@50 (sec/4_experiment.tex, "Our VIB" row), and the 16 one-decimal values
     # of the original eval_fashion.py run (keep_rest.zip: fashion_eval_logs/
     # vib_hd256_tok1_lslr0.2_lsmax116_vib0.0001__epoch_01_step_01500.log). Where the paper rounds this
     # model to one decimal it writes 22.2 (22.15 rounded up); the exact value rounds to 22.1, as that log
     # printed. all_fashion_metrics.json's 22.18 is a different evaluator (its "fast path").
     'fashion_paper_2dp': {'10': '22.15', '50': '40.76'},
     'fashion_log_1dp': {'dress': ('3.4', '11.6', '16.9', '36.4'), 'shirt': ('8.3', '19.3', '25.4', '41.5'),
                         'toptee': ('7.2', '17.7', '24.2', '44.4'), 'avg': ('6.3', '16.2', '22.1', '40.8')}},
]
EXTRAS = [
    {'label': 'run2_e02s0500', 'sha256': '0f62fec180c372d7d59f85398512fea1b651806917b6d5bdc9387a8272d81bd9',
     'path': EXPERIMENT / '4M_checkpoints_rerun' / 'vib_hd256_tok1_lslr0.2_lsmax116_vib0.1024_vwarm0.3_seedR'
             / 'epoch_02_step_00500.pt',
     'why': 'second VIB run at its CIRCO-val argmax (the paper used epoch_01_step_03000)'},
    {'label': 'seed3407_e02s0500', 'sha256': 'ac6f3ead5dc37b838ec6931209f3f101b80d088246426a8175dfae86057b17a2',
     'path': EXPERIMENT / '4M_checkpoints_seed3407' / 'vib_hd256_tok1_lslr0.2_lsmax116_vib0.1024_vwarm0.3_seed3407'
             / 'epoch_02_step_00500.pt',
     'why': 'third VIB run (seed 3407) at its CIRCO-val argmax'},
    {'label': 'margin_e01s3500', 'sha256': 'e3b136b1de689c1f53ea7c881440c1903c128cfa8ed2b2096094c9b33b5bfdec',
     'path': CK / 'base_hd256_tok1_margin0.2_mw0.4_lslr0.4_lsmax116' / 'epoch_01_step_03500.pt',
     'why': 'margin remedy at its CIRCO-val argmax (the paper used epoch_01_step_01500)'},
    {'label': 'noise_e00s3500', 'sha256': '61e4145c9e3afaab37c7c2627ba42e6dde76e280e00f271762962ca8707b80b7',
     'path': CK / 'base_hd256_tok1_lslr0.2_lsmax116_noise0.1' / 'epoch_00_step_03500.pt',
     'why': 'additive-noise remedy at its CIRCO-val argmax (the paper used epoch_01_step_00500)'},
]

FASHION_TOL    = 1e-3   # avg recall (fraction); one query flip moves a 3-category average by ~1.7e-4
CIRCO_TOL      = 2e-3   # CIRCO-val mAP@10 (fraction) vs a stored result; a broken setup moves it by >1e-2
VAL_QUERY_TOL  = 2      # known models: CIRCO-val queries (of 220) whose AP@10 may differ from the stored one
TEST_QUERY_TOL = 8      # known models: CIRCO-test queries (of 800, 1%) whose top-50 may differ from the original


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
    """Earlier checkpoint of the unit-norm run with identical bytes (final.pt -> epoch_00/checkpoint.pt)."""
    for other, sha in UNIT_NORM.items():
        if other == rel:
            return None
        if sha == UNIT_NORM[rel]:
            return other
    return None


def val_file(out_dir, key):
    return out_dir / 'circo_val' / f'{key}.json'


def fiq_file(out_dir, key):
    return out_dir / 'fashion_val' / f'{key}.json'


def all_entries():
    """(role, name, path, sha256, expected pseudo_norm) for every checkpoint the run touches."""
    rows = [('known', e['label'], e['path'], e['sha256'], 0.0) for e in KNOWN]
    rows += [('gate' if rel in GATE else 'sweep', rel, RUN_DIR / rel, sha, 1.0) for rel, sha in UNIT_NORM.items()]
    rows += [('extra', e['label'], e['path'], e['sha256'], 0.0) for e in EXTRAS]
    return rows


# ---------------------------------------------------------------------------
# Step 0: preflight
# ---------------------------------------------------------------------------
def query_image_problems():
    """Every CIRCO query needs its reference image and every Fashion-IQ val query both images."""
    out = []
    info = CIRCO_DIR / 'COCO2017_unlabeled' / 'annotations' / 'image_info_unlabeled2017.json'
    if not info.exists():
        return [f'Missing {info}: run evaluation/prepare_eval_data.py']
    names = {i['id']: i['file_name'] for i in read_json(info)['images']}
    missing = 0
    for split in ('val', 'test'):
        ann = CIRCO_DIR / 'annotations' / f'{split}.json'
        if not ann.exists():
            out.append(f'Missing {ann}')
            continue
        for q in read_json(ann):
            fn = names.get(q['reference_img_id'])
            missing += fn is None or not (CIRCO_DIR / 'COCO2017_unlabeled' / 'unlabeled2017' / fn).exists()
    if missing:
        out.append(f'{missing} CIRCO queries lack their reference image: run evaluation/prepare_eval_data.py')
    images = FIQ_DIR / 'images' / 'images'
    for c in FIQ_CATS:
        cap = FIQ_DIR / 'captions' / f'cap.{c}.val.json'
        if not cap.exists():
            out.append(f'Missing {cap}')
            continue
        lacking = sum(not ((images / f'{a["candidate"]}.png').exists() and (images / f'{a["target"]}.png').exists())
                      for a in read_json(cap))
        if lacking:
            out.append(f'Fashion-IQ {c}: {lacking} val queries lack an image: run evaluation/prepare_eval_data.py')
    return out


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
        problems.append(f'OpenAI CLIP is not importable ({e}).')
    for mod in ('numpy', 'PIL', 'yaml'):
        try:
            __import__(mod)
        except Exception as e:                               # noqa: BLE001
            problems.append(f'Python package {mod} is not importable: {e}')

    if REPO_ROOT != CODES_ROOT.resolve():
        problems.append(f'This clone is at {REPO_ROOT}, but the loaders hardcode {CODES_ROOT}. Clone it there.')
    if not CONFIG_YAML.exists():
        problems.append(f'Missing {CONFIG_YAML}')

    # Data the repo's own code opens directly.
    needed = [CIRCO_DIR / 'circo_repo' / 'src' / 'dataset.py', CIRCO_DIR / 'circo_repo' / 'src' / 'evaluation.py',
              GALLERY, FIQ_DIR / 'images' / 'images']
    needed += [FIQ_DIR / f'gallery_cache_{c}_ViT-L-14.pt' for c in FIQ_CATS]
    needed += [FIQ_DIR / 'captions' / f'cap.{c}.val.json' for c in FIQ_CATS]
    needed += [FIQ_DIR / 'image_splits' / f'split.{c}.val.json' for c in FIQ_CATS]
    for p in needed:
        if not p.exists():
            problems.append(f'Missing data: {p}')
    problems.extend(query_image_problems())
    for e in KNOWN + EXTRAS:
        key = key_for(e['path'])
        if e in KNOWN and not (STORED_VAL / f'{key}.json').exists():
            problems.append(f'Missing stored result {STORED_VAL / (key + ".json")}')
        if e in KNOWN and not (ORIGINAL_SUBMISSIONS / f'{key}__submission.json').exists():
            problems.append(f'Missing original submission for {e["label"]}')
    if not (EXPERIMENT / 'eval_data' / 'prepared.json').exists():
        warnings.append('eval_data/prepared.json not found: was evaluation/prepare_eval_data.py run?')

    # Every checkpoint: present, byte-identical to the archived copy, pseudo_norm as trained.
    for role, name, path, sha, pn_expected in all_entries():
        entry = {'role': role, 'path': str(path), 'sha256_expected': sha}
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
            entry.update(pseudo_norm=pn, mapper_type=ck.get('mapper_type'))
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
# Evaluate one checkpoint (each benchmark saved as soon as it finishes)
# ---------------------------------------------------------------------------
def fashion_eval_cfg(device):
    """The eval-side Config that eval_fashion.py's CLI builds (data dir, batch size, workers)."""
    from config.config import load_from_yaml
    cfg = load_from_yaml(str(CONFIG_YAML))
    cfg.device = device
    return cfg


def run_circo_val(ckpt, key, pn_expected, out_dir, device):
    path = val_file(out_dir, key)
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
    s, p, m = v2['stickiness'], v1['stickiness'], v2['map_atk']
    log(f'[RESULT] CIRCO-val  {key}  mAP@5/10/25/50 = '
        f'{100*m["@5"]:.2f}/{100*m["@10"]:.2f}/{100*m["@25"]:.2f}/{100*m["@50"]:.2f}  '
        f'Ref@10 stick = {100*s["r10_ref"]:.1f}%  simDelta = {s["sim_delta"]:+.4f}  '
        f'P(gt) = {p["p_target_mean"]:.4f}')
    if not same:
        log('  WARNING: eval_circo and eval_circo_v2 report different mAP for the same model')
    return rec


def run_fashion(ckpt, key, out_dir, cfg):
    path = fiq_file(out_dir, key)
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
    rec = {'checkpoint': str(ckpt), 'key': key, 'evaluated_at': now(),
           'seconds': round(time.time() - t0, 1), 'per_category': per_cat, 'avg': avg}
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
    for target in (val_file, fiq_file):
        dst = target(out_dir, key)
        if not dst.exists():
            rec = dict(read_json(target(out_dir, src_key)))
            rec.update(checkpoint=str(RUN_DIR / rel), key=key, same_file_as=src_key)
            write_json(dst, rec)
    log(f'[RESULT] {rel} is byte-identical to {src_rel} (same sha256): results reused')
    write_summary(out_dir)


# ---------------------------------------------------------------------------
# CIRCO test files
# ---------------------------------------------------------------------------
def _skip_legacy_lookup(_checkpoint_path):
    log('  (legacy val-metric lookup skipped: those files predate this re-evaluation)')


def registry(out_dir):
    p = out_dir / 'circo_test' / 'test_files.json'
    return read_json(p) if p.exists() else {}


def register(out_dir, name, info):
    reg = registry(out_dir)
    reg[name] = info
    write_json(out_dir / 'circo_test' / 'test_files.json', reg)


def produce_test_file(ckpt, dest, pn_expected, device):
    """Write the CIRCO test file of ckpt to dest with eval_circo_test.generate_test_submission."""
    from evaluation import eval_circo_test
    seen = {}
    original_load = eval_circo_test.load_model

    def checked_load(path, dev):            # record what the test-side loader actually built
        model, name = original_load(path, dev)
        seen['pseudo_norm'] = float(model.pseudo_norm)
        seen['tau'] = float(model.clip.logit_scale.exp().item())
        return model, name

    staging = dest.parent / '_staging'
    eval_circo_test.load_model = checked_load
    eval_circo_test._print_existing_val_metrics = _skip_legacy_lookup
    # Write into a staging folder so metrics/circo_test_submissions/ (the originals) is never touched.
    eval_circo_test.SUBMISSION_DIR = staging
    try:
        with cudnn_settings(deterministic=False):
            produced = eval_circo_test.generate_test_submission(ckpt, device, CIRCO_DIR, GALLERY)
    finally:
        eval_circo_test.load_model = original_load
    free_gpu()
    if seen.get('pseudo_norm') != pn_expected:
        raise RuntimeError(f'test-side loader built pseudo_norm={seen.get("pseudo_norm")}, expected {pn_expected}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(produced, dest)
    with contextlib.suppress(OSError):
        staging.rmdir()
    return {'file': dest.name, 'folder': dest.parent.name, 'sha256': sha256_file(dest),
            'n_queries': len(read_json(dest)), 'checkpoint': str(ckpt), 'pseudo_norm_in_model': seen['pseudo_norm'],
            'tau': seen['tau'], 'created_at': now(), 'code_commit': git_commit()}


def compare_submissions(new_path, original_path):
    a, b = read_json(new_path), read_json(original_path)
    common = set(a) & set(b)
    identical = sum(a[q] == b[q] for q in common)
    same_set = sum(a[q] != b[q] and set(a[q]) == set(b[q]) for q in common)
    return {'original_file': original_path.name, 'original_sha256': sha256_file(original_path),
            'identical_file': sha256_file(new_path) == sha256_file(original_path),
            'n_queries': len(set(a) | set(b)), 'only_in_new': len(set(a) - set(b)),
            'only_in_original': len(set(b) - set(a)), 'identical_lists': identical,
            'same_set_other_order': same_set, 'different_sets': len(common) - identical - same_set}


def test_file_ok(cmp, tol):
    return cmp['only_in_new'] == 0 and cmp['only_in_original'] == 0 and cmp['n_queries'] - cmp['identical_lists'] <= tol


def new_test_file(ckpt, name, pn_expected, device, out_dir, fields):
    """A test file for the CIRCO server (circo_test/to_upload/)."""
    dest = out_dir / 'circo_test' / 'to_upload' / name
    reg = registry(out_dir)
    if dest.exists() and name in reg:
        log(f'  test file  : already done ({name})')
        return reg[name]
    info = produce_test_file(ckpt, dest, pn_expected, device)
    info.update(fields, server='https://circo.micc.unifi.it/evaluation', server_result=None)
    register(out_dir, name, info)
    log(f'  test file  : to_upload/{name}  ({info["n_queries"]} queries, sha256 {info["sha256"][:16]}...)')
    return info


def recheck_test_file(ckpt, key, pn_expected, device, out_dir):
    """Regenerate an already-submitted test file (circo_test/rechecks/) and compare it with the original."""
    name = f'{key}__regenerated__submission.json'
    dest = out_dir / 'circo_test' / 'rechecks' / name
    reg = registry(out_dir)
    if dest.exists() and name in reg and 'comparison' in reg[name]:
        log(f'  test file  : already re-checked ({name})')
        return reg[name]
    info = produce_test_file(ckpt, dest, pn_expected, device)
    info['comparison'] = compare_submissions(dest, ORIGINAL_SUBMISSIONS / f'{key}__submission.json')
    register(out_dir, name, info)
    c = info['comparison']
    log(f'  test file  : vs original {c["identical_lists"]}/{c["n_queries"]} identical top-50 lists, '
        f'file identical: {"yes" if c["identical_file"] else "no"}')
    return info


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def compare_fashion(rec, ref, tol):
    deltas = {k: rec['avg'][k] - r for k, r in zip(KS, ref)}
    ok = all(abs(d) <= tol for d in deltas.values())
    exact = all(d == 0 for d in deltas.values())
    return ok, exact, deltas


def compare_fashion_printed(rec, e):
    """Fashion-IQ values that survive only as printed text, against what this record prints at the same
    precision: the paper's two-decimal averages and the original log's one-decimal values."""
    rows = [('paper', f'avg R@{k}', ref, f'{100 * rec["avg"][k]:.2f}') for k, ref in e['fashion_paper_2dp'].items()]
    for cat, refs in e['fashion_log_1dp'].items():
        vals = rec['avg'] if cat == 'avg' else rec['per_category'][cat]
        rows += [('original log', f'{cat} R@{k}', ref, f'{100 * vals[k]:.1f}') for k, ref in zip(KS, refs)]
    return [dict(zip(('source', 'value', 'reference', 'now'), r)) for r in rows]


def fmt_fashion_row(name, rec, ref, deltas, status):
    cells = '  '.join(f'R@{k} {100*r:6.3f}->{100*rec["avg"][k]:6.3f} ({100*deltas[k]:+.3f})'
                      for k, r in zip(KS, ref))
    return f'  {name:24s} {cells}  {status}'


def compare_val(now_v2, stored):
    """Per-query comparison of a CIRCO-val result with a stored eval_circo_v2 result."""
    qn, qs = now_v2['per_query']['query_id'], stored['per_query']['query_id']
    pos = {q: i for i, q in enumerate(qs)}
    per_k = {}
    for k in ('@5', '@10', '@25', '@50'):
        a, b = now_v2['per_query']['ap_at_k'][k], stored['per_query']['ap_at_k'][k]
        mism = [q for i, q in enumerate(qn) if q in pos and abs(a[i] - b[pos[q]]) > 1e-9]
        per_k[k] = {'map_now': float(now_v2['map_atk'][k]), 'map_stored': float(stored['map_atk'][k]),
                    'delta': float(now_v2['map_atk'][k]) - float(stored['map_atk'][k]),
                    'n_query_mismatch': len(mism), 'mismatched_queries': mism[:20]}
    return {'n_queries_now': len(qn), 'n_queries_stored': len(qs),
            'missing': sorted(set(qs) - set(qn))[:20], 'unexpected': sorted(set(qn) - set(qs))[:20],
            'per_k': per_k, 'ref10_stickiness_now': now_v2['stickiness']['r10_ref'],
            'ref10_stickiness_stored': stored['stickiness']['r10_ref']}


def val_ok(cmp, args):
    k10 = cmp['per_k']['@10']
    return (not cmp['missing'] and not cmp['unexpected'] and abs(k10['delta']) <= args.circo_tol
            and k10['n_query_mismatch'] <= args.val_query_tol)


def fmt_val(cmp):
    k10 = cmp['per_k']['@10']
    return (f'CIRCO-val mAP@10 stored {100*k10["map_stored"]:.4f} now {100*k10["map_now"]:.4f} '
            f'(delta {100*k10["delta"]:+.4f} pts), AP@10 differs on {k10["n_query_mismatch"]}/{cmp["n_queries_now"]} queries')


def check_known(e, out_dir, args):
    key = key_for(e['path'])
    c, f = read_json(val_file(out_dir, key)), read_json(fiq_file(out_dir, key))
    v = compare_val(c['v2'], read_json(STORED_VAL / f'{key}.json'))
    ok_v = val_ok(v, args)
    if 'fashion_ref' in e:
        ok_f, exact, deltas = compare_fashion(f, e['fashion_ref'], args.fashion_tol)
        fashion = {'reference': dict(zip(KS, e['fashion_ref'])), 'now': f['avg'], 'deltas': deltas, 'exact': exact}
        f_line = fmt_fashion_row('Fashion-IQ (training)', f, e['fashion_ref'], deltas,
                                 'exact' if exact else ('ok' if ok_f else 'FAIL'))
    else:
        rows = compare_fashion_printed(f, e)
        bad = [r for r in rows if r['now'] != r['reference']]
        ok_f = not bad
        fashion = {'reference_paper_2dp': e['fashion_paper_2dp'], 'reference_log_1dp': e['fashion_log_1dp'],
                   'now': f['avg'], 'now_per_category': f['per_category'], 'rows': rows, 'mismatches': bad}
        paper = [r for r in rows if r['source'] == 'paper']
        logged = [r for r in rows if r['source'] == 'original log']
        f_line = (f'  Fashion-IQ paper (2 dp) ' + ', '.join(f'{r["value"]} {r["reference"]} -> {r["now"]}' for r in paper)
                  + f'; original log (1 dp) {sum(r["now"] == r["reference"] for r in logged)}/{len(logged)} '
                  f'values identical  {"ok" if ok_f else "FAIL"}')
        f_line += ''.join(f'\n    differs: {r["source"]} {r["value"]} {r["reference"]} -> now {r["now"]}' for r in bad)
    t = registry(out_dir)[f'{key}__regenerated__submission.json']['comparison']
    ok_t = test_file_ok(t, args.test_query_tol)
    ok_pn = c['pseudo_norm_in_model'] == 0.0
    ok = ok_v and ok_f and ok_t and ok_pn
    log(f'  {e["label"]}: {e["paper"]}')
    log(f'  {fmt_val(v)}  {"ok" if ok_v else "FAIL"}')
    log(f_line)
    log(f'  CIRCO test file vs original: {t["identical_lists"]}/{t["n_queries"]} identical top-50 lists, '
        f'{t["same_set_other_order"]} reordered, {t["different_sets"]} different; file identical: '
        f'{"yes" if t["identical_file"] else "no"}  {"ok" if ok_t else "FAIL"}')
    log(f'  pseudo_norm in model = {c["pseudo_norm_in_model"]}  {"ok" if ok_pn else "FAIL"}')
    log(f'  => {"PASS" if ok else "FAIL"}')
    return ok, {'ok': ok, 'paper': e['paper'], 'circo_val': v, 'fashion': fashion, 'circo_test': t,
                'pseudo_norm_in_model': c['pseudo_norm_in_model']}


def check_gate(args, out_dir):
    rows, ok_all = [], True
    for rel, (ref, source) in GATE.items():
        key = key_for(RUN_DIR / rel)
        f, c = read_json(fiq_file(out_dir, key)), read_json(val_file(out_dir, key))
        ok, exact, deltas = compare_fashion(f, ref, args.fashion_tol)
        ok = ok and c['pseudo_norm_in_model'] == 1.0
        ok_all &= ok
        log(fmt_fashion_row(rel, f, ref, deltas, 'exact' if exact else ('ok' if ok else 'FAIL')))
        rows.append({'checkpoint': rel, 'reference_file': source, 'reference': dict(zip(KS, ref)),
                     'now': f['avg'], 'deltas': deltas, 'exact': exact, 'ok': ok,
                     'pseudo_norm_in_model': c['pseudo_norm_in_model']})
    write_json(out_dir / 'checks' / 'unit_norm_gate.json',
               {'ok': ok_all, 'rows': rows, 'tolerance': args.fashion_tol, 'checked_at': now()})
    return ok_all


# ---------------------------------------------------------------------------
# Selection (paper's rule, Table S1)
# ---------------------------------------------------------------------------
def select(out_dir):
    rows, identical = [], {}
    for rel in UNIT_NORM:
        src = first_with_same_file(rel)
        if src is not None:
            identical[rel] = src
            continue
        key = key_for(RUN_DIR / rel)
        c, f = read_json(val_file(out_dir, key)), read_json(fiq_file(out_dir, key))
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
# summary.csv (rebuilt after every evaluation from the saved JSONs)
# ---------------------------------------------------------------------------
SUMMARY_COLUMNS = [
    'role', 'name', 'checkpoint', 'same_file_as', 'sha256', 'pseudo_norm', 'tau',
    'circo_val_map5', 'circo_val_map10', 'circo_val_map25', 'circo_val_map50',
    'circo_val_r5', 'circo_val_r10', 'circo_val_r25', 'circo_val_r50',
    'ref10_stickiness', 'sim_delta', 'sim_ref_mean', 'sim_gt_mean', 'median_rank_ref',
    'p_gt_target', 'p_gt_set',
    'fiq_r1', 'fiq_r5', 'fiq_r10', 'fiq_r50',
    'fiq_dress_r10', 'fiq_dress_r50', 'fiq_shirt_r10', 'fiq_shirt_r50', 'fiq_toptee_r10', 'fiq_toptee_r50',
    'reference_fiq_r10', 'reference_fiq_r50', 'stored_circo_val_map10', 'val_ap10_query_mismatches',
    'test_file', 'test_vs_original', 'selected',
]


def write_summary(out_dir):
    selected = None
    if (out_dir / 'selection_unit_norm.json').exists():
        selected = read_json(out_dir / 'selection_unit_norm.json')['selected']['checkpoint']
    reg = registry(out_dir)
    known_refs = {e['label']: e.get('fashion_ref') for e in KNOWN}
    rows = []
    for role, name, path, sha, _ in all_entries():
        key = key_for(path)
        row = dict.fromkeys(SUMMARY_COLUMNS, '')
        row.update(role=role, name=name, checkpoint=str(path.relative_to(EXPERIMENT)), sha256=sha,
                   selected='yes' if role in ('gate', 'sweep') and name == selected else '')
        ref = GATE[name][0] if role == 'gate' else known_refs.get(name) if role == 'known' else None
        if ref:
            row.update(reference_fiq_r10=ref[2], reference_fiq_r50=ref[3])
        vp, fp, sp = val_file(out_dir, key), fiq_file(out_dir, key), STORED_VAL / f'{key}.json'
        if vp.exists():
            c = read_json(vp)
            v2, st, pg = c['v2'], c['v2']['stickiness'], c['v1']['stickiness']
            row.update(same_file_as=c.get('same_file_as', ''), pseudo_norm=c['pseudo_norm_in_model'],
                       tau=round(c['tau'], 4), ref10_stickiness=st['r10_ref'], sim_delta=st['sim_delta'],
                       sim_ref_mean=st['sim_ref_mean'], sim_gt_mean=st['sim_gt_mean'],
                       median_rank_ref=st['median_rank_ref'],
                       p_gt_target=pg['p_target_mean'], p_gt_set=pg['p_gtset_mean'])
            for k in ('5', '10', '25', '50'):
                row[f'circo_val_map{k}'] = v2['map_atk'][f'@{k}']
                row[f'circo_val_r{k}'] = v2['recall_atk'][f'@{k}']
            if sp.exists() and role in ('known', 'extra'):
                cmp = compare_val(v2, read_json(sp))
                row.update(stored_circo_val_map10=cmp['per_k']['@10']['map_stored'],
                           val_ap10_query_mismatches=cmp['per_k']['@10']['n_query_mismatch'])
        if fp.exists():
            f = read_json(fp)
            for k in KS:
                row[f'fiq_r{k}'] = f['avg'][k]
            for cat in FIQ_CATS:
                row[f'fiq_{cat}_r10'] = f['per_category'][cat]['10']
                row[f'fiq_{cat}_r50'] = f['per_category'][cat]['50']
        for fname, info in reg.items():
            if info.get('checkpoint') == str(path):
                if info['folder'] == 'to_upload':
                    row['test_file'] = fname
                elif 'comparison' in info:
                    cmpt = info['comparison']
                    row['test_vs_original'] = f'{cmpt["identical_lists"]}/{cmpt["n_queries"]} identical'
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
        import PIL
        info.update(numpy=numpy.__version__, pillow=PIL.__version__)
    except Exception:                                        # noqa: BLE001
        pass
    return info


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out_dir', type=Path, default=REPO_ROOT / 'results' / 'reeval_2026_11')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--preflight_only', action='store_true',
                   help='check packages, GPU, data and checkpoints, then exit')
    p.add_argument('--fashion_tol', type=float, default=FASHION_TOL)
    p.add_argument('--circo_tol', type=float, default=CIRCO_TOL)
    p.add_argument('--val_query_tol', type=int, default=VAL_QUERY_TOL)
    p.add_argument('--test_query_tol', type=int, default=TEST_QUERY_TOL)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    banner(f'Re-evaluation   {now()}')
    log(f'  code   : {REPO_ROOT}  (commit {git_commit()})')
    log(f'  output : {out_dir}')

    # ---- Step 0 -------------------------------------------------------------
    banner('Step 0/6  Preflight')
    problems, warnings, ckpts = preflight()
    for name, e in ckpts.items():
        state = 'ok  ' if e.get('sha256') == e['sha256_expected'] else 'FAIL'
        tau = f'{e["tau"]:.2f}' if 'tau' in e else '-'
        log(f'  {state} {e["role"]:5s} {name:24s} sha256 {e.get("sha256", "missing")[:16]}  '
            f'pseudo_norm {e.get("pseudo_norm", "-")}  tau {tau}')
    for w in warnings:
        log(f'  note: {w}')
    info = environment(args)
    write_json(out_dir / 'preflight.json', {'checked_at': now(), 'environment': info,
                                            'problems': problems, 'warnings': warnings, 'checkpoints': ckpts})
    history = read_json(out_dir / 'run_info.json') if (out_dir / 'run_info.json').exists() else []
    history.append(info)
    write_json(out_dir / 'run_info.json', history)
    if problems:
        log('\nPREFLIGHT FAILED:')
        for pr in problems:
            log(f'  - {pr}')
        sys.exit(1)
    log(f'\n  Preflight passed: {len(ckpts)} checkpoints; torch {info.get("torch")}, CUDA {info.get("cuda")}, '
        f'Pillow {info.get("pillow")}, GPU {info.get("gpu")}')
    if args.preflight_only:
        return

    cfg = fashion_eval_cfg(args.device)

    # ---- Step 1 -------------------------------------------------------------
    banner('Step 1/6  Known models: must reproduce their stored results and original test files')
    known_ok, known_report = True, {}
    for e in KNOWN:
        evaluate_checkpoint(e['path'], 0.0, out_dir, args.device, cfg)
        recheck_test_file(e['path'], key_for(e['path']), 0.0, args.device, out_dir)
    log()
    for e in KNOWN:
        ok, rep = check_known(e, out_dir, args)
        known_ok &= ok
        known_report[e['label']] = rep
        log()
    write_json(out_dir / 'checks' / 'known_models.json',
               {'ok': known_ok, 'models': known_report, 'checked_at': now(),
                'tolerances': {'circo_map10': args.circo_tol, 'fashion': args.fashion_tol,
                               'val_queries': args.val_query_tol, 'test_queries': args.test_query_tol}})
    write_summary(out_dir)
    if not known_ok:
        banner('STOPPED: a known model did not reproduce, so the rebuilt setup differs from the original. '
               'Nothing new was made. Send run_log.txt and checks/known_models.json.')
        sys.exit(2)

    # ---- Step 2 -------------------------------------------------------------
    banner(f'Step 2/6  Gate: {len(GATE)} unit-norm checkpoints vs their training-time Fashion-IQ evals')
    for rel in GATE:
        evaluate_checkpoint(RUN_DIR / rel, 1.0, out_dir, args.device, cfg)
    log()
    if not check_gate(args, out_dir):
        banner('STOPPED: the unit-norm gate did not reproduce. No test file was made. '
               'Send run_log.txt and checks/unit_norm_gate.json.')
        sys.exit(3)

    # ---- Step 3 -------------------------------------------------------------
    remaining = [rel for rel in UNIT_NORM if rel not in GATE]
    banner(f'Step 3/6  Sweep: the remaining {len(remaining)} unit-norm checkpoint files')
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
        if left:
            log(f'  progress   : {i}/{len(remaining)} done, about '
                f'{left * sum(durations) / len(durations) / 60:.0f} min left in the sweep')
    if failures:
        banner(f'STOPPED: {len(failures)} unit-norm checkpoint(s) failed: {failures}. Rerun to retry them.')
        sys.exit(4)

    # ---- Step 4 -------------------------------------------------------------
    banner('Step 4/6  Unit-norm selection: argmax CIRCO-val mAP@10 (paper, Table S1)')
    selection = select(out_dir)
    write_json(out_dir / 'selection_unit_norm.json', selection)
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
    sel = selection['selected']
    banner(f'Step 5/6  CIRCO test file for the selected unit-norm checkpoint ({sel["checkpoint"]})')
    sel_key = key_for(RUN_DIR / sel['checkpoint'])
    new_test_file(RUN_DIR / sel['checkpoint'], f'{sel_key}__with_projection__submission.json', 1.0,
                  args.device, out_dir,
                  {'role': 'unit-norm, selected', 'checkpoint_sha256': UNIT_NORM[sel['checkpoint']],
                   'circo_val_map10': sel['circo_val_map10'], 'fiq_r10': sel['fiq_r10'], 'fiq_r50': sel['fiq_r50'],
                   'selection_rule': selection['rule']})

    # ---- Step 6 -------------------------------------------------------------
    banner(f'Step 6/6  Extras: {len(EXTRAS)} checkpoints the paper still needs numbers for')
    extras_report, extra_failures = {}, []
    for e in EXTRAS:
        key = key_for(e['path'])
        try:
            log(f'\n  {e["label"]}: {e["why"]}')
            evaluate_checkpoint(e['path'], 0.0, out_dir, args.device, cfg)
            rep = {'why': e['why'], 'key': key}
            if (STORED_VAL / f'{key}.json').exists():
                rep['circo_val_vs_stored'] = compare_val(read_json(val_file(out_dir, key))['v2'],
                                                         read_json(STORED_VAL / f'{key}.json'))
                log(f'  {fmt_val(rep["circo_val_vs_stored"])}')
            if (ORIGINAL_SUBMISSIONS / f'{key}__submission.json').exists():
                rep['circo_test_vs_original'] = recheck_test_file(e['path'], key, 0.0, args.device, out_dir)['comparison']
            else:
                c, f = read_json(val_file(out_dir, key)), read_json(fiq_file(out_dir, key))
                rep['new_test_file'] = new_test_file(
                    e['path'], f'{key}__submission.json', 0.0, args.device, out_dir,
                    {'role': e['why'], 'checkpoint_sha256': e['sha256'],
                     'circo_val_map10': float(c['v2']['map_atk']['@10']),
                     'fiq_r10': f['avg']['10'], 'fiq_r50': f['avg']['50']})['file']
            extras_report[e['label']] = rep
        except Exception:                                    # noqa: BLE001
            log(traceback.format_exc())
            log(f'  FAILED: {e["label"]} (continuing with the others)')
            extra_failures.append(e['label'])
            free_gpu()
        write_json(out_dir / 'checks' / 'extras.json', extras_report)
    write_summary(out_dir)

    # ---- Done ---------------------------------------------------------------
    uploads = sorted((out_dir / 'circo_test' / 'to_upload').glob('*__submission.json'))
    banner(f'{"DONE" if not extra_failures else "DONE EXCEPT EXTRAS " + str(extra_failures)} in '
           f'{(time.time() - t_start) / 60:.1f} min   {now()}')
    log(f'  results : {out_dir}')
    log('  upload these at https://circo.micc.unifi.it/evaluation and keep each result email:')
    for u in uploads:
        log(f'    circo_test/to_upload/{u.name}')
    log('  re-checked against the original submissions (no upload needed): circo_test/rechecks/')
    if extra_failures:
        log('  rerun the same command to retry the failed extras')
        sys.exit(5)


if __name__ == '__main__':
    main()

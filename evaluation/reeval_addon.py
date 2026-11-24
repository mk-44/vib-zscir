"""Add-on to reeval.py, run once on the same machine after it: the release check, Table 1's logit-scale
(tau) rows with the provenance of their CIRCO test files, the margin and additive-noise checkpoints the
paper currently reports, and optionally the ~1.5M token studies of the supplement.

Why
    reeval.py (results/reeval_2026_11/) re-evaluated the unit-norm run, two known models and four extra
    checkpoints. Left for this add-on:
    1 release   checkpoints_to_share/{baseline_deterministic,vib_beta0.1,vib_beta0.1_rerun}.pt are
                mapper-only copies of the baseline, the VIB headline and the second VIB run
                (epoch_01_step_03000). Through the fixed loaders each must reproduce its original
                exactly: CIRCO-val AP on every query, Fashion-IQ to 6 dp, and all 800 top-50 lists of
                the original CIRCO test file (same order, so the same server score; byte identity is
                recorded too).
    2 tau       Table 1's Frozen (epoch_00_step_01500), Free (epoch_02) and Capped (epoch_01_step_01000)
                rows took Fashion-IQ from the exploratory pipeline of paper_results/*full_val.csv, which
                differs from the canonical evaluator by up to ~0.1 points; they get canonical numbers.
                Every run below shares the run id base_hd256_tok1, so the four test files
                metrics/circo_test_submissions/base_hd256_tok1__*__submission.json cannot be told apart
                by name. Every checkpoint that could have written one regenerates it; each original must be
                reproduced by its presumed source and by no other candidate (all 800 top-50 lists, or all
                but at most reeval's TEST_QUERY_TOL near-tie flips, reported as 'near').
    3 remedies  the margin (epoch_01_step_01500) and additive-noise (epoch_01_step_00500) checkpoints
                behind the paper's current rows: canonical numbers and a recheck of their test files.
    4 1.5M      (--with_1p5m) the ~1.5M token studies (supplement Sec. D.5, Table S10): CIRCO val and
                Fashion-IQ, no test files. Table S9's per-token ablation needs its own script.

What it runs
    reeval.py's own functions (run_circo_val, run_fashion, recheck_test_file, compare_val), so the
    evaluation code, the cuDNN settings and the file formats are those of the main run. reeval.py's
    results, metrics/ and the original submissions are only read.

Outputs (results/reeval_2026_11/addon/)
    release/, tau/<run>/, remedies/, small_scale/   circo_val/, fashion_val/, circo_test/ as in reeval.py
    entries/<label>.json                            one record per checkpoint: status, checks, numbers
    tau_provenance.json                             which candidates reproduce each original test file
    summary.json                                    all entries, the verdicts and the paper-number table
    preflight.json, run_info.json                   environment, inputs and invocation history

Exit codes: 1 preflight, 2 release check, 3 tau provenance, 4 any other failure. Every entry runs before
a non-zero exit, except after a failed preflight. Rerunning the same command resumes: finished entries
are skipped, and an interrupted one reuses whatever it had already saved.

Usage (after reeval.py has finished, on the same machine and environment)
    PYTHONPATH=/notebooks/experiment/codes python -u evaluation/reeval_addon.py --preflight_only
    PYTHONPATH=/notebooks/experiment/codes python -u evaluation/reeval_addon.py --with_1p5m
"""

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import reeval as R          # noqa: E402  (paths, helpers and evaluation steps of the main run)


# ---------------------------------------------------------------------------
# Checkpoints (paths relative to /notebooks/experiment, sha256 of the archived copy in keep_checkpoints.zip)
# ---------------------------------------------------------------------------
TAU_RUN = {   # every folder whose run id is base_hd256_tok1 and that holds one of the four tags
    'frozen'       : '4M_checkpoints/checkpoints_frozen_ls/base_hd256_tok1',            # tau frozen at 100
    'free'         : '4M_checkpoints/checkpoints_trainable_ls/base_hd256_tok1',         # tau free, -> 144
    'capped'       : '4M_checkpoints/checkpoints_trainable_ls_clipped/base_hd256_tok1', # tau capped at 115
    '1500k'        : 'checkpoints_1500k/base_hd256_tok1',                               # ~1.5M runs: epoch_02 only
    '1500k_caption': 'checkpoints_1500k_caption_aware/base_hd256_tok1',
    '1500k_free'   : 'checkpoints_1500k_trainable_ls/base_hd256_tok1',
}
TAU_TAGS = ('epoch_00_step_01500', 'epoch_02', 'epoch_01_step_01000', 'epoch_00_step_03500')

# Training-time Fashion-IQ eval of the second VIB run at epoch_01_step_03000 (its step_evals/
# step_03000_epoch_01.json in keep_checkpoints.zip; train.py rounds each value to 6 dp).
RERUN_E01S3000_FASHION = {
    'dress' : {'1': 0.036192, '5': 0.11998,  '10': 0.164601, '50': 0.36936},
    'shirt' : {'1': 0.078508, '5': 0.196271, '10': 0.25368,  '50': 0.421982},
    'toptee': {'1': 0.072922, '5': 0.18562,  '10': 0.242733, '50': 0.433452},
    'avg'   : {'1': 0.062541, '5': 0.16729,  '10': 0.220338, '50': 0.408265},
}


def _e(label, group, kind, rel, sha256, out_sub, key, **extra):
    """kind: 'full' = CIRCO val + Fashion-IQ + test file; 'test' = test file only; 'eval' = no test file."""
    entry = {'label': label, 'group': group, 'kind': kind, 'rel': rel, 'sha256': sha256,
             'out_sub': out_sub, 'key': key}
    entry.update(extra)
    return entry


def _csv(source, f10, f50, c10, stick):
    """Values of the exploratory pipeline (keep_rest.zip paper_results/), printed next to the canonical ones."""
    return {'source': source, 'fiq_r10': f10, 'fiq_r50': f50, 'circo_val_map10': c10, 'ref10_stickiness': stick}


RELEASE = [
    _e('release_baseline', 'release', 'full', 'checkpoints_to_share/baseline_deterministic.pt',
       'b9f20cfe7b6a6ae6cb512e538c57f826b9734d430563b86610cb2254570a4d86', 'release',
       'base_hd256_tok1_lslr0.2_lsmax116__epoch_02_step_03000', stored_val=True, fashion_ref='main',
       paper='baseline (Tab. 2, S6): F 24.99 / 45.11, CIRCO val 12.12, test 11.39'),
    _e('release_vib', 'release', 'full', 'checkpoints_to_share/vib_beta0.1.pt',
       '4dc644057cd8d6803a3dfe5ea95cc8e47d0afb31a50fa79b393e395ea55935ed', 'release',
       'vib_hd256_tok1_lslr0.2_lsmax116_vib0.0001__epoch_01_step_01500', stored_val=True, fashion_ref='main',
       paper='VIB beta=0.1 (Tab. 2, S6): F 22.15 / 40.76, CIRCO val 12.38, test 13.22'),
    _e('release_vib_rerun', 'release', 'full', 'checkpoints_to_share/vib_beta0.1_rerun.pt',
       '60d996173a2e572de70ace35767427850e8040d4c2365eccb31861a28469e92e', 'release',
       'vib_hd256_tok1_lslr0.2_lsmax116_vib0.1024_vwarm0.3_seedR__epoch_01_step_03000', stored_val=True,
       fashion_ref=RERUN_E01S3000_FASHION,
       paper='second VIB run e01s3000 (Tab. S14): CIRCO val 11.82, stick 67.3, P(gt) 0.0419, test 13.28; '
             'S14 Fashion 22.00 / 40.91 came from eval_fashion_v2 (training-time 22.03 / 40.83)'),
]

TAU = [
    # Paper rows and presumed sources of the four original test files: full evaluation.
    _e('tau_frozen_e00s1500', 'tau', 'full', f'{TAU_RUN["frozen"]}/epoch_00_step_01500.pt',
       'ad1619417a923e9e5d4de69f23eeb6661dc5846fbe2e3ffa0451494ec0fadf0a', 'tau/frozen',
       'base_hd256_tok1__epoch_00_step_01500', presumed=True, row='Frozen',
       paper='Table 1 Frozen: F 23.4, C val 10.0, C test 8.3',
       csv=_csv('FROZEN_ls__base_hd256_tok1__full_val.csv', 0.2338, 0.4233, 0.1003, 0.4864)),
    _e('tau_free_epoch02', 'tau', 'full', f'{TAU_RUN["free"]}/epoch_02.pt',
       '1e83d3e05a5d6535ddee4e2c5240cef474b489b51042a8debf66bacf48867017', 'tau/free',
       'base_hd256_tok1__epoch_02', presumed=True, row='Free (uncapped)',
       paper='Table 1 Free (uncapped): F 21.9, C val 10.2, C test 8.4',
       csv=_csv('NAIVE_trainable_ls__base_hd256_tok1__full_val.csv', 0.2188, 0.4005, 0.1015, 0.7727)),
    _e('tau_capped_e01s1000', 'tau', 'full', f'{TAU_RUN["capped"]}/epoch_01_step_01000.pt',
       '059db9c10e664ba41c5ed4a8f54df3f5d54bf2b9c08d5dbe8f84d46eab0954c3', 'tau/capped',
       'base_hd256_tok1__epoch_01_step_01000', presumed=True, row='Capped',
       paper='Table 1 Capped (115), fast lr: F 12.6, C val 6.3, C test 5.8',
       csv=_csv('CLIPPED_ls__base_hd256_tok1__full_val.csv', 0.1258, 0.2655, 0.0634, 0.9045)),
    _e('tau_free_e00s3500', 'tau', 'full', f'{TAU_RUN["free"]}/epoch_00_step_03500.pt',
       'ec524f129ca7daa79b3deada014f6611ed088560c7b54a78d1f37d9b4d6bdcfa', 'tau/free',
       'base_hd256_tok1__epoch_00_step_03500', presumed=True, stored_val=True, row='(free, val argmax)',
       paper='no paper row: the free run\'s CIRCO-val argmax; a stored val result and an unexplained '
             'submission carry this name',
       csv=_csv('NAIVE_trainable_ls__base_hd256_tok1__full_val.csv', 0.2386, 0.4216, 0.1220, 0.7364)),
    # Other checkpoints with the same file names: test file only.
    _e('alt_free_e00s1500', 'tau', 'test', f'{TAU_RUN["free"]}/epoch_00_step_01500.pt',
       'f9eb47d9987896f4a9f0e53c26042e9940b6cb4acea00bca9f56c560b7fefde9', 'tau/free',
       'base_hd256_tok1__epoch_00_step_01500'),
    _e('alt_capped_e00s1500', 'tau', 'test', f'{TAU_RUN["capped"]}/epoch_00_step_01500.pt',
       'bcbc6fedeb010f67e2e78e4cf2a712cfb86735b8419c82d04bd91f8c406cb598', 'tau/capped',
       'base_hd256_tok1__epoch_00_step_01500'),
    _e('alt_frozen_epoch02', 'tau', 'test', f'{TAU_RUN["frozen"]}/epoch_02.pt',
       '5da2cf6b7e5f5fe10e8fd6db5ea7d1a8cef442704274c51755ac48f462fcf735', 'tau/frozen',
       'base_hd256_tok1__epoch_02'),
    _e('alt_1500k_epoch02', 'tau', 'test', f'{TAU_RUN["1500k"]}/epoch_02.pt',
       '8c7aeae389da8ea5b9a3962927d19c288de9ff390537c7e11432efde3b029390', 'tau/1500k',
       'base_hd256_tok1__epoch_02'),
    _e('alt_1500k_caption_epoch02', 'tau', 'test', f'{TAU_RUN["1500k_caption"]}/epoch_02.pt',
       '112882994eb65ae4f20d566cd1a4c32ba4eeb7c88a5095cc158d271b3066c8f0', 'tau/1500k_caption',
       'base_hd256_tok1__epoch_02'),
    _e('alt_1500k_free_epoch02', 'tau', 'test', f'{TAU_RUN["1500k_free"]}/epoch_02.pt',
       '5bd2666ceb10c2c4390ab8fdcff5b0f818f92eeb1f6b54d41d747a6e810cf99b', 'tau/1500k_free',
       'base_hd256_tok1__epoch_02'),
    _e('alt_frozen_e01s1000', 'tau', 'test', f'{TAU_RUN["frozen"]}/epoch_01_step_01000.pt',
       '27475ade76ebf84e8084e086ddf6f47f8f8473142170f600e0477a9a4c48aa54', 'tau/frozen',
       'base_hd256_tok1__epoch_01_step_01000'),
    _e('alt_free_e01s1000', 'tau', 'test', f'{TAU_RUN["free"]}/epoch_01_step_01000.pt',
       '7fece38a4503b05f14b69e5cca38597ea937a093e15feba741292db2c9d0d6c2', 'tau/free',
       'base_hd256_tok1__epoch_01_step_01000'),
    _e('alt_frozen_e00s3500', 'tau', 'test', f'{TAU_RUN["frozen"]}/epoch_00_step_03500.pt',
       '02fb13f9462ec5431631cba24e4767b357ea51c5f52dba93fb3a0b47acb67896', 'tau/frozen',
       'base_hd256_tok1__epoch_00_step_03500'),
    _e('alt_capped_e00s3500', 'tau', 'test', f'{TAU_RUN["capped"]}/epoch_00_step_03500.pt',
       'f00bed82b4b71e8d0a735c3b00ca90e59c9c3ccfa12d1ca62e1e4d6bc8537402', 'tau/capped',
       'base_hd256_tok1__epoch_00_step_03500'),
]

REMEDIES = [
    _e('margin_e01s1500', 'remedy', 'full',
       '4M_checkpoints/base_hd256_tok1_margin0.2_mw0.4_lslr0.4_lsmax116/epoch_01_step_01500.pt',
       '21f0286f65293c3ec1732f4f98e612eeb382d4835984f3ca61695bd075bb3e8b', 'remedies',
       'base_hd256_tok1_margin0.2_mw0.4_lslr0.4_lsmax116__epoch_01_step_01500', stored_val=True, row='+ margin',
       paper='Table 1 / S8 margin: F 23.6, C val 10.8 (S8 10.80), stick 65.0, C test 11.4',
       csv=_csv('MARGIN__base_hd256_tok1_m0.2_mw0.4_lslr0.4__full_val.csv', 0.2363, 0.4268, 0.1080, 0.6500)),
    _e('noise_e01s0500', 'remedy', 'full',
       '4M_checkpoints/base_hd256_tok1_lslr0.2_lsmax116_noise0.1/epoch_01_step_00500.pt',
       '4297e667e9cbad945c59e2135281055e99ec12c69db29415ef2f50bc852c45b7', 'remedies',
       'base_hd256_tok1_lslr0.2_lsmax116_noise0.1__epoch_01_step_00500', row='+ additive noise',
       paper='Table 1 / S8 additive noise: F 21.7, C val 8.6 (S8 8.65), stick 87.7, C test 9.1',
       csv=_csv('NOISE__base_hd256_tok1_noise0.1__CORRECTED.csv', 0.2169, 0.3949, 0.0865, 0.8773)),
]

SMALL = [   # ~1.5M images, free logit scale; matched to the supplement by the inspection CSVs.
            # The four mergers have stored per-query CIRCO-val results (metrics/circo/), checked query by query.
    _e('small_tok1_e04', 'small', 'eval', 'checkpoints_1500k_trainable_ls/base_hd256_tok1/epoch_04.pt',
       'd4b6743dab830910ceae1a5390f90fbc43b3f59752a1ad1c2d53defd323d3611', 'small_scale',
       'base_hd256_tok1__epoch_04', row='1 token', paper='D.5: one token, Fashion R@10 24.59',
       csv=_csv('trainable_ls__tok1__inspection.csv', 0.2459, 0.4296, 0.1180, 0.8545)),
    _e('small_tok2_e06', 'small', 'eval', 'checkpoints_1500k_trainable_ls/base_hd256_tok2/epoch_06.pt',
       'edacac1aee78ec8fc7596d3d51586d5885d427babfbfb61f69ce3ebed833e338', 'small_scale',
       'base_hd256_tok2__epoch_06', row='2 tokens', paper='D.5: two tokens, Fashion R@10 24.42',
       csv=_csv('trainable_ls__tok2__inspection.csv', 0.2442, 0.4425, 0.1005, 0.6955)),
    _e('small_tok4_e01', 'small', 'eval', 'checkpoints_1500k_trainable_ls/base_hd256_tok4/epoch_01.pt',
       'cb8c54ce42268b40cee31b7fdf44144e8ccd20cb09f65d738ddfbcf23a4d7c7f', 'small_scale',
       'base_hd256_tok4__epoch_01', row='4 tokens',
       paper='D.5 / Table S9 "all four combined": Fashion R@10 8.80, CIRCO mAP@10 2.78, stick 99.55',
       csv=_csv('trainable_ls__tok4__inspection.csv', 0.0880, 0.1954, 0.0278, 0.9955)),
    _e('small_tok4_merge1_learned_e07', 'small', 'eval', 'checkpoints_1500k_merger/base_hd256_tok4_merge1/epoch_07.pt',
       'fe6fceb45dedae9014d28ccb101efa36115554106625e2734da174296f310eec', 'small_scale',
       'base_hd256_tok4_merge1__epoch_07', stored_val=True, row='tok4->1 learned',
       paper='Table S10 tok4->1 learned: CIRCO mAP@10 10.3, stick 98.2; D.5 text: Fashion R@10 24.1',
       csv=_csv('merger__tok4_merge1_learned__inspection.csv', 0.2409, 0.4180, 0.1027, 0.9818)),
    _e('small_tok4_merge1_weighted_e05', 'small', 'eval',
       'checkpoints_1500k_merger/base_hd256_tok4_merge1_weighted/epoch_05.pt',
       '727ec7680b2a6a40a769e3567ba6966139f3497f61894dfe4521c67a37d3dff5', 'small_scale',
       'base_hd256_tok4_merge1_weighted__epoch_05', stored_val=True, row='tok4->1 weighted',
       paper='Table S10 tok4->1 weighted: CIRCO mAP@10 8.2, stick 55.9',
       csv=_csv('merger__tok4_merge1_weighted__inspection.csv', 0.2202, 0.3989, 0.0825, 0.5591)),
    _e('small_tok2_merge1_learned_e00', 'small', 'eval',
       'checkpoints_1500k_merger/base_hd256_tok2_merge1_learned/epoch_00.pt',
       '21efbf3d345b56afdff4decfddda576c8abaf6894520616ff1ca861514b69d73', 'small_scale',
       'base_hd256_tok2_merge1_learned__epoch_00', stored_val=True, row='tok2->1 learned',
       paper='Table S10 tok2->1 learned: CIRCO mAP@10 8.0, stick 80.5',
       csv=_csv('merger__tok2_merge1_learned__inspection.csv', 0.2353, 0.4200, 0.0802, 0.8045)),
    _e('small_tok2_merge1_weighted_e00', 'small', 'eval',
       'checkpoints_1500k_merger/base_hd256_tok2_merge1_weighted/epoch_00.pt',
       '6180c798b7a5f5d5de84114841e67bfd078b3faa4e79228f8c9f3207a51b6e94', 'small_scale',
       'base_hd256_tok2_merge1_weighted__epoch_00', stored_val=True, row='tok2->1 weighted',
       paper='Table S10 tok2->1 weighted: CIRCO mAP@10 10.0, stick 77.7',
       csv=_csv('merger__tok2_merge1_weighted__inspection.csv', 0.2098, 0.3892, 0.1003, 0.7773)),
]


def all_entries(with_1p5m):
    return RELEASE + TAU + REMEDIES + (SMALL if with_1p5m else [])


def _self_check():
    """Distinct files, and no two entries that would write the same result file."""
    every = all_entries(True)
    assert len({e['label'] for e in every}) == len(every), 'duplicate label'
    assert len({e['sha256'] for e in every}) == len(every), 'two entries point at identical files'
    assert len({(e['out_sub'], e['key']) for e in every}) == len(every), 'two entries share a result file'
    for tag in TAU_TAGS:
        presumed = [e for e in TAU if e['key'] == f'base_hd256_tok1__{tag}' and e.get('presumed')]
        assert len(presumed) == 1, f'exactly one presumed source per original ({tag})'


_self_check()


def ckpt_path(e):
    return R.EXPERIMENT / e['rel']


# ---------------------------------------------------------------------------
# Step 0: preflight
# ---------------------------------------------------------------------------
def environment_problems():
    """Packages, GPU, the clone's location and the evaluation data (the checks reeval.py's preflight makes)."""
    problems = []
    try:
        import torch
        if not torch.cuda.is_available():
            problems.append('No CUDA GPU is visible to PyTorch.')
    except Exception as e:                                   # noqa: BLE001
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
    if R.REPO_ROOT != R.CODES_ROOT.resolve():
        problems.append(f'This clone is at {R.REPO_ROOT}, but the loaders hardcode {R.CODES_ROOT}.')
    if not R.CONFIG_YAML.exists():
        problems.append(f'Missing {R.CONFIG_YAML}')
    needed = [R.CIRCO_DIR / 'circo_repo' / 'src' / 'dataset.py', R.CIRCO_DIR / 'circo_repo' / 'src' / 'evaluation.py',
              R.GALLERY, R.FIQ_DIR / 'images' / 'images']
    needed += [R.FIQ_DIR / f'gallery_cache_{c}_ViT-L-14.pt' for c in R.FIQ_CATS]
    needed += [R.FIQ_DIR / 'captions' / f'cap.{c}.val.json' for c in R.FIQ_CATS]
    needed += [R.FIQ_DIR / 'image_splits' / f'split.{c}.val.json' for c in R.FIQ_CATS]
    problems += [f'Missing data: {p}' for p in needed if not p.exists()]
    problems += R.query_image_problems()
    return problems


def input_problems(entries, main_out):
    """Files this run compares against: the main run's verified setup and results, stored metrics."""
    problems = []
    known = main_out / 'checks' / 'known_models.json'
    if not known.exists():
        problems.append(f'Missing {known}: run evaluation/reeval.py first (its Step 1 verifies this setup)')
    elif not R.read_json(known).get('ok'):
        problems.append(f'{known} says the known models did not reproduce: the setup is not verified')
    for e in entries:
        if e.get('stored_val') and not (R.STORED_VAL / f'{e["key"]}.json').exists():
            problems.append(f'Missing stored result {R.STORED_VAL / (e["key"] + ".json")}')
        if e['kind'] in ('full', 'test') and not (R.ORIGINAL_SUBMISSIONS / f'{e["key"]}__submission.json').exists():
            problems.append(f'Missing original submission {e["key"]}__submission.json')
        if e.get('fashion_ref') == 'main' and not R.fiq_file(main_out, e['key']).exists():
            problems.append(f'Missing {R.fiq_file(main_out, e["key"])} from the main run')
    return problems


def load_meta(path):
    """What the loaders will read from a checkpoint (torch.load, CPU)."""
    import torch
    ck = torch.load(path, map_location='cpu')
    cfg = ck.get('config', {}) or {}
    meta = {'has_mapper_state': 'mapper_state' in ck, 'mapper_type': ck.get('mapper_type'),
            'pseudo_norm': float(cfg.get('pseudo_norm', 0.0) or 0.0), 'num_tokens': int(cfg.get('num_tokens', 1) or 1),
            'merge_tokens': cfg.get('merge_tokens'), 'merge_type': cfg.get('merge_type', 'learned'),
            'has_merger_state': ck.get('merger_state') is not None,
            'tau': math.exp(float(ck['logit_scale'])) if ck.get('logit_scale') is not None else None}
    del ck
    return meta


def checkpoint_problems(entries):
    rows, problems = {}, []
    for e in entries:
        p = ckpt_path(e)
        row = {'group': e['group'], 'kind': e['kind'], 'path': str(p), 'sha256_expected': e['sha256']}
        rows[e['label']] = row
        if not p.exists():
            problems.append(f'Missing checkpoint {p}: unzip paperspace_addon_bundle.zip into {R.EXPERIMENT}')
            continue
        row['sha256'] = R.sha256_file(p)
        if row['sha256'] != e['sha256']:
            problems.append(f'{p}: sha256 {row["sha256"][:16]}... differs from the archived copy {e["sha256"][:16]}...')
            continue
        try:
            meta = load_meta(p)
        except Exception as ex:                              # noqa: BLE001
            problems.append(f'{p}: torch.load failed: {ex}')
            continue
        row.update(meta)
        if not meta['has_mapper_state'] or not meta['mapper_type']:
            problems.append(f'{p}: no mapper_state / mapper_type entry')
        if meta['pseudo_norm'] != 0.0:
            problems.append(f'{p}: pseudo_norm={meta["pseudo_norm"]}, expected absent or 0.0')
        wants_merger = meta['merge_tokens'] is not None and meta['merge_tokens'] < meta['num_tokens']
        if wants_merger != meta['has_merger_state']:
            problems.append(f'{p}: merge_tokens={meta["merge_tokens"]} but merger_state '
                            f'{"present" if meta["has_merger_state"] else "missing"}')
    return rows, problems


# ---------------------------------------------------------------------------
# One checkpoint
# ---------------------------------------------------------------------------
def entry_file(out_dir, label):
    return out_dir / 'entries' / f'{label}.json'


def numbers(c, f):
    v2, v1 = c['v2'], c['v1']
    exact = {k: sum(f['per_category'][cat][k] for cat in R.FIQ_CATS) / len(R.FIQ_CATS) for k in R.KS}
    return {'circo_val_map': {k: float(v2['map_atk'][f'@{k}']) for k in ('5', '10', '25', '50')},
            'circo_val_recall': {k: float(v2['recall_atk'][f'@{k}']) for k in ('5', '10', '25', '50')},
            'ref10_stickiness': float(v2['stickiness']['r10_ref']), 'sim_delta': float(v2['stickiness']['sim_delta']),
            'p_gt': float(v1['stickiness']['p_target_mean']), 'tau': c['tau'],
            'pseudo_norm_in_model': c['pseudo_norm_in_model'],
            'fiq_avg_exact': exact,                    # mean of the unrounded per-category recalls: print this
            'fiq_avg_train_style': f['avg'],           # train.py's 6-dp averaging: compare with step_evals
            'fiq_per_category': f['per_category']}


def fashion_reference(e, main_out):
    ref = e.get('fashion_ref')
    if ref == 'main':
        rec = R.read_json(R.fiq_file(main_out, e['key']))
        return dict(rec['per_category'], avg=rec['avg']), f'main run {R.fiq_file(main_out, e["key"]).name}'
    return ref, 'training-time step_evals'


def fashion_check(f, ref):
    rows = []
    for cat in R.FIQ_CATS + ('avg',):
        for k in R.KS:
            now = f['avg'][k] if cat == 'avg' else round(f['per_category'][cat][k], 6)
            want = round(float(ref[cat][k]), 6)
            rows.append({'value': f'{cat} R@{k}', 'reference': want, 'now': now, 'same': now == want})
    return all(r['same'] for r in rows), rows


def val_check(c, key):
    cmp = R.compare_val(c['v2'], R.read_json(R.STORED_VAL / f'{key}.json'))
    ok = (not cmp['missing'] and not cmp['unexpected'] and cmp['n_queries_now'] == cmp['n_queries_stored']
          and all(v['n_query_mismatch'] == 0 for v in cmp['per_k'].values()))
    return ok, cmp


def test_match(cmp):
    """'exact': all 800 top-50 lists identical and in order (same server score, even if the JSON formatting
    differs); 'near': at most reeval's TEST_QUERY_TOL lists differ (near-tie flips); else 'different'."""
    if cmp['only_in_new'] or cmp['only_in_original']:
        return 'different'
    differ = cmp['n_queries'] - cmp['identical_lists']
    return 'exact' if differ == 0 else 'near' if differ <= R.TEST_QUERY_TOL else 'different'


def evaluate_entry(e, out_dir, main_out, args, cfg):
    ckpt, od, key = ckpt_path(e), out_dir / e['out_sub'], e['key']
    rec = {'label': e['label'], 'group': e['group'], 'kind': e['kind'], 'checkpoint': str(ckpt),
           'sha256': e['sha256'], 'key': key, 'out_dir': str(od), 'paper': e.get('paper'), 'row': e.get('row'),
           'csv': e.get('csv'), 'presumed_source_of': f'{key}__submission.json' if e.get('presumed') else None,
           'started_at': R.now()}
    checks = {}
    c = f = None
    if e['kind'] in ('full', 'eval'):
        c = R.run_circo_val(ckpt, key, 0.0, od, args.device)
        f = R.run_fashion(ckpt, key, od, cfg)
        rec['numbers'] = numbers(c, f)
        checks['pseudo_norm_off'] = {'ok': c['pseudo_norm_in_model'] == 0.0, 'now': c['pseudo_norm_in_model']}
    if e['kind'] in ('full', 'test'):
        t = R.recheck_test_file(ckpt, key, 0.0, args.device, od)
        match = test_match(t['comparison'])
        rec['test'] = {'file': t['file'], 'sha256': t['sha256'], 'n_queries': t['n_queries'], 'match': match,
                       'comparison': t['comparison']}
        if e['group'] == 'release':      # a release copy must give exactly the original's rankings
            checks['test_lists_identical'] = {'ok': match == 'exact', 'match': match,
                                              'file_identical': t['comparison']['identical_file']}
        elif e['group'] == 'remedy':
            checks['test_reproduced'] = {'ok': match in ('exact', 'near'), 'match': match,
                                         'file_identical': t['comparison']['identical_file']}
    if e.get('stored_val'):
        ok, cmp = val_check(c, key)
        checks['circo_val_per_query'] = {'ok': ok, 'comparison': cmp}
    if e.get('fashion_ref') is not None:
        ref, source = fashion_reference(e, main_out)
        ok, rows = fashion_check(f, ref)
        checks['fashion_6dp'] = {'ok': ok, 'reference': source, 'rows': rows}
    rec.update(checks=checks, ok=all(v['ok'] for v in checks.values()), status='done', finished_at=R.now())
    return rec


def log_entry(rec):
    ch = rec.get('checks', {})
    if 'circo_val_per_query' in ch:
        R.log(f'  {R.fmt_val(ch["circo_val_per_query"]["comparison"])}  '
              f'{"ok" if ch["circo_val_per_query"]["ok"] else "DIFFERS"}')
    if 'fashion_6dp' in ch:
        rows = ch['fashion_6dp']['rows']
        R.log(f'  Fashion-IQ vs {ch["fashion_6dp"]["reference"]}: {sum(r["same"] for r in rows)}/{len(rows)} '
              f'values identical at 6 dp  {"ok" if ch["fashion_6dp"]["ok"] else "DIFFERS"}')
        for r in rows:
            if not r['same']:
                R.log(f'    differs: {r["value"]} {r["reference"]:.6f} -> now {r["now"]:.6f}')
    if 'test' in rec:
        cmp = rec['test']['comparison']
        R.log(f'  test file  : vs {cmp["original_file"]}: {cmp["identical_lists"]}/{cmp["n_queries"]} identical '
              f'top-50 lists ({rec["test"]["match"]}), file identical: {"yes" if cmp["identical_file"] else "no"}')
    if rec.get('checks'):
        R.log(f'  => {rec["label"]}: {"PASS" if rec["ok"] else "FAIL"}')


def run_entry(e, out_dir, main_out, args, cfg, position):
    path = entry_file(out_dir, e['label'])
    if path.exists():
        rec = R.read_json(path)
        if rec.get('status') == 'done':
            R.log(f'  {position} {e["label"]}: already done ({"PASS" if rec["ok"] else "FAIL"})')
            return rec
    R.log()
    R.log(f'--- {position} {e["label"]}: {e["rel"]}   [{R.now()}]')
    if e.get('paper'):
        R.log(f'  paper      : {e["paper"]}')
    try:
        rec = evaluate_entry(e, out_dir, main_out, args, cfg)
    except Exception:                                        # noqa: BLE001
        R.log(traceback.format_exc())
        R.log(f'  FAILED: {e["label"]} (continuing with the others; rerun the same command to retry it)')
        rec = {'label': e['label'], 'group': e['group'], 'kind': e['kind'], 'checkpoint': str(ckpt_path(e)),
               'key': e['key'], 'status': 'failed', 'ok': False, 'error': traceback.format_exc(limit=4),
               'finished_at': R.now()}
        R.free_gpu()
    R.write_json(path, rec)
    if rec['status'] == 'done':
        log_entry(rec)
    return rec


# ---------------------------------------------------------------------------
# tau provenance
# ---------------------------------------------------------------------------
def tau_provenance(recs):
    rows = []
    for tag in TAU_TAGS:
        key = f'base_hd256_tok1__{tag}'
        cands = [e for e in TAU if e['key'] == key]
        presumed = next(e['label'] for e in cands if e.get('presumed'))
        matched, differ, unchecked, how = [], [], [], {}
        for e in cands:
            r = recs.get(e['label'])
            if not r or r.get('status') != 'done':
                unchecked.append(e['label'])
                continue
            how[e['label']] = {'match': r['test']['match'], 'file_identical': r['test']['comparison']['identical_file'],
                               'identical_lists': r['test']['comparison']['identical_lists']}
            (matched if r['test']['match'] in ('exact', 'near') else differ).append(e['label'])
        ok = matched == [presumed] and not unchecked
        rows.append({'original': f'{key}__submission.json', 'presumed_source': presumed,
                     'presumed_rel': next(e['rel'] for e in cands if e['label'] == presumed),
                     'identical': matched, 'different': differ, 'not_checked': unchecked, 'per_candidate': how,
                     'ok': ok})
    return all(r['ok'] for r in rows), rows


def log_provenance(rows):
    for r in rows:
        if r['ok']:
            h = r['per_candidate'][r['presumed_source']]
            R.log(f'  {r["original"]:52s} <- {r["presumed_source"]}: {h["identical_lists"]}/800 lists identical '
                  f'({h["match"]}, file identical: {"yes" if h["file_identical"] else "no"}); '
                  f'{len(r["different"])} other candidate(s) differ  ok')
        else:
            got = ', '.join(r['identical']) or 'no candidate'
            R.log(f'  {r["original"]:52s} reproduced by {got}; presumed {r["presumed_source"]}'
                  f'{" (not checked: " + ", ".join(r["not_checked"]) + ")" if r["not_checked"] else ""}  FAIL')


# ---------------------------------------------------------------------------
# Paper-number table
# ---------------------------------------------------------------------------
def _p(x, dp):
    return '-' if x is None else f'{100 * x:.{dp}f}'


def paper_table(recs, entries):
    rows = []
    for e in entries:
        r = recs.get(e['label'])
        if e['kind'] == 'test' or not r or r.get('status') != 'done':
            continue
        n = r['numbers']
        row = {'label': e['label'], 'group': e['group'], 'row': e.get('row'), 'checkpoint': e['rel'],
               'fiq_r10': n['fiq_avg_exact']['10'], 'fiq_r50': n['fiq_avg_exact']['50'],
               'circo_val_map10': n['circo_val_map']['10'], 'ref10_stickiness': n['ref10_stickiness'],
               'p_gt': n['p_gt'], 'tau': n['tau'], 'paper': e.get('paper'), 'csv': e.get('csv')}
        if 'test' in r:
            row['test_match'] = r['test']['match']
            row['test_file_identical_to_original'] = r['test']['comparison']['identical_file']
            row['original_test_file'] = r['test']['comparison']['original_file']
        rows.append(row)
    return rows


def log_paper_table(rows):
    head = (f'  {"entry":32s} {"F R@10":>7s} {"(1dp)":>6s} {"F R@50":>7s} {"C val@10":>8s} {"(1dp)":>6s} '
            f'{"stick":>6s} {"P(gt)":>7s} {"tau":>7s}  test file / csv (F10, C10, stick)')
    for group, title in (('release', 'Release copies (must equal their originals)'),
                         ('tau', "Table 1 logit-scale rows (canonical; the paper prints one decimal)"),
                         ('remedy', "Table 1 / S8 remedies at the paper's current checkpoints"),
                         ('small', '~1.5M token studies (Sec. D.5, Table S10; two decimals in the paper)')):
        g = [r for r in rows if r['group'] == group]
        if not g:
            continue
        R.log(f'\n  {title}')
        R.log(head)
        for r in g:
            test = ''
            if 'test_match' in r:
                test = {'exact': 'test: reproduces original', 'near': 'test: reproduces original (near)',
                        'different': 'test: DIFFERS'}[r['test_match']]
            csv = r.get('csv')
            ref = (f'csv {100*csv["fiq_r10"]:.2f}, {100*csv["circo_val_map10"]:.2f}, '
                   f'{100*csv["ref10_stickiness"]:.1f}') if csv else ''
            R.log(f'  {r["label"]:32s} {_p(r["fiq_r10"], 2):>7s} {_p(r["fiq_r10"], 1):>6s} {_p(r["fiq_r50"], 2):>7s} '
                  f'{_p(r["circo_val_map10"], 2):>8s} {_p(r["circo_val_map10"], 1):>6s} '
                  f'{_p(r["ref10_stickiness"], 1):>6s} {r["p_gt"]:7.4f} {r["tau"]:7.2f}  {"; ".join(x for x in (test, ref) if x)}')
            if r.get('paper'):
                R.log(f'  {"":32s} paper: {r["paper"]}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out_dir', type=Path, default=REPO_ROOT / 'results' / 'reeval_2026_11' / 'addon')
    p.add_argument('--main_out', type=Path, default=REPO_ROOT / 'results' / 'reeval_2026_11',
                   help="reeval.py's output folder (read only)")
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--with_1p5m', action='store_true', help='also run the ~1.5M token studies (Step 4)')
    p.add_argument('--preflight_only', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    out_dir, main_out = args.out_dir, args.main_out
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    entries = all_entries(args.with_1p5m)

    R.banner(f'Re-evaluation add-on   {R.now()}')
    R.log(f'  code     : {R.REPO_ROOT}  (commit {R.git_commit()})')
    R.log(f'  output   : {out_dir}')
    R.log(f'  main run : {main_out}  (read only)')
    R.log(f'  entries  : {len(entries)} checkpoints{" (with the ~1.5M studies)" if args.with_1p5m else ""}')

    # ---- Step 0 -------------------------------------------------------------
    R.banner('Step 0/4  Preflight')
    problems = environment_problems() + input_problems(entries, main_out)
    ckpts, ck_problems = checkpoint_problems(entries)
    problems += ck_problems
    for label, row in ckpts.items():
        state = 'ok  ' if row.get('sha256') == row['sha256_expected'] else 'FAIL'
        tau = f'{row["tau"]:.2f}' if row.get('tau') else '-'
        R.log(f'  {state} {row["group"]:7s} {label:32s} sha256 {row.get("sha256", "missing")[:16]}  '
              f'tokens {row.get("num_tokens", "-")}  merge {row.get("merge_tokens", "-")}  tau {tau}')
    info = R.environment(args)
    R.write_json(out_dir / 'preflight.json', {'checked_at': R.now(), 'environment': info, 'problems': problems,
                                              'checkpoints': ckpts})
    history = R.read_json(out_dir / 'run_info.json') if (out_dir / 'run_info.json').exists() else []
    history.append(info)
    R.write_json(out_dir / 'run_info.json', history)
    if problems:
        R.log('\nPREFLIGHT FAILED:')
        for pr in problems:
            R.log(f'  - {pr}')
        sys.exit(1)
    R.log(f'\n  Preflight passed: {len(ckpts)} checkpoints; torch {info.get("torch")}, GPU {info.get("gpu")}')
    if args.preflight_only:
        return

    cfg = R.fashion_eval_cfg(args.device)
    recs = {}

    def run_group(group_entries):
        for i, e in enumerate(group_entries, 1):
            recs[e['label']] = run_entry(e, out_dir, main_out, args, cfg, f'[{i}/{len(group_entries)}]')

    # ---- Step 1 -------------------------------------------------------------
    R.banner('Step 1/4  Release check: checkpoints_to_share/ must reproduce their originals exactly')
    run_group(RELEASE)

    # ---- Step 2 -------------------------------------------------------------
    R.banner(f'Step 2/4  Table 1 logit-scale rows, and which checkpoint wrote each base_hd256_tok1__* test file '
             f'({len(TAU)} candidates)')
    run_group(TAU)
    R.log()
    tau_ok, prov = tau_provenance(recs)
    R.write_json(out_dir / 'tau_provenance.json', {'ok': tau_ok, 'rows': prov, 'checked_at': R.now()})
    log_provenance(prov)

    # ---- Step 3 -------------------------------------------------------------
    R.banner("Step 3/4  Margin and additive-noise checkpoints behind the paper's current rows")
    run_group(REMEDIES)

    # ---- Step 4 -------------------------------------------------------------
    if args.with_1p5m:
        R.banner('Step 4/4  ~1.5M token studies (Sec. D.5, Table S10): CIRCO val and Fashion-IQ')
        run_group(SMALL)
    else:
        R.banner('Step 4/4  ~1.5M token studies: skipped (pass --with_1p5m to run them)')

    # ---- Verdict ------------------------------------------------------------
    def bad(group):
        return [e['label'] for e in entries if e['group'] == group and not recs.get(e['label'], {}).get('ok')]

    release_bad, tau_bad = bad('release'), bad('tau')
    other_bad = bad('remedy') + bad('small')
    code = 2 if release_bad else 3 if (tau_bad or not tau_ok) else 4 if other_bad else 0
    table = paper_table(recs, entries)
    R.banner('Numbers for the paper (Fashion-IQ is the exact mean of the per-category recalls)')
    log_paper_table(table)
    R.write_json(out_dir / 'summary.json', {
        'finished_at': R.now(), 'exit_code': code, 'with_1p5m': args.with_1p5m,
        'failed': {'release': release_bad, 'tau': tau_bad, 'tau_provenance_ok': tau_ok, 'other': other_bad},
        'tau_provenance': prov, 'paper_table': table, 'entries': recs})

    verdict = {0: 'DONE: every check passed', 2: f'RELEASE CHECK FAILED: {release_bad}',
               3: f'TAU PROVENANCE FAILED: {tau_bad or "see tau_provenance.json"}',
               4: f'DONE EXCEPT: {other_bad}'}[code]
    R.banner(f'{verdict}   {(time.time() - t_start) / 60:.1f} min   {R.now()}')
    R.log(f'  results : {out_dir}  (summary.json, tau_provenance.json, entries/)')
    if code:
        R.log('  rerun the same command to retry failed entries; finished ones are skipped')
        sys.exit(code)


if __name__ == '__main__':
    main()

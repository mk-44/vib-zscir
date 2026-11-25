"""Final evaluation-only run for the arXiv version: the canonical numbers the paper still needs, the token-norm /
sigma / KL probes, a Table S13 input-regime control, and a dump of query embeddings and model internals for the
CPU analyses on the Mac. Run once on the A6000 machine, after evaluation/reeval.py and evaluation/reeval_addon.py,
in the same environment. Nothing is trained.

Why
    Every paper row now uses one checkpoint rule (argmax CIRCO-val mAP@10), and the mechanism section answers a
    reviewer with new metrics. Still missing: (1) canonical CIRCO-val / Fashion-IQ numbers and CIRCO test files for
    several best-val rows; (2) a source for every token norm, sigma and KL value the paper prints; (3) raw reference
    features and query variants (token-only, text-only, mean token, swaps), which need the CIRCO images and a GPU;
    (4) a check of the input regime behind Table S13 (neg_analysis_all.py feeds unit-norm cached features to the
    mapper, which was trained on raw CLIP features of norm ~18.5).

What it runs
    Canonical numbers: reeval.py's own functions (run_circo_val, run_fashion, new_test_file, recheck_test_file,
    produce_test_file, compare_submissions) and reeval_addon.py's checks, so evaluators, cuDNN settings and file
    formats are those of the two previous runs. Probes: the functions of probe_token_norm.py, probe_sigma.py and
    probe_kl.py on their own reference set. S13 control: neg_analysis_all.py's load_dataset, load_model and
    compute_metrics. Dump: the model's own pieces (clip.encoder_image_unnorm, mapper, merger, pseudo_norm,
    _build_token_sequence, clip.forward_text_transform). Composed queries keep the canonical batch structure
    (CIRCO one query at a time, as eval_circo_v2 / eval_circo_test; Fashion-IQ in eval_fashion's batches), so the
    self-check expects exact agreement with the canonical results (Fashion-IQ references made on the earlier
    machine may differ by near-tie flips, see FIQ_QUERY_TOL).

Steps (a failed preflight or gate stops the run; after that every item runs and failures are recorded)
    0 preflight  packages and versions, GPU, data, the previous runs' results and stored metrics, every checkpoint:
                 present, sha256 equal to the archived copy, mapper type / tokens / merger / pseudo_norm as expected
    1 gate       baseline e02s3000 and VIB e01s1500 reproduce results/reeval_2026_11: every CIRCO-val query exactly
                 (AP and recall at every K, reference rank, best-GT rank); Fashion-IQ within FIQ_QUERY_TOL queries per
                 category at every K (it encodes its reference images on the GPU in every run, and another machine's
                 rounding can reorder a near-tie at a rank cut; CIRCO uses cached image features)
    2 canonical  CIRCO val + Fashion-IQ (+ CIRCO test file) for: capped tau e00s1500 (its test file must equal the
                 add-on's, which was uploaded); beta=0.5 e01s2500 and beta=1.0 e01s1000 (epoch-1 resumed with KL
                 weight 5e-4 / 1e-3 under the averaged-KL code; new test files); KL_mu-only e01s1000 (test recheck);
                 the ~1.5M token study at its val argmax (tok1 epoch_01, tok2 epoch_08, tok4 epoch_00; no test
                 files); the matched end-of-epoch-0 beta sweep (beta 0 / 0.1 / 0.2 / 0.5 / 1.0, summed-KL code; new
                 test files); beta=0.2 e02s3000 and baseline epoch_01 (test rechecks), which the dump self-check
                 needs as canonical references
    3 probes     token norms (20 checkpoints) and sigma / KL (the 9 VIB-type ones) on the 220 CIRCO-val references
    4 dump       G0: model-independent encodes (raw reference features v_raw, text-only and "a photo of" text
                 features), then per model the tokens (VIB: log-variances and per-reference KL) and the query
                 variants V1 composed, V2 token-only, V3 token-only on unit-norm input (S13 regime), V4 mean token +
                 caption, V5 caption swap, V6 reference swap (V4-V6 for the 8 core models); self-check per model
    5 sampled    S1: 8 seeded posterior samples z = mu + sigma*eps -> composed CIRCO queries for VIB, run 2 and
                 beta=0.2, plus a matched-noise control for the baseline
    6 attention  G10: EOT-row attention mass per text layer on SOT / prompt words / pseudo-token / modifier / EOT
    7 S13        neg_analysis_all.py's own pipeline on eval_data/circo (unit-norm cached features), baseline and VIB
    8 summary    summary.json, the paper-number table, the CIRCO upload list with sha256, dump/MANIFEST.sha256

Outputs (results/final_2026_11/; dump/ holds the big arrays and is packed separately)
    entries/<label>.json            one record per item: status, checks, numbers
    gate/, canonical/<group>/       circo_val/ and fashion_val/ records in reeval.py's format
    circo_test/to_upload/           NEW CIRCO test files for the server; to_upload.sha256 lists them
    circo_test/rechecks/            regenerated copies of already-submitted files; circo_test/test_files.json
    probes/                         token_norm_probe.json, sigma_probe.json, kl_probe.json (the probe scripts'
                                    formats), per_reference.json
    s13/s13_control.json            the S13 control
    checks/                         gate.json, selfcheck.json
    summary.json, preflight.json, run_info.json, run_log.txt (the nohup redirect)
    dump/                           dump_info.json, shared/, models/<label>/, sampled/<label>/, attention/<label>/,
                                    MANIFEST.sha256 (sha256sum format)

Exit codes: 1 preflight, 2 gate, 3 dump self-check, 4 any other failure, 0 everything passed. Every item runs before
a non-zero exit, except after a failed preflight or gate. Rerunning the same command resumes: finished items are
skipped (the gate is re-compared from its saved measurements), failed ones are retried.

Usage (repo at /notebooks/experiment/codes; the three bundles unpacked into /notebooks/experiment)
    PYTHONPATH=/notebooks/experiment/codes python -u evaluation/final_run.py --preflight_only
    PYTHONPATH=/notebooks/experiment/codes python -u evaluation/final_run.py
"""

import argparse
import contextlib
import gc
import math
import os
import platform
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import reeval as R          # noqa: E402  paths, helpers and evaluation steps of the main run
from evaluation import reeval_addon as A    # noqa: E402  the add-on's checks (numbers, val_check, fashion_check, ...)


# ---------------------------------------------------------------------------
# Fixed inputs
# ---------------------------------------------------------------------------
REEVAL_OUT    = REPO_ROOT / 'results' / 'reeval_2026_11'        # reeval.py's results (read only)
ADDON_OUT     = REEVAL_OUT / 'addon'                             # reeval_addon.py's results (read only)
ALL_CIRCO     = REPO_ROOT / 'all_model_comparison' / 'all_circo_metrics.json'   # May 2026 sweep, eval_circo.py
ALL_CIRCO_PREFIX = '/notebooks/experiment/'                      # its keys are absolute checkpoint paths
STORED_PROBES = {'token_norm': REPO_ROOT / 'metrics' / 'token_norm_probe.json',
                 'sigma': REPO_ROOT / 'metrics' / 'sigma_probe.json',
                 'kl': REPO_ROOT / 'metrics' / 'kl_probe.json'}
STORED_S13    = REPO_ROOT / 'metrics' / 'neg_structure' / 'circo'
PAPER_S13     = {'baseline': 0.167, 'vib': 0.210}   # Table S13: token-only query-to-source similarity
CLIP_WEIGHTS  = (Path(os.path.expanduser('~/.cache/clip')) / 'ViT-L-14.pt', 932768134)
EXPECTED_VERSIONS = {'python': '3.11.7', 'torch': '2.1.1+cu121', 'torchvision': '0.16.1+cu121', 'numpy': '1.26.3',
                     'pillow': '9.5.0', 'clip_commit': 'd05afc436d78f1c48dc0dbf8e5980a9d471f35f6'}
MIN_FREE_GB   = 4.0      # the dump is ~1.3 GB and its tarball about as much again

BATCH         = 256      # text-encoder batch of the batched variants (V2-V6, text-only, S1)
ATTN_BATCH    = 128
SWAP_SEEDS    = (1, 2)   # derangements for V5 / V6
S1_SAMPLES    = 8
S1_SEED0      = 101      # sample k draws eps from numpy.random.default_rng(S1_SEED0 + k)
SELF_N        = 16       # encode_from_vraw vs model.encode_query on this many CIRCO-val queries
SELF_EQ_TOL   = 1e-5     # max |encode_from_vraw - encode_query|
SELF_MAP_TOL  = 5e-4     # |delta CIRCO-val mAP@10| (fraction, = 0.05 points)
SELF_REF10_TOL = 1       # |delta Ref@10| in queries
FIQ_N         = {'dress': 2017, 'shirt': 2038, 'toptee': 1961}   # Fashion-IQ val queries per category
FIQ_QUERY_TOL = 2        # Fashion-IQ queries per category (gate: every K; dump self-check: R@10) that may differ from a
                         # reference made on the earlier machine: near-tie flips at a rank cut, 0.1 points
ATTN_TOL      = 1e-5     # attention-capturing pass vs normal pass, max |delta q|
PROBE_TOL     = 1.5e-4   # stored probe values are rounded to 4 dp
REIMPL_RTOL   = 1e-6     # this file's token / sigma / KL formulas vs the probe scripts' on identical inputs
GROUPS        = ('sot', 'prompt', 'pseudo', 'modifier', 'eot')


# ---------------------------------------------------------------------------
# Checkpoints (paths relative to /notebooks/experiment, sha256 of the archived copy in keep_checkpoints.zip)
# bundle: paperspace_bundle.zip (main), paperspace_addon_bundle.zip (addon), paperspace_final_bundle.zip (final)
# ---------------------------------------------------------------------------
BUNDLES = {'main': 'paperspace_bundle.zip', 'addon': 'paperspace_addon_bundle.zip',
           'final': 'paperspace_final_bundle.zip'}


def _ck(rel, sha256, mapper, title, pn=0.0, tokens=1, bundle='final'):
    return {'rel': rel, 'sha256': sha256, 'mapper': mapper, 'title': title, 'pn': pn, 'tokens': tokens,
            'bundle': bundle}


_B = '4M_checkpoints/base_hd256_tok1_lslr0.2_lsmax116'
_V = '4M_checkpoints/vib_hd256_tok1_lslr0.2_lsmax116_vib'
_N = '4M_checkpoints/base_hd256_tok1_lslr0.2_lsmax116_noise0.1'
_RESUMED = 'epoch-1 resumed with KL weight {w} (averaged-KL code)'
CK = {
    'baseline': _ck(f'{_B}/epoch_02_step_03000.pt',
                    'b6618d3e0da5f6aecb59e2ea08b272caaed87f2b8829af1de2321c28dcc39d4f', 'base',
                    'baseline (bounded tau) e02s3000', bundle='main'),
    'vib': _ck(f'{_V}0.0001/epoch_01_step_01500.pt',
               'cfbab52c8b00dff9f5ee0c3e64c5ea43607cf36bf038ab93a622e2865fb7b784', 'vib',
               'VIB beta=0.1 (summed-KL code, effective 0.1024) e01s1500', bundle='main'),
    'run2': _ck('4M_checkpoints_rerun/vib_hd256_tok1_lslr0.2_lsmax116_vib0.1024_vwarm0.3_seedR/epoch_02_step_00500.pt',
                '0f62fec180c372d7d59f85398512fea1b651806917b6d5bdc9387a8272d81bd9', 'vib',
                'VIB run 2 (beta 0.1024, averaged-KL code) e02s0500', bundle='main'),
    'beta02': _ck(f'{_V}0.0002/epoch_02_step_03000.pt',
                  '482bb7dc48fba2a64ab7b5ff30b34a1b5f35755b323ac8205661bec16003c980', 'vib',
                  'VIB beta=0.2 e02s3000'),
    'margin': _ck('4M_checkpoints/base_hd256_tok1_margin0.2_mw0.4_lslr0.4_lsmax116/epoch_01_step_03500.pt',
                  'e3b136b1de689c1f53ea7c881440c1903c128cfa8ed2b2096094c9b33b5bfdec', 'base',
                  'margin remedy e01s3500', bundle='main'),
    'noise_e00s3500': _ck(f'{_N}/epoch_00_step_03500.pt',
                          '61e4145c9e3afaab37c7c2627ba42e6dde76e280e00f271762962ca8707b80b7', 'base',
                          'additive-noise remedy e00s3500', bundle='main'),
    'noise_e01s0500': _ck(f'{_N}/epoch_01_step_00500.pt',
                          '4297e667e9cbad945c59e2135281055e99ec12c69db29415ef2f50bc852c45b7', 'base',
                          'additive-noise remedy e01s0500', bundle='addon'),
    'noise_e00s0500': _ck(f'{_N}/epoch_00_step_00500.pt',
                          '4ccfc0ed185fafacda6a69cdacde2428172e2a0cd73e2da4f80545f51017491e', 'base',
                          'additive-noise remedy e00s0500'),
    'unit_norm': _ck('4M_checkpoints/base_hd256_tok1_lslr0.2_lsmax116_pnorm1.0/epoch_01_step_02000.pt',
                     '8e86392b0b5ae23fcf7853408f98144e2e35a10778b46b1a59ed88f63e839b3c', 'base',
                     'unit-norm remedy e01s2000 (projection on)', pn=1.0, bundle='main'),
    'klmu': _ck('4M_checkpoints/base_hd256_tok1_lslr0.2_lsmax116_l2tok0.1024_seedR/epoch_01_step_01000.pt',
                'b3fa883abd29b430699d80cea4cf130c449dc4a01b34163a96448011c2afed7b', 'base',
                'KL_mu-only (deterministic + L2 on the token, weight 0.1024) e01s1000'),
    'tau_frozen': _ck('4M_checkpoints/checkpoints_frozen_ls/base_hd256_tok1/epoch_00_step_01500.pt',
                      'ad1619417a923e9e5d4de69f23eeb6661dc5846fbe2e3ffa0451494ec0fadf0a', 'base',
                      'tau frozen at 100, e00s1500 (val argmax)', bundle='addon'),
    'tau_free': _ck('4M_checkpoints/checkpoints_trainable_ls/base_hd256_tok1/epoch_00_step_03500.pt',
                    'ec524f129ca7daa79b3deada014f6611ed088560c7b54a78d1f37d9b4d6bdcfa', 'base',
                    'tau free, e00s3500 (val argmax)', bundle='addon'),
    'tau_capped': _ck('4M_checkpoints/checkpoints_trainable_ls_clipped/base_hd256_tok1/epoch_00_step_01500.pt',
                      'bcbc6fedeb010f67e2e78e4cf2a712cfb86735b8419c82d04bd91f8c406cb598', 'base',
                      'tau capped at 115, e00s1500 (val argmax)', bundle='addon'),
    'beta05_e01s2500': _ck(f'{_V}0.0005/epoch_01_step_02500.pt',
                           '24190c0d968fe82fbc07e2b0fb1271be535b3bc51f61c184c56e6451b37de28e', 'vib',
                           'beta=0.5 e01s2500: ' + _RESUMED.format(w='5e-4')),
    'beta10_e01s1000': _ck(f'{_V}0.001/epoch_01_step_01000.pt',
                           '5628dd0edbfaf4d124951c7402f545e7717e5de0fca780e6c0e85f6d7d9bd3b7', 'vib',
                           'beta=1.0 e01s1000: ' + _RESUMED.format(w='1e-3')),
    'baseline_e01': _ck(f'{_B}/epoch_01/checkpoint.pt',
                        '21ff420c96db941ce7f1fa2da65e4e5c1daefcfdc05b1e7149778c8a9c84fdbb', 'base',
                        'baseline epoch_01 (same-run control)'),
    'small_tok1_e01': _ck('checkpoints_1500k_trainable_ls/base_hd256_tok1/epoch_01.pt',
                          '176c954e010d1561c0392af8f4f1470a8c3607c06fd96798968437f3ee16ec0f', 'base',
                          '~1.5M images, 1 token, epoch_01 (val argmax)'),
    'small_tok2_e08': _ck('checkpoints_1500k_trainable_ls/base_hd256_tok2/epoch_08.pt',
                          '667a00e1af31e19fef6c63b589a78e1fb243cea4629994ec4fa9653b13c204dd', 'base',
                          '~1.5M images, 2 tokens, epoch_08 (val argmax)', tokens=2),
    'small_tok4_e00': _ck('checkpoints_1500k_trainable_ls/base_hd256_tok4/epoch_00.pt',
                          '14fb4030e1798a970e6c9382f9dcfe6cf19db3911920d62d56851918a20c0ad5', 'base',
                          '~1.5M images, 4 tokens, epoch_00 (val argmax)', tokens=4),
    'sweep_b0_e00': _ck(f'{_B}/epoch_00/checkpoint.pt',
                        '54a493e7fa99e84b85fd7e4cbdec0e98efdc9472f30f4c5c3c2393f07247d3db', 'base',
                        'end of epoch 0, beta=0 (baseline run)'),
    'sweep_b01_e00': _ck(f'{_V}0.0001/epoch_00/checkpoint.pt',
                         '5c1f5f58bba93b06b58e597e0aed2545a0000e59481146e8adc9285c53961a7a', 'vib',
                         'end of epoch 0, beta=0.1 (summed-KL code, effective 0.1024)'),
    'sweep_b02_e00': _ck(f'{_V}0.0002/epoch_00/checkpoint.pt',
                         '09dd160e3c2599e27162c557603d0510cdbc34635bccd3a96f3c31b15161e071', 'vib',
                         'end of epoch 0, beta=0.2 (summed-KL code, effective 0.2048)'),
    'sweep_b05_e00': _ck(f'{_V}0.0005/epoch_00/checkpoint.pt',
                         'b15d1a76dfb7e0547935b7a5a052fad1635f020d82a4ee652a991ed352230d97', 'vib',
                         'end of epoch 0, beta=0.5 (summed-KL code, effective 0.512)'),
    'sweep_b10_e00': _ck(f'{_V}0.001/epoch_00/checkpoint.pt',
                         '90b0611971d3f8ab2f65c17e5940570ca47a91851e7370f269a76d43a2c3bd97', 'vib',
                         'end of epoch 0, beta=1.0 (summed-KL code, effective 1.024)'),
}

GATE_MODELS = ('baseline', 'vib')

RESUMED_NOTE = ('These checkpoints were trained in epoch 0 with the old summed-KL code (effective beta 0.512 / '
                '1.024) and resumed for epoch 1 with the averaged-KL code at vib_beta 5e-4 / 1e-3, so their KL rose '
                'to ~150 nats: they are not clean beta=0.5 / 1.0 models.')


def _row(label, group, out_sub, test=None, stored_val=False, stored_agg=False, paper=None, **extra):
    """test: None, 'new' (to_upload), 'recheck_repo' (vs metrics/circo_test_submissions) or 'recheck_file'."""
    e = {'label': label, 'group': group, 'out_sub': out_sub, 'test': test, 'stored_val': stored_val,
         'stored_agg': stored_agg, 'paper': paper}
    e.update(extra)
    return e


CANONICAL = [
    _row('tau_capped', 'tau', 'canonical/tau_capped', test='recheck_file', stored_agg=True,
         test_ref=('addon', 'tau/capped/circo_test/rechecks/base_hd256_tok1__epoch_00_step_01500__regenerated__submission.json'),
         test_ref_sha256='12dcbda685c0e72e9727f066058604a181a5b1946d9673d09926e0ab20f7b351',
         test_name='tau_capped__base_hd256_tok1__epoch_00_step_01500__regenerated__submission.json',
         paper='Table 1 Capped under best val: CIRCO val 7.71 (all_circo_metrics), test 6.14 (server, 2026-11-25); '
               'no canonical Fashion-IQ yet'),
    _row('beta05_e01s2500', 'beta_resumed', 'canonical/beta_resumed', test='new', stored_val=True,
         paper='best-val beta=0.5 row: stored CIRCO val 9.00 (metrics/circo); no test file yet', note=RESUMED_NOTE),
    _row('beta10_e01s1000', 'beta_resumed', 'canonical/beta_resumed', test='new', stored_val=True,
         paper='best-val beta=1.0 row: stored CIRCO val 9.05 (metrics/circo); no test file yet', note=RESUMED_NOTE),
    _row('klmu', 'klmu', 'canonical/klmu', test='recheck_repo',
         paper='KL_mu-only: CIRCO val 8.90 (paper_results CSV), test 7.6 (receipt pending)'),
    _row('small_tok1_e01', 'small', 'canonical/small_1p5m', stored_agg=True,
         paper='~1.5M token study (D.5 / S10) at the val argmax; all_circo_metrics CIRCO val 11.83'),
    _row('small_tok2_e08', 'small', 'canonical/small_1p5m', stored_agg=True, paper='all_circo_metrics CIRCO val 10.61'),
    _row('small_tok4_e00', 'small', 'canonical/small_1p5m', stored_agg=True, paper='all_circo_metrics CIRCO val 3.70'),
    _row('sweep_b0_e00', 'sweep', 'canonical/sweep_e00', test='new', stored_agg=True,
         paper='matched epoch-0 sweep: all_circo_metrics CIRCO val 11.02'),
    _row('sweep_b01_e00', 'sweep', 'canonical/sweep_e00', test='new', stored_agg=True, paper='CIRCO val 11.79'),
    _row('sweep_b02_e00', 'sweep', 'canonical/sweep_e00', test='new', stored_agg=True, paper='CIRCO val 10.69'),
    _row('sweep_b05_e00', 'sweep', 'canonical/sweep_e00', test='new', stored_val=True, stored_agg=True,
         paper='CIRCO val 7.23'),
    _row('sweep_b10_e00', 'sweep', 'canonical/sweep_e00', test='new', stored_val=True, stored_agg=True,
         paper='CIRCO val 3.49'),
    _row('beta02', 'reference', 'canonical/references', test='recheck_repo', stored_val=True, stored_agg=True,
         paper='beta=0.2: CIRCO val 11.36, test 12.38 (verified receipt)'),
    _row('baseline_e01', 'reference', 'canonical/references', test='recheck_repo', stored_val=True, stored_agg=True,
         paper='baseline epoch_01: CIRCO val 11.35 (all_circo_metrics), test 11.10 (verified receipt)'),
]
CANON = {e['label']: e for e in CANONICAL}

PROBE_MODELS = ('baseline', 'vib', 'run2', 'beta02', 'margin', 'noise_e00s3500', 'noise_e01s0500', 'noise_e00s0500',
                'unit_norm', 'klmu', 'tau_frozen', 'tau_free', 'tau_capped', 'beta05_e01s2500', 'beta10_e01s1000',
                'sweep_b0_e00', 'sweep_b01_e00', 'sweep_b02_e00', 'sweep_b05_e00', 'sweep_b10_e00')
DUMP_CORE   = ('baseline', 'vib', 'run2', 'beta02', 'margin', 'noise_e00s3500', 'unit_norm', 'klmu')
DUMP_EXTRA  = ('tau_frozen', 'tau_free', 'tau_capped', 'beta05_e01s2500', 'beta10_e01s1000', 'baseline_e01',
               'sweep_b0_e00', 'sweep_b01_e00', 'sweep_b02_e00', 'sweep_b05_e00', 'sweep_b10_e00')
S1_MODELS   = ('vib', 'run2', 'beta02')     # S1 control: the baseline with matched isotropic noise
G10_MODELS  = DUMP_CORE
S13_MODELS  = ('baseline', 'vib')

# Canonical references of the dump self-check. val / fiq: ('reeval',) reeval_2026_11/{circo_val,fashion_val}/<key>,
# ('addon', sub) addon/<sub>/..., ('final',) this run's Step 2 record. test: ('repo',) the original submission,
# ('reeval_upload', name), ('addon_file', rel), ('final_new',) this run's to_upload file.
_RV, _RP = ('reeval',), ('repo',)
DUMP_REF = {
    'baseline': (_RV, _RV, _RP), 'vib': (_RV, _RV, _RP), 'run2': (_RV, _RV, _RP),
    'margin': (_RV, _RV, ('reeval_upload', 'base_hd256_tok1_margin0.2_mw0.4_lslr0.4_lsmax116__epoch_01_step_03500__submission.json')),
    'noise_e00s3500': (_RV, _RV, ('reeval_upload', 'base_hd256_tok1_lslr0.2_lsmax116_noise0.1__epoch_00_step_03500__submission.json')),
    'unit_norm': (_RV, _RV, ('reeval_upload', 'base_hd256_tok1_lslr0.2_lsmax116_pnorm1.0__epoch_01_step_02000__with_projection__submission.json')),
    'tau_frozen': (('addon', 'tau/frozen'), ('addon', 'tau/frozen'), _RP),
    'tau_free': (('addon', 'tau/free'), ('addon', 'tau/free'), _RP),
    'tau_capped': (('final',), ('final',), ('addon_file', CANON['tau_capped']['test_ref'][1])),
    'beta02': (('final',), ('final',), _RP), 'klmu': (('final',), ('final',), _RP),
    'baseline_e01': (('final',), ('final',), _RP),
    'beta05_e01s2500': (('final',), ('final',), ('final_new',)), 'beta10_e01s1000': (('final',), ('final',), ('final_new',)),
    'sweep_b0_e00': (('final',), ('final',), ('final_new',)), 'sweep_b01_e00': (('final',), ('final',), ('final_new',)),
    'sweep_b02_e00': (('final',), ('final',), ('final_new',)), 'sweep_b05_e00': (('final',), ('final',), ('final_new',)),
    'sweep_b10_e00': (('final',), ('final',), ('final_new',)),
}

VARIANTS = {
    'V1': 'composed P(z_ref, caption): "a photo of [z] <caption>", the canonical encode_query path',
    'V2': 'token-only P(z_ref): "a photo of [z]" (_build_token_sequence without text)',
    'V3': 'token-only on unit-norm input: z = mapper(v_raw / ||v_raw||) (+ merger, pseudo_norm), the Table S13 regime',
    'V4': "mean token + caption: z_bar = mean of the final tokens over the dataset's unique references "
          '(CIRCO 1,017 / Fashion-IQ 4,315), re-projected to rho when pseudo_norm > 0',
    'V5_s<seed>': 'caption swap P(z_i, caption_pi(i)); pi = saved derangement (within split / category) of that seed',
    'V6_s<seed>': 'reference swap P(z_pi(i), caption_i)',
}


def _self_check_tables():
    assert all(l in CK for l in PROBE_MODELS + DUMP_CORE + DUMP_EXTRA + S1_MODELS + S13_MODELS + GATE_MODELS)
    assert all(e['label'] in CK for e in CANONICAL) and len(CANON) == len(CANONICAL)
    assert len({c['sha256'] for c in CK.values()}) == len(CK), 'two labels point at identical files'
    assert len({c['rel'] for c in CK.values()}) == len(CK)
    assert set(DUMP_REF) == set(DUMP_CORE + DUMP_EXTRA)
    assert 'seed3407' not in ' '.join(c['rel'] for c in CK.values()), 'seed3407 is not reported'
    assert [l for l, c in CK.items() if c['pn'] != 0.0] == ['unit_norm']
    keys = [(e['out_sub'], R.key_for(Path(CK[e['label']]['rel']))) for e in CANONICAL]
    assert len(set(keys)) == len(keys), 'two canonical rows share a result file'
    for label, (v, f, t) in DUMP_REF.items():
        assert (v[0] == 'final') == (label in CANON) and (t[0] == 'final_new') == (CANON.get(label, {}).get('test') == 'new')


_self_check_tables()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def ckpt_path(label):
    return R.EXPERIMENT / CK[label]['rel']


def key_of(label):
    return R.key_for(ckpt_path(label))


def entry_file(out, label):
    return out / 'entries' / f'{label}.json'


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def batched(n, size):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


def pct(x, dp):
    return '-' if x is None else f'{100 * x:.{dp}f}'


def free_memory():
    gc.collect()
    R.free_gpu()


def run_item(out, label, title, fn, recs, resume=True):
    """Run one item: skip it if a finished record exists, record a failure instead of stopping the run."""
    path = entry_file(out, label)
    if resume and path.exists():
        rec = R.read_json(path)
        if rec.get('status') == 'done':
            R.log(f'  {label}: already done ({"PASS" if rec.get("ok") else "FAIL"})')
            recs[label] = rec
            return rec
    R.log()
    R.log(f'--- {label}: {title}   [{R.now()}]')
    t0 = time.time()
    try:
        rec = fn()
        rec.update(label=label, title=title, status='done', seconds=round(time.time() - t0, 1), finished_at=R.now())
        rec.setdefault('ok', True)
    except Exception:                                         # noqa: BLE001
        R.log(traceback.format_exc())
        R.log(f'  FAILED: {label} (continuing with the others; rerun the same command to retry it)')
        rec = {'label': label, 'title': title, 'status': 'failed', 'ok': False,
               'error': traceback.format_exc(limit=8), 'finished_at': R.now()}
    free_memory()
    R.write_json(path, rec)
    R.log(f'  => {label}: {"PASS" if rec["ok"] else ("FAILED" if rec["status"] == "failed" else "FAIL")}'
          f'  ({rec.get("seconds", "-")} s)')
    recs[label] = rec
    return rec


# ---------------------------------------------------------------------------
# Comparisons (plain Python)
# ---------------------------------------------------------------------------
def compare_perquery(now, ref, ks=('@5', '@10', '@25', '@50')):
    """Query-by-query comparison of two eval_circo_v2 per_query blocks."""
    pos = {q: i for i, q in enumerate(ref['query_id'])}
    qn = now['query_id']
    common = [(i, pos[q]) for i, q in enumerate(qn) if q in pos]
    mism = {}
    for k in ks:
        mism[f'ap{k}'] = [qn[i] for i, j in common if abs(now['ap_at_k'][k][i] - ref['ap_at_k'][k][j]) > 1e-9]
        mism[f'recall{k}'] = [qn[i] for i, j in common
                              if abs(now['recall_at_k'][k][i] - ref['recall_at_k'][k][j]) > 1e-9]
    for f in ('rank_ref', 'best_gt_rank'):
        mism[f] = [qn[i] for i, j in common if now[f][i] != ref[f][j]]

    def maxdiff(f):
        d = [abs(now[f][i] - ref[f][j]) for i, j in common if now[f][i] is not None and ref[f][j] is not None]
        return max(d) if d else None

    out = {'n_now': len(qn), 'n_reference': len(ref['query_id']), 'n_common': len(common),
           'missing': sorted(set(ref['query_id']) - set(qn))[:20], 'unexpected': sorted(set(qn) - set(pos))[:20],
           'mismatch_counts': {k: len(v) for k, v in mism.items()},
           'mismatched_queries': {k: v[:20] for k, v in mism.items() if v},
           'max_abs_diff_sim_ref': maxdiff('sim_ref'), 'max_abs_diff_sim_gt_mean': maxdiff('sim_gt_mean'),
           'map10_now': mean([now['ap_at_k']['@10'][i] for i, _ in common]),
           'map10_reference': mean([ref['ap_at_k']['@10'][j] for _, j in common]),
           'ref10_now': sum(now['rank_ref'][i] <= 10 for i, _ in common),
           'ref10_reference': sum(ref['rank_ref'][j] <= 10 for _, j in common)}
    out['exact'] = (not out['missing'] and not out['unexpected'] and len(qn) == len(ref['query_id'])
                    and all(v == 0 for v in out['mismatch_counts'].values()))
    return out


def compare_lists(sub, ref_path):
    """CIRCO test top-50 lists (query id -> 50 image ids) against a submission file."""
    ref = R.read_json(ref_path)
    common = set(sub) & set(ref)
    identical = sum(sub[q] == ref[q] for q in common)
    same_set = sum(sub[q] != ref[q] and set(sub[q]) == set(ref[q]) for q in common)
    return {'reference_file': str(ref_path), 'n_queries': len(set(sub) | set(ref)),
            'only_in_now': len(set(sub) - set(ref)), 'only_in_reference': len(set(ref) - set(sub)),
            'identical_lists': identical, 'same_set_other_order': same_set,
            'different_sets': len(common) - identical - same_set}


def fiq_query_diff(cat, now, ref):
    """Difference in Fashion-IQ hits (queries) between two recall fractions of one category."""
    return round(now * FIQ_N[cat]) - round(ref * FIQ_N[cat])


def compare_fashion_recalls(now, ref):
    """Per-category Fashion-IQ recalls ({cat: {'1','5','10','50'}}) against a reeval.py fashion_val record.
    ok: R@10 within FIQ_QUERY_TOL queries in every category; 'exact' says whether every value is identical."""
    rows = []
    for cat in R.FIQ_CATS:
        for k in R.KS:
            a, b = now[cat][k], ref['per_category'][cat][k]
            rows.append({'value': f'{cat} R@{k}', 'k': k, 'now': a, 'reference': b, 'same': round(a, 6) == round(b, 6),
                         'query_diff': fiq_query_diff(cat, a, b)})
    avg_now = mean([now[c]['10'] for c in R.FIQ_CATS])
    avg_ref = mean([ref['per_category'][c]['10'] for c in R.FIQ_CATS])
    ok = all(abs(r['query_diff']) <= FIQ_QUERY_TOL for r in rows if r['k'] == '10')
    return {'ok': ok, 'exact': all(r['same'] for r in rows), 'avg_r10_now': avg_now, 'avg_r10_reference': avg_ref,
            'query_tol': FIQ_QUERY_TOL, 'rows': rows}


def gate_fashion_check(f, ref_f):
    """The gate's Fashion-IQ part: every category and K within FIQ_QUERY_TOL queries of reeval_2026_11 (each row also
    says whether its 6-dp value is identical; the avg rows follow from the categories)."""
    _, rows = A.fashion_check(f, dict(ref_f['per_category'], avg=ref_f['avg']))
    for r in rows:
        cat = r['value'].split()[0]
        if cat in FIQ_N:
            r['query_diff'] = fiq_query_diff(cat, r['now'], r['reference'])
    ok = all(abs(r['query_diff']) <= FIQ_QUERY_TOL for r in rows if 'query_diff' in r)
    return ok, rows


def validate_submission(path):
    """CIRCO server format: every test query id, 50 unique int image ids each, the reference never listed."""
    sub = R.read_json(path)
    refs = {str(a['id']): int(a['reference_img_id']) for a in R.read_json(R.CIRCO_DIR / 'annotations' / 'test.json')}
    bad = [] if set(sub) == set(refs) else ['query ids differ from the CIRCO test split']
    for q, lst in sub.items():
        if not (isinstance(lst, list) and len(lst) == 50 and len(set(lst)) == 50
                and all(isinstance(x, int) and not isinstance(x, bool) for x in lst)):
            bad.append(f'query {q}: not 50 unique integer ids')
        elif q in refs and refs[q] in lst:
            bad.append(f'query {q}: lists its own reference image')
    size = path.stat().st_size
    if size >= 5 * 2 ** 20:
        bad.append(f'{size} bytes: above the 5 MiB upload limit')
    return {'ok': not bad, 'n_queries': len(sub), 'bytes': size, 'problems': bad[:10]}


def _all_circo():
    if not hasattr(_all_circo, 'data'):
        _all_circo.data = R.read_json(ALL_CIRCO) if ALL_CIRCO.exists() else {}
    return _all_circo.data


def aggregate_check(label, c):
    """CIRCO-val mAP@10 and Ref@10 against the May 2026 sweep (all_circo_metrics.json, eval_circo.py)."""
    d = _all_circo().get(ALL_CIRCO_PREFIX + CK[label]['rel'])
    if d is None or d.get('eval_status') not in (None, 'success'):
        return {'ok': False, 'reason': f'no successful entry in {ALL_CIRCO.name}'}
    st, v2 = d['circo'], c['v2']
    now10, ref10 = float(v2['map_atk']['@10']), float(st['map_atk']['@10'])
    r_now, r_ref = float(v2['stickiness']['r10_ref']), float(st['stickiness']['r10_ref'])
    n = int(v2.get('n_queries', 220))
    return {'ok': abs(now10 - ref10) <= R.CIRCO_TOL and abs(r_now - r_ref) * n <= 2 + 1e-9,
            'source': f'{ALL_CIRCO.name} (eval_circo.py, {d.get("evaluated_at")})', 'map10_now': now10,
            'map10_stored': ref10, 'delta': now10 - ref10, 'exact': now10 == ref10, 'ref10_now': r_now,
            'ref10_stored': r_ref, 'tau_stored': d.get('logit_scale')}


def stored_probe(kind, rel):
    """The stored probe result (metrics/*_probe.json) for a checkpoint, matched by path."""
    p = STORED_PROBES[kind]
    if not p.exists():
        return None
    d = R.read_json(p)
    res = d.get('results', d) if kind != 'sigma' else d
    for name, r in res.items():
        if isinstance(r, dict) and str(r.get('checkpoint', '')).endswith(rel):
            return dict(r, _source=f'{p.name}:{name}')
    return None


# ---------------------------------------------------------------------------
# Step 0: preflight
# ---------------------------------------------------------------------------
def environment_problems(device):
    problems = []
    try:
        import torch
        if device.startswith('cuda') and not torch.cuda.is_available():
            problems.append('No CUDA GPU is visible to PyTorch.')
    except Exception as e:                                    # noqa: BLE001
        problems.append(f'PyTorch is not importable: {e}')
    try:
        import clip
        if 'ViT-L/14' not in clip.available_models():
            problems.append('The installed clip package does not offer ViT-L/14.')
    except Exception as e:                                    # noqa: BLE001
        problems.append(f'OpenAI CLIP is not importable ({e}).')
    for mod in ('numpy', 'PIL', 'yaml'):
        try:
            __import__(mod)
        except Exception as e:                                # noqa: BLE001
            problems.append(f'Python package {mod} is not importable: {e}')
    if R.REPO_ROOT != R.CODES_ROOT.resolve():
        problems.append(f'This clone is at {R.REPO_ROOT}, but the loaders hardcode {R.CODES_ROOT}.')
    if not R.CONFIG_YAML.exists():
        problems.append(f'Missing {R.CONFIG_YAML}')
    needed = [R.CIRCO_DIR / 'circo_repo' / 'src' / 'dataset.py', R.CIRCO_DIR / 'circo_repo' / 'src' / 'evaluation.py',
              R.GALLERY, R.FIQ_DIR / 'images' / 'images', R.CIRCO_DIR / 'annotations' / 'test.json']
    needed += [R.FIQ_DIR / f'gallery_cache_{c}_ViT-L-14.pt' for c in R.FIQ_CATS]
    needed += [R.FIQ_DIR / 'captions' / f'cap.{c}.val.json' for c in R.FIQ_CATS]
    needed += [R.FIQ_DIR / 'image_splits' / f'split.{c}.val.json' for c in R.FIQ_CATS]
    problems += [f'Missing data: {p}' for p in needed if not p.exists()]
    problems += R.query_image_problems()
    return problems


def versions():
    v = {'python': platform.python_version()}
    try:
        import torch
        v.update(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                 gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        import torchvision
        v['torchvision'] = torchvision.__version__
        import numpy
        import PIL
        v.update(numpy=numpy.__version__, pillow=PIL.__version__)
    except Exception as e:                                    # noqa: BLE001
        v['error'] = str(e)
    try:
        import importlib.metadata as md
        import json
        import re
        d = json.loads(md.distribution('clip').read_text('direct_url.json') or '{}')
        commit = d.get('vcs_info', {}).get('commit_id')
        if not commit:
            m = re.search(r'[0-9a-f]{40}', d.get('url', ''))
            commit = m.group(0) if m else None
        v.update(clip_commit=commit, clip_source=d.get('url'))
    except Exception as e:                                    # noqa: BLE001
        v.update(clip_commit=None, clip_source=f'unknown ({e})')
    return v


def version_warnings(v):
    w = [f'{k} is {v.get(k)}, the previous runs used {want}' for k, want in EXPECTED_VERSIONS.items()
         if v.get(k) != want]
    path, size = CLIP_WEIGHTS
    if not path.exists():
        w.append(f'{path} is missing: clip.load will download it (932 MB, needs network)')
    elif path.stat().st_size != size:
        w.append(f'{path} has {path.stat().st_size} bytes, expected {size}: clip.load will re-download it')
    return w


def reference_paths(label, out):
    """(val record, Fashion-IQ record, test file) the dump self-check compares this model with."""
    v, f, t = DUMP_REF[label]
    key = key_of(label)

    def rec_path(spec, kind):
        if spec[0] == 'reeval':
            return (R.val_file if kind == 'val' else R.fiq_file)(REEVAL_OUT, key)
        if spec[0] == 'addon':
            return (R.val_file if kind == 'val' else R.fiq_file)(ADDON_OUT / spec[1], key)
        return (R.val_file if kind == 'val' else R.fiq_file)(out / CANON[label]['out_sub'], key)

    if t[0] == 'repo':
        test = R.ORIGINAL_SUBMISSIONS / f'{key}__submission.json'
    elif t[0] == 'reeval_upload':
        test = REEVAL_OUT / 'circo_test' / 'to_upload' / t[1]
    elif t[0] == 'addon_file':
        test = ADDON_OUT / t[1]
    else:
        test = out / 'circo_test' / 'to_upload' / f'{key}__submission.json'
    return {'val': rec_path(v, 'val'), 'fiq': rec_path(f, 'fiq'), 'test': test,
            'produced_by_this_run': v[0] == 'final'}


def input_problems(args, out):
    problems, warnings = [], []
    known = REEVAL_OUT / 'checks' / 'known_models.json'
    if not known.exists():
        problems.append(f'Missing {known}: the main re-evaluation results must be on this machine')
    elif not R.read_json(known).get('ok'):
        problems.append(f'{known} says the known models did not reproduce')
    if (ADDON_OUT / 'summary.json').exists() and R.read_json(ADDON_OUT / 'summary.json').get('exit_code') != 0:
        warnings.append('addon/summary.json does not record exit code 0')
    for label in GATE_MODELS:
        for p in (R.val_file(REEVAL_OUT, key_of(label)), R.fiq_file(REEVAL_OUT, key_of(label))):
            if not p.exists():
                problems.append(f'Missing gate reference {p}')
    if not args.skip_dump:
        for label in DUMP_CORE + DUMP_EXTRA:
            refs = reference_paths(label, out)
            if refs['produced_by_this_run']:
                continue
            problems += [f'Missing dump reference {refs[k]} ({label})' for k in ('val', 'fiq', 'test')
                         if not refs[k].exists()]
    if not args.skip_canonical:
        for e in CANONICAL:
            key = key_of(e['label'])
            if e['stored_val'] and not (R.STORED_VAL / f'{key}.json').exists():
                problems.append(f'Missing stored result {R.STORED_VAL / (key + ".json")}')
            if e['test'] == 'recheck_repo' and not (R.ORIGINAL_SUBMISSIONS / f'{key}__submission.json').exists():
                problems.append(f'Missing original submission {key}__submission.json')
            if e['test'] == 'recheck_file':
                p = ADDON_OUT / e['test_ref'][1]
                if not p.exists():
                    problems.append(f'Missing {p}')
                elif R.sha256_file(p) != e['test_ref_sha256']:
                    problems.append(f'{p}: sha256 differs from the uploaded file ({e["test_ref_sha256"][:16]}...)')
            if e['stored_agg'] and (ALL_CIRCO_PREFIX + CK[e['label']]['rel']) not in _all_circo():
                problems.append(f'{CK[e["label"]]["rel"]} is not in {ALL_CIRCO}')
    for kind, p in STORED_PROBES.items():
        if not p.exists():
            warnings.append(f'Missing {p}: the {kind} probe cannot be compared with its stored value')
    for label in S13_MODELS:
        if not (STORED_S13 / f'{key_of(label)}.json').exists():
            warnings.append(f'Missing stored S13 record for {label}')
    out.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out).free / 1e9
    if free_gb < MIN_FREE_GB:
        problems.append(f'Only {free_gb:.1f} GB free under {out}; the dump needs about {MIN_FREE_GB:.0f} GB')
    return problems, warnings


def checkpoint_problems():
    rows, problems = {}, []
    for label, c in CK.items():
        p = ckpt_path(label)
        row = {'bundle': c['bundle'], 'path': str(p), 'sha256_expected': c['sha256']}
        rows[label] = row
        if not p.exists():
            problems.append(f'Missing checkpoint {p}: unzip {BUNDLES[c["bundle"]]} into {R.EXPERIMENT}')
            continue
        row['sha256'] = R.sha256_file(p)
        if row['sha256'] != c['sha256']:
            problems.append(f'{p}: sha256 {row["sha256"][:16]}... differs from the archived copy {c["sha256"][:16]}...')
            continue
        try:
            meta = A.load_meta(p)
        except Exception as ex:                               # noqa: BLE001
            problems.append(f'{p}: torch.load failed: {ex}')
            continue
        row.update(meta)
        if not meta['has_mapper_state'] or meta['mapper_type'] != c['mapper']:
            problems.append(f'{p}: mapper_type {meta["mapper_type"]!r}, expected {c["mapper"]!r} with a mapper_state')
        if meta['pseudo_norm'] != c['pn']:
            problems.append(f'{p}: pseudo_norm={meta["pseudo_norm"]}, expected {c["pn"]}')
        if meta['num_tokens'] != c['tokens']:
            problems.append(f'{p}: num_tokens={meta["num_tokens"]}, expected {c["tokens"]}')
        wants_merger = meta['merge_tokens'] is not None and meta['merge_tokens'] < meta['num_tokens']
        if wants_merger != meta['has_merger_state']:
            problems.append(f'{p}: merge_tokens={meta["merge_tokens"]} but merger_state '
                            f'{"present" if meta["has_merger_state"] else "missing"}')
    return rows, problems


# ---------------------------------------------------------------------------
# Step 1: gate
# ---------------------------------------------------------------------------
def gate_item(label, out, args, cfg):
    ckpt, key, od = ckpt_path(label), key_of(label), out / 'gate'
    c = R.run_circo_val(ckpt, key, 0.0, od, args.device)
    f = R.run_fashion(ckpt, key, od, cfg)
    ref_c = R.read_json(R.val_file(REEVAL_OUT, key))
    ref_f = R.read_json(R.fiq_file(REEVAL_OUT, key))
    pq = compare_perquery(c['v2']['per_query'], ref_c['v2']['per_query'])
    agg = {k: c['v2'][k] == ref_c['v2'][k] for k in ('map_atk', 'recall_atk', 'semantic_map_at10', 'stickiness')}
    ok_f, rows = gate_fashion_check(f, ref_f)
    checks = {
        'circo_val_per_query': {'ok': pq['exact'], **pq},
        'circo_val_map_recall': {'ok': agg['map_atk'] and agg['recall_atk'], 'same': agg},
        'tau': {'ok': c['tau'] == ref_c['tau'], 'now': c['tau'], 'reference': ref_c['tau']},
        'pseudo_norm_off': {'ok': c['pseudo_norm_in_model'] == 0.0, 'now': c['pseudo_norm_in_model']},
        'fashion': {'ok': ok_f, 'exact': all(r['same'] for r in rows), 'query_tol': FIQ_QUERY_TOL, 'rows': rows},
    }
    ok = all(v['ok'] for v in checks.values())
    R.log(f'  CIRCO val vs reeval_2026_11: {pq["n_common"]}/{pq["n_reference"]} queries, mismatches '
          f'{sum(pq["mismatch_counts"].values())} (AP/recall at all K, reference and best-GT ranks), max |d sim_ref| '
          f'{pq["max_abs_diff_sim_ref"]}  {"ok" if checks["circo_val_per_query"]["ok"] else "DIFFERS"}')
    worst = max(abs(r['query_diff']) for r in rows if 'query_diff' in r)
    R.log(f'  Fashion-IQ vs reeval_2026_11: {sum(r["same"] for r in rows)}/{len(rows)} values identical at 6 dp; '
          f'largest change {worst} quer{"y" if worst == 1 else "ies"} in a category (allowed {FIQ_QUERY_TOL})  '
          f'{"ok" if ok_f else "DIFFERS"}')
    for r in rows:
        if not r['same']:
            qd = r.get('query_diff')
            note = '' if qd is None else f'  ({qd:+d} quer{"y" if abs(qd) == 1 else "ies"})'
            R.log(f'    differs: {r["value"]} {r["reference"]:.6f} -> now {r["now"]:.6f}{note}')
    return {'checkpoint': str(ckpt), 'key': key, 'sha256': CK[label]['sha256'], 'numbers': A.numbers(c, f),
            'reference': {'circo_val': str(R.val_file(REEVAL_OUT, key)), 'fashion': str(R.fiq_file(REEVAL_OUT, key))},
            'checks': checks, 'ok': ok}


# ---------------------------------------------------------------------------
# Step 2: canonical rows
# ---------------------------------------------------------------------------
def recheck_against(ckpt, name, ref_path, pn, device, out):
    """Regenerate a test file into circo_test/rechecks/<name> and compare it with ref_path."""
    dest = out / 'circo_test' / 'rechecks' / name
    reg = R.registry(out)
    if dest.exists() and name in reg and 'comparison' in reg[name]:
        R.log(f'  test file  : already re-checked ({name})')
        return reg[name]
    info = R.produce_test_file(ckpt, dest, pn, device)
    info['comparison'] = R.compare_submissions(dest, Path(ref_path))
    R.register(out, name, info)
    c = info['comparison']
    R.log(f'  test file  : vs {c["original_file"]}: {c["identical_lists"]}/{c["n_queries"]} identical top-50 lists, '
          f'file identical: {"yes" if c["identical_file"] else "no"}')
    return info


def canonical_item(e, out, args, cfg):
    label = e['label']
    ck, ckpt, key, od = CK[label], ckpt_path(label), key_of(label), out / e['out_sub']
    c = R.run_circo_val(ckpt, key, ck['pn'], od, args.device)
    f = R.run_fashion(ckpt, key, od, cfg)
    n = A.numbers(c, f)
    rec = {'group': e['group'], 'checkpoint': str(ckpt), 'sha256': ck['sha256'], 'key': key, 'out_dir': str(od),
           'paper': e['paper'], 'note': e.get('note'), 'numbers': n}
    checks = {'pseudo_norm': {'ok': c['pseudo_norm_in_model'] == ck['pn'], 'now': c['pseudo_norm_in_model'],
                              'expected': ck['pn']}}
    if e['stored_val']:
        ok, cmp = A.val_check(c, key)
        checks['stored_val_per_query'] = {'ok': ok, 'source': str(R.STORED_VAL / f'{key}.json'), 'comparison': cmp}
        R.log(f'  stored     : {R.fmt_val(cmp)}  {"ok" if ok else "DIFFERS"}')
    if e['stored_agg']:
        checks['stored_val_aggregate'] = agg = aggregate_check(label, c)
        if 'map10_stored' in agg:
            R.log(f'  stored     : {agg["source"]}: mAP@10 {100 * agg["map10_stored"]:.4f} -> now '
                  f'{100 * agg["map10_now"]:.4f}, Ref@10 {100 * agg["ref10_stored"]:.2f} -> {100 * agg["ref10_now"]:.2f}'
                  f'  {"ok" if agg["ok"] else "DIFFERS"}')
    fields = {'label': label, 'role': ck['title'], 'checkpoint_sha256': ck['sha256'],
              'circo_val_map10': n['circo_val_map']['10'], 'fiq_r10': n['fiq_avg_exact']['10'],
              'fiq_r50': n['fiq_avg_exact']['50']}
    if e['test'] == 'new':
        info = R.new_test_file(ckpt, f'{key}__submission.json', ck['pn'], args.device, out, fields)
        v = validate_submission(out / 'circo_test' / 'to_upload' / info['file'])
        rec['test'] = {'kind': 'new', 'file': info['file'], 'sha256': info['sha256'], 'n_queries': info['n_queries'],
                       'tau_test_side': info['tau'], 'validation': v}
        checks['test_file_valid'] = {'ok': v['ok'], **v}
    elif e['test'] in ('recheck_repo', 'recheck_file'):
        if e['test'] == 'recheck_repo':
            info = R.recheck_test_file(ckpt, key, ck['pn'], args.device, out)
            sha_ok, accept = True, ('exact', 'near')
        else:
            ref = ADDON_OUT / e['test_ref'][1]
            info = recheck_against(ckpt, e['test_name'], ref, ck['pn'], args.device, out)
            sha_ok, accept = info['comparison']['original_sha256'] == e['test_ref_sha256'], ('exact',)
        match = A.test_match(info['comparison'])
        rec['test'] = {'kind': 'recheck', 'file': info['file'], 'sha256': info['sha256'], 'match': match,
                       'tau_test_side': info['tau'], 'comparison': info['comparison']}
        checks['test_reproduced'] = {'ok': match in accept and sha_ok, 'match': match,
                                     'file_identical': info['comparison']['identical_file'],
                                     'reference_sha256_ok': sha_ok}
    rec['checks'] = checks
    rec['ok'] = all(v['ok'] for v in checks.values())
    R.log(f'  numbers    : CIRCO val mAP@10 {100 * n["circo_val_map"]["10"]:.4f}  Ref@10 {100 * n["ref10_stickiness"]:.1f}%'
          f'  Fashion-IQ R@10/50 {100 * n["fiq_avg_exact"]["10"]:.4f}/{100 * n["fiq_avg_exact"]["50"]:.4f}'
          f'  tau {n["tau"]:.4f}')
    return rec


# ---------------------------------------------------------------------------
# Step 3: probes
# ---------------------------------------------------------------------------
_CACHE = {}


def probe_item(label, args):
    if 'probe_refs' not in _CACHE:
        _CACHE['probe_refs'] = nx_probe_refs(args)
    raw = nx_probe(label, args, _CACHE['probe_refs'])
    rec = {'checkpoint': str(ckpt_path(label)), 'key': key_of(label), **raw}
    checks = {}
    for kind, fields in (('token_norm', ('mean', 'std', 'min', 'max')), ('sigma', ('sigma_bar', 'sigma_min', 'mu_norm')),
                         ('kl', ('kl_mu_nats', 'kl_sigma_nats', 'kl_total_nats'))):
        stored = stored_probe(kind, CK[label]['rel'])
        if stored is None or kind not in raw:
            continue
        rows = [{'field': f, 'stored': stored[f], 'now': raw[kind][f],
                 'same': abs(raw[kind][f] - stored[f]) <= PROBE_TOL} for f in fields]
        checks[f'stored_{kind}'] = {'ok': all(r['same'] for r in rows), 'source': stored['_source'], 'rows': rows}
        R.log(f'  stored {kind:10s}: {stored["_source"]}: ' + ', '.join(f'{r["field"]} {r["stored"]} -> {r["now"]}'
                                                                     for r in rows)
              + f'  {"ok" if checks[f"stored_{kind}"]["ok"] else "DIFFERS"}')
    rec['checks'] = checks
    rec['ok'] = all(v['ok'] for v in checks.values())
    t = raw['token_norm']
    line = f'  token norm : mean {t["mean"]:.4f}  std {t["std"]:.4f}  min {t["min"]:.4f}  max {t["max"]:.4f}  n {t["n_refs"]}'
    if 'sigma' in raw:
        line += (f'   sigma_bar {raw["sigma"]["sigma_bar"]:.4f}  sigma_min {raw["sigma"]["sigma_min"]:.4f}  '
                 f'||mu|| {raw["sigma"]["mu_norm"]:.4f}   KL_mu {raw["kl"]["kl_mu_nats"]:.4f}  '
                 f'KL_sigma {raw["kl"]["kl_sigma_nats"]:.4f} nats')
    R.log(line)
    return rec


def write_probe_files(out, recs):
    """Combined probe files in the formats of probe_token_norm.py / probe_sigma.py / probe_kl.py."""
    done = {l: recs[f'probe_{l}'] for l in PROBE_MODELS if recs.get(f'probe_{l}', {}).get('status') == 'done'}
    if not done:
        return
    any_rec = next(iter(done.values()))
    n_refs, n_q = any_rec['token_norm']['n_refs'], any_rec.get('n_val_queries')
    tn = {'backbone': 'ViT-L/14', 'n_unique_refs': n_refs, 'n_val_queries': n_q,
          'results': {l: {k: r['token_norm'][k] for k in ('checkpoint', 'n_refs', 'mean', 'std', 'min', 'max')}
                      for l, r in done.items()}}
    sg = {l: {k: r['sigma'][k] for k in ('checkpoint', 'sigma_bar', 'sigma_min', 'mu_norm')}
          for l, r in done.items() if 'sigma' in r}
    kl = {'backbone': 'ViT-L/14', 'n_unique_refs': n_refs,
          'results': {l: {k: v for k, v in r['kl'].items() if not k.startswith('per_ref')}
                      for l, r in done.items() if 'kl' in r}}
    per = {'reference_ids': any_rec.get('reference_ids'),
           'models': {l: {'token_norm': r['token_norm']['per_ref'],
                          **({'sigma_mean': r['sigma']['per_ref_sigma_mean'], 'sigma_min': r['sigma']['per_ref_sigma_min'],
                              'mu_norm': r['sigma']['per_ref_mu_norm'], 'kl_mu': r['kl']['per_ref_kl_mu'],
                              'kl_sigma': r['kl']['per_ref_kl_sigma']} if 'sigma' in r else {})}
                      for l, r in done.items()}}
    for name, obj in (('token_norm_probe.json', tn), ('sigma_probe.json', sg), ('kl_probe.json', kl),
                      ('per_reference.json', per)):
        R.write_json(out / 'probes' / name, obj)


# ---------------------------------------------------------------------------
# Step 4: dump
# ---------------------------------------------------------------------------
def shared_item(ddir, args, cfg):
    meta = nx_shared(ddir / 'shared', args, cfg)
    R.write_json(ddir / 'shared' / 'info.json', dict(meta, status='done', created_at=R.now()))
    return {'counts': meta['counts'], 'n_arrays': len(meta['arrays']),
            'bytes': sum(a['bytes'] for a in meta['arrays']), 'ok': True}


def ensure_shared(out, args, cfg, recs):
    """The model-independent encodes (dump/shared/), needed by Steps 4-6; made (or retried) once per invocation."""
    if 'shared' in _CACHE or _CACHE.get('shared_failed'):
        return _CACHE.get('shared')
    ddir = out / 'dump'
    rec = run_item(out, 'dump_shared', 'model-independent encodes: v_raw, text features, index, derangements',
                   lambda: shared_item(ddir, args, cfg), recs)
    if rec['status'] != 'done':
        _CACHE['shared_failed'] = True
        return None
    _CACHE['shared'] = nx_load_shared(ddir / 'shared')
    return _CACHE['shared']


def evaluate_selfcheck(raw, refs):
    """Compare the dump's own retrieval results with the canonical records (plain Python)."""
    checks = {'encode_from_vraw': {'ok': raw['encode_max_abs_diff'] < SELF_EQ_TOL,
                                   'max_abs_diff': raw['encode_max_abs_diff'], 'n_queries': raw['encode_n'],
                                   'batch16_max_abs_diff_info': raw.get('encode_max_abs_diff_batch16')}}
    if refs['val'].exists():
        cmp = compare_perquery(raw['val_perquery'], R.read_json(refs['val'])['v2']['per_query'])
        ok = (not cmp['missing'] and not cmp['unexpected'] and abs(cmp['map10_now'] - cmp['map10_reference']) < SELF_MAP_TOL
              and abs(cmp['ref10_now'] - cmp['ref10_reference']) <= SELF_REF10_TOL)
        checks['circo_val'] = {'ok': ok, 'reference': str(refs['val']), **cmp}
    else:
        checks['circo_val'] = {'ok': None, 'reason': 'reference record missing', 'reference': str(refs['val'])}
    if refs['test'].exists():
        cmp = compare_lists(raw['test_lists'], refs['test'])
        ok = (not cmp['only_in_now'] and not cmp['only_in_reference']
              and cmp['n_queries'] - cmp['identical_lists'] <= R.TEST_QUERY_TOL)
        checks['circo_test_lists'] = {'ok': ok, 'exact': cmp['identical_lists'] == cmp['n_queries'], **cmp}
    else:
        checks['circo_test_lists'] = {'ok': None, 'reason': 'reference file missing', 'reference': str(refs['test'])}
    if refs['fiq'].exists():
        cmp = compare_fashion_recalls(raw['fashion_recalls'], R.read_json(refs['fiq']))
        checks['fashion_recalls'] = {'reference': str(refs['fiq']), **cmp}
    else:
        checks['fashion_recalls'] = {'ok': None, 'reason': 'reference record missing', 'reference': str(refs['fiq'])}
    checks['probe_formulas'] = raw['probe_reimpl']
    states = [c['ok'] for c in checks.values()]
    ok = False if False in states else None if None in states else True
    exact = (raw['encode_max_abs_diff'] == 0.0 and checks['circo_val'].get('exact') is True
             and checks['circo_test_lists'].get('exact') is True and checks['fashion_recalls'].get('exact') is True)
    return {'ok': ok, 'exact': exact, 'checks': checks}


def log_selfcheck(sc):
    c = sc['checks']
    R.log(f'  self-check : encode_from_vraw vs encode_query max |d| {c["encode_from_vraw"]["max_abs_diff"]:.2e}'
          f' ({c["encode_from_vraw"]["n_queries"]} queries)  {_ok(c["encode_from_vraw"]["ok"])}')
    v = c['circo_val']
    if 'map10_now' in v:
        R.log(f'               CIRCO val mAP@10 {100 * v["map10_reference"]:.4f} -> {100 * v["map10_now"]:.4f}, Ref@10 '
              f'{v["ref10_reference"]} -> {v["ref10_now"]} queries, per-query mismatches '
              f'{sum(v["mismatch_counts"].values())}  {_ok(v["ok"])}')
    else:
        R.log(f'               CIRCO val: {v.get("reason")}  {_ok(v["ok"])}')
    t = c['circo_test_lists']
    R.log(f'               CIRCO test: {t.get("identical_lists", "-")}/{t.get("n_queries", "-")} top-50 lists identical '
          f'to the canonical file  {_ok(t["ok"])}')
    fr = c['fashion_recalls']
    if 'avg_r10_now' in fr:
        R.log(f'               Fashion-IQ R@10 {100 * fr["avg_r10_reference"]:.4f} -> {100 * fr["avg_r10_now"]:.4f} '
              f'({"exact" if fr["exact"] else "not exact"})  {_ok(fr["ok"])}')
    else:
        R.log(f'               Fashion-IQ: {fr.get("reason")}  {_ok(fr["ok"])}')
    R.log(f'               probe formulas on identical inputs: max rel |d| {c["probe_formulas"].get("max_rel_diff")}'
          f'  {_ok(c["probe_formulas"]["ok"])}')
    R.log(f'  => self-check {_ok(sc["ok"])}{" (exact)" if sc["exact"] else ""}')


def _ok(x):
    return {True: 'ok', False: 'FAIL', None: 'NOT CHECKED'}[x]


def dump_model_item(label, core, out, args, cfg, recs):
    sh = ensure_shared(out, args, cfg, recs)
    if sh is None:
        raise RuntimeError('the shared encodes (dump_shared) are not available')
    mdir = out / 'dump' / 'models' / label
    refs = reference_paths(label, out)
    missing = [str(refs[k]) for k in ('val', 'fiq', 'test') if not refs[k].exists()]
    if missing:      # fail (and retry on the next run) rather than write a dump nothing can verify
        raise RuntimeError(f'canonical reference(s) for the self-check missing: {missing}'
                           + (f'; Step 2 entry {label} must finish first' if refs['produced_by_this_run'] else ''))
    if core and args.skip_fashion_swaps:
        R.log('  --skip_fashion_swaps: Fashion-IQ V4-V6 are not written')
    raw = nx_dump_model(label, core, mdir, args, cfg, sh)
    sc = evaluate_selfcheck(raw['selfcheck_raw'], refs)
    log_selfcheck(sc)
    info = {'label': label, 'title': CK[label]['title'], 'checkpoint': CK[label]['rel'], 'sha256': CK[label]['sha256'],
            'key': key_of(label), 'core': core, 'mapper': CK[label]['mapper'], 'tau': raw['tau'],
            'pseudo_norm_in_model': raw['pseudo_norm'], 'num_tokens': raw['num_tokens'], 'arrays': raw['arrays'],
            'notes': raw.get('notes', {}), 'selfcheck': sc, 'status': 'done', 'created_at': R.now()}
    R.write_json(mdir / 'info.json', info)
    return {'core': core, 'checkpoint': CK[label]['rel'], 'selfcheck': sc, 'ok': sc['ok'] is True,
            'n_arrays': len(raw['arrays']), 'bytes': sum(a['bytes'] for a in raw['arrays'])}


def write_dump_info(out):
    ddir = out / 'dump'
    if not (ddir / 'shared' / 'info.json').exists():
        return
    R.write_json(ddir / 'dump_info.json', {
        'written_at': R.now(), 'code_commit': R.git_commit(), 'versions': versions(),
        'variants': VARIANTS,
        'text_features': {'text_raw': 'encode_text(tokenize(caption)) (Fashion-IQ caption = captions[0] + " and " '
                                      '+ captions[1], as eval_fashion.build_queries)',
                          'text_aphoto': 'encode_text(tokenize("a photo of " + caption))',
                          'concept_aphoto': 'CIRCO only: encode_text(tokenize("a photo of " + shared_concept))'},
        'dtypes': 'CIRCO v_raw, tokens, V1-V3 and all text features fp32; V4-V6, Fashion-IQ query variants and '
                  'tokens, sampled queries fp16 (Fashion-IQ v_raw fp32)',
        'batching': 'CIRCO tokens and V1 one reference / query at a time (eval_circo_v2, eval_circo_test); Fashion-IQ '
                    f'v_raw and V1 in eval_fashion batches of cfg.batch_size; other variants in batches of {BATCH}',
        'models': {'core': list(DUMP_CORE), 'extra': list(DUMP_EXTRA),
                   'titles': {l: CK[l]['title'] for l in DUMP_CORE + DUMP_EXTRA}},
        'sampled': {'models': list(S1_MODELS), 'control': 'baseline', 'samples': S1_SAMPLES,
                    'eps': f'numpy.random.default_rng({S1_SEED0} + k).standard_normal((n_refs, T, D), float32), '
                           'saved in shared/s1_eps.npy', 'seeds': [S1_SEED0 + k for k in range(S1_SAMPLES)]},
        'attention': {'models': list(G10_MODELS), 'groups': list(GROUPS), 'heads': 'averaged'},
        'swap_seeds': list(SWAP_SEEDS), 'excluded': 'seed3407 (not reported)'})


# ---------------------------------------------------------------------------
# Steps 5-7
# ---------------------------------------------------------------------------
def s1_item(label, out, args, cfg, recs, control=False):
    sh = ensure_shared(out, args, cfg, recs)
    if sh is None:
        raise RuntimeError('the shared encodes (dump_shared) are not available')
    sigma2 = None
    if control:
        v = recs.get('s1_vib', {})
        if v.get('status') != 'done':
            raise RuntimeError('the control needs s1_vib (its mean sigma^2)')
        sigma2 = v['sigma2_mean']
    rec = nx_s1(label, sigma2, out / 'dump' / 'sampled' / ('baseline_control' if control else label), args, sh)
    q = rec['quick']
    R.log(f'  sampled    : val mAP@10 {100 * q["val_map10_mean"]:.2f} +- {100 * q["val_map10_sd"]:.2f} over '
          f'{S1_SAMPLES} samples (deterministic token {100 * q["val_map10_det"]:.2f}); Ref@10 '
          f'{100 * q["val_ref10_mean"]:.1f}% (det {100 * q["val_ref10_det"]:.1f}%); test top-10 Jaccard vs det '
          f'{q["test_jaccard10_mean"]:.3f}')
    return rec


def g10_item(label, out, args, cfg, recs):
    sh = ensure_shared(out, args, cfg, recs)
    if sh is None:
        raise RuntimeError('the shared encodes (dump_shared) are not available')
    rec = nx_g10(label, out / 'dump' / 'attention' / label, args, sh)
    rec['ok'] = rec['max_abs_diff'] < ATTN_TOL and rec['group_sum_max_dev'] < 1e-4 and rec['restored']
    for v in ('V1', 'V2'):
        m = rec['summary'][v]['pseudo_mass_by_layer']
        R.log(f'  attention  : {v} EOT-row mass on the pseudo-token, layers 1-12: ' + ' '.join(f'{x:.3f}' for x in m))
    R.log(f'  attention  : capturing pass vs normal pass max |d q| {rec["max_abs_diff"]:.2e}, wrappers removed: '
          f'{rec["restored"]}')
    return rec


def s13_item(out, args):
    rec = nx_s13(args)
    for label in S13_MODELS:
        r = rec['models'][label]
        st = r.get('stored')
        r['reproduces_stored'] = (st is not None and abs(r['now']['sim_p_vraw_mean'] - st['sim_p_vraw_mean']) <= 5e-4)
        r['reproduces_paper'] = abs(round(r['now']['sim_p_vraw_mean'], 3) - PAPER_S13[label]) <= 5e-4
        R.log(f'  S13 {label:8s}: token-only query-to-source similarity {r["now"]["sim_p_vraw_mean"]:.5f} '
              f'(stored {st["sim_p_vraw_mean"] if st else "-"}, paper {PAPER_S13[label]}); P(gt) '
              f'{r["now"]["prob_on_gt_mean"]:.4f}; reproduces stored: {r["reproduces_stored"]}, paper: '
              f'{r["reproduces_paper"]}')
    rec['unit_norm_regime_confirmed'] = rec['input_norm_max_dev'] < 1e-4
    R.write_json(out / 's13' / 's13_control.json', rec)
    rec['ok'] = True          # a finding either way; only an exception fails this item
    return rec


# ---------------------------------------------------------------------------
# Step 8: summary and the paper table
# ---------------------------------------------------------------------------
TABLE_GROUPS = (('gate', 'Gate: reproduced from reeval_2026_11 (CIRCO exactly, Fashion-IQ within '
                         f'{FIQ_QUERY_TOL} queries per category)'),
                ('tau', 'Table 1 tau row at its val argmax'),
                ('beta_resumed', 'beta rows at best val: ' + _RESUMED.format(w='5e-4 / 1e-3') + '\n  ' + RESUMED_NOTE),
                ('klmu', 'KL_mu-only ablation'),
                ('sweep', 'Matched end-of-epoch-0 beta sweep (all trained with the summed-KL code)'),
                ('small', '~1.5M token study at the val argmax'),
                ('reference', 'Canonical rows the dump self-check needed'))


def paper_rows(recs):
    rows = []
    for label in GATE_MODELS:
        r = recs.get(f'gate_{label}')
        if r and r.get('status') == 'done':
            rows.append(dict(_table_row(label, r), group='gate'))
    for e in CANONICAL:
        r = recs.get(e['label'])
        if r and r.get('status') == 'done':
            rows.append(dict(_table_row(e['label'], r), group=e['group']))
    return rows


def _table_row(label, r):
    n = r['numbers']
    row = {'label': label, 'title': CK[label]['title'], 'checkpoint': CK[label]['rel'],
           'fiq_r10': n['fiq_avg_exact']['10'], 'fiq_r50': n['fiq_avg_exact']['50'],
           'circo_val_map': n['circo_val_map'], 'circo_val_recall': n['circo_val_recall'],
           'ref10_stickiness': n['ref10_stickiness'], 'sim_delta': n['sim_delta'], 'p_gt': n['p_gt'], 'tau': n['tau'],
           'fiq_per_category': n['fiq_per_category'], 'paper': r.get('paper'), 'ok': r.get('ok')}
    if 'test' in r:
        row['test'] = {k: r['test'].get(k) for k in ('kind', 'file', 'sha256', 'match')}
    return row


def log_paper_table(rows):
    head = (f'  {"entry":17s} {"F R@10 exact":>12s} {"2dp":>6s} {"1dp":>5s} {"F R@50":>8s} {"C@10 exact":>10s} '
            f'{"2dp":>6s} {"1dp":>5s} {"C@5":>6s} {"C@25":>6s} {"C@50":>6s} {"stick":>5s} {"P(gt)":>6s} {"tau":>8s}  test')
    for group, title in TABLE_GROUPS:
        g = [r for r in rows if r['group'] == group]
        if not g:
            continue
        R.log(f'\n  {title}')
        R.log(head)
        for r in g:
            m = r['circo_val_map']
            t = r.get('test') or {}
            test = {'new': f'NEW {t.get("file")}', 'recheck': f'recheck: {t.get("match")}'}.get(t.get('kind'), '-')
            R.log(f'  {r["label"]:17s} {pct(r["fiq_r10"], 6):>12s} {pct(r["fiq_r10"], 2):>6s} {pct(r["fiq_r10"], 1):>5s} '
                  f'{pct(r["fiq_r50"], 2):>8s} {pct(m["10"], 6):>10s} {pct(m["10"], 2):>6s} {pct(m["10"], 1):>5s} '
                  f'{pct(m["5"], 2):>6s} {pct(m["25"], 2):>6s} {pct(m["50"], 2):>6s} {pct(r["ref10_stickiness"], 1):>5s} '
                  f'{r["p_gt"]:6.4f} {r["tau"]:8.4f}  {test}{"" if r["ok"] else "  (CHECK FAILED)"}')
            R.log(f'  {"":17s} {r["title"]}' + (f'; {r["paper"]}' if r.get('paper') else ''))


def probe_rows(recs):
    rows = []
    for label in PROBE_MODELS:
        r = recs.get(f'probe_{label}')
        if r and r.get('status') == 'done':
            row = {'label': label, 'title': CK[label]['title'],
                   **{f'token_norm_{k}': r['token_norm'][k] for k in ('mean', 'std', 'min', 'max', 'n_refs')}}
            if 'sigma' in r:
                row.update({k: r['sigma'][k] for k in ('sigma_bar', 'sigma_min', 'mu_norm')})
                row.update({k: r['kl'][k] for k in ('kl_mu_nats', 'kl_sigma_nats', 'kl_total_nats', 'kl_total_bits')})
            rows.append(row)
    return rows


def log_probe_table(rows):
    R.log(f'\n  {"probe":17s} {"||z|| mean":>10s} {"std":>7s} {"min":>7s} {"max":>7s} {"sigma_bar":>9s} '
          f'{"sigma_min":>9s} {"||mu||":>7s} {"KL_mu":>8s} {"KL_sig":>8s} {"KL bits":>8s}  '
          f'({rows[0]["token_norm_n_refs"]} CIRCO-val references; KL in nats, total also in bits)')
    for r in rows:
        vib = 'sigma_bar' in r
        f = (lambda k, w, d: f'{r[k]:{w}.{d}f}' if vib else f'{"-":>{w}s}')
        R.log(f'  {r["label"]:17s} {r["token_norm_mean"]:10.4f} {r["token_norm_std"]:7.4f} {r["token_norm_min"]:7.4f} '
              f'{r["token_norm_max"]:7.4f} {f("sigma_bar", 9, 4)} {f("sigma_min", 9, 4)} {f("mu_norm", 7, 3)} '
              f'{f("kl_mu_nats", 8, 3)} {f("kl_sigma_nats", 8, 3)} {f("kl_total_bits", 8, 3)}')


def upload_list(out):
    reg = R.registry(out)
    rows = []
    for p in sorted((out / 'circo_test' / 'to_upload').glob('*__submission.json')):
        info = reg.get(p.name, {})
        rows.append({'file': p.name, 'sha256': R.sha256_file(p), 'bytes': p.stat().st_size,
                     'n_queries': info.get('n_queries'), 'label': info.get('label'), 'role': info.get('role'),
                     'checkpoint': info.get('checkpoint'), 'circo_val_map10': info.get('circo_val_map10')})
    if rows:
        (out / 'circo_test' / 'to_upload.sha256').write_text(''.join(f'{r["sha256"]}  {r["file"]}\n' for r in rows))
    return rows


def write_manifest(ddir):
    if not ddir.exists():
        return None
    files = sorted(p for p in ddir.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256' and not p.name.endswith('.tmp'))
    lines = [f'{R.sha256_file(p)}  {p.relative_to(ddir)}\n' for p in files]
    (ddir / 'MANIFEST.sha256').write_text(''.join(lines))
    return {'root': str(ddir), 'n_files': len(files), 'bytes': sum(p.stat().st_size for p in files),
            'manifest': 'dump/MANIFEST.sha256'}


def exit_code(recs, labels):
    selfcheck = [l for l in labels if l.startswith('dump_') and l != 'dump_shared'
                 and recs.get(l, {}).get('status') == 'done' and recs[l]['selfcheck']['ok'] is False]
    other = [l for l in labels if l not in selfcheck and not recs.get(l, {}).get('ok')]
    return (3 if selfcheck else 4 if other else 0), selfcheck, other


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out_dir', type=Path, default=REPO_ROOT / 'results' / 'final_2026_11')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--preflight_only', action='store_true', help='check everything, evaluate nothing')
    p.add_argument('--skip_canonical', action='store_true',
                   help='skip Step 2 (canonical rows); the 13 dump models whose self-check reference Step 2 makes then fail')
    p.add_argument('--skip_probes', action='store_true', help='skip Step 3')
    p.add_argument('--skip_dump', action='store_true', help='skip Step 4 (the shared encodes still run if 5/6 need them)')
    p.add_argument('--skip_fashion_swaps', action='store_true', help='Step 4: no Fashion-IQ V4-V6 (saves ~370 MB)')
    p.add_argument('--skip_sampled', action='store_true', help='skip Step 5 (S1)')
    p.add_argument('--skip_attention', action='store_true', help='skip Step 6 (G10)')
    p.add_argument('--skip_s13', action='store_true', help='skip Step 7')
    return p.parse_args()


def main():
    args = parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    R.banner(f'Final evaluation run   {R.now()}')
    R.log(f'  code   : {R.REPO_ROOT}  (commit {R.git_commit()})')
    R.log(f'  output : {out}')
    R.log(f'  reads  : {REEVAL_OUT} (main re-evaluation and add-on), {R.STORED_VAL.parent}')
    skipped = [k[5:] for k, v in vars(args).items() if k.startswith('skip_') and v]
    if skipped:
        R.log(f'  skipped: {", ".join(skipped)}')

    # ---- Step 0 -------------------------------------------------------------
    R.banner('Step 0/8  Preflight')
    v = versions()
    problems = environment_problems(args.device)
    p_in, warnings = input_problems(args, out)
    rows, p_ck = checkpoint_problems()
    problems += p_in + p_ck
    warnings = version_warnings(v) + warnings
    for label, row in rows.items():
        state = 'ok  ' if row.get('sha256') == row['sha256_expected'] else 'FAIL'
        tau = f'{row["tau"]:.2f}' if row.get('tau') else '-'
        R.log(f'  {state} {row["bundle"]:5s} {label:16s} sha256 {row.get("sha256", "missing")[:16]}  '
              f'{row.get("mapper_type", "-"):4s} tokens {row.get("num_tokens", "-")}  pseudo_norm '
              f'{row.get("pseudo_norm", "-")}  tau {tau}')
    R.log(f'  versions: ' + ', '.join(f'{k} {v.get(k)}' for k in ('python', 'torch', 'torchvision', 'numpy', 'pillow',
                                                                 'clip_commit', 'gpu')))
    for w in warnings:
        R.log(f'  note: {w}')
    info = dict(R.environment(args), versions=v)
    R.write_json(out / 'preflight.json', {'checked_at': R.now(), 'environment': info, 'problems': problems,
                                          'warnings': warnings, 'checkpoints': rows})
    history = R.read_json(out / 'run_info.json') if (out / 'run_info.json').exists() else []
    history.append(info)
    R.write_json(out / 'run_info.json', history)
    if problems:
        R.log('\nPREFLIGHT FAILED:')
        for pr in problems:
            R.log(f'  - {pr}')
        sys.exit(1)
    R.log(f'\n  Preflight passed: {len(rows)} checkpoints; torch {v.get("torch")}, GPU {v.get("gpu")}')
    if args.preflight_only:
        return

    cfg = R.fashion_eval_cfg(args.device)
    recs, labels = {}, []

    # ---- Step 1 -------------------------------------------------------------
    R.banner('Step 1/8  Gate: baseline and VIB must reproduce reeval_2026_11 (CIRCO exactly, Fashion-IQ within '
             f'{FIQ_QUERY_TOL} queries per category)')
    for label in GATE_MODELS:
        run_item(out, f'gate_{label}', CK[label]['title'], lambda label=label: gate_item(label, out, args, cfg), recs,
                 resume=False)
    gate_ok = all(recs[f'gate_{l}'].get('ok') for l in GATE_MODELS)
    R.write_json(out / 'checks' / 'gate.json', {'ok': gate_ok, 'checked_at': R.now(),
                                                'models': {l: recs[f'gate_{l}'] for l in GATE_MODELS}})
    if not gate_ok:
        R.banner('STOPPED: the gate did not reproduce, so this setup differs from the one that made the canonical '
                 'numbers. Nothing else was run. Send run_log.txt and checks/gate.json. To measure again from scratch, '
                 'delete results/final_2026_11/gate/ first.')
        sys.exit(2)

    # ---- Step 2 -------------------------------------------------------------
    if args.skip_canonical:
        R.banner('Step 2/8  Canonical rows: skipped (--skip_canonical)')
    else:
        R.banner(f'Step 2/8  Canonical rows: {len(CANONICAL)} checkpoints (CIRCO val, Fashion-IQ, CIRCO test file)')
        for i, e in enumerate(CANONICAL, 1):
            labels.append(e['label'])
            run_item(out, e['label'], f'[{i}/{len(CANONICAL)}] {CK[e["label"]]["title"]}  ({CK[e["label"]]["rel"]})',
                     lambda e=e: canonical_item(e, out, args, cfg), recs)

    # ---- Step 3 -------------------------------------------------------------
    if args.skip_probes:
        R.banner('Step 3/8  Probes: skipped (--skip_probes)')
    else:
        R.banner(f'Step 3/8  Probes: token norm ({len(PROBE_MODELS)} checkpoints), sigma and KL (VIB-type), '
                 f'220 CIRCO-val references')
        for i, label in enumerate(PROBE_MODELS, 1):
            labels.append(f'probe_{label}')
            run_item(out, f'probe_{label}', f'[{i}/{len(PROBE_MODELS)}] {CK[label]["title"]}',
                     lambda label=label: probe_item(label, args), recs)
        write_probe_files(out, recs)

    # ---- Step 4 -------------------------------------------------------------
    if args.skip_dump:
        R.banner('Step 4/8  Dump: skipped (--skip_dump)')
    else:
        models = DUMP_CORE + DUMP_EXTRA
        R.banner(f'Step 4/8  Dump (G0): shared encodes, then {len(models)} models ({len(DUMP_CORE)} core with V1-V6, '
                 f'{len(DUMP_EXTRA)} extras with V1-V3), each with a self-check')
        labels.append('dump_shared')
        ensure_shared(out, args, cfg, recs)
        for i, label in enumerate(models, 1):
            labels.append(f'dump_{label}')
            run_item(out, f'dump_{label}', f'[{i}/{len(models)}] {CK[label]["title"]}',
                     lambda label=label: dump_model_item(label, label in DUMP_CORE, out, args, cfg, recs), recs)
        write_dump_info(out)
        R.write_json(out / 'checks' / 'selfcheck.json', {
            'checked_at': R.now(), 'models': {l: recs.get(f'dump_{l}', {}).get('selfcheck') for l in models}})

    # ---- Step 5 -------------------------------------------------------------
    if args.skip_sampled:
        R.banner('Step 5/8  Sampled tokens (S1): skipped (--skip_sampled)')
    else:
        R.banner(f'Step 5/8  Sampled tokens (S1): {S1_SAMPLES} posterior samples for {", ".join(S1_MODELS)}, '
                 f'and the baseline with matched isotropic noise')
        for label in S1_MODELS:
            labels.append(f's1_{label}')
            run_item(out, f's1_{label}', f'{CK[label]["title"]}: z = mu + sigma * eps',
                     lambda label=label: s1_item(label, out, args, cfg, recs), recs)
        labels.append('s1_baseline_control')
        run_item(out, 's1_baseline_control', 'baseline z + s * eps, s^2 = mean sigma^2 of VIB e01s1500',
                 lambda: s1_item('baseline', out, args, cfg, recs, control=True), recs)

    # ---- Step 6 -------------------------------------------------------------
    if args.skip_attention:
        R.banner('Step 6/8  Attention (G10): skipped (--skip_attention)')
    else:
        R.banner(f'Step 6/8  Attention (G10): EOT-row attention mass per text layer, {len(G10_MODELS)} core models')
        for label in G10_MODELS:
            labels.append(f'g10_{label}')
            run_item(out, f'g10_{label}', CK[label]['title'], lambda label=label: g10_item(label, out, args, cfg, recs),
                     recs)

    # ---- Step 7 -------------------------------------------------------------
    if args.skip_s13:
        R.banner('Step 7/8  S13 control: skipped (--skip_s13)')
    else:
        R.banner("Step 7/8  S13 control: neg_analysis_all.py's own pipeline on eval_data/circo (baseline, VIB)")
        labels.append('s13_control')
        run_item(out, 's13_control', 'Table S13 in its original (unit-norm input) regime',
                 lambda: s13_item(out, args), recs)

    # ---- Step 8 -------------------------------------------------------------
    R.banner('Step 8/8  Summary')
    code, selfcheck_bad, other_bad = exit_code(recs, labels)
    table, probes = paper_rows(recs), probe_rows(recs)
    uploads = upload_list(out)
    write_dump_info(out)
    dump = write_manifest(out / 'dump')
    R.banner('Numbers for the paper (Fashion-IQ = exact mean of the per-category recalls; percent)')
    log_paper_table(table)
    if probes:
        log_probe_table(probes)
    s13 = recs.get('s13_control', {})
    R.write_json(out / 'summary.json', {
        'finished_at': R.now(), 'exit_code': code, 'minutes': round((time.time() - t_start) / 60, 1),
        'argv': sys.argv, 'code_commit': R.git_commit(),
        'failed': {'dump_selfcheck': selfcheck_bad, 'other': other_bad},
        'paper_table': table, 'probe_table': probes,
        's13_control': {k: s13.get(k) for k in ('models', 'unit_norm_regime_confirmed', 'pt_files')} if s13 else None,
        'sampled': {l: recs.get(l, {}).get('quick') for l in [f's1_{m}' for m in S1_MODELS] + ['s1_baseline_control']},
        'attention': {m: recs.get(f'g10_{m}', {}).get('summary') for m in G10_MODELS},
        'to_upload': uploads, 'dump': dump, 'entries': recs})
    R.log('\n  CIRCO test files to upload (circo_test/to_upload/, listed in circo_test/to_upload.sha256):')
    for u in uploads:
        R.log(f'    {u["sha256"]}  {u["file"]}  ({u["label"]}, val mAP@10 {pct(u["circo_val_map10"], 2)})')
    if dump:
        R.log(f'  dump: {dump["n_files"]} files, {dump["bytes"] / 1e6:.1f} MB under {dump["root"]} (dump/MANIFEST.sha256)')
    verdict = {0: 'DONE: every check passed', 3: f'DUMP SELF-CHECK FAILED: {selfcheck_bad}',
               4: f'DONE EXCEPT: {other_bad}'}[code]
    if code == 3 and other_bad:
        verdict += f'; also failed: {other_bad}'
    R.banner(f'{verdict}   {(time.time() - t_start) / 60:.1f} min   {R.now()}')
    R.log(f'  results: {out}  (summary.json, entries/, checks/)')
    if code:
        R.log('  rerun the same command to retry failed items; finished ones are skipped')
        sys.exit(code)


# ===========================================================================
# Numerical layer (torch / numpy / CLIP). Everything above is plain Python and is exercised by the CPU mock.
# ===========================================================================
def save_array(path, arr, dtype):
    """Atomic .npy write; returns the record dump/*/info.json lists."""
    import numpy as np
    a = arr.detach().cpu().numpy() if hasattr(arr, 'detach') else np.asarray(arr)
    a = np.ascontiguousarray(a.astype(dtype, copy=False))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        np.save(f, a, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return {'file': path.name, 'shape': list(a.shape), 'dtype': str(a.dtype), 'bytes': path.stat().st_size,
            'sha256': R.sha256_file(path)}


def load_array(path):
    import numpy as np
    import torch
    return torch.from_numpy(np.load(path, allow_pickle=False))


def nx_load_model(label, device):
    """The model exactly as eval_circo_v2.load_model builds it (the canonical CIRCO loader)."""
    from evaluation import eval_circo_v2
    model, clip_name = eval_circo_v2.load_model(ckpt_path(label), device)
    if clip_name != 'ViT-L/14':
        raise RuntimeError(f'{label}: backbone {clip_name}')
    if float(model.pseudo_norm) != CK[label]['pn']:
        raise RuntimeError(f'{label}: model.pseudo_norm={model.pseudo_norm}, expected {CK[label]["pn"]}')
    if model.use_multilayer:
        raise RuntimeError(f'{label}: multilayer mappers need intermediate features; not supported here')
    return model


def raw_tokens(model, v):
    """Mapper (+ merger) output before the unit-norm projection; a VIB mapper in eval mode returns mu."""
    z = model.mapper(v)
    if model.merger is not None:
        z = model.merger(z)
    return z


def project(model, z):
    """Pic2Word's fixed-norm edit z <- rho * z / ||z|| (unit-norm runs only)."""
    import torch.nn.functional as F
    return F.normalize(z, dim=-1) * model.pseudo_norm if model.pseudo_norm > 0 else z


def tokens_from_vraw(model, v):
    return project(model, raw_tokens(model, v))


def encode_tokens(model, z, text_tokens=None):
    emb, eot = model._build_token_sequence(z, text_tokens)
    return model.clip.forward_text_transform(emb, eot)


def encode_from_vraw(model, v, text_tokens=None):
    """Pic2Word.encode_query without its first line (the image encoder): the same operations on the same v_raw."""
    return encode_tokens(model, tokens_from_vraw(model, v), text_tokens)


def vib_mu_logvar(model, v):
    """mu and the clamped log-variance exactly as VIBMapper.forward / probe_sigma.py / probe_kl.py build them."""
    m = model.mapper
    b = v.shape[0]
    h = m.encoder(v)
    mu = m.fc_mu(h).view(b, m.num_tokens, m.token_dim)
    logvar = m.fc_logvar(h).view(b, m.num_tokens, m.token_dim).clamp(-10.0, getattr(m, 'logvar_clip_max', 2.0))
    return mu, logvar


def vib_stats(mu, logvar):
    """Per reference: sigma mean, sigma min, ||mu|| (probe_sigma.py), KL_mu, KL_sigma in nats (probe_kl.py)."""
    import torch
    b = mu.shape[0]
    sigma = (0.5 * logvar).exp()
    return torch.stack([sigma.float().mean(dim=(1, 2)), sigma.float().amin(dim=(1, 2)),
                        mu.float().norm(dim=-1).reshape(b, -1).mean(dim=1),
                        (0.5 * mu.pow(2).sum(dim=(1, 2))).float(),
                        (0.5 * (logvar.exp() - logvar - 1).sum(dim=(1, 2))).float()], dim=1)


VIB_STATS_COLUMNS = ['sigma_mean', 'sigma_min', 'mu_norm', 'kl_mu_nats', 'kl_sigma_nats']


def composed(model, z, captions, device, size=BATCH):
    """Batched composed queries P(z_i, caption_i)."""
    import clip
    import torch
    out = []
    for s in batched(len(captions), size):
        tok = clip.tokenize(captions[s], truncate=True).to(device)
        out.append(encode_tokens(model, z[s], tok).float().cpu())
    return torch.cat(out)


def token_only(model, z, size=BATCH):
    import torch
    return torch.cat([encode_tokens(model, z[s]).float().cpu() for s in batched(z.shape[0], size)])


def tokens_batched(model, v, size=BATCH):
    """(final tokens, norms before the projection) for a [N, 768] tensor of v_raw."""
    import torch
    zs, ns = [], []
    for s in batched(v.shape[0], size):
        zr = raw_tokens(model, v[s])
        zs.append(project(model, zr))
        ns.append(zr.norm(dim=-1))
    return torch.cat(zs), torch.cat(ns)


def derangement(keys_ref, keys_text, seed):
    """Seeded permutation pi with keys_ref[pi[i]] != keys_ref[i] and keys_text[pi[i]] != keys_text[i] for every i."""
    n = len(keys_ref)
    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)

    def bad(i, j):
        return keys_ref[j] == keys_ref[i] or keys_text[j] == keys_text[i]

    for _ in range(50):
        viol = [i for i in range(n) if bad(i, perm[i])]
        if not viol:
            return perm
        for i in viol:
            if not bad(i, perm[i]):
                continue
            order = list(range(n))
            rng.shuffle(order)
            for j in order:
                if j != i and not bad(i, perm[j]) and not bad(j, perm[i]):
                    perm[i], perm[j] = perm[j], perm[i]
                    break
    raise RuntimeError(f'no valid derangement of {n} items (seed {seed})')


def nx_gallery(device):
    """The CIRCO gallery cache, loaded once, on the device, as eval_circo_v2 loads it."""
    if 'gallery' not in _CACHE:
        import torch
        cache = torch.load(R.GALLERY, map_location='cpu')
        if cache['clip_model'] != 'ViT-L/14':
            raise RuntimeError(f'gallery cache is for {cache["clip_model"]}')
        _CACHE['gallery'] = {'emb': cache['embeddings'].to(device), 'img_ids': cache['img_ids'],
                             'id2idx': cache['img_id_to_idx']}
        del cache
    return _CACHE['gallery']


def nx_shared(sdir, args, cfg):
    """v_raw of every CIRCO reference (one image at a time) and every Fashion-IQ query (eval_fashion's batches),
    text-only features, the row index of both datasets and the saved derangements."""
    import clip
    import PIL.Image
    import torch
    from evaluation import eval_circo_v2, eval_fashion
    dev = args.device
    model = nx_load_model('baseline', dev)        # only its frozen CLIP is used: identical in every model
    preprocess = model.clip.preprocess             # = clip.load(...)[1] of the canonical loaders
    arrays = []

    # CIRCO: queries in annotation order (val then test), unique references in order of first use
    circo_ds, _ = eval_circo_v2._load_circo_official(R.CIRCO_DIR)
    queries, refs, ref_row, splits = [], [], {}, {}
    for split in ('val', 'test'):
        ds = circo_ds(data_path=str(R.CIRCO_DIR), split=split, mode='relative', preprocess=preprocess)
        splits[split] = [len(queries), len(ds.annotations)]
        for qi, ann in enumerate(ds.annotations):
            rid = str(ann['reference_img_id'])
            if rid not in ref_row:
                ref_row[rid] = len(refs)
                refs.append({'ref_img_id': ann['reference_img_id'],
                             'path': str(ds.img_paths[ds.img_ids_indexes_map[rid]])})
            queries.append({'split': split, 'index': qi, 'query_id': ann['id'], 'ref_img_id': ann['reference_img_id'],
                            'ref_row': ref_row[rid], 'caption': ann['relative_caption'],
                            'shared_concept': ann['shared_concept'], 'target_img_id': ann.get('target_img_id'),
                            'gt_img_ids': ann.get('gt_img_ids')})
    with torch.no_grad(), R.cudnn_settings(deterministic=False):          # eval_circo_v2's settings
        vr = torch.cat([model.clip.encoder_image_unnorm(
            preprocess(PIL.Image.open(r['path']).convert('RGB')).unsqueeze(0).to(dev)).float().cpu() for r in refs])
        caps = [q['caption'] for q in queries]
        texts = {'circo_text_raw': caps, 'circo_text_aphoto': ['a photo of ' + c for c in caps],
                 'circo_concept_aphoto': ['a photo of ' + q['shared_concept'] for q in queries]}
        feats = {name: nx_encode_texts(model, t, dev) for name, t in texts.items()}
    arrays.append(save_array(sdir / 'circo_vraw.npy', vr, 'float32'))
    arrays += [save_array(sdir / f'{name}.npy', f, 'float32') for name, f in feats.items()]
    derange = {'circo': {}, 'fashion': {}}
    for split, (start, n) in splits.items():
        q = queries[start:start + n]
        derange['circo'][split] = {str(s): derangement([x['ref_img_id'] for x in q], [x['caption'] for x in q], s)
                                   for s in SWAP_SEEDS}
    R.write_json(sdir / 'circo_index.json', {'queries': queries, 'refs': refs, 'splits': splits,
                                             'rows': 'circo_vraw rows = refs; query rows = queries (val then test)'})

    # Fashion-IQ: queries exactly as eval_fashion.build_queries keeps them, v_raw in its batches
    eval_dir = Path(cfg.eval_data_dir) / 'fashion_iq'
    images_dir = eval_dir / 'images' / 'images'
    cats, fvr, mods_all, uniq, uniq_first, query_uniq = {}, [], [], {}, [], []
    with torch.no_grad(), R.cudnn_settings(deterministic=True):           # eval_fashion's settings
        for cat in eval_fashion.CATEGORIES:
            anns = eval_fashion.load_annotations(cat, 'val', eval_dir)
            names = eval_fashion.load_split(cat, 'val', eval_dir)
            _, name_to_idx = eval_fashion.build_gallery(names, images_dir, preprocess, cfg.batch_size, cfg.num_workers)
            loader, mods, gt = eval_fashion.build_queries(anns, name_to_idx, images_dir, preprocess, cfg.batch_size,
                                                          cfg.num_workers, verbose=False)
            kept = [a for a in anns if a['candidate'] in name_to_idx and a['target'] in name_to_idx
                    and (images_dir / f'{a["candidate"]}.png').exists() and (images_dir / f'{a["target"]}.png').exists()]
            cands = [p.stem for p in loader.dataset.image_paths]
            if (len(kept) != len(mods) or [a['candidate'] for a in kept] != cands
                    or [a['captions'][0] + ' and ' + a['captions'][1] for a in kept] != mods
                    or [name_to_idx[a['target']] for a in kept] != gt.tolist()):
                raise RuntimeError(f'Fashion-IQ {cat}: the query index differs from eval_fashion.build_queries')
            start, sizes = len(mods_all), []
            for batch in loader:
                v = model.clip.encoder_image_unnorm(batch[0].to(dev))
                fvr.append(v.float().cpu())
                sizes.append(int(v.shape[0]))
            for i, c in enumerate(cands):
                if c not in uniq:
                    uniq[c] = len(uniq_first)
                    uniq_first.append(start + i)
                query_uniq.append(uniq[c])
            cats[cat] = {'start': start, 'n': len(mods), 'batches': sizes, 'gallery_names': names,
                         'candidates': cands, 'targets': [a['target'] for a in kept],
                         'candidate_gallery_row': [name_to_idx[c] for c in cands], 'target_gallery_row': gt.tolist()}
            mods_all += mods
            derange['fashion'][cat] = {str(s): derangement(cands, mods, s) for s in SWAP_SEEDS}
        ffeats = {'fashion_text_raw': nx_encode_texts(model, mods_all, dev),
                  'fashion_text_aphoto': nx_encode_texts(model, ['a photo of ' + m for m in mods_all], dev)}
    arrays.append(save_array(sdir / 'fashion_vraw_query.npy', torch.cat(fvr), 'float32'))
    arrays += [save_array(sdir / f'{name}.npy', f, 'float32') for name, f in ffeats.items()]
    R.write_json(sdir / 'fashion_index.json', {
        'categories': cats, 'modifiers': mods_all, 'unique_candidates': list(uniq), 'unique_first_query': uniq_first,
        'query_unique_row': query_uniq,
        'rows': 'fashion_vraw_query / text / V1 rows = queries (dress, shirt, toptee; eval_fashion order); V2 / V3 / '
                'token rows = unique_candidates (v_raw of the first query that uses the image)'})
    R.write_json(sdir / 'derangements.json', dict(derange, rule='pi[i] != i with a different reference and caption; '
                                                              'CIRCO within split, Fashion-IQ within category'))
    del model
    free_memory()
    counts = {'circo_queries': len(queries), 'circo_refs': len(refs), 'fashion_queries': len(mods_all),
              'fashion_unique_candidates': len(uniq_first)}
    R.log(f'  shared     : {counts}')
    return {'arrays': arrays, 'counts': counts}


def nx_encode_texts(model, texts, device, size=BATCH):
    import clip
    import torch
    return torch.cat([model.clip.encode_text(clip.tokenize(texts[s], truncate=True).to(device)).float().cpu()
                      for s in batched(len(texts), size)])


class Shared:
    """dump/shared/ loaded back (fp32 arrays round-trip exactly)."""
    def __init__(self, sdir):
        ci = R.read_json(sdir / 'circo_index.json')
        self.queries, self.refs, self.splits = ci['queries'], ci['refs'], ci['splits']
        self.fashion = R.read_json(sdir / 'fashion_index.json')
        self.derange = R.read_json(sdir / 'derangements.json')
        self.vraw = load_array(sdir / 'circo_vraw.npy')
        self.fvraw = load_array(sdir / 'fashion_vraw_query.npy')
        self.dir = sdir

    def circo_perm(self, seed):
        """Derangement of seed over all CIRCO queries (val and test parts each within their split)."""
        perm = []
        for split in ('val', 'test'):
            start, _ = self.splits[split]
            perm += [start + j for j in self.derange['circo'][split][str(seed)]]
        return perm

    def fashion_perm(self, seed):
        perm = []
        for cat, info in self.fashion['categories'].items():
            perm += [info['start'] + j for j in self.derange['fashion'][cat][str(seed)]]
        return perm


def nx_load_shared(sdir):
    return Shared(sdir)


def nx_val_perquery(q, sh, gal, ranks=(5, 10, 25, 50)):
    """eval_circo_v2.eval_circo's per-query block, scoring each V1 row against the gallery exactly as it does."""
    import numpy as np
    from evaluation.eval_circo_v2 import _compute_query_ap_at_k, _compute_query_recall_at_k
    g, img_ids, id2idx = gal['emb'], gal['img_ids'], gal['id2idx']
    pq = {'query_id': [], 'rank_ref': [], 'best_gt_rank': [], 'sim_ref': [], 'sim_gt_mean': [],
          'ap_at_k': {f'@{k}': [] for k in ranks}, 'recall_at_k': {f'@{k}': [] for k in ranks}}
    start, n = sh.splits['val']
    for i in range(start, start + n):
        a = sh.queries[i]
        scores = (q[i:i + 1].clone() @ g.T).squeeze(0)     # a fresh [1, 768] query, as encode_query returns it
        ref_idx = id2idx[str(a['ref_img_id'])]
        gt_indices = [id2idx[str(x)] for x in a['gt_img_ids'] if str(x) in id2idx]
        sim_gt = scores[gt_indices].cpu().tolist() if gt_indices else []
        gt_ranks = [(scores > scores[gi]).sum().item() + 1 for gi in gt_indices]
        pq['query_id'].append(a['query_id'])
        pq['sim_ref'].append(scores[ref_idx].item())
        pq['rank_ref'].append((scores > scores[ref_idx]).sum().item() + 1)
        pq['sim_gt_mean'].append(float(np.mean(sim_gt)) if sim_gt else None)
        pq['best_gt_rank'].append(min(gt_ranks) if gt_ranks else None)
        masked = scores.clone()
        masked[ref_idx] = float('-inf')
        top = [int(img_ids[j]) for j in masked.topk(50).indices.cpu().tolist()]
        for k in ranks:
            pq['ap_at_k'][f'@{k}'].append(_compute_query_ap_at_k(top, a['gt_img_ids'], k))
            pq['recall_at_k'][f'@{k}'].append(_compute_query_recall_at_k(top, a['gt_img_ids'], k))
    return pq


def nx_test_lists(q, sh, gal):
    """eval_circo_test.generate_test_submission's top-50 lists for the V1 rows of the test split."""
    g, img_ids, id2idx = gal['emb'], gal['img_ids'], gal['id2idx']
    start, n = sh.splits['test']
    sub = {}
    for i in range(start, start + n):
        a = sh.queries[i]
        scores = (q[i:i + 1].clone() @ g.T).squeeze(0)
        scores[id2idx[str(a['ref_img_id'])]] = float('-inf')
        sub[str(a['query_id'])] = [int(img_ids[j]) for j in scores.topk(50).indices.cpu().tolist()]
    return sub


def nx_probe_refs(args):
    """The probe scripts' reference set: unique CIRCO-val references, preprocessed, in their order."""
    import clip
    import PIL.Image
    import torch
    from evaluation import eval_circo_v2
    from evaluation.probe_token_norm import _unique_reference_paths
    circo_ds, _ = eval_circo_v2._load_circo_official(R.CIRCO_DIR)
    _, preprocess = clip.load('ViT-L/14', device='cpu')
    ds = circo_ds(data_path=str(R.CIRCO_DIR), split='val', mode='relative', preprocess=preprocess)
    paths = _unique_reference_paths(ds)
    ids, seen = [], set()
    for q in range(len(ds)):
        rid = str(ds.annotations[q]['reference_img_id'])
        if rid not in seen:
            seen.add(rid)
            ids.append(int(rid))
    tensors = torch.stack([preprocess(PIL.Image.open(p).convert('RGB')) for p in paths])
    return {'tensors': tensors, 'reference_ids': ids, 'n_val_queries': len(ds)}


def nx_probe(label, args, refs):
    import math as _m
    from evaluation import probe_kl, probe_sigma, probe_token_norm
    from evaluation.eval_circo_v2 import load_model
    model, cm = load_model(ckpt_path(label), args.device)
    if cm != 'ViT-L/14':
        raise RuntimeError(f'{label}: backbone {cm}')
    t, path = refs['tensors'], str(ckpt_path(label))
    with R.cudnn_settings(deterministic=False):
        n = probe_token_norm.token_norms(model, t, args.device)
        rec = {'pseudo_norm_in_model': float(model.pseudo_norm), 'tau': float(model.clip.logit_scale.exp().item()),
               'reference_ids': refs['reference_ids'], 'n_val_queries': refs['n_val_queries'],
               'token_norm': {'checkpoint': path, 'n_refs': int(n.numel()), 'mean': round(float(n.mean()), 4),
                              'std': round(float(n.std()), 4), 'min': round(float(n.min()), 4),
                              'max': round(float(n.max()), 4), 'per_ref': [float(x) for x in n]}}
        if hasattr(model.mapper, 'fc_logvar'):
            sm, smin, mn = probe_sigma.sigma_stats(model, t, args.device)
            rec['sigma'] = {'checkpoint': path, 'sigma_bar': round(float(sm.mean()), 4),
                            'sigma_min': round(float(smin.mean()), 4), 'mu_norm': round(float(mn.mean()), 4),
                            'per_ref_sigma_mean': [float(x) for x in sm], 'per_ref_sigma_min': [float(x) for x in smin],
                            'per_ref_mu_norm': [float(x) for x in mn]}
            klmu, klsig = probe_kl.kl_terms(model, t, args.device)
            mu_n, sig_n = float(klmu.mean()), float(klsig.mean())
            ln2 = _m.log(2)
            rec['kl'] = {'checkpoint': path, 'kl_mu_nats': round(mu_n, 4), 'kl_sigma_nats': round(sig_n, 4),
                         'kl_total_nats': round(mu_n + sig_n, 4), 'kl_mu_bits': round(mu_n / ln2, 4),
                         'kl_sigma_bits': round(sig_n / ln2, 4), 'kl_total_bits': round((mu_n + sig_n) / ln2, 4),
                         'per_ref_kl_mu': [float(x) for x in klmu], 'per_ref_kl_sigma': [float(x) for x in klsig]}
    del model
    return rec


def nx_probe_reimpl(model, vib, device):
    """This file's token / sigma / KL formulas against the probe scripts' functions, on identical inputs."""
    import torch
    from evaluation import probe_kl, probe_sigma, probe_token_norm
    if 'probe_refs' not in _CACHE:
        _CACHE['probe_refs'] = nx_probe_refs(argparse.Namespace(device=device))
    t = _CACHE['probe_refs']['tensors']
    norms, stats = [], []
    for i in range(0, t.shape[0], 64):                 # the probes' batches
        v = model.clip.encoder_image_unnorm(t[i:i + 64].to(device))
        norms.append(tokens_from_vraw(model, v).norm(dim=-1).reshape(-1).float().cpu())
        if vib:
            stats.append(vib_stats(*vib_mu_logvar(model, v)).cpu())
    pairs = {'token_norm': (torch.cat(norms), probe_token_norm.token_norms(model, t, device))}
    if vib:
        s = torch.cat(stats)
        sm, smin, mn = probe_sigma.sigma_stats(model, t, device)
        klmu, klsig = probe_kl.kl_terms(model, t, device)
        pairs.update(sigma_mean=(s[:, 0], sm), sigma_min=(s[:, 1], smin), mu_norm=(s[:, 2], mn),
                     kl_mu=(s[:, 3], klmu), kl_sigma=(s[:, 4], klsig))
    rel = {k: float(((a - b).abs() / b.abs().clamp_min(1.0)).max()) for k, (a, b) in pairs.items()}
    worst = max(rel.values())
    return {'ok': worst <= REIMPL_RTOL, 'max_rel_diff': worst, 'per_quantity': rel, 'n_refs': int(t.shape[0])}


def nx_dump_model(label, core, mdir, args, cfg, sh):
    import clip
    import PIL.Image
    import torch
    import torch.nn.functional as F
    from evaluation import eval_fashion, retrieval
    dev = args.device
    model = nx_load_model(label, dev)
    vib = hasattr(model.mapper, 'fc_logvar')
    arrays, notes, raw = [], {}, {}

    def put(name, t, dtype):
        arrays.append(save_array(mdir / name, t, dtype))

    queries = sh.queries
    caps, rows = [q['caption'] for q in queries], [q['ref_row'] for q in queries]
    with torch.no_grad(), R.cudnn_settings(deterministic=False):          # CIRCO: eval_circo_v2's settings
        v = sh.vraw.to(dev)
        # tokens one reference at a time (a fresh [1, 768] input, as the image encoder hands it to the mapper inside
        # encode_query): the values encode_query builds for that image at batch 1
        zr = [raw_tokens(model, v[r:r + 1].clone()) for r in range(v.shape[0])]
        z = torch.cat([project(model, x) for x in zr])
        put('circo_tokens.npy', z, 'float32')
        put('circo_token_norm_raw.npy', torch.cat(zr).norm(dim=-1), 'float32')
        if vib:
            mu, lv = (torch.cat(x) for x in zip(*[vib_mu_logvar(model, v[s]) for s in batched(v.shape[0], BATCH)]))
            put('circo_logvar.npy', lv, 'float32')
            put('circo_vib_stats.npy', vib_stats(mu, lv), 'float32')
            notes.update(vib_stats_columns=VIB_STATS_COLUMNS, logvar='clamped to (-10, logvar_clip_max) as in the model',
                         token_is_mu='a VIB token is mu (no merger, pseudo_norm 0); tokens one reference at a time, '
                                     f'mu / logvar in batches of {BATCH}',
                         circo_tokens_vs_mu_max_abs_diff=float((mu - z).abs().max()))
        # V1 one query at a time, as eval_circo_v2 / eval_circo_test encode each query
        q1 = torch.cat([encode_tokens(model, z[r:r + 1], clip.tokenize([c], truncate=True).to(dev))
                        for c, r in zip(caps, rows)])
        put('circo_V1.npy', q1, 'float32')
        put('circo_V2.npy', token_only(model, z), 'float32')
        z3, _ = tokens_batched(model, F.normalize(v, dim=-1))
        put('circo_tokens_unitnorm_input.npy', z3, 'float32')
        put('circo_V3.npy', token_only(model, z3), 'float32')
        if core:
            zbar = project(model, z.mean(dim=0, keepdim=True))
            put('circo_V4.npy', composed(model, zbar.expand(len(caps), -1, -1), caps, dev), 'float16')
            zq = z[rows]
            for seed in SWAP_SEEDS:
                perm = sh.circo_perm(seed)
                put(f'circo_V5_s{seed}.npy', composed(model, zq, [caps[perm[i]] for i in range(len(caps))], dev), 'float16')
                put(f'circo_V6_s{seed}.npy', composed(model, z[[rows[perm[i]] for i in range(len(caps))]], caps, dev),
                    'float16')
        # self-check (a): encode_from_vraw == encode_query, one query at a time and as one batch of SELF_N
        pre = model.clip.preprocess
        val0 = sh.splits['val'][0]
        idx = list(range(val0, val0 + min(SELF_N, sh.splits['val'][1])))
        imgs = [pre(PIL.Image.open(sh.refs[rows[i]]['path']).convert('RGB')).unsqueeze(0).to(dev) for i in idx]
        toks = [clip.tokenize([caps[i]], truncate=True).to(dev) for i in idx]
        raw['encode_max_abs_diff'] = max(float((model.encode_query(im, tk) - q1[i:i + 1]).abs().max())
                                         for i, im, tk in zip(idx, imgs, toks))
        raw['encode_n'] = len(idx)
        raw['encode_max_abs_diff_batch16'] = float((model.encode_query(torch.cat(imgs), torch.cat(toks))
                                                    - encode_from_vraw(model, v[[rows[i] for i in idx]], torch.cat(toks)))
                                                   .abs().max())
        # (b), (c): CIRCO val per query and test top-50 lists, scored as the canonical code does
        gal = nx_gallery(dev)
        raw['val_perquery'] = nx_val_perquery(q1, sh, gal)
        raw['test_lists'] = nx_test_lists(q1, sh, gal)
        # (e): the probe formulas this file re-implements (also used for the per-reference KL below)
        raw['probe_reimpl'] = nx_probe_reimpl(model, vib, dev)
        del v, q1

    fi = sh.fashion
    with torch.no_grad(), R.cudnn_settings(deterministic=True):           # Fashion-IQ: eval_fashion's settings
        eval_dir = Path(cfg.eval_data_dir) / 'fashion_iq'
        fv, mods, recalls, fq1 = sh.fvraw, fi['modifiers'], {}, []
        for cat, info in fi['categories'].items():
            qf, pos = [], info['start']
            for size in info['batches']:                       # (d): eval_fashion's batches and CPU scoring
                tok = clip.tokenize(mods[pos:pos + size], truncate=True).to(dev)
                qf.append(encode_from_vraw(model, fv[pos:pos + size].clone().to(dev), tok).cpu())
                pos += size
            qf = torch.cat(qf)
            gallery = eval_fashion.load_gallery_cache(cat, cfg.clip_model, eval_dir)
            rec = retrieval.recall_at_k(qf @ gallery.T, torch.tensor(info['target_gallery_row'], dtype=torch.long),
                                        k_values=[1, 5, 10, 50])
            recalls[cat] = {str(k): float(x) for k, x in rec.items()}
            fq1.append(qf)
        raw['fashion_recalls'] = recalls
        put('fashion_V1.npy', torch.cat(fq1), 'float16')
        if core:
            vu = fv[fi['unique_first_query']].to(dev)
            zu, nu = tokens_batched(model, vu)
            put('fashion_tokens.npy', zu, 'float16')
            put('fashion_token_norm_raw.npy', nu, 'float32')
            if vib:
                mu, lv = (torch.cat(x) for x in zip(*[vib_mu_logvar(model, vu[s]) for s in batched(vu.shape[0], BATCH)]))
                put('fashion_logvar.npy', lv, 'float16')
                put('fashion_vib_stats.npy', vib_stats(mu, lv), 'float32')
            put('fashion_V2.npy', token_only(model, zu), 'float16')
            put('fashion_V3.npy', token_only(model, tokens_batched(model, F.normalize(vu, dim=-1))[0]), 'float16')
            if not args.skip_fashion_swaps:
                zq, _ = tokens_batched(model, fv.to(dev))
                zbar = project(model, zu.mean(dim=0, keepdim=True))
                put('fashion_V4.npy', composed(model, zbar.expand(len(mods), -1, -1), mods, dev), 'float16')
                for seed in SWAP_SEEDS:
                    perm = sh.fashion_perm(seed)
                    put(f'fashion_V5_s{seed}.npy', composed(model, zq, [mods[perm[i]] for i in range(len(mods))], dev),
                        'float16')
                    put(f'fashion_V6_s{seed}.npy', composed(model, zq[perm], mods, dev), 'float16')
    out = {'arrays': arrays, 'notes': notes, 'selfcheck_raw': raw, 'tau': float(model.clip.logit_scale.exp().item()),
           'pseudo_norm': float(model.pseudo_norm), 'num_tokens': int(model.mapper.num_tokens)}
    del model
    return out


def nx_s1_eps(sh, shape):
    """Shared standard-normal draws (numpy PCG64, so the Mac can regenerate them), saved once."""
    import numpy as np
    p = sh.dir / 's1_eps.npy'
    if not p.exists():
        eps = np.stack([np.random.default_rng(S1_SEED0 + k).standard_normal(shape, dtype=np.float32)
                        for k in range(S1_SAMPLES)])
        save_array(p, eps, 'float32')
    return load_array(p)


def nx_quick_metrics(q, sh, gal):
    """Val AP@10 and Ref@10, and top-10 lists, with the reference masked as in the canonical code (batched)."""
    from evaluation.eval_circo_v2 import _compute_query_ap_at_k
    g, img_ids, id2idx = gal['emb'], gal['img_ids'], gal['id2idx']
    ap10, ref10, top10 = [], [], {}
    for s in batched(len(sh.queries), BATCH):
        scores = q[s].to(g.device) @ g.T
        for row, i in zip(scores, range(s.start, s.stop)):
            a = sh.queries[i]
            ref_idx = id2idx[str(a['ref_img_id'])]
            if a['split'] == 'val':
                ref10.append(int((row > row[ref_idx]).sum().item() + 1 <= 10))
            row[ref_idx] = float('-inf')
            top = [int(img_ids[j]) for j in row.topk(50).indices.cpu().tolist()]
            top10[i] = top[:10]
            if a['split'] == 'val':
                ap10.append(_compute_query_ap_at_k(top, a['gt_img_ids'], 10))
    return {'val_map10': mean(ap10), 'val_ref10': mean(ref10), 'top10': top10}


def nx_s1(label, sigma2, sdir, args, sh):
    import statistics
    import torch
    dev = args.device
    model = nx_load_model(label, dev)
    caps, rows = [q['caption'] for q in sh.queries], [q['ref_row'] for q in sh.queries]
    rec = {'checkpoint': CK[label]['rel'], 'samples': S1_SAMPLES, 'seeds': [S1_SEED0 + k for k in range(S1_SAMPLES)]}
    with torch.no_grad(), R.cudnn_settings(deterministic=False):
        v = sh.vraw.to(dev)
        if sigma2 is None:
            if not hasattr(model.mapper, 'fc_logvar'):
                raise RuntimeError(f'{label} is not a VIB model')
            mu, lv = (torch.cat(x) for x in zip(*[vib_mu_logvar(model, v[s]) for s in batched(v.shape[0], BATCH)]))
            base, scale = mu, (0.5 * lv).exp()
            det = tokens_batched(model, v)[0]                  # mu (+ merger, pseudo_norm): the inference token
            rec.update(kind='posterior samples z = mu + exp(0.5 * clamp(logvar, -10, clip_max)) * eps, then merger '
                            'and pseudo_norm as Pic2Word.forward', sigma2_mean=float((scale ** 2).mean()))
        else:
            base = det = tokens_batched(model, v)[0]
            scale = torch.full_like(base, math.sqrt(sigma2))
            rec.update(kind='matched-noise control: z = z_det + s * eps, s^2 = mean sigma^2 of VIB e01s1500 over the '
                            'CIRCO references and dimensions', s=math.sqrt(sigma2), s2=sigma2)
        eps = nx_s1_eps(sh, tuple(base.shape))
        gal = nx_gallery(dev)
        qd = composed(model, det[rows], caps, dev)
        m_det = nx_quick_metrics(qd, sh, gal)
        outs, norms, maps, refs, jac_test, jac_val = [], [], [], [], [], []
        for k in range(S1_SAMPLES):
            zk = base + scale * eps[k].to(dev)
            if sigma2 is None:
                if model.merger is not None:
                    zk = model.merger(zk)
                zk = project(model, zk)
            norms.append(zk.norm(dim=-1).reshape(zk.shape[0], -1).mean(dim=1).cpu())
            qk = composed(model, zk[rows], caps, dev)
            outs.append(qk)
            m = nx_quick_metrics(qk, sh, gal)
            maps.append(m['val_map10'])
            refs.append(m['val_ref10'])
            for i, t in m['top10'].items():
                j = len(set(t) & set(m_det['top10'][i])) / len(set(t) | set(m_det['top10'][i]))
                (jac_val if sh.queries[i]['split'] == 'val' else jac_test).append(j)
    rec['arrays'] = [save_array(sdir / 'circo_sampled_queries.npy', torch.stack(outs), 'float16'),
                     save_array(sdir / 'sample_token_norms.npy', torch.stack(norms), 'float32')]
    rec['quick'] = {'val_map10_per_sample': maps, 'val_map10_mean': mean(maps), 'val_map10_sd': statistics.pstdev(maps),
                    'val_ref10_per_sample': refs, 'val_ref10_mean': mean(refs), 'val_map10_det': m_det['val_map10'],
                    'val_ref10_det': m_det['val_ref10'], 'test_jaccard10_mean': mean(jac_test),
                    'val_jaccard10_mean': mean(jac_val), 'note': 'scored in batches (not the canonical one-query GEMV); '
                                                                'exploratory'}
    R.write_json(sdir / 'info.json', {k: v for k, v in rec.items()})
    del model
    return rec


@contextlib.contextmanager
def capture_text_attention(model):
    """Temporarily call every text resblock's attention with need_weights=True (head-averaged weights)."""
    blocks = list(model.clip.text_transformer.resblocks)
    if any('attention' in vars(b) for b in blocks):
        raise RuntimeError('a text block already has an instance-level attention override')
    store = [None] * len(blocks)

    def make(blk, li):
        def attention(x):                                     # CLIP's ResidualAttentionBlock.attention + weights
            blk.attn_mask = blk.attn_mask.to(dtype=x.dtype, device=x.device) if blk.attn_mask is not None else None
            y, w = blk.attn(x, x, x, need_weights=True, attn_mask=blk.attn_mask, average_attn_weights=True)
            store[li] = w
            return y
        return attention

    try:
        for li, blk in enumerate(blocks):
            blk.attention = make(blk, li)
        yield store
    finally:
        for blk in blocks:
            if 'attention' in vars(blk):
                del blk.attention


def nx_g10(label, gdir, args, sh):
    import clip
    import torch
    dev = args.device
    model = nx_load_model(label, dev)
    caps, rows = [q['caption'] for q in sh.queries], [q['ref_row'] for q in sh.queries]
    p_eot = model._prompt_eot_idx                             # "a photo of": SOT a photo of -> 4
    arrays, summary, maxdiff, sum_dev = [], {}, 0.0, 0.0
    with torch.no_grad(), R.cudnn_settings(deterministic=False):
        z = tokens_batched(model, sh.vraw.to(dev))[0]
        t = z.shape[1]
        for variant in ('V1', 'V2'):
            n = len(caps) if variant == 'V1' else z.shape[0]
            masses, eot_rows, eots, trunc = [], [], [], []
            for s in batched(n, ATTN_BATCH):
                if variant == 'V1':
                    tok = clip.tokenize(caps[s], truncate=True).to(dev)
                    emb, eot = model._build_token_sequence(z[rows[s]], tok)
                    full = p_eot + t + tok.argmax(dim=-1)        # SOT..of + tokens + caption words + EOT
                    trunc += (full > model.max_token_len).tolist()
                else:
                    emb, eot = model._build_token_sequence(z[s])
                    trunc += [False] * emb.shape[0]
                q0 = model.clip.forward_text_transform(emb, eot)
                with capture_text_attention(model) as store:
                    q1 = model.clip.forward_text_transform(emb, eot)
                maxdiff = max(maxdiff, float((q1 - q0).abs().max()))
                w = torch.stack(store, dim=1)                   # [B, layers, L, S]
                b = w.shape[0]
                rowsw = w[torch.arange(b, device=w.device), :, eot, :]          # [B, layers, S]
                pos = torch.arange(w.shape[-1], device=w.device).unsqueeze(0).expand(b, -1)
                e = eot.unsqueeze(1)
                masks = torch.stack([pos == 0, (pos >= 1) & (pos < p_eot), (pos >= p_eot) & (pos < p_eot + t),
                                     (pos >= p_eot + t) & (pos < e), (pos == e)], dim=1).float()
                m = torch.einsum('bls,bgs->blg', rowsw, masks)
                sum_dev = max(sum_dev, float((m.sum(dim=-1) - 1).abs().max()))
                masses.append(m.cpu())
                eot_rows.append(rowsw.cpu())
                eots.append(eot.cpu())
            masses = torch.cat(masses)
            arrays += [save_array(gdir / f'circo_attn_groups_{variant}.npy', masses, 'float32'),
                       save_array(gdir / f'circo_attn_eotrow_{variant}.npy', torch.cat(eot_rows), 'float16'),
                       save_array(gdir / f'circo_attn_eotpos_{variant}.npy', torch.cat(eots), 'int16')]
            summary[variant] = {'n': n, 'n_truncated': int(sum(trunc)),
                                'mean_mass_by_layer': masses.mean(dim=0).tolist(),
                                'pseudo_mass_by_layer': masses[:, :, 2].mean(dim=0).tolist(),
                                'modifier_mass_by_layer': masses[:, :, 3].mean(dim=0).tolist()}
    restored = all('attention' not in vars(blk) for blk in model.clip.text_transformer.resblocks)
    rec = {'checkpoint': CK[label]['rel'], 'groups': list(GROUPS), 'layers': len(model.clip.text_transformer.resblocks),
           'positions': f'SOT 0, prompt 1..{p_eot - 1}, pseudo {p_eot}..{p_eot}+T-1, modifier up to EOT-1, EOT',
           'max_abs_diff': maxdiff, 'group_sum_max_dev': sum_dev, 'restored': restored, 'summary': summary,
           'arrays': arrays}
    R.write_json(gdir / 'info.json', rec)
    del model
    return rec


def nx_s13(args):
    """neg_analysis_all.py's own load_dataset / load_model / compute_metrics and main()'s aggregation."""
    import torch
    saved = list(sys.path)
    try:
        from evaluation import neg_analysis_all as NA         # inserts codes/ and codes/evaluation into sys.path
    finally:
        sys.path[:] = saved
    NA.DEVICE = args.device
    pt_files = [{'path': str(p.relative_to(R.CIRCO_DIR)), 'bytes': p.stat().st_size}
                for p in sorted(R.CIRCO_DIR.rglob('*.pt'))]
    with R.cudnn_settings(deterministic=False):
        batches = NA.load_dataset(str(R.CIRCO_DIR), 'circo')
        norms = torch.cat([b.norm(dim=-1) for b in batches])
        rec = {'pt_files': pt_files, 'n_batches': len(batches), 'batch_size': int(batches[0].shape[0]),
               'input_norm_min': float(norms.min()), 'input_norm_max': float(norms.max()),
               'input_norm_max_dev': float((norms - 1).abs().max()), 'models': {}}
        for label in S13_MODELS:
            model, ls, meta = NA.load_model(str(ckpt_path(label)))
            all_m, n_q = None, 0
            for b in batches:
                bm = NA.compute_metrics(model, b, ls)
                if all_m is None:
                    all_m = {k: [] for k in bm}
                for k, x in bm.items():
                    all_m[k].extend(x)
                n_q += b.shape[0]
            res = {'n_queries': n_q, 'logit_scale': ls}
            for k, xs in all_m.items():                      # neg_analysis_all.main's mean / population std
                mn = sum(xs) / len(xs)
                res[f'{k}_mean'] = mn
                res[f'{k}_std'] = (sum((x - mn) ** 2 for x in xs) / len(xs)) ** 0.5
            sp = STORED_S13 / f'{key_of(label)}.json'
            stored = R.read_json(sp) if sp.exists() else None
            rec['models'][label] = {'checkpoint': CK[label]['rel'], 'meta': meta, 'now': res, 'paper': PAPER_S13[label],
                                    'stored': {k: v for k, v in stored.items() if k.endswith('_mean')} if stored else None,
                                    'stored_file': str(sp)}
            del model
            free_memory()
    return rec


if __name__ == '__main__':
    main()

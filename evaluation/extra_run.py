"""Eval-only extra run for the arXiv version (nothing is trained): the four-token study at its val argmax with the token
geometry of Supp. C, the Fashion-IQ mean-token / swap queries, fp32 Fashion-IQ per-query ranks, Table S13 in the trained
input regime, the run-2 release copy and the sampling-noise curve. Run once on the final run's A6000 machine, after
evaluation/final_run.py, in the same environment. No file of the earlier runs is changed; the only files written
outside results/extra_2026_11/ are the new release copy (E5) and the COCO images E4 downloads (E4 also deletes its own
incomplete downloads: an image listed in e4_s13/downloaded_files.txt that does not decode, a leftover .part file).

Why
    E1  Supp. C's four-token paragraph (token norms 4.30-7.11, ||v|| 18.58, pairwise token cosines 0.03-0.15 mean 0.09,
        token-v cosines -0.026..+0.007, query-reference cosine at most 0.39) has no source, and Table S9's per-token rows
        were measured at epoch_01, not at the val argmax epoch_00.
    E2  The final run skipped the Fashion-IQ mean-token (V4) and swap (V5, V6) queries (--skip_fashion_swaps).
    E3  The dump keeps Fashion-IQ queries in fp16, which cannot reproduce near-tie ranks: fp32 per-query ranks.
    E4  Table S13 fed unit-norm cached features to a mapper trained on raw CLIP features (norm ~18.6).
    E5  checkpoints_to_share/ holds run 2 at e01s3000; the paper now reports run 2 at its val argmax e02s0500.
    E6  S1 measured one noise scale (a = 1): the curve between the deterministic token (a = 0) and a full sample.

What it reuses (no evaluation step is re-implemented)
    The final run's dump (results/final_2026_11/dump/shared: v_raw of every CIRCO reference, one image at a time, and of
    every Fashion-IQ query in eval_fashion's batches; the saved derangements; S1's eps) and final_run.py's own pieces on
    it: raw_tokens / project / encode_tokens (Pic2Word.encode_query after the image encoder), nx_val_perquery
    (eval_circo_v2's one-query scoring), composed / tokens_batched / vib_mu_logvar / nx_quick_metrics (S1). Fashion-IQ
    is scored as eval_fashion scores it (CPU queries @ gallery.T per category, retrieval.recall_at_k). The release
    recheck uses reeval.py's run_circo_val / run_fashion / recheck_test_file and reeval_addon.py's rules; E4 uses
    neg_analysis_all.py's load_dataset / load_model / compute_metrics. The final run's dump self-check showed this path
    reproduces the canonical evaluators exactly (encode max |d| 0, CIRCO per query, 800 test lists, Fashion-IQ).

Items (each resumable; E1-E6 can be skipped with --skip_eN)
    0 preflight  packages and versions, GPU, every input (data, the earlier runs' records, the dump files read, checked
                 against dump/MANIFEST.sha256), every checkpoint's sha256 against the table, free disk
    1 gate       tok4 e00 "all four combined" through this run's path must reproduce the final run's canonical
                 small_tok4_e00 record exactly: CIRCO val per query (mAP@10 0.0369626, Ref@10 0.990909) and Fashion-IQ
                 per category (R@10 0.0866530); a failure stops the run (exit 2)
    E1 four-token study at tok4 e00 (val argmax) and e01 (the paper's old pick): tokens 0..3 alone and all four combined
       ("token i alone" = z[:, i:i+1] through the same _build_token_sequence path: the prompt carries only that token
       where the model puts its tokens). CIRCO val mAP@5/10/25/50, Ref@10, query-reference cosine; Fashion-IQ R@K per
       category and average; token geometry over the 220 val references and all 1,017 references. e01 is compared
       with the add-on's record (CIRCO exactly, Fashion-IQ within FIQ_QUERY_TOL)
    E2 Fashion-IQ V4 (mean token over the 4,315 unique candidates + caption, re-projected to rho when pseudo_norm > 0),
       V5 (caption swap) and V6 (reference swap) with the dump's derangements (seeds 1, 2) for the 8 core models, built
       in eval_fashion's batches; fp32 features and per-query ranks. Self-check: V1 built the same way reproduces the
       canonical Fashion-IQ recalls (every K within FIQ_QUERY_TOL; exactness recorded)
    E3 fp32 Fashion-IQ per-query ranks of the 19 dumped models (rank of the target, of the reference, which stays in
       the gallery, and of the target with the reference removed). R@1/5/10/50 recomputed from the saved ints must
       equal this run's canonical recall (exact ties at a cut-off are counted and allowed); reported against the final
       run's values (within FIQ_QUERY_TOL required) and the earlier sessions' records
    E5 run-2 release copy checkpoints_to_share/vib_beta0.1_rerun_e02s0500.pt built with the transformation of the
       existing copies (verified first: rebuilding vib_beta0.1.pt and baseline_deterministic.pt from their sources must
       give the same file), never overwriting a file; then the release check: CIRCO val per query, Fashion-IQ and the
       800 test lists identical to the source checkpoint, the stored records and the original submission (f2bcf8fb...)
    E6 noise-scale curve: z = mu + a * sigma * eps (vib, run2, beta02) and z = z_det + a * s * eps (baseline control,
       s as in S1), a in E6_ALPHAS, eps = dump/shared/s1_eps.npy; CIRCO val mAP@10 and Ref@10 per a (mean and
       population sd over the 8 samples) with S1's scoring. Self-check: a = 0 equals the deterministic value (S1's and
       the canonical per-query record), a = 1 reproduces S1's per-sample values
    E4 (last: it downloads ~5,000 COCO images) Table S13 on the 5,120 images neg_analysis_all.load_dataset selects
       (seed-42 permutation of the CIRCO gallery cache rows, first MAX_BATCHES x BATCH_SIZE): images fetched into
       eval_data/circo/COCO2017_unlabeled/unlabeled2017 like prepare_eval_data.py does, every selected image decoded
       in full first (prepare_eval_data.image_ok's verify() passes a download the server cut short; such a file of
       this run is fetched again), encoded with encoder_image_unnorm and checked against the cached features;
       neg_analysis_all.compute_metrics in the raw (trained) and the unit-norm (Table S13) regime for E4_MODELS.
       Self-check: the unit-norm regime reproduces the final run's s13_control exactly

Exact checks and the host
    CIRCO (E1, E5), S1 (E6) and S13 (E4) score on the GPU and have reproduced bit for bit across sessions and hosts (CIRCO
    val of all 19 dumped models against the earlier sessions' records, S13 against the stored May values), so they are
    compared exactly. Fashion-IQ scores on the CPU, where another host can flip a near-tie at a rank cut: it is compared
    within final_run.FIQ_QUERY_TOL queries per category, except in the gate (exact unless --gate_fiq_tol) and in E1's
    "all four == gate" check, which is exact when E1 runs on the host and boot that measured the gate and within
    FIQ_QUERY_TOL otherwise (a restart part-way through the run). host_fingerprint() is saved in every entry. E5 measures
    the source and the release copy in one process: a retry of E5 measures both again.

Outputs (results/extra_2026_11/; dump/ holds the arrays)
    entries/<item>.json             one record per item: status, checks, numbers
    gate/tok4_e00_all.json          the gate measurement (a rerun re-compares it; delete gate/ to measure again)
    e1/<label>_circo_perquery.json  E1 per-query CIRCO blocks of every variant
    e4_s13/selection.json           the 5,120 images (gallery row, COCO id, file)
    e4_s13/downloaded_files.txt     every COCO image this run downloaded (each was absent before; for the cleanup)
    e5_release/                     the transformation check, the release check's canonical records
    dump/models/<label>/            E1 tokens, CIRCO queries, Fashion-IQ ranks; E2 fp32 V1 / V4-V6 and ranks; E3 V1 ranks
    dump/e4_s13/, dump/e6_noise/    E4 raw features, their cosine to the cached ones, per-query metrics; E6 per-query
                                    AP@10 and reference ranks
    summary.json, preflight.json, run_info.json, run_log.txt, MANIFEST.sha256 (every file but itself and the log)

Exit codes: 1 preflight, 2 gate (it did not reproduce, or it raised: the message says which), 3 a self-check failed,
4 an item failed (exception, missing input, failed download) or the summary could not be written, 0 all passed. Every
item runs before a non-zero exit, except after a failed preflight or gate. Rerunning the same command resumes: finished
items are skipped (also one whose self-check failed: delete its entries/<item>.json to measure it again), failed ones are
retried.

Usage (repo at /notebooks/experiment/codes, results/final_2026_11 with dump/ and results/reeval_2026_11 with addon/ in
place; ~/venv and the CLIP weights in ~/.cache/clip are on the machine's local disk, so rebuild them after a restart).
The console output is also written to results/extra_2026_11/run_log.txt, unless stdout already is that file (a nohup
redirect with >>).
    PYTHONPATH=/notebooks/experiment/codes ~/venv/bin/python -u evaluation/extra_run.py --preflight_only
    PYTHONPATH=/notebooks/experiment/codes ~/venv/bin/python -u evaluation/extra_run.py
"""

import argparse
import itertools
import math
import os
import shutil
import statistics
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import reeval as R          # noqa: E402  paths, helpers and canonical evaluation steps
from evaluation import reeval_addon as A    # noqa: E402  release-check rules, checkpoint metadata
from evaluation import final_run as F       # noqa: E402  checkpoint table, comparisons, numerical pieces, the dump


# ---------------------------------------------------------------------------
# Fixed inputs
# ---------------------------------------------------------------------------
FINAL_OUT = REPO_ROOT / 'results' / 'final_2026_11'      # the final run (read only)
DUMP      = FINAL_OUT / 'dump'                           # its dump (read only)
DUMP_MANIFEST_FILES = 329                                # dump/MANIFEST.sha256 lines written by the final run
MIN_FREE_GB = 3.0        # what bytes_added() lists (~1.75 GB with every item) plus the two tarballs (~0.9 GB)
PACK_GB     = 0.9        # extra_results.tar.gz + extra_dump.tar.gz (fp32 arrays barely compress)
N_CIRCO_REFS, N_CIRCO_VAL, D_FEAT = 1017, 220, 768       # dump/shared: unique CIRCO references, val queries
COCO_MEAN_BYTES = 154e3  # COCO unlabeled2017: ~19 GB for 123,403 images

SMALL4_E01 = next(e for e in A.SMALL if e['label'] == 'small_tok4_e01')
_RELEASE = {e['label']: e for e in A.RELEASE}
XCK = {    # checkpoints this run reads that are not in final_run.CK (sha256 from reeval_addon.py's tables)
    'small_tok4_e01': {'rel': SMALL4_E01['rel'], 'sha256': SMALL4_E01['sha256'], 'mapper': 'base', 'pn': 0.0, 'tokens': 4,
                       'title': "~1.5M images, 4 tokens, epoch_01 (the paper's Table S9 checkpoint)"},
    'release_vib': {'rel': _RELEASE['release_vib']['rel'], 'sha256': _RELEASE['release_vib']['sha256'], 'mapper': 'vib',
                    'pn': 0.0, 'tokens': 1, 'title': 'checkpoints_to_share copy of VIB e01s1500'},
    'release_baseline': {'rel': _RELEASE['release_baseline']['rel'], 'sha256': _RELEASE['release_baseline']['sha256'],
                         'mapper': 'base', 'pn': 0.0, 'tokens': 1,
                         'title': 'checkpoints_to_share copy of baseline e02s3000'},
}

# E1
E1_MODELS   = ('small_tok4_e00', 'small_tok4_e01')
E1_VARIANTS = ('t0', 't1', 't2', 't3', 'all')
E1_REFS = {'small_tok4_e00': ('final', 'canonical/small_1p5m'), 'small_tok4_e01': ('addon', 'small_scale')}
GATE_MODEL = 'small_tok4_e00'
E1_EXPECTED = {   # the reference records must hold these values (final run e00; add-on e01): CIRCO mAP@10, Ref@10, F R@10
    'small_tok4_e00': (0.03696260221260221, 0.990909090909091, 0.08665298290070943),
    'small_tok4_e01': (0.027829545454545454, 0.9954545454545455, 0.08818373886618358)}
CIRCO_OFFICIAL_TOL = 1e-12   # per-query mean vs the official compute_metrics value (float summation order only)
PAPER_S9 = {'t0': (17.73, 3.56, 3.18), 't1': (18.41, 3.39, 0.91), 't2': (17.88, 3.97, 6.36), 't3': (16.76, 3.51, 3.18),
            'all': (8.80, 2.78, 99.55)}   # Table S9 (epoch_01, exploratory pipeline): F R@10, CIRCO mAP@10, Ref@10
PAPER_GEOMETRY = ('token norms 4.30-7.11; ||v|| 18.58; pairwise token cosines 0.03-0.15 (mean 0.09); token-v cosines '
                  '-0.026..+0.007; no combined query above cosine 0.39 with the reference')
# E2 / E3
E2_MODELS = F.DUMP_CORE
E2_VARIANTS = ('V1', 'V4') + tuple(f'V{v}_s{s}' for v in (5, 6) for s in F.SWAP_SEEDS)
E3_MODELS = F.DUMP_CORE + F.DUMP_EXTRA
FIQ_K = (1, 5, 10, 50)
RANK_COLUMNS = ['rank_target', 'ties_target', 'rank_ref', 'ties_ref', 'rank_target_noref', 'rank_tokref'] + \
               [f'hit@{k}' for k in FIQ_K]
RANK_NOTE = ('rank = 1 + #(gallery scores strictly above); ties = #(other gallery rows with exactly the same score); '
             'ref = the query\'s reference (candidate) image, which stays in the Fashion-IQ gallery; rank_target_noref = '
             'target rank with the reference row removed; rank_tokref = rank of the image whose token the query carries '
             '(V6: the swapped-in reference; V4: -1); hit@K = target in torch.topk(sims, K), i.e. the hits '
             'retrieval.recall_at_k counts')
# E4
E4_MODELS = ('baseline', 'vib', 'run2', 'beta02', 'margin', 'noise_e00s3500', 'unit_norm')
E4_CHECK_MODELS = F.S13_MODELS              # the final run's s13_control has these two
E4_ENCODE_BATCH = 64                        # build_circo_gallery.py's default batch
E4_MIN_COS = 0.99                           # re-encoded (fp32) vs cached (fp16 encoder) feature of the same image
E4_V2_TOL = 1e-4                            # compute_metrics(raw) positive similarity vs the dump's V2 cosine
E4_REGIMES = ('raw', 'unit_norm')
E4_N_IMAGES = 5 * 1024                      # neg_analysis_all MAX_BATCHES x BATCH_SIZE (for the size estimate only)
E4_LEDGER = 'downloaded_files.txt'          # e4_s13/: every image this run started to download (it was absent then)
E4_LOWEST = 10                              # images with the lowest re-encoded vs cached cosine listed in the entry
# E5
NEW_RELEASE_REL = 'checkpoints_to_share/vib_beta0.1_rerun_e02s0500.pt'
RUN2_SUBMISSION_SHA256 = 'f2bcf8fb7bf985877b12a9b1e1752a49aa77b694ea30d30629c07164e46f923f'
RELEASE_PAIRS = (('release_vib', 'vib', 0.1024), ('release_baseline', 'baseline', None))
RELEASE_KEYS = ('mapper_state', 'config', 'mapper_kwargs', 'merger_state', 'mapper_type', 'logit_scale')
RELEASE_DROPPED = ('epoch', 'step', 'optimizer', 'metrics', 'total_steps')
MACHINE_PREFIX, DATA_ROOT = '/notebooks/experiment', '<data_root>'
# E6
E6_MODELS = F.S1_MODELS
E6_ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)

# A6000 estimates (seconds) from the final run's run_log: a model load ~6 s; canonical CIRCO val ~0.7 min and
# Fashion-IQ ~1.65 min per model (with image encoding); text-only encodes on the dump's v_raw are much cheaper
EST = {'gate': 25, 'e1': 80, 'e2': 45, 'e3': 15, 'e5': 380, 'e6': 60, 'e4_images': 150, 'e4_model': 25}


def _self_check_tables():
    assert set(E1_MODELS) <= set(F.CK) | set(XCK) and GATE_MODEL in E1_MODELS and set(E1_REFS) == set(E1_MODELS)
    assert all(l in F.CK for l in E2_MODELS + E3_MODELS + E4_MODELS + E6_MODELS)
    assert set(E2_MODELS) <= set(E3_MODELS) and set(E4_CHECK_MODELS) <= set(E4_MODELS) and set(E4_MODELS) <= set(E2_MODELS)
    assert E6_ALPHAS[0] == 0.0 and 1.0 in E6_ALPHAS and len(set(E6_ALPHAS)) == len(E6_ALPHAS)
    assert len({c['sha256'] for c in list(F.CK.values()) + list(XCK.values())}) == len(F.CK) + len(XCK)
    assert all(c['pn'] == 0.0 for c in XCK.values())


_self_check_tables()
_XC = {}       # per-process caches (shared encodes, Fashion-IQ galleries, E4 batches)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def ck_info(label):
    return F.CK[label] if label in F.CK else XCK[label]


def ck_path(label):
    return R.EXPERIMENT / ck_info(label)['rel']


def key_of(label):
    return R.key_for(ck_path(label))


def ck_title(label):
    return f'{ck_info(label)["title"]}  ({ck_info(label)["rel"]})'


def check(ok, **info):
    return dict(ok=bool(ok), **info)


def all_ok(checks):
    return all(c['ok'] for c in checks.values())


def fiq_avg(rec):
    """Exact mean of the per-category recalls (reeval_addon.numbers' fiq_avg_exact)."""
    return {k: sum(rec[c][k] for c in R.FIQ_CATS) / len(R.FIQ_CATS) for k in R.KS}


def recall_cmp(now, ref, tol):
    """Fashion-IQ recalls {cat: {k: fraction}} against a reference; ok = every category and K within tol queries."""
    rows = []
    for cat in R.FIQ_CATS:
        for k in R.KS:
            a, b = now[cat][k], ref[cat][k]
            rows.append({'value': f'{cat} R@{k}', 'now': a, 'reference': b, 'same': a == b,
                         'query_diff': F.fiq_query_diff(cat, a, b)})
    worst = max(abs(r['query_diff']) for r in rows)
    return {'ok': worst <= tol, 'exact': all(r['same'] for r in rows), 'max_query_diff': worst, 'query_tol': tol,
            'differing': [r for r in rows if not r['same']]}


def final_run_fiq(label):
    """The final run's own Fashion-IQ recalls of a dumped model (its dump self-check, computed on this machine)."""
    rows = R.read_json(FINAL_OUT / 'entries' / f'dump_{label}.json')['selfcheck']['checks']['fashion_recalls']['rows']
    out = {c: {} for c in R.FIQ_CATS}
    for r in rows:
        cat, k = r['value'].split(' R@')
        out[cat][k] = r['now']
    return out


def circo_summary(pq):
    """CIRCO-val aggregates of an eval_circo_v2 per-query block (means in query order, as final_run.mean)."""
    gts = [x for x in pq['sim_gt_mean'] if x is not None]
    ranks = pq['rank_ref']
    return {'n_queries': len(pq['query_id']),
            'map': {k: F.mean(pq['ap_at_k'][f'@{k}']) for k in ('5', '10', '25', '50')},
            'recall': {k: F.mean(pq['recall_at_k'][f'@{k}']) for k in ('5', '10', '25', '50')},
            'ref1': F.mean([int(r <= 1) for r in ranks]), 'ref10': F.mean([int(r <= 10) for r in ranks]),
            'median_rank_ref': statistics.median(ranks), 'sim_ref_mean': F.mean(pq['sim_ref']),
            'sim_ref_max': max(pq['sim_ref']), 'sim_gt_mean': F.mean(gts), 'sim_delta': F.mean(pq['sim_ref']) - F.mean(gts)}


def rank_check(summary):
    """Rank table vs canonical recall: hit columns equal recall_at_k exactly, ranks consistent with the hits, and
    recall from the ranks equal to it except where an exact score tie straddles the cut-off."""
    problems, exact, ambiguous = [], True, 0
    for cat, s in summary.items():
        n = s['n']
        for k, x in s['at_k'].items():
            ambiguous += x['tie_ambiguous']
            if x['from_hits'] != x['canonical']:
                problems.append(f'{cat} R@{k}: hit columns give {x["from_hits"]}, recall_at_k {x["canonical"]}')
            if x['violations']:
                problems.append(f'{cat} R@{k}: {x["violations"]} queries whose rank contradicts the topk hit')
            if x['from_ranks'] != x['canonical']:
                exact = False
                gap = round((x['from_ranks'] - x['canonical']) * n)
                if not (x['from_ranks_pessimistic'] <= x['canonical'] <= x['from_ranks'] and gap <= x['tie_ambiguous']):
                    problems.append(f'{cat} R@{k}: ranks give {x["from_ranks"]}, canonical {x["canonical"]} '
                                    f'({x["tie_ambiguous"]} tie-ambiguous queries)')
    return check(not problems, exact=exact, tie_ambiguous_total=ambiguous, problems=problems[:20])


def rank_numbers(summary):
    """Per category and averaged: R@K from hits (canonical), [ties-for, ties-against] bounds, reference in the top K,
    recall with the reference removed, share of queries whose reference outranks the target."""
    cats = list(summary)
    out = {'per_category': summary, 'avg': {}}
    for k in (str(x) for x in FIQ_K):
        for f in ('from_hits', 'from_ranks', 'from_ranks_pessimistic', 'ref_in_top', 'recall_noref'):
            out['avg'].setdefault(f, {})[k] = sum(summary[c]['at_k'][k][f] for c in cats) / len(cats)
    out['avg']['ref_above_target'] = sum(summary[c]['ref_above_target'] for c in cats) / len(cats)
    return out


def log_fiq(prefix, rec):
    a = fiq_avg(rec)
    R.log(f'  {prefix:24s} R@10 ' + ' '.join(f'{c} {100 * rec[c]["10"]:5.2f}' for c in R.FIQ_CATS)
          + f'  avg {100 * a["10"]:.4f}  R@50 avg {100 * a["50"]:.4f}')


def host_fingerprint():
    """The machine that scores Fashion-IQ on its CPU: the kernel boot id (new on every boot and on any other host; a
    container shares its host's), the CPU model and count and torch's CPU threads; the hostname where /proc is missing.
    Boot id and hostname are stored as sha256 prefixes, so the records carry no machine identifier."""
    import hashlib
    import platform

    def digest(s):
        return hashlib.sha256(s.encode()).hexdigest()[:16] if s else None

    fp = {'hostname': digest(platform.node()), 'cpu_count': os.cpu_count(), 'boot_id': None, 'cpu_model': None,
          'torch_threads': None}
    try:
        fp['boot_id'] = digest(Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except OSError:
        pass
    try:
        fp['cpu_model'] = next(ln.split(':', 1)[1].strip() for ln in Path('/proc/cpuinfo').read_text().splitlines()
                               if ln.startswith('model name'))
    except (OSError, StopIteration):
        fp['cpu_model'] = platform.processor() or None
    try:
        import torch
        fp['torch_threads'] = torch.get_num_threads()
    except Exception:                                         # noqa: BLE001
        pass
    return fp


def same_host(a, b):
    """Same boot of the same machine (boot id when both have one, else hostname) with the same CPU and threads."""
    if not a or not b:
        return False
    keys = ('cpu_model', 'cpu_count', 'torch_threads') + (('boot_id',) if a.get('boot_id') and b.get('boot_id')
                                                           else ('hostname',))
    return all(a.get(k) == b.get(k) for k in keys)


def cephfs_usage(path):
    """(directory, quota bytes, used bytes) of the nearest CephFS quota at or above path, or None. Paperspace's
    /notebooks is a network (Ceph) mount whose statfs can show the whole cluster rather than the plan's quota."""
    get = getattr(os, 'getxattr', None)                       # Linux only
    if get is None:
        return None
    for p in [Path(path)] + list(Path(path).parents):
        try:
            quota, used = int(get(str(p), 'ceph.quota.max_bytes')), int(get(str(p), 'ceph.dir.rbytes'))
        except (OSError, ValueError):
            continue
        if quota > 0:
            return str(p), quota, used
    return None


def bytes_added(args, out):
    """What this run writes, from the array shapes (JSON records and the release copy are a few MB)."""
    n_fiq = sum(F.FIQ_N.values())
    ranks = n_fiq * len(RANK_COLUMNS) * 4
    parts = {}
    if not args.skip_e1:
        parts['E1 tokens, queries, ranks'] = len(E1_MODELS) * (N_CIRCO_REFS * 4 * D_FEAT * 4 + len(E1_VARIANTS) * (
            N_CIRCO_VAL * D_FEAT * 4 + ranks))
    if not args.skip_e2:
        parts['E2 fp32 arrays'] = len(E2_MODELS) * (len(E2_VARIANTS) * n_fiq * D_FEAT * 4 + (len(E2_VARIANTS) - 1) * ranks)
    if not args.skip_e3:
        parts['E3 ranks'] = len(E3_MODELS) * ranks
    if not args.skip_e4:
        parts['E4 arrays'] = E4_N_IMAGES * (D_FEAT * 4 + 4 + len(E4_MODELS) * len(E4_REGIMES) * 11 * 8)
        done = out / 'entries' / 'e4_images.json'
        if not (done.exists() and R.read_json(done).get('status') == 'done'):
            parts['E4 COCO images (at most)'] = E4_N_IMAGES * COCO_MEAN_BYTES
    return parts


# ---------------------------------------------------------------------------
# Step 0: preflight
# ---------------------------------------------------------------------------
def needed_labels(args):
    need = {GATE_MODEL}
    if not args.skip_e1:
        need |= set(E1_MODELS)
    if not args.skip_e2:
        need |= set(E2_MODELS)
    if not args.skip_e3:
        need |= set(E3_MODELS)
    if not args.skip_e4:
        need |= set(E4_MODELS) | {'baseline'}
    if not args.skip_e5:
        need |= {'run2'} | {p[0] for p in RELEASE_PAIRS} | {p[1] for p in RELEASE_PAIRS}
    if not args.skip_e6:
        need |= set(E6_MODELS) | {'baseline'}
    return sorted(need, key=lambda l: (l not in F.CK, list(F.CK).index(l) if l in F.CK else 0, l))


def dump_files(args):
    """The dump files this run reads (relative to dump/)."""
    files = ['shared/circo_vraw.npy', 'shared/fashion_vraw_query.npy', 'shared/circo_index.json',
             'shared/fashion_index.json', 'shared/derangements.json', 'shared/info.json', 'dump_info.json']
    if not args.skip_e6:
        files.append('shared/s1_eps.npy')
    if not args.skip_e4:
        files += [f'models/{l}/circo_V2.npy' for l in E4_MODELS]
    return files


def record_inputs(args):
    """Records of the earlier runs that the comparisons read."""
    files = []
    for label in E1_MODELS if not args.skip_e1 else (GATE_MODEL,):
        where, sub = E1_REFS[label]
        base = (FINAL_OUT if where == 'final' else F.ADDON_OUT) / sub
        files += [R.val_file(base, key_of(label)), R.fiq_file(base, key_of(label))]
    fiq_models = (E2_MODELS if not args.skip_e2 else ()) + (E3_MODELS if not args.skip_e3 else ())
    for label in dict.fromkeys(fiq_models):
        files += [F.reference_paths(label, FINAL_OUT)['fiq'], FINAL_OUT / 'entries' / f'dump_{label}.json']
    if not args.skip_e4:
        files.append(FINAL_OUT / 's13' / 's13_control.json')
    if not args.skip_e5:
        key = key_of('run2')
        files += [R.val_file(F.REEVAL_OUT, key), R.fiq_file(F.REEVAL_OUT, key), R.STORED_VAL / f'{key}.json',
                  R.ORIGINAL_SUBMISSIONS / f'{key}__submission.json']
    if not args.skip_e6:
        files += [FINAL_OUT / 'entries' / f's1_{l}.json' for l in E6_MODELS + ('baseline_control',)]
        files += [F.reference_paths(l, FINAL_OUT)['val'] for l in E6_MODELS + ('baseline',)]
    return list(dict.fromkeys(files))


def input_problems(args, out):
    problems, warnings = [], []
    problems += [f'Missing input from an earlier run: {p}' for p in record_inputs(args) if not Path(p).exists()]
    for label, want in E1_EXPECTED.items():
        vp, fp = e1_reference(label)
        if (label == GATE_MODEL or not args.skip_e1) and vp.exists() and fp.exists():
            v2, per = R.read_json(vp)['v2'], R.read_json(fp)['per_category']
            have = (float(v2['map_atk']['@10']), float(v2['stickiness']['r10_ref']), fiq_avg(per)['10'])
            if have != want:
                problems.append(f'{vp.name} ({label}): reference values {have} differ from the expected {want}')
    if not args.skip_e5:
        sub = R.ORIGINAL_SUBMISSIONS / f'{key_of("run2")}__submission.json'
        if sub.exists() and R.sha256_file(sub) != RUN2_SUBMISSION_SHA256:
            problems.append(f'{sub}: sha256 differs from the uploaded file ({RUN2_SUBMISSION_SHA256[:16]}...)')
        new = R.EXPERIMENT / NEW_RELEASE_REL
        if new.exists():
            warnings.append(f'{new} exists already: E5 reuses it only if it is identical to the rebuilt copy')
    if not args.skip_e4:
        s13 = FINAL_OUT / 's13' / 's13_control.json'
        if s13.exists() and not R.read_json(s13).get('unit_norm_regime_confirmed'):
            problems.append(f'{s13} does not confirm the unit-norm regime')
    # the dump: every file this run reads must match dump/MANIFEST.sha256
    man_path = DUMP / 'MANIFEST.sha256'
    if not man_path.exists():
        problems.append(f'Missing {man_path}')
        manifest = {}
    else:
        lines = man_path.read_text().splitlines()
        manifest = {ln.split('  ', 1)[1]: ln.split('  ', 1)[0] for ln in lines if '  ' in ln}
        if len(lines) != DUMP_MANIFEST_FILES:
            warnings.append(f'{man_path} lists {len(lines)} files, the final run wrote {DUMP_MANIFEST_FILES}')
    checked = {}
    for rel in dump_files(args):
        p = DUMP / rel
        if not p.exists():
            problems.append(f'Missing dump file {p}')
        elif rel not in manifest:
            problems.append(f'{rel} is not in dump/MANIFEST.sha256')
        else:
            checked[rel] = R.sha256_file(p) == manifest[rel]
            if not checked[rel]:
                problems.append(f'dump/{rel}: sha256 differs from dump/MANIFEST.sha256')
    out.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out).free / 1e9
    adds = bytes_added(args, out)
    disk = {'free_gb': round(free_gb, 1), 'adds_gb': {k: round(v / 1e9, 3) for k, v in adds.items()},
            'adds_total_gb': round(sum(adds.values()) / 1e9, 2), 'pack_gb': PACK_GB, 'needs_gb': MIN_FREE_GB}
    if free_gb < MIN_FREE_GB:
        problems.append(f'Only {free_gb:.1f} GB free under {out}; this run needs about {MIN_FREE_GB:.0f} GB')
    ceph = cephfs_usage(out)
    if ceph:
        qdir, quota, used = ceph
        disk.update(quota_dir=qdir, quota_gb=round(quota / 1e9, 2), used_gb=round(used / 1e9, 2))
        if (quota - used) / 1e9 < MIN_FREE_GB:
            problems.append(f'The storage quota of {qdir} leaves {(quota - used) / 1e9:.1f} GB ({used / 1e9:.1f} of '
                            f'{quota / 1e9:.1f} GB used); this run and its tarballs need about {MIN_FREE_GB:.0f} GB')
    else:
        warnings.append(f'the volume reports {free_gb:,.0f} GB free; on a shared network mount that need not be your '
                        f'storage quota: check that about {MIN_FREE_GB:.0f} GB of it are left (this run adds about '
                        f'{disk["adds_total_gb"]:.2f} GB, the two tarballs about {PACK_GB} GB)')
    return problems, warnings, {'dump_files_checked': checked, 'free_gb': round(free_gb, 1), 'disk': disk}


def checkpoint_problems(labels):
    """Present, sha256 equal to the table, mapper type / tokens / merger / pseudo_norm as expected (as final_run)."""
    rows, problems = {}, []
    for label in labels:
        c, p = ck_info(label), ck_path(label)
        row = rows[label] = {'path': str(p), 'sha256_expected': c['sha256']}
        if not p.exists():
            problems.append(f'Missing checkpoint {p}')
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


def estimate(args):
    steps = [('gate (tok4 e00, all four)', EST['gate'])]
    if not args.skip_e1:
        steps.append((f'E1 four-token study ({len(E1_MODELS)} checkpoints x {len(E1_VARIANTS)} variants)',
                      EST['e1'] * len(E1_MODELS)))
    if not args.skip_e2:
        steps.append((f'E2 Fashion-IQ V4-V6 ({len(E2_MODELS)} models)', EST['e2'] * len(E2_MODELS)))
    if not args.skip_e3:
        steps.append((f'E3 Fashion-IQ ranks ({len(E3_MODELS)} models)', EST['e3'] * len(E3_MODELS)))
    if not args.skip_e5:
        steps.append(('E5 release copy + canonical recheck (2x CIRCO val, 2x Fashion-IQ, 1 test file)', EST['e5']))
    if not args.skip_e6:
        steps.append((f'E6 noise curve ({len(E6_MODELS) + 1} models x {len(E6_ALPHAS)} scales)',
                      EST['e6'] * (len(E6_MODELS) + 1)))
    if not args.skip_e4:
        steps.append(('E4 images (selection, encoding; plus the COCO download time)', EST['e4_images']))
        steps.append((f'E4 S13 metrics ({len(E4_MODELS)} models x 2 regimes)', EST['e4_model'] * len(E4_MODELS)))
    return steps


# ---------------------------------------------------------------------------
# Step 1: gate
# ---------------------------------------------------------------------------
def e1_reference(label):
    where, sub = E1_REFS[label]
    base = (FINAL_OUT if where == 'final' else F.ADDON_OUT) / sub
    return R.val_file(base, key_of(label)), R.fiq_file(base, key_of(label))


def compare_all_four(meas, label, fiq_tol):
    """'all four combined' against the canonical record of the checkpoint (final run e00 / add-on e01)."""
    vp, fp = e1_reference(label)
    rc, rf = R.read_json(vp), R.read_json(fp)
    pq = F.compare_perquery(meas['circo_perquery'], rc['v2']['per_query'])
    s = circo_summary(meas['circo_perquery'])
    off = float(rc['v2']['map_atk']['@10'])
    fr = recall_cmp(meas['fashion_recalls'], rf['per_category'], fiq_tol)
    return {
        'circo_val_per_query': check(pq['exact'], reference=str(vp), **pq),
        'circo_map10_official': check(abs(s['map']['10'] - off) <= CIRCO_OFFICIAL_TOL, now=s['map']['10'], official=off,
                                      abs_diff=abs(s['map']['10'] - off)),
        'circo_ref10': check(s['ref10'] == float(rc['v2']['stickiness']['r10_ref']), now=s['ref10'],
                             reference=float(rc['v2']['stickiness']['r10_ref'])),
        'fashion': dict(fr, reference=str(fp), avg_r10_now=fiq_avg(meas['fashion_recalls'])['10'],
                        avg_r10_reference=fiq_avg(rf['per_category'])['10'])}


def gate_item(out, args, cfg):
    path = out / 'gate' / 'tok4_e00_all.json'
    if path.exists():
        meas = R.read_json(path)
        R.log(f'  measurement: re-compared from {path} (delete gate/ to measure again)')
    else:
        raw = xn_e1(GATE_MODEL, ('all',), None, args, cfg, xn_shared())
        v = raw['variants']['all']
        meas = {'label': GATE_MODEL, 'checkpoint': ck_info(GATE_MODEL)['rel'], 'measured_at': R.now(),
                'circo_perquery': v['circo_perquery'], 'fashion_recalls': v['fashion_recalls'],
                'tau': raw['tau'], 'versions': F.versions(), 'host': host_fingerprint()}
        R.write_json(path, meas)
    checks = compare_all_four(meas, GATE_MODEL, args.gate_fiq_tol)
    s = circo_summary(meas['circo_perquery'])
    pq, fr = checks['circo_val_per_query'], checks['fashion']
    R.log(f'  CIRCO val vs final run: {pq["n_common"]}/{pq["n_reference"]} queries, mismatches '
          f'{sum(pq["mismatch_counts"].values())} (AP/recall at all K, reference and best-GT ranks); mAP@10 '
          f'{s["map"]["10"]:.10f} (official {checks["circo_map10_official"]["official"]:.10f}), Ref@10 {s["ref10"]:.6f}'
          f'  {"ok" if pq["ok"] else "DIFFERS"}')
    R.log(f'  Fashion-IQ vs final run: avg R@10 {fr["avg_r10_now"]:.10f} (reference {fr["avg_r10_reference"]:.10f}); '
          f'largest change {fr["max_query_diff"]} queries (allowed {args.gate_fiq_tol})  {"ok" if fr["ok"] else "DIFFERS"}')
    for r in fr['differing']:
        R.log(f'    differs: {r["value"]} {r["reference"]:.6f} -> now {r["now"]:.6f} ({r["query_diff"]:+d} queries)')
    return {'label': GATE_MODEL, 'checkpoint': ck_info(GATE_MODEL)['rel'], 'measurement': str(path),
            'numbers': {'circo': s, 'fashion': meas['fashion_recalls'], 'fashion_avg': fiq_avg(meas['fashion_recalls'])},
            'checks': checks, 'ok': all_ok(checks)}


# ---------------------------------------------------------------------------
# E1: four-token study
# ---------------------------------------------------------------------------
def e1_item(label, out, args, cfg):
    mdir = out / 'dump' / 'models' / label
    raw = xn_e1(label, E1_VARIANTS, mdir, args, cfg, xn_shared())
    numbers, perquery = {}, {}
    for var, v in raw['variants'].items():
        perquery[var] = v['circo_perquery']
        numbers[var] = {'circo': circo_summary(v['circo_perquery']), 'fashion': v['fashion_recalls'],
                        'fashion_avg': fiq_avg(v['fashion_recalls']), 'fashion_ranks': rank_numbers(v['rank_summary'])}
    R.write_json(out / 'e1' / f'{label}_circo_perquery.json', {'label': label, 'variants': perquery})
    checks = {f'fashion_ranks_{var}': rank_check(v['rank_summary']) for var, v in raw['variants'].items()}
    meas = {'circo_perquery': perquery['all'], 'fashion_recalls': raw['variants']['all']['fashion_recalls']}
    if label == GATE_MODEL:     # the same computation as the gate's measurement: identical on the gate's host and boot
        g = R.read_json(out / 'gate' / 'tok4_e00_all.json')
        pq = F.compare_perquery(meas['circo_perquery'], g['circo_perquery'])
        here = host_fingerprint()
        same = same_host(g.get('host'), here)
        fr = recall_cmp(meas['fashion_recalls'], g['fashion_recalls'], 0 if same else F.FIQ_QUERY_TOL)
        checks['all_equals_gate'] = check(pq['exact'] and fr['ok'], circo_per_query_exact=pq['exact'],
                                          fashion_identical=fr['exact'], fashion_max_query_diff=fr['max_query_diff'],
                                          fashion_query_tol=fr['query_tol'], same_host_as_gate=same,
                                          host_gate=g.get('host'), host_now=here,
                                          rule='CIRCO per query exact (GPU scoring); Fashion-IQ identical on the host '
                                               'and boot that measured the gate, within FIQ_QUERY_TOL queries on '
                                               'another (CPU scoring)')
        R.log(f'  all four vs the gate measurement: CIRCO per query {"exact" if pq["exact"] else "DIFFERS"}; Fashion-IQ '
              f'{"identical" if fr["exact"] else str(fr["max_query_diff"]) + " queries max"} '
              f'({"same host" if same else "another host or boot: within " + str(F.FIQ_QUERY_TOL) + " queries"})  '
              f'{"ok" if checks["all_equals_gate"]["ok"] else "DIFFERS"}')
    cmp = compare_all_four(meas, label, F.FIQ_QUERY_TOL)
    checks['all_vs_canonical_circo'] = cmp['circo_val_per_query']
    checks['all_vs_canonical_circo_official_map10'] = cmp['circo_map10_official']
    checks['all_vs_canonical_fashion'] = cmp['fashion']
    for var in E1_VARIANTS:
        c, f = numbers[var]['circo'], numbers[var]['fashion_avg']
        R.log(f'  {var:4s} CIRCO mAP@5/10/25/50 ' + '/'.join(f'{100 * c["map"][k]:.2f}' for k in ('5', '10', '25', '50'))
              + f'  Ref@10 {100 * c["ref10"]:6.2f}%  q-ref cos max {c["sim_ref_max"]:.3f} mean {c["sim_ref_mean"]:.3f}  '
              f'Fashion R@10/50 {100 * f["10"]:.2f}/{100 * f["50"]:.2f}')
    R.write_json(mdir / 'info_e1.json', {'label': label, 'checkpoint': ck_info(label)['rel'], 'item': 'E1',
                                         'arrays': raw['arrays'], 'rank_columns': RANK_COLUMNS, 'rank_note': RANK_NOTE,
                                         'variants': list(E1_VARIANTS), 'created_at': R.now(),
                                         'batching': 'tokens and CIRCO queries one reference / query at a time '
                                                     '(eval_circo_v2); Fashion-IQ in eval_fashion batches'})
    g = raw['geometry']['val']
    norms = ' '.join(f'{g["token_norm"][str(i)]["mean"]:.2f}' for i in range(4))
    R.log(f'  geometry ({g["n_refs"]} val refs): token norms {norms}'
          + f'; ||v|| {g["v_norm"]["mean"]:.2f}; pairwise cos {g["pairwise_cos_over_pairs"]["min"]:.3f}..'
          f'{g["pairwise_cos_over_pairs"]["max"]:.3f} (mean {g["pairwise_cos_over_pairs"]["mean"]:.3f}); token-v cos '
          f'{g["token_v_cos_over_tokens"]["min"]:+.4f}..{g["token_v_cos_over_tokens"]["max"]:+.4f}')
    a = cmp['fashion']
    R.log(f'  all four vs {"final run" if label == GATE_MODEL else "add-on"} record: CIRCO per query '
          f'{"exact" if cmp["circo_val_per_query"]["ok"] else "DIFFERS"}; Fashion-IQ '
          f'{"exact" if a["exact"] else str(a["max_query_diff"]) + " queries max"}  {"ok" if a["ok"] else "DIFFERS"}')
    return {'label': label, 'checkpoint': ck_info(label)['rel'], 'sha256': ck_info(label)['sha256'], 'tau': raw['tau'],
            'variants': {v: ('the four tokens together (= encode_query)' if v == 'all' else
                             f'token {v[1]} alone: z[:, {v[1]}:{int(v[1]) + 1}] through _build_token_sequence')
                         for v in E1_VARIANTS},
            'numbers': numbers, 'geometry': raw['geometry'], 'paper_table_s9_e01': PAPER_S9,
            'paper_geometry': PAPER_GEOMETRY, 'arrays': raw['arrays'], 'rank_columns': RANK_COLUMNS,
            'rank_note': RANK_NOTE, 'checks': checks, 'ok': all_ok(checks)}


# ---------------------------------------------------------------------------
# E2: Fashion-IQ mean-token and swap queries; E3: Fashion-IQ ranks
# ---------------------------------------------------------------------------
def e2_item(label, out, args, cfg):
    mdir = out / 'dump' / 'models' / label
    raw = xn_e2(label, mdir, args, cfg, xn_shared())
    refs = F.reference_paths(label, FINAL_OUT)
    v1 = raw['variants']['V1']['fashion_recalls']
    canon = recall_cmp(v1, R.read_json(refs['fiq'])['per_category'], F.FIQ_QUERY_TOL)
    same_machine = recall_cmp(v1, final_run_fiq(label), F.FIQ_QUERY_TOL)
    checks = {'v1_vs_canonical': dict(canon, reference=str(refs['fiq'])),
              'v1_vs_final_run_dump': dict(same_machine, reference=f'final run entries/dump_{label}.json')}
    checks.update({f'ranks_{v}': rank_check(x['rank_summary']) for v, x in raw['variants'].items()})
    numbers = {v: {'fashion': x['fashion_recalls'], 'fashion_avg': fiq_avg(x['fashion_recalls']),
                   'ranks': rank_numbers(x['rank_summary'])} for v, x in raw['variants'].items()}
    for v in E2_VARIANTS:
        log_fiq(v, raw['variants'][v]['fashion_recalls'])
    R.log(f'  V1 vs canonical: {"exact" if canon["exact"] else str(canon["max_query_diff"]) + " queries max"}; vs the '
          f'final run on this machine: {"exact" if same_machine["exact"] else str(same_machine["max_query_diff"]) + " max"}'
          f'  {"ok" if canon["ok"] and same_machine["ok"] else "DIFFERS"}')
    R.write_json(mdir / 'info_e2.json', {
        'label': label, 'checkpoint': F.CK[label]['rel'], 'item': 'E2', 'arrays': raw['arrays'],
        'variants': {'V1': 'composed (canonical; the self-check): fashion_V1_fp32.npy; its ranks are E3\'s '
                           'fashion_V1_ranks.npy',
                     'V4': F.VARIANTS['V4'], 'V5_s<seed>': F.VARIANTS['V5_s<seed>'], 'V6_s<seed>': F.VARIANTS['V6_s<seed>']},
        'batching': "eval_fashion's batches of cfg.batch_size per category (dump/shared/fashion_index.json); tokens of "
                    'each query computed in its own batch; V4 z_bar = mean of those tokens over the unique candidates',
        'rank_columns': RANK_COLUMNS, 'rank_note': RANK_NOTE, 'zbar_norm': raw['zbar_norm'], 'created_at': R.now()})
    return {'label': label, 'checkpoint': F.CK[label]['rel'], 'numbers': numbers, 'arrays': raw['arrays'],
            'tau': raw['tau'], 'pseudo_norm_in_model': raw['pseudo_norm'], 'zbar_norm': raw['zbar_norm'],
            'checks': checks, 'ok': all_ok(checks)}


def e3_item(label, out, args, cfg):
    mdir = out / 'dump' / 'models' / label
    raw = xn_e3(label, mdir, args, cfg, xn_shared())
    refs = F.reference_paths(label, FINAL_OUT)
    rec = raw['fashion_recalls']
    final = recall_cmp(rec, final_run_fiq(label), F.FIQ_QUERY_TOL)
    earlier = recall_cmp(rec, R.read_json(refs['fiq'])['per_category'], F.FIQ_QUERY_TOL)
    checks = {'ranks_vs_this_run_recall': rank_check(raw['rank_summary']),
              'recall_vs_final_run': dict(final, reference=f'final run entries/dump_{label}.json')}
    numbers = {'fashion': rec, 'fashion_avg': fiq_avg(rec), 'ranks': rank_numbers(raw['rank_summary']),
               'vs_canonical_record': dict(earlier, reference=str(refs['fiq']))}
    rk = numbers['ranks']['avg']
    R.log(f'  canonical R@1/5/10/50 avg ' + '/'.join(f'{100 * numbers["fashion_avg"][k]:.2f}' for k in R.KS)
          + f'; reference in top 1/10 {100 * rk["ref_in_top"]["1"]:.2f}/{100 * rk["ref_in_top"]["10"]:.2f}%; R@10 without '
          f'the reference {100 * rk["recall_noref"]["10"]:.2f}; tie-ambiguous '
          f'{checks["ranks_vs_this_run_recall"]["tie_ambiguous_total"]}')
    R.log(f'  vs final run: {"exact" if final["exact"] else str(final["max_query_diff"]) + " queries max"}; vs the '
          f'canonical record: {"exact" if earlier["exact"] else str(earlier["max_query_diff"]) + " queries max"}  '
          f'{"ok" if all_ok(checks) else "DIFFERS"}')
    R.write_json(mdir / 'info_e3.json', {'label': label, 'checkpoint': F.CK[label]['rel'], 'item': 'E3',
                                         'arrays': raw['arrays'], 'rank_columns': RANK_COLUMNS, 'rank_note': RANK_NOTE,
                                         'created_at': R.now()})
    return {'label': label, 'checkpoint': F.CK[label]['rel'], 'numbers': numbers, 'arrays': raw['arrays'],
            'checks': checks, 'ok': all_ok(checks)}


# ---------------------------------------------------------------------------
# E4: Table S13 in the trained input regime
# ---------------------------------------------------------------------------
def e4_images_item(out, args, cfg):
    sel = xn_e4_select(args)
    R.write_json(out / 'e4_s13' / 'selection.json', {
        'rule': 'neg_analysis_all.load_dataset: every 2-D [N, 768] embedding cache (N >= 64) under eval_data/circo in '
                'sorted order, concatenated, torch.randperm(N, generator=torch.Generator().manual_seed(42)), first '
                'MAX_BATCHES x BATCH_SIZE rows, batches of BATCH_SIZE', **sel})
    fetch = xn_e4_fetch(sel['images'], args, out / 'e4_s13')
    if fetch['missing']:          # a download problem: fail the item so that a rerun retries the missing images
        raise RuntimeError(f'{len(fetch["missing"])} of the {sel["n"]} images are missing or incomplete after the '
                           f'download, e.g. {fetch["missing"][:5]}; rerun to retry them')
    enc = xn_e4_encode(sel['images'], out / 'dump' / 'e4_s13', args, cfg)
    checks = {'selection_equals_load_dataset': check(sel['equals_load_dataset'], n=sel['n']),
              'images_present_and_decoded': check(True, n=sel['n'], downloaded=fetch['downloaded'],
                                                  removed_incomplete=fetch['removed_incomplete'],
                                                  ledger=f'e4_s13/{E4_LEDGER}'),
              'images_match_cached_features': check(enc['cos_min'] >= E4_MIN_COS, min_cos=E4_MIN_COS,
                                                    **{k: v for k, v in enc.items() if k != 'arrays'})}
    R.log(f'  {sel["n"]} images ({sel["n_batches"]} batches of {sel["batch_size"]}) from {sel["pt_files"]}; '
          f'{fetch["present_before"]} were present, {fetch["downloaded"]} downloaded'
          + (f', {len(fetch["removed_incomplete"])} incomplete ones of an earlier attempt fetched again'
             if fetch['removed_incomplete'] else '') + '; every one decodes in full')
    if enc['cos_min'] < E4_MIN_COS:
        R.log(f'  lowest cosines (file, cos): {[(x["file_name"], round(x["cos"], 5)) for x in enc["lowest"]]}; to '
              f'measure again, replace those images and delete entries/e4_images.json')
    R.log(f'  re-encoded vs cached feature: cos min {enc["cos_min"]:.5f} mean {enc["cos_mean"]:.6f}; own row nearest '
          f'{enc["nearest_is_self"]}/{sel["n"]}; ||v_raw|| mean {enc["vraw_norm_mean"]:.3f}  '
          f'{"ok" if all_ok(checks) else "DIFFERS"}')
    return {'n': sel['n'], 'fetch': fetch, 'encode': enc, 'arrays': enc['arrays'], 'checks': checks, 'ok': all_ok(checks)}


def e4_model_item(label, out, args, recs):
    if recs.get('e4_images', {}).get('status') != 'done' or not recs['e4_images'].get('ok'):
        raise RuntimeError('e4_images did not finish with every check passed (see entries/e4_images.json; a failed '
                           'image check is kept on a rerun until that entry is deleted)')
    raw = xn_e4_model(label, out / 'dump' / 'e4_s13', args, xn_shared())
    checks = {'raw_regime_equals_dump_V2': check(raw['v2_max_abs_diff'] <= E4_V2_TOL, max_abs_diff=raw['v2_max_abs_diff'],
                                                 tol=E4_V2_TOL, n=raw['v2_n'])}
    if label in E4_CHECK_MODELS:
        ref = R.read_json(FINAL_OUT / 's13' / 's13_control.json')['models'][label]['now']
        now = raw['regimes']['unit_norm']['aggregate']
        diff = {k: (now.get(k), v) for k, v in ref.items() if now.get(k) != v}
        checks['unit_norm_equals_s13_control'] = check(not diff, differing=diff, n_values=len(ref))
    for reg in E4_REGIMES:
        a = raw['regimes'][reg]['aggregate']
        R.log(f'  {reg:9s}: positive sim {a["sim_p_vraw_mean"]:.5f}  P(gt) {a["prob_on_gt_mean"]:.4f}  gt-hardest '
              f'{a["gt_minus_hardest_mean"]:+.4f}  in-batch R@1 {100 * raw["regimes"][reg]["inbatch_r1"]:.2f}%  '
              f'input norm {raw["regimes"][reg]["input_norm_mean"]:.3f}')
    R.log(f'  checks: ' + ', '.join(f'{k} {"ok" if v["ok"] else "FAIL"}' for k, v in checks.items()))
    return {'label': label, 'checkpoint': F.CK[label]['rel'], 'logit_scale': raw['logit_scale'], 'meta': raw['meta'],
            'regimes': raw['regimes'], 'arrays': raw['arrays'], 'checks': checks, 'ok': all_ok(checks)}


# ---------------------------------------------------------------------------
# E5: run-2 release copy
# ---------------------------------------------------------------------------
def release_copy(ck, vib_beta=None):
    """The transformation behind checkpoints_to_share/ (inferred key by key from the three existing copies and their
    sources on 2026-11-26; rebuilding each from its source gives the same pickled data and tensors): keep
    RELEASE_KEYS in that order with mapper_state as a plain dict, drop RELEASE_DROPPED; in the config replace the
    machine prefix of train_data_dir / eval_data_dir with <data_root>, clear output_dir ('') and resume (None) and set
    margin_delay 0.0 (margin is off in these runs); store the effective beta when vib_beta is given (the summed-KL
    headline run: vib_beta 1e-4 x batch 1024 = 0.1024; run 2 already stores 0.1024)."""
    unknown = set(ck) - set(RELEASE_KEYS) - set(RELEASE_DROPPED)
    missing = set(RELEASE_KEYS) - set(ck)
    if unknown or missing:
        raise RuntimeError(f'unexpected checkpoint layout: extra keys {sorted(unknown)}, missing {sorted(missing)}')
    cfg = dict(ck['config'])
    if float(cfg.get('margin_weight', 0.0) or 0.0) != 0.0:
        raise RuntimeError('the release rule was inferred from margin-off runs only (margin_delay -> 0.0)')
    for k in ('train_data_dir', 'eval_data_dir'):
        if isinstance(cfg.get(k), str) and cfg[k].startswith(MACHINE_PREFIX):
            cfg[k] = DATA_ROOT + cfg[k][len(MACHINE_PREFIX):]
    cfg['output_dir'], cfg['resume'], cfg['margin_delay'] = '', None, 0.0
    if vib_beta is not None:
        cfg['vib_beta'] = vib_beta
    left = {k: v for k, v in cfg.items() if isinstance(v, str) and '/notebooks' in v}
    if left:
        raise RuntimeError(f'machine paths left in the config: {left}')
    return {'mapper_state': dict(ck['mapper_state']), 'config': cfg, 'mapper_kwargs': ck['mapper_kwargs'],
            'merger_state': ck['merger_state'], 'mapper_type': ck['mapper_type'], 'logit_scale': ck['logit_scale']}


def e5_item(out, args, cfg):
    odir = out / 'e5_release'
    for sub in ('source', 'release'):      # a retry: reeval.py would reuse these, possibly measured on another host
        if (odir / sub).exists():
            shutil.rmtree(odir / sub)
            R.log(f'  removed e5_release/{sub}/ of an earlier attempt: source and release are measured in one process')
    build = xn_e5_build(args, odir)
    R.write_json(odir / 'transform.json', {'rule': release_copy.__doc__, 'pairs': build['pairs'], 'new': build['new']})
    pairs_ok = all(p['equivalent'] for p in build['pairs'])
    checks = {'transform_reproduces_existing_copies': check(pairs_ok, pairs={p['release']: {
        k: p[k] for k in ('bytes_identical', 'equivalent', 'deep_equal', 'members_differing')} for p in build['pairs']})}
    if not pairs_ok:
        R.log('  the transformation does not reproduce the existing release copies: no new file written')
        return {'build': build, 'checks': checks, 'ok': False}
    new = R.EXPERIMENT / NEW_RELEASE_REL
    key, src = key_of('run2'), ck_path('run2')
    state = 'created' if build['new']['created'] else 'already present, identical'
    R.log(f'  new copy  : {new} sha256 {build["new"]["sha256"]} ({state})')
    R.log('  source    : canonical CIRCO val and Fashion-IQ of the source checkpoint on this machine')
    c_src = R.run_circo_val(src, key, 0.0, odir / 'source', args.device)
    f_src = R.run_fashion(src, key, odir / 'source', cfg)
    R.log('  release   : canonical CIRCO val, Fashion-IQ and CIRCO test file of the new copy')
    c_rel = R.run_circo_val(new, key, 0.0, odir / 'release', args.device)
    f_rel = R.run_fashion(new, key, odir / 'release', cfg)
    t = R.recheck_test_file(new, key, 0.0, args.device, odir / 'release')
    pq_src = F.compare_perquery(c_rel['v2']['per_query'], c_src['v2']['per_query'])
    same_agg = {k: c_rel['v2'][k] == c_src['v2'][k] for k in ('map_atk', 'recall_atk', 'stickiness')}
    ok_st, cmp_st = A.val_check(c_rel, key)
    pq_re = F.compare_perquery(c_rel['v2']['per_query'], R.read_json(R.val_file(F.REEVAL_OUT, key))['v2']['per_query'])
    fiq_src = recall_cmp(f_rel['per_category'], f_src['per_category'], 0)
    fiq_re = recall_cmp(f_rel['per_category'], R.read_json(R.fiq_file(F.REEVAL_OUT, key))['per_category'], F.FIQ_QUERY_TOL)
    match = A.test_match(t['comparison'])
    checks.update({
        'new_copy_equals_source_where_loaders_read': check(build['new']['loader_fields_equal'],
                                                           differing=build['new']['loader_fields_differing'],
                                                           config_changes=build['new']['config_differing']),
        'circo_val_equals_source': check(pq_src['exact'] and all(same_agg.values()) and c_rel['tau'] == c_src['tau'],
                                         per_query=pq_src['mismatch_counts'], aggregates=same_agg),
        'circo_val_equals_stored_record': check(ok_st, source=str(R.STORED_VAL / f'{key}.json'),
                                                mismatches={k: v['n_query_mismatch'] for k, v in cmp_st['per_k'].items()}),
        'circo_val_equals_reeval_record': check(pq_re['exact'], per_query=pq_re['mismatch_counts']),
        'fashion_equals_source': fiq_src,
        'fashion_vs_reeval_record': fiq_re,
        'test_lists_identical_to_original': check(match == 'exact'
                                                  and t['comparison']['original_sha256'] == RUN2_SUBMISSION_SHA256,
                                                  match=match, identical_lists=t['comparison']['identical_lists'],
                                                  n_queries=t['comparison']['n_queries'],
                                                  original_sha256=t['comparison']['original_sha256'],
                                                  file_identical=t['comparison']['identical_file']),
        'pseudo_norm_off': check(c_rel['pseudo_norm_in_model'] == 0.0 == c_src['pseudo_norm_in_model'])})
    n = A.numbers(c_rel, f_rel)
    for k, v in checks.items():
        R.log(f'  {k:45s} {"ok" if v["ok"] else "FAIL"}')
    R.log(f'  numbers   : CIRCO val mAP@10 {100 * n["circo_val_map"]["10"]:.4f}  Ref@10 {100 * n["ref10_stickiness"]:.1f}%  '
          f'Fashion-IQ R@10/50 {100 * n["fiq_avg_exact"]["10"]:.4f}/{100 * n["fiq_avg_exact"]["50"]:.4f}')
    return {'new_file': str(new), 'sha256': build['new']['sha256'], 'bytes': build['new']['bytes'],
            'created': build['new']['created'], 'source': str(src), 'key': key, 'numbers': n, 'build': build,
            'test_file': t['file'], 'test_sha256': t['sha256'], 'checks': checks, 'ok': all_ok(checks)}


# ---------------------------------------------------------------------------
# E6: noise-scale curve
# ---------------------------------------------------------------------------
def e6_item(label, out, args, recs, control=False):
    name = 'baseline_control' if control else label
    s1 = R.read_json(FINAL_OUT / 'entries' / f's1_{name}.json')
    sigma2 = None
    if control:
        sigma2 = R.read_json(FINAL_OUT / 'entries' / 's1_vib.json')['sigma2_mean']
        if sigma2 != s1['s2']:
            raise RuntimeError(f's1_vib sigma2_mean {sigma2} differs from the control s^2 {s1["s2"]}')
    raw = xn_e6(label, sigma2, out / 'dump' / 'e6_noise' / name, args, xn_shared())
    ref = R.read_json(F.reference_paths(label, FINAL_OUT)['val'])['v2']['per_query']
    curve = raw['curve']
    a0 = curve[E6_ALPHAS.index(0.0)]
    a1 = curve[E6_ALPHAS.index(1.0)]
    q = s1['quick']
    ref_ap, ref_r10 = ref['ap_at_k']['@10'], [int(r <= 10) for r in ref['rank_ref']]
    ap0, r0 = a0['per_query_ap10'][0], [int(r <= 10) for r in a0['per_query_rank_ref'][0]]
    checks = {
        'a0_equals_s1_deterministic': check(a0['map10'][0] == q['val_map10_det'] and a0['ref10'][0] == q['val_ref10_det'],
                                            now=(a0['map10'][0], a0['ref10'][0]), s1=(q['val_map10_det'], q['val_ref10_det'])),
        'a0_equals_canonical_per_query': check(abs(F.mean(ap0) - F.mean(ref_ap)) < F.SELF_MAP_TOL
                                               and abs(sum(r0) - sum(ref_r10)) <= F.SELF_REF10_TOL,
                                               exact=ap0 == ref_ap and r0 == ref_r10, map10_now=F.mean(ap0),
                                               map10_canonical=F.mean(ref_ap), ref10_now=sum(r0), ref10_canonical=sum(ref_r10),
                                               ap10_queries_differing=sum(x != y for x, y in zip(ap0, ref_ap))),
        'a1_reproduces_s1': check(a1['map10'] == q['val_map10_per_sample'] and a1['ref10'] == q['val_ref10_per_sample'],
                                  now_map10=a1['map10'], s1_map10=q['val_map10_per_sample'], now_ref10=a1['ref10'],
                                  s1_ref10=q['val_ref10_per_sample']),
        'scoring_equals_nx_quick_metrics': check(raw['quick_check']['equal'], **raw['quick_check']),
        'summary_recomputed_from_saved_arrays': check(raw['recomputed_equal'])}
    for a, c in zip(E6_ALPHAS, curve):
        R.log(f'  a = {a:4.2f}: mAP@10 {100 * c["map10_mean"]:6.2f} +- {100 * c["map10_sd"]:5.2f}  Ref@10 '
              f'{100 * c["ref10_mean"]:5.1f}% +- {100 * c["ref10_sd"]:4.1f}  mean ||z|| {c["token_norm_mean"]:.3f}')
    R.log(f'  checks: ' + ', '.join(f'{k} {"ok" if v["ok"] else "FAIL"}' for k, v in checks.items()))
    return {'label': name, 'checkpoint': F.CK[label]['rel'], 'kind': raw['kind'], 'alphas': list(E6_ALPHAS),
            'samples': F.S1_SAMPLES, 'scale': raw['scale'],
            'curve': [{k: v for k, v in c.items() if not k.startswith('per_query')} for c in curve],
            'scoring': "S1's (final_run.nx_quick_metrics): queries composed in batches of 256, scored in batches, "
                       'reference masked; sd = population sd over the 8 samples; a = 0 computed once (every sample '
                       'is the deterministic token)', 'arrays': raw['arrays'], 'checks': checks, 'ok': all_ok(checks)}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def tables(recs):
    """Every new number, by item (the full values are in the entries)."""
    t = {}
    for label in E1_MODELS:
        r = recs.get(f'e1_{label}', {})
        if r.get('status') == 'done':
            t.setdefault('e1', {})[label] = {
                v: {'circo_map': n['circo']['map'], 'ref10': n['circo']['ref10'], 'sim_ref_max': n['circo']['sim_ref_max'],
                    'sim_ref_mean': n['circo']['sim_ref_mean'], 'fiq_r10': n['fashion_avg']['10'],
                    'fiq_r50': n['fashion_avg']['50'], 'fiq_per_category': n['fashion']} for v, n in r['numbers'].items()}
            t.setdefault('e1_geometry', {})[label] = r['geometry']
    for label in E2_MODELS:
        r = recs.get(f'e2_{label}', {})
        if r.get('status') == 'done':
            t.setdefault('e2', {})[label] = {v: {'fiq_r10': n['fashion_avg']['10'], 'fiq_r50': n['fashion_avg']['50'],
                                                 'fiq_per_category': n['fashion']} for v, n in r['numbers'].items()}
    for label in E3_MODELS:
        r = recs.get(f'e3_{label}', {})
        if r.get('status') == 'done':
            a = r['numbers']['ranks']['avg']
            t.setdefault('e3', {})[label] = {
                'fiq_recall': r['numbers']['fashion_avg'], 'ref_in_top': a['ref_in_top'], 'recall_noref': a['recall_noref'],
                'ref_above_target': a['ref_above_target'], 'bounds_ties_for': a['from_ranks'],
                'bounds_ties_against': a['from_ranks_pessimistic']}
    for label in E4_MODELS:
        r = recs.get(f'e4_{label}', {})
        if r.get('status') == 'done':
            t.setdefault('e4', {})[label] = {reg: dict(x['aggregate'], inbatch_r1=x['inbatch_r1'])
                                             for reg, x in r['regimes'].items()}
    r = recs.get('e5_release', {})
    if r.get('status') == 'done' and 'sha256' in r:
        t['e5'] = {'file': r['new_file'], 'sha256': r['sha256'], 'bytes': r['bytes'], 'ok': r['ok'],
                   'circo_val_map10': r['numbers']['circo_val_map']['10'], 'fiq_r10': r['numbers']['fiq_avg_exact']['10']}
    for name in E6_MODELS + ('baseline_control',):
        r = recs.get(f'e6_{name}', {})
        if r.get('status') == 'done':
            t.setdefault('e6', {})[name] = r['curve']
    return t


def log_tables(t):
    pc = lambda x, d=2: '-' if x is None else f'{100 * x:.{d}f}'     # noqa: E731
    if 'e1' in t:
        R.log('\n  E1 four-token study (percent; Fashion-IQ = exact mean of the categories; paper Table S9 at epoch_01)')
        R.log(f'  {"checkpoint":16s} {"tokens":6s} {"C@5":>6s} {"C@10":>6s} {"C@25":>6s} {"C@50":>6s} {"Ref@10":>7s} '
              f'{"qref max":>8s} {"F R@10":>7s} {"F R@50":>7s}  paper (F, C@10, Ref@10)')
        for label, rows in t['e1'].items():
            for v, x in rows.items():
                paper = ('F {:.2f}, C {:.2f}, stick {:.2f}'.format(*PAPER_S9[v]) if label == 'small_tok4_e01'
                         else '')
                R.log(f'  {label:16s} {v:6s} ' + ' '.join(f'{pc(x["circo_map"][k]):>6s}' for k in ('5', '10', '25', '50'))
                      + f' {pc(x["ref10"]):>7s} {x["sim_ref_max"]:8.3f} {pc(x["fiq_r10"]):>7s} {pc(x["fiq_r50"]):>7s}  {paper}')
        for label, geo in t['e1_geometry'].items():
            for part, g in geo.items():
                R.log(f'  {label} geometry ({part}): token norms ' + ' '.join(
                    f'{g["token_norm"][str(i)]["mean"]:.2f}' for i in range(len(g['token_norm']))) +
                    f'; ||v|| {g["v_norm"]["mean"]:.2f}; pairwise cos {g["pairwise_cos_over_pairs"]["min"]:.3f}..'
                    f'{g["pairwise_cos_over_pairs"]["max"]:.3f} mean {g["pairwise_cos_over_pairs"]["mean"]:.3f}; token-v '
                    f'cos {g["token_v_cos_over_tokens"]["min"]:+.4f}..{g["token_v_cos_over_tokens"]["max"]:+.4f}')
        R.log(f'  paper (Supp. C, no source): {PAPER_GEOMETRY}')
    if 'e2' in t:
        R.log('\n  E2 Fashion-IQ query variants: avg R@10 / R@50 (percent)')
        R.log(f'  {"model":16s} ' + ' '.join(f'{v:>13s}' for v in E2_VARIANTS))
        for label, rows in t['e2'].items():
            R.log(f'  {label:16s} ' + ' '.join(f'{pc(rows[v]["fiq_r10"]):>6s}/{pc(rows[v]["fiq_r50"]):>6s}'
                                                for v in E2_VARIANTS))
    if 'e3' in t:
        R.log('\n  E3 Fashion-IQ fp32 ranks (avg over categories, percent): recall R@1/5/10/50, reference in the top '
              '1/10, R@10 without the reference, reference above the target, R@1 tie bounds')
        for label, x in t['e3'].items():
            R.log(f'  {label:16s} ' + '/'.join(pc(x['fiq_recall'][k]) for k in R.KS)
                  + f'   ref@1/10 {pc(x["ref_in_top"]["1"])}/{pc(x["ref_in_top"]["10"])}   R@10 noref '
                  f'{pc(x["recall_noref"]["10"])}   ref>target {pc(x["ref_above_target"])}   R@1 '
                  f'[{pc(x["bounds_ties_against"]["1"])}, {pc(x["bounds_ties_for"]["1"])}]')
    if 'e4' in t:
        R.log('\n  E4 Table S13 (5,120 CIRCO gallery images): raw = trained input regime, unit_norm = the paper\'s')
        R.log(f'  {"model":16s} {"regime":10s} {"pos sim":>8s} {"P(gt)":>7s} {"gt-hard":>8s} {"R@1 in-batch":>12s} '
              f'{"neg mean":>8s} {"eff negs":>8s}')
        for label, regs in t['e4'].items():
            for reg, a in regs.items():
                R.log(f'  {label:16s} {reg:10s} {a["sim_p_vraw_mean"]:8.4f} {a["prob_on_gt_mean"]:7.4f} '
                      f'{a["gt_minus_hardest_mean"]:+8.4f} {pc(a["inbatch_r1"]):>12s} {a["neg_mean_mean"]:8.4f} '
                      f'{a["effective_negs_mean"]:8.1f}')
    if 'e5' in t:
        e = t['e5']
        R.log(f'\n  E5 release copy: {e["file"]}  sha256 {e["sha256"]}  ({e["bytes"]} bytes)  checks '
              f'{"ok" if e["ok"] else "FAILED"}; CIRCO val mAP@10 {pc(e["circo_val_map10"], 4)}, Fashion-IQ R@10 '
              f'{pc(e["fiq_r10"], 4)}')
    if 'e6' in t:
        R.log('\n  E6 noise-scale curve: CIRCO val mAP@10 / Ref@10 (percent, mean +- population sd over 8 samples)')
        R.log(f'  {"model":16s} ' + ' '.join(f'{"a=" + str(a):>17s}' for a in E6_ALPHAS))
        for name, curve in t['e6'].items():
            R.log(f'  {name:16s} ' + ' '.join(f'{pc(c["map10_mean"]):>5s}+-{pc(c["map10_sd"]):<4s}/{pc(c["ref10_mean"], 1):>5s}'
                                              for c in curve))


def write_manifest(out):
    skip = {'MANIFEST.sha256', 'run_log.txt'}            # top level only
    files = sorted(p for p in out.rglob('*') if p.is_file() and p.relative_to(out).as_posix() not in skip
                   and not p.name.endswith('.tmp'))
    (out / 'MANIFEST.sha256').write_text(''.join(f'{R.sha256_file(p)}  {p.relative_to(out)}\n' for p in files))
    return {'n_files': len(files), 'bytes': sum(p.stat().st_size for p in files), 'manifest': 'MANIFEST.sha256'}


def exit_code(recs, labels):
    selfcheck = [l for l in labels if recs.get(l, {}).get('status') == 'done' and not recs[l].get('ok')]
    other = [l for l in labels if recs.get(l, {}).get('status') != 'done']
    return (3 if selfcheck else 4 if other else 0), selfcheck, other


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
class _Tee:
    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, 'a')          # noqa: SIM115

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self.stream.fileno()


def start_log(out):
    """Mirror stdout into out/run_log.txt, unless stdout is already that file (a nohup redirect)."""
    path = out / 'run_log.txt'
    try:
        same = path.exists() and os.path.samestat(os.fstat(sys.stdout.fileno()), os.stat(path))
    except (OSError, ValueError, AttributeError):         # io.UnsupportedOperation is both an OSError and a ValueError
        same = False
    if same:
        return lambda: None
    old = sys.stdout
    sys.stdout = _Tee(old, path)

    def restore():
        sys.stdout.flush()
        sys.stdout.file.close()
        sys.stdout = old
    return restore


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out_dir', type=Path, default=REPO_ROOT / 'results' / 'extra_2026_11')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--preflight_only', action='store_true', help='check everything, evaluate nothing')
    p.add_argument('--gate_fiq_tol', type=int, default=0,
                   help='Fashion-IQ queries per category and K the gate may differ by (default 0 = exact; CIRCO is '
                        'always exact). Raise to 2 only if the gate stops on a near-tie flip on another host')
    for n, what in (('e1', 'four-token study'), ('e2', 'Fashion-IQ V4-V6'), ('e3', 'Fashion-IQ ranks'),
                    ('e4', 'Table S13 trained regime (downloads COCO images)'), ('e5', 'run-2 release copy'),
                    ('e6', 'noise-scale curve')):
        p.add_argument(f'--skip_{n}', action='store_true', help=f'skip {n.upper()} ({what})')
    p.add_argument('--download_workers', type=int, default=16, help='E4: parallel COCO image downloads')
    p.add_argument('--no_download', action='store_true', help='E4: fail instead of downloading missing images')
    return p.parse_args()


def main():
    args = parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    restore = start_log(out)
    try:
        _main(args, out)
    finally:
        restore()


def _main(args, out):
    t_start = time.time()
    R.banner(f'Extra evaluation run (eval only, nothing trained)   {R.now()}')
    R.log(f'  code   : {R.REPO_ROOT}  (commit {R.git_commit()})')
    R.log(f'  output : {out}')
    R.log(f'  reads  : {FINAL_OUT} (final run and its dump), {F.REEVAL_OUT} (re-evaluation and add-on)')
    skipped = [k[5:] for k, v in vars(args).items() if k.startswith('skip_') and v]
    if skipped:
        R.log(f'  skipped: {", ".join(skipped)}')

    # ---- Step 0 -------------------------------------------------------------
    R.banner('Step 0  Preflight')
    v = F.versions()
    problems = F.environment_problems(args.device)
    p_in, warnings, info_in = input_problems(args, out)
    labels_ck = needed_labels(args)
    rows, p_ck = checkpoint_problems(labels_ck)
    problems += p_in + p_ck
    warnings = F.version_warnings(v) + warnings
    for label, row in rows.items():
        state = 'ok  ' if row.get('sha256') == row['sha256_expected'] else 'FAIL'
        R.log(f'  {state} {label:18s} sha256 {row.get("sha256", "missing")[:16]}  {row.get("mapper_type", "-"):4s} '
              f'tokens {row.get("num_tokens", "-")}  pseudo_norm {row.get("pseudo_norm", "-")}')
    disk = info_in['disk']
    R.log(f'  dump files checked against dump/MANIFEST.sha256: {sum(info_in["dump_files_checked"].values())}/'
          f'{len(dump_files(args))}')
    R.log(f'  disk: this run adds about {disk["adds_total_gb"]:.2f} GB ('
          + ', '.join(f'{k} {v:.3f}' for k, v in disk['adds_gb'].items()) + f'), the tarballs about {PACK_GB} GB; the '
          f'volume reports {disk["free_gb"]:,.1f} GB free'
          + (f'; CephFS quota of {disk["quota_dir"]}: {disk["used_gb"]:.2f} of {disk["quota_gb"]:.2f} GB used'
             if 'quota_gb' in disk else '; no storage quota visible'))
    R.log('  versions: ' + ', '.join(f'{k} {v.get(k)}' for k in ('python', 'torch', 'torchvision', 'numpy', 'pillow',
                                                                 'clip_commit', 'gpu')))
    for w in warnings:
        R.log(f'  note: {w}')
    info = dict(R.environment(args), versions=v, host=host_fingerprint())
    R.write_json(out / 'preflight.json', {'checked_at': R.now(), 'environment': info, 'problems': problems,
                                          'warnings': warnings, 'checkpoints': rows, 'inputs': info_in})
    history = R.read_json(out / 'run_info.json') if (out / 'run_info.json').exists() else []
    history.append(info)
    R.write_json(out / 'run_info.json', history)
    if problems:
        R.log('\nPREFLIGHT FAILED:')
        for pr in problems:
            R.log(f'  - {pr}')
        sys.exit(1)
    steps = estimate(args)
    R.log(f'\n  Preflight passed: {len(rows)} checkpoints; torch {v.get("torch")}, GPU {v.get("gpu")}')
    R.log('  Runtime estimate (A6000, from the final run\'s timings; finished items are skipped on a rerun):')
    for name, sec in steps:
        R.log(f'    {name:78s} ~{sec / 60:4.1f} min')
    R.log(f'    {"total (plus the COCO download in E4: ~5,000 images, ~0.8 GB)":78s} ~{sum(s for _, s in steps) / 60:4.0f} min')
    if args.preflight_only:
        return

    cfg = R.fashion_eval_cfg(args.device)
    recs, labels = {}, ['gate']

    # ---- Step 1 -------------------------------------------------------------
    R.banner('Step 1  Gate: tok4 e00 "all four combined" through this run\'s path must reproduce the final run\'s '
             'canonical record (CIRCO per query; Fashion-IQ ' + ('exactly' if not args.gate_fiq_tol else
                                                                  f'within {args.gate_fiq_tol} queries') + ')')

    def with_host(fn):                                        # every entry records the machine it was measured on
        return lambda: dict(fn(), host=host_fingerprint())

    F.run_item(out, 'gate', ck_title(GATE_MODEL), with_host(lambda: gate_item(out, args, cfg)), recs, resume=False)
    R.write_json(out / 'checks' / 'gate.json', {'ok': bool(recs['gate'].get('ok')), 'checked_at': R.now(),
                                                'gate': recs['gate']})
    if recs['gate'].get('status') != 'done':
        R.banner('STOPPED: the gate raised an error before it could compare (traceback above and in checks/gate.json), '
                 'so this is not a reproduction failure and --gate_fiq_tol would not help. Nothing else was run. Fix '
                 'the cause (GPU, a missing file, ...) and rerun the same command.')
        sys.exit(2)
    if not recs['gate'].get('ok'):
        R.banner('STOPPED: the gate did not reproduce, so this path or this setup differs from the one behind the final '
                 'run. Nothing else was run. Send run_log.txt and checks/gate.json. If CIRCO is exact and Fashion-IQ '
                 'differs by at most 2 queries per category (a near-tie flip on another host), rerun with '
                 '--gate_fiq_tol 2 (the saved measurement is re-compared); to measure again, delete gate/ first.')
        sys.exit(2)

    def items(title, todo):
        R.banner(title)
        for i, (label, name, fn) in enumerate(todo, 1):
            labels.append(label)
            F.run_item(out, label, f'[{i}/{len(todo)}] {name}', with_host(fn), recs)

    if not args.skip_e1:
        items(f'E1  Four-token study: {len(E1_MODELS)} checkpoints x {len(E1_VARIANTS)} variants, token geometry',
              [(f'e1_{l}', ck_title(l), (lambda l=l: e1_item(l, out, args, cfg))) for l in E1_MODELS])
    if not args.skip_e2:
        items(f'E2  Fashion-IQ V4 / V5 / V6 for the {len(E2_MODELS)} core models (fp32, eval_fashion batches)',
              [(f'e2_{l}', ck_title(l), (lambda l=l: e2_item(l, out, args, cfg))) for l in E2_MODELS])
    if not args.skip_e3:
        items(f'E3  Fashion-IQ fp32 per-query ranks for the {len(E3_MODELS)} dumped models',
              [(f'e3_{l}', ck_title(l), (lambda l=l: e3_item(l, out, args, cfg))) for l in E3_MODELS])
    if not args.skip_e5:
        items('E5  Run-2 release copy and its release check',
              [('e5_release', f'{NEW_RELEASE_REL} from {F.CK["run2"]["rel"]}', lambda: e5_item(out, args, cfg))])
    if not args.skip_e6:
        items(f'E6  Noise-scale curve: a in {list(E6_ALPHAS)}, {F.S1_SAMPLES} samples (S1\'s eps)',
              [(f'e6_{l}', f'{ck_title(l)}: z = mu + a * sigma * eps', (lambda l=l: e6_item(l, out, args, recs)))
               for l in E6_MODELS] +
              [('e6_baseline_control', 'baseline z_det + a * s * eps (s as in S1)',
                lambda: e6_item('baseline', out, args, recs, control=True))])
    if not args.skip_e4:
        items('E4  Table S13 in the trained input regime: the 5,120 images of neg_analysis_all.load_dataset',
              [('e4_images', 'select, fetch and encode the images', lambda: e4_images_item(out, args, cfg))] +
              [(f'e4_{l}', ck_title(l), (lambda l=l: e4_model_item(l, out, args, recs))) for l in E4_MODELS])

    # ---- Summary ------------------------------------------------------------
    try:
        code = summarize(out, recs, labels, skipped, t_start)
    except Exception:                                         # noqa: BLE001
        R.log(traceback.format_exc())
        R.banner('SUMMARY FAILED: the tables, summary.json or MANIFEST.sha256 could not be written (traceback above). '
                 'Every item record is in entries/; rerun the same command to write them again (finished items are '
                 'skipped).')
        sys.exit(4)
    if code:
        R.log('  rerun the same command to retry failed items; finished ones are skipped')
        sys.exit(code)


def summarize(out, recs, labels, skipped, t_start):
    R.banner('Summary')
    code, selfcheck_bad, other_bad = exit_code(recs, labels)
    t = tables(recs)
    R.banner('New numbers (full values in summary.json and entries/)')
    log_tables(t)
    R.write_json(out / 'summary.json', {
        'finished_at': R.now(), 'exit_code': code, 'minutes': round((time.time() - t_start) / 60, 1),
        'argv': sys.argv, 'code_commit': R.git_commit(), 'skipped': skipped,
        'failed': {'selfcheck': selfcheck_bad, 'other': other_bad}, 'tables': t,
        'manifest': 'MANIFEST.sha256: sha256sum format, every file under the output folder except itself and run_log.txt',
        'entries': recs})
    manifest = write_manifest(out)
    R.log(f'  MANIFEST.sha256: {manifest["n_files"]} files, {manifest["bytes"] / 1e6:.1f} MB')
    verdict = {0: 'DONE: every check passed', 3: f'SELF-CHECK FAILED: {selfcheck_bad}',
               4: f'DONE EXCEPT (failed items): {other_bad}'}[code]
    if code == 3 and other_bad:
        verdict += f'; also failed: {other_bad}'
    R.banner(f'{verdict}   {(time.time() - t_start) / 60:.1f} min   {R.now()}')
    R.log(f'  results: {out}  (summary.json, entries/, dump/, MANIFEST.sha256)')
    return code


# ===========================================================================
# Numerical layer (torch / numpy / CLIP). Everything above is plain Python and is exercised by the CPU mock.
# ===========================================================================
def xn_shared():
    """dump/shared loaded back (fp32 arrays round-trip exactly), once per process."""
    if 'shared' not in _XC:
        _XC['shared'] = F.Shared(DUMP / 'shared')
    return _XC['shared']


def xn_load(label, device):
    """The model as eval_circo_v2.load_model builds it (final_run.nx_load_model for its own labels)."""
    if label in F.CK:
        return F.nx_load_model(label, device)
    from evaluation import eval_circo_v2
    model, clip_name = eval_circo_v2.load_model(ck_path(label), device)
    if clip_name != 'ViT-L/14':
        raise RuntimeError(f'{label}: backbone {clip_name}')
    if float(model.pseudo_norm) != ck_info(label)['pn']:
        raise RuntimeError(f'{label}: model.pseudo_norm={model.pseudo_norm}, expected {ck_info(label)["pn"]}')
    if model.use_multilayer:
        raise RuntimeError(f'{label}: multilayer mappers are not supported here')
    return model


def fashion_gallery(cat, cfg):
    if ('fiq', cat) not in _XC:
        from evaluation import eval_fashion
        _XC[('fiq', cat)] = eval_fashion.load_gallery_cache(cat, cfg.clip_model, Path(cfg.eval_data_dir) / 'fashion_iq')
    return _XC[('fiq', cat)]


def fashion_batches(sh):
    """(category, row slice) of eval_fashion's batches (cfg.batch_size per category, as dump/shared recorded them)."""
    for cat, info in sh.fashion['categories'].items():
        pos = info['start']
        for size in info['batches']:
            yield cat, slice(pos, pos + size)
            pos += size
        if pos != info['start'] + info['n']:
            raise RuntimeError(f'Fashion-IQ {cat}: batches do not cover the category')


def rank_table(sims, gt, ref, tokref=None):
    """Per-query int32 columns RANK_COLUMNS from one category's similarity matrix (the canonical CPU scores)."""
    import torch
    i = torch.arange(sims.shape[0])
    s_gt, s_ref = sims[i, gt].unsqueeze(1), sims[i, ref].unsqueeze(1)
    rank_gt = (sims > s_gt).sum(1) + 1
    cols = [rank_gt, (sims == s_gt).sum(1) - 1, (sims > s_ref).sum(1) + 1, (sims == s_ref).sum(1) - 1,
            rank_gt - (s_ref.squeeze(1) > s_gt.squeeze(1)).long()]
    if tokref is None:
        cols.append(torch.full_like(rank_gt, -1))
    else:
        cols.append((sims > sims[i, tokref].unsqueeze(1)).sum(1) + 1)
    for k in FIQ_K:
        cols.append((torch.topk(sims, k, dim=-1).indices == gt.unsqueeze(1)).any(dim=1).long())
    return torch.stack(cols, dim=1).to(torch.int32)


def rank_summary(table, sh, recalls):
    """Recall and rank numbers from the saved int table (per category)."""
    t = table.long()
    out = {}
    for cat, info in sh.fashion['categories'].items():
        s, n = info['start'], info['n']
        x = t[s:s + n]
        r, ties, rr, rn = x[:, 0], x[:, 1], x[:, 2], x[:, 4]
        at_k = {}
        for j, k in enumerate(FIQ_K):
            hit = x[:, 6 + j].bool()
            opt, pes = r <= k, (r + ties) <= k
            at_k[str(k)] = {'canonical': recalls[cat][str(k)], 'from_hits': int(hit.sum()) / n,
                            'from_ranks': int(opt.sum()) / n, 'from_ranks_pessimistic': int(pes.sum()) / n,
                            'tie_ambiguous': int((opt & ~pes).sum()), 'violations': int(((hit & ~opt) | (pes & ~hit)).sum()),
                            'ref_in_top': int((rr <= k).sum()) / n, 'recall_noref': int((rn <= k).sum()) / n}
        out[cat] = {'n': n, 'at_k': at_k, 'ref_above_target': int((rr < r).sum()) / n,
                    'median_rank_target': statistics.median(r.tolist()), 'median_rank_ref': statistics.median(rr.tolist()),
                    'n_target_tied': int((ties > 0).sum())}
    return out


def fashion_score(feats, sh, cfg, tokref=None):
    """eval_fashion's scoring (CPU sims = queries @ gallery.T per category, retrieval.recall_at_k) and the rank table.
    feats: {category: list of per-batch CPU fp32 query features}; tokref: {category: gallery rows} or None."""
    import torch
    from evaluation import retrieval
    recalls, tables_ = {}, []
    for cat, info in sh.fashion['categories'].items():
        q = torch.cat(feats[cat])                       # a fresh tensor per category, as eval_fashion / nx_dump_model
        g = fashion_gallery(cat, cfg)
        if q.shape[0] != info['n'] or g.shape[0] != len(info['gallery_names']):
            raise RuntimeError(f'Fashion-IQ {cat}: {q.shape[0]} queries / {g.shape[0]} gallery rows')
        sims = q @ g.T
        gt = torch.tensor(info['target_gallery_row'], dtype=torch.long)
        rec = retrieval.recall_at_k(sims, gt, k_values=list(FIQ_K))
        recalls[cat] = {str(k): float(x) for k, x in rec.items()}
        ref = torch.tensor(info['candidate_gallery_row'], dtype=torch.long)
        tr = None if tokref is None else torch.tensor(tokref[cat], dtype=torch.long)
        tables_.append(rank_table(sims, gt, ref, tr))
    table = torch.cat(tables_)
    return recalls, table, rank_summary(table, sh, recalls)


def token_geometry(z, v):
    """Supp. C's four-token geometry: z [N, T, D] final tokens, v [N, D] raw image features (float64, CPU)."""
    import torch.nn.functional as TF
    z, v = z.double().cpu(), v.double().cpu()
    t = z.shape[1]
    zu, vu = TF.normalize(z, dim=-1), TF.normalize(v, dim=-1)

    def st(x):
        return {'mean': float(x.mean()), 'std': float(x.std(unbiased=False)), 'min': float(x.min()), 'max': float(x.max())}

    norms = z.norm(dim=-1)
    pairs = {f'{i}-{j}': (zu[:, i] * zu[:, j]).sum(-1) for i, j in itertools.combinations(range(t), 2)}
    tv = {str(i): (zu[:, i] * vu).sum(-1) for i in range(t)}
    pm, tm = [float(x.mean()) for x in pairs.values()], [float(x.mean()) for x in tv.values()]
    return {'n_refs': int(z.shape[0]), 'token_norm': {str(i): st(norms[:, i]) for i in range(t)},
            'token_norm_means_range': [min(float(norms[:, i].mean()) for i in range(t)),
                                       max(float(norms[:, i].mean()) for i in range(t))],
            'v_norm': st(v.norm(dim=-1)),
            'pairwise_cos': {p: st(x) for p, x in pairs.items()},
            'pairwise_cos_over_pairs': {'min': min(pm), 'max': max(pm), 'mean': sum(pm) / len(pm)},
            'token_v_cos': {i: st(x) for i, x in tv.items()},
            'token_v_cos_over_tokens': {'min': min(tm), 'max': max(tm)},
            'token_v_cos_all_refs': {'min': min(float(x.min()) for x in tv.values()),
                                     'max': max(float(x.max()) for x in tv.values())},
            'definitions': 'final tokens (mapper output; no merger, pseudo_norm 0); v = encoder_image_unnorm output; '
                           'cosines per reference, then mean (min / max) over references; std = population'}


def _sel(variant):
    return slice(None) if variant == 'all' else slice(int(variant[1]), int(variant[1]) + 1)


def xn_e1(label, variants, mdir, args, cfg, sh):
    """Tokens one reference at a time; CIRCO val queries one at a time (eval_circo_v2); Fashion-IQ in eval_fashion's
    batches; every variant through encode_tokens, the single tokens as z[:, i:i+1]."""
    import clip
    import torch
    dev = args.device
    model = xn_load(label, dev)
    if model.merger is not None or model.mapper.num_tokens != 4 or float(model.pseudo_norm) != 0.0:
        raise RuntimeError(f'{label}: expected 4 tokens without merger and projection')
    start, n = sh.splits['val']
    if start != 0:
        raise RuntimeError('CIRCO val rows must come first in dump/shared/circo_index.json')
    caps, rows = [q['caption'] for q in sh.queries[:n]], [q['ref_row'] for q in sh.queries[:n]]
    res, arrays = {'variants': {v: {} for v in variants}}, []
    with torch.no_grad(), R.cudnn_settings(deterministic=False):          # CIRCO: eval_circo_v2's settings
        v = sh.vraw.to(dev)
        zr = [F.raw_tokens(model, v[r:r + 1].clone()) for r in range(v.shape[0])]
        z = torch.cat([F.project(model, x) for x in zr])
        gal = F.nx_gallery(dev)
        for var in variants:
            s = _sel(var)
            q = torch.cat([F.encode_tokens(model, z[r:r + 1, s], clip.tokenize([c], truncate=True).to(dev))
                           for c, r in zip(caps, rows)])
            res['variants'][var]['circo_perquery'] = F.nx_val_perquery(q, sh, gal)
            if mdir is not None:
                arrays.append(F.save_array(mdir / f'circo_V1_{var}.npy', q, 'float32'))
        if mdir is not None:
            arrays.append(F.save_array(mdir / 'circo_tokens.npy', z, 'float32'))
            val_rows = sorted(set(rows))
            res['geometry'] = {'val': token_geometry(z[val_rows], v[val_rows]), 'all': token_geometry(z, v)}
            res['geometry']['val']['refs'] = (f'the {len(val_rows)} CIRCO val references (rows {val_rows[0]}..'
                                              f'{val_rows[-1]} of dump/shared/circo_vraw)')
            res['geometry']['all']['refs'] = f'all {z.shape[0]} rows of dump/shared/circo_vraw (val and test references)'
    fi, mods = sh.fashion, sh.fashion['modifiers']
    feats = {var: {c: [] for c in fi['categories']} for var in variants}
    with torch.no_grad(), R.cudnn_settings(deterministic=True):           # Fashion-IQ: eval_fashion's settings
        for cat, s in fashion_batches(sh):
            tok = clip.tokenize(mods[s], truncate=True).to(dev)
            zb = F.tokens_from_vraw(model, sh.fvraw[s].clone().to(dev))
            for var in variants:
                feats[var][cat].append(F.encode_tokens(model, zb[:, _sel(var)], tok).cpu())
    tokref = {c: info['candidate_gallery_row'] for c, info in fi['categories'].items()}
    for var in variants:
        rec, table, summ = fashion_score(feats[var], sh, cfg, tokref)
        res['variants'][var].update(fashion_recalls=rec, rank_summary=summ)
        if mdir is not None:
            arrays.append(F.save_array(mdir / f'fashion_{var}_ranks.npy', table, 'int32'))
    res.update(arrays=arrays, tau=float(model.clip.logit_scale.exp().item()))
    del model
    F.free_memory()
    return res


def xn_e2(label, mdir, args, cfg, sh):
    import clip
    import torch
    dev = args.device
    model = F.nx_load_model(label, dev)
    fi, mods = sh.fashion, sh.fashion['modifiers']
    cats = list(fi['categories'])
    perms = {seed: sh.fashion_perm(seed) for seed in F.SWAP_SEEDS}
    cand = [None] * len(mods)                         # gallery row of every query's reference (within its category)
    for cat, info in fi['categories'].items():
        for j, row in enumerate(info['candidate_gallery_row']):
            cand[info['start'] + j] = row
    feats = {v: {c: [] for c in cats} for v in E2_VARIANTS}
    with torch.no_grad(), R.cudnn_settings(deterministic=True):           # eval_fashion's settings
        zs = []
        for cat, s in fashion_batches(sh):
            tok = clip.tokenize(mods[s], truncate=True).to(dev)
            zb = F.tokens_from_vraw(model, sh.fvraw[s].clone().to(dev))
            zs.append(zb)
            feats['V1'][cat].append(F.encode_tokens(model, zb, tok).cpu())
        z = torch.cat(zs)                              # each query's token, computed in its eval_fashion batch
        zbar = F.project(model, z[fi['unique_first_query']].mean(dim=0, keepdim=True))
        for cat, s in fashion_batches(sh):
            tok = clip.tokenize(mods[s], truncate=True).to(dev)
            feats['V4'][cat].append(F.encode_tokens(model, zbar.expand(s.stop - s.start, -1, -1), tok).cpu())
            for seed, perm in perms.items():
                tokp = clip.tokenize([mods[perm[i]] for i in range(s.start, s.stop)], truncate=True).to(dev)
                feats[f'V5_s{seed}'][cat].append(F.encode_tokens(model, z[s], tokp).cpu())
                feats[f'V6_s{seed}'][cat].append(F.encode_tokens(model, z[perm[s.start:s.stop]], tok).cpu())
    res, arrays = {'variants': {}}, []
    for var in E2_VARIANTS:
        tokref = None
        if var.startswith('V6'):
            perm = perms[int(var.split('_s')[1])]
            tokref = {c: [cand[perm[i]] for i in range(info['start'], info['start'] + info['n'])]
                      for c, info in fi['categories'].items()}
        elif var != 'V4':
            tokref = {c: info['candidate_gallery_row'] for c, info in fi['categories'].items()}
        rec, table, summ = fashion_score(feats[var], sh, cfg, tokref)
        res['variants'][var] = {'fashion_recalls': rec, 'rank_summary': summ}
        name = 'V1_fp32' if var == 'V1' else var       # the final dump's fashion_V1.npy is fp16; V1 ranks are E3's
        arrays.append(F.save_array(mdir / f'fashion_{name}.npy', torch.cat([torch.cat(feats[var][c]) for c in cats]),
                                   'float32'))
        if var != 'V1':
            arrays.append(F.save_array(mdir / f'fashion_{var}_ranks.npy', table, 'int32'))
    res.update(arrays=arrays, tau=float(model.clip.logit_scale.exp().item()), pseudo_norm=float(model.pseudo_norm),
               zbar_norm=float(zbar.norm(dim=-1).mean()))
    del model
    F.free_memory()
    return res


def xn_e3(label, mdir, args, cfg, sh):
    import clip
    import torch
    dev = args.device
    model = F.nx_load_model(label, dev)
    fi, mods = sh.fashion, sh.fashion['modifiers']
    feats = {c: [] for c in fi['categories']}
    with torch.no_grad(), R.cudnn_settings(deterministic=True):           # eval_fashion's batches and settings
        for cat, s in fashion_batches(sh):
            tok = clip.tokenize(mods[s], truncate=True).to(dev)
            feats[cat].append(F.encode_from_vraw(model, sh.fvraw[s].clone().to(dev), tok).cpu())
    tokref = {c: info['candidate_gallery_row'] for c, info in fi['categories'].items()}
    rec, table, summ = fashion_score(feats, sh, cfg, tokref)
    arrays = [F.save_array(mdir / 'fashion_V1_ranks.npy', table, 'int32')]
    del model
    F.free_memory()
    return {'fashion_recalls': rec, 'rank_summary': summ, 'arrays': arrays}


def _neg_analysis(args):
    saved = list(sys.path)
    try:
        from evaluation import neg_analysis_all as NA         # inserts codes/ and codes/evaluation into sys.path
    finally:
        sys.path[:] = saved
    NA.DEVICE = args.device
    return NA


def _e4_unit_batches(args):
    if 'e4_unit' not in _XC:
        NA = _neg_analysis(args)
        _XC['e4_unit'] = NA.load_dataset(str(R.CIRCO_DIR), 'circo')
    return _XC['e4_unit']


def xn_e4_select(args):
    """The rows neg_analysis_all.load_dataset picks, mapped back to gallery images (verified equal to its batches)."""
    import torch
    NA = _neg_analysis(args)
    parts, owners = [], []
    for f in sorted(Path(R.CIRCO_DIR).rglob('*.pt')):
        try:
            d = torch.load(f, map_location='cpu', weights_only=False)
        except Exception:                                     # noqa: BLE001  (load_dataset skips these too)
            continue
        emb = d.get('embeddings') if isinstance(d, dict) else d
        if isinstance(emb, torch.Tensor) and emb.dim() == 2 and emb.shape[1] == 768 and emb.shape[0] >= 64:
            parts.append(emb.float())
            owners.append((f, d.get('img_ids') if isinstance(d, dict) else None, emb.shape[0]))
    v = torch.cat(parts)
    perm = torch.randperm(v.shape[0], generator=torch.Generator().manual_seed(42))[:NA.MAX_BATCHES * NA.BATCH_SIZE]
    batches = _e4_unit_batches(args)
    sel = torch.cat(batches)
    equal = bool(torch.equal(sel, v[perm][:sel.shape[0]]))
    offsets = list(itertools.accumulate([0] + [o[2] for o in owners]))
    info = R.read_json(R.CIRCO_DIR / 'COCO2017_unlabeled' / 'annotations' / 'image_info_unlabeled2017.json')
    by_id = {img['id']: img for img in info['images']}
    images = []
    for row in perm[:sel.shape[0]].tolist():
        o = max(i for i, off in enumerate(offsets[:-1]) if off <= row)
        f, ids, _ = owners[o]
        if ids is None:
            raise RuntimeError(f'{f} has no img_ids: its rows cannot be mapped to images')
        img = by_id[int(ids[row - offsets[o]])]
        images.append({'row': row, 'cache': str(Path(f).relative_to(R.CIRCO_DIR)), 'cache_row': row - offsets[o],
                       'img_id': img['id'], 'file_name': img['file_name'], 'coco_url': img.get('coco_url')})
    return {'n': len(images), 'n_batches': len(batches), 'batch_size': int(batches[0].shape[0]),
            'pt_files': [str(Path(o[0]).relative_to(R.CIRCO_DIR)) for o in owners], 'equals_load_dataset': equal,
            'images': images}


def coco_image_path(img):
    return R.CIRCO_DIR / 'COCO2017_unlabeled' / 'unlabeled2017' / img['file_name']


def image_signature(images):
    """[file name, size, mtime_ns] of every selected image: the saved v_raw is reused only for exactly these files."""
    out = []
    for im in images:
        st = coco_image_path(im).stat()
        out.append([im['file_name'], st.st_size, st.st_mtime_ns])
    return out


def vraw_reusable(adir, images):
    """True if adir/vraw.npy was encoded from these files as they are now (names, order, sizes, mtimes)."""
    info = adir / 'vraw_info.json'
    if not ((adir / 'vraw.npy').exists() and info.exists()):
        return False
    try:
        return R.read_json(info).get('images') == image_signature(images)
    except OSError:                                           # an image vanished: encode again (and fail there)
        return False


def image_decodes(path):
    """True if PIL decodes the whole file. prepare_eval_data.image_ok (open + verify) does not decode JPEG data, so a
    download the server cut short passes it; load() raises 'image file is truncated' (LOAD_TRUNCATED_IMAGES off)."""
    import PIL.Image
    try:
        with PIL.Image.open(path) as im:
            im.load()
        return True
    except Exception:                                         # noqa: BLE001
        return False


def xn_e4_fetch(images, args, ldir):
    """Download the missing images as prepare_eval_data.py does (its download / image_ok; the server is probed first),
    after decoding every present one in full. ldir/E4_LEDGER lists each image this run started to download (it was
    absent then): such a file that does not decode (the server cut it short, or a kill) is removed and fetched again;
    any other one that does not decode stops the item with its name. A downloaded file is kept only if it decodes."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import PIL.ImageFile
    from evaluation import prepare_eval_data as P
    if PIL.ImageFile.LOAD_TRUNCATED_IMAGES:
        raise RuntimeError('PIL.ImageFile.LOAD_TRUNCATED_IMAGES is set in this process: a truncated image would decode '
                           'without an error')
    ledger = Path(ldir) / E4_LEDGER
    ours = set(ledger.read_text().split()) if ledger.exists() else set()
    present = [im for im in images if coco_image_path(im).exists()]
    with ThreadPoolExecutor(max_workers=max(1, args.download_workers)) as pool:
        decodes = list(pool.map(lambda im: image_decodes(coco_image_path(im)), present))
    broken = [im for im, ok in zip(present, decodes) if not ok]
    foreign = [im['file_name'] for im in broken if im['file_name'] not in ours]
    if foreign:
        raise RuntimeError(f'{len(foreign)} selected images that this run did not download do not decode, e.g. '
                           f'{foreign[:5]} (in {coco_image_path(broken[0]).parent}): replace them and rerun')
    for im in broken:
        coco_image_path(im).unlink()
    if broken:
        R.log(f'  removed {len(broken)} incomplete images of an earlier attempt, fetched again: '
              f'{[im["file_name"] for im in broken][:5]}')
    todo = [im for im in images if not coco_image_path(im).exists()]
    res = {'needed': len(images), 'present_before': len(images) - len(todo),     # present and decoding in full
           'removed_incomplete': [im['file_name'] for im in broken], 'downloaded': 0, 'failed': [], 'missing': [],
           'downloaded_files': []}
    if todo and args.no_download:
        raise RuntimeError(f'{len(todo)} images are missing and --no_download is set')
    lock = threading.Lock()

    def fetch(im, retries=3, timeout=60):
        dest = coco_image_path(im)
        part = dest.with_name(dest.name + '.part')
        with lock:                                            # recorded before the first byte is written
            if im['file_name'] not in ours:
                ledger.parent.mkdir(parents=True, exist_ok=True)
                with open(ledger, 'a') as fh:
                    fh.write(im['file_name'] + '\n')
                ours.add(im['file_name'])
        for _ in range(2):
            if part.exists():       # a killed attempt's: start over (a complete one would make a Range request fail)
                part.unlink()
            P.download(im.get('coco_url') or P.COCO_IMG_URL.format(im['file_name']), dest, retries=retries,
                       timeout=timeout)
            if P.image_ok(dest) and image_decodes(dest):
                return None
            dest.unlink()
        return im['file_name']

    if todo:
        R.log(f'  downloading {len(todo)} COCO images ({args.download_workers} parallel)')
        try:
            if fetch(todo[0], retries=2, timeout=30):
                raise RuntimeError('not a valid image')
        except Exception as e:                                # noqa: BLE001
            raise RuntimeError(f'the COCO image server is not reachable ({e}); rerun later or fetch the images') from e
        res['downloaded_files'].append(todo[0]['file_name'])
        t0, failed = time.time(), []
        with ThreadPoolExecutor(max_workers=max(1, args.download_workers)) as pool:
            for im, bad in zip(todo[1:], pool.map(lambda x: _safe_fetch(fetch, x), todo[1:])):
                (failed if bad else res['downloaded_files']).append(im['file_name'])
        res.update(downloaded=len(res['downloaded_files']), failed=failed, seconds=round(time.time() - t0, 1))
    res['missing'] = [im['file_name'] for im in images if not coco_image_path(im).exists()]
    return res


def _safe_fetch(fetch, im):
    try:
        return fetch(im)
    except Exception:                                         # noqa: BLE001
        return im['file_name']


class _Images:
    """Preprocessed images in selection order; a file that cannot be read stops the run (no blank substitutes)."""
    def __init__(self, paths, preprocess):
        self.paths, self.preprocess = paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import PIL.Image
        try:
            with PIL.Image.open(self.paths[i]) as im:
                return self.preprocess(im.convert('RGB'))
        except Exception as e:                                # noqa: BLE001  (a DataLoader worker would not name it)
            raise RuntimeError(f'{self.paths[i]}: {type(e).__name__}: {e}') from e


def xn_e4_encode(images, adir, args, cfg):
    import torch
    import torch.nn.functional as TF
    from torch.utils.data import DataLoader
    dev = args.device
    path = adir / 'vraw.npy'
    if vraw_reusable(adir, images):
        vr = F.load_array(path)
        R.log(f'  v_raw     : reused {path} (encoded from these files)')
        arrays = [R.read_json(adir / 'vraw_info.json')['array']]
    else:
        model = F.nx_load_model('baseline', dev)            # only its frozen CLIP is used
        sig = image_signature(images)
        loader = DataLoader(_Images([coco_image_path(im) for im in images], model.clip.preprocess),
                            batch_size=E4_ENCODE_BATCH, shuffle=False, num_workers=cfg.num_workers)
        out = []
        with torch.no_grad(), R.cudnn_settings(deterministic=True):
            for batch in loader:
                out.append(model.clip.encoder_image_unnorm(batch.to(dev)).float().cpu())
        vr = torch.cat(out)
        if image_signature(images) != sig:
            raise RuntimeError('selected images changed while they were encoded')
        arrays = [F.save_array(path, vr, 'float32')]
        R.write_json(adir / 'vraw_info.json', {'array': arrays[0], 'file_names': [im['file_name'] for im in images],
                                               'images': sig, 'images_columns': ['file_name', 'bytes', 'mtime_ns'],
                                               'encoder': 'clip.encoder_image_unnorm (fp32 CLIP), CLIP preprocess, '
                                                          f'batches of {E4_ENCODE_BATCH}, cuDNN deterministic'})
        del model
        F.free_memory()
    unit = torch.cat(_e4_unit_batches(args))
    cos = (TF.normalize(vr, dim=-1) * unit).sum(-1)
    nearest = (TF.normalize(vr, dim=-1) @ unit.T).argmax(dim=1)
    arrays.append(F.save_array(adir / 'vraw_cos.npy', cos, 'float32'))
    low = torch.argsort(cos)[:E4_LOWEST].tolist()
    return {'cos_min': float(cos.min()), 'cos_mean': float(cos.mean()),
            'lowest': [{'file_name': images[i]['file_name'], 'row': images[i]['row'], 'cos': float(cos[i])} for i in low],
            'nearest_is_self': int((nearest == torch.arange(len(images))).sum()),
            'vraw_norm_mean': float(vr.norm(dim=-1).mean()), 'vraw_norm_min': float(vr.norm(dim=-1).min()),
            'vraw_norm_max': float(vr.norm(dim=-1).max()), 'arrays': arrays}


def xn_e4_model(label, adir, args, sh):
    """neg_analysis_all.compute_metrics in both input regimes (nx_s13's aggregation: mean and population std)."""
    import numpy as np
    import torch
    import torch.nn.functional as TF
    NA = _neg_analysis(args)
    unit = _e4_unit_batches(args)
    vr = F.load_array(adir / 'vraw.npy')
    raw = [vr[i:i + b.shape[0]] for i, b in zip(itertools.accumulate([0] + [b.shape[0] for b in unit]), unit)]
    regimes, arrays = {}, []
    with R.cudnn_settings(deterministic=False):
        model, ls, meta = NA.load_model(str(F.ckpt_path(label)))
        for reg, batches in (('raw', raw), ('unit_norm', unit)):
            all_m, n_q = None, 0
            for b in batches:
                bm = NA.compute_metrics(model, b, ls)
                if all_m is None:
                    all_m = {k: [] for k in bm}
                for k, x in bm.items():
                    all_m[k].extend(x)
                n_q += b.shape[0]
            agg = {'n_queries': n_q, 'logit_scale': ls}
            for k, xs in all_m.items():
                mn = sum(xs) / len(xs)
                agg[f'{k}_mean'] = mn
                agg[f'{k}_std'] = (sum((x - mn) ** 2 for x in xs) / len(xs)) ** 0.5
            cols = list(all_m)
            arr = np.array([all_m[k] for k in cols], dtype=np.float64).T
            arrays.append(F.save_array(adir / f'{label}__{reg}.npy', arr, 'float64'))
            regimes[reg] = {'aggregate': agg, 'columns': cols,
                            'inbatch_r1': sum(x > 0 for x in all_m['gt_minus_hardest']) / n_q,
                            'input_norm_mean': float(torch.cat(batches).norm(dim=-1).mean())}
        v2 = torch.from_numpy(np.load(DUMP / 'models' / label / 'circo_V2.npy'))
        pos = (v2 * TF.normalize(sh.vraw, dim=-1)).sum(-1)
        m = NA.compute_metrics(model, sh.vraw, ls)
        d = float((pos - torch.tensor(m['sim_p_vraw'])).abs().max())
    del model
    F.free_memory()
    return {'logit_scale': ls, 'meta': meta, 'regimes': regimes, 'arrays': arrays, 'v2_max_abs_diff': d,
            'v2_n': int(sh.vraw.shape[0])}


def _ckpt_members(path):
    import zipfile
    with zipfile.ZipFile(path) as z:
        return {n.split('/', 1)[1] if '/' in n else n: z.read(n) for n in z.namelist()}


def _deep_equal(a, b, where='', out=None):
    import torch
    out = [] if out is None else out
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a, b)):
            out.append(where)
    elif isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            out.append(f'{where} keys {list(a)} != {list(b)}')
        for k in a:
            if k in b:
                _deep_equal(a[k], b[k], f'{where}/{k}', out)
    elif type(a) is not type(b) or a != b:
        out.append(f'{where}: {a!r} != {b!r}')
    return out


def compare_checkpoint_files(built, existing):
    """Rebuilt vs existing release file: bytes, zip members and (loaded) values. torch.save's .data/serialization_id
    is <hash of the record names: 20 digits><crc32 of the data: 20 digits>; the first half uses the C++ library's
    std::hash, so it can differ between platforms while every record is identical."""
    import torch
    a, b = _ckpt_members(built), _ckpt_members(existing)
    sid = '.data/serialization_id'
    differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    crc_same = a.get(sid, b'')[20:] == b.get(sid, b'')[20:] and len(a.get(sid, b'')) == 40
    deep = _deep_equal(torch.load(built, map_location='cpu'), torch.load(existing, map_location='cpu'))
    return {'bytes_identical': Path(built).read_bytes() == Path(existing).read_bytes(),
            'members_differing': differing, 'serialization_id_crc_part_identical': crc_same, 'deep_equal': not deep,
            'deep_differences': deep[:20],
            'equivalent': not deep and crc_same and set(differing) <= {sid}}


def _loader_fields_equal(new, src):
    """Every field the loaders read is the same in the release copy and the source (mapper and merger weights, mapper
    type and kwargs, logit scale, the model fields of the config)."""
    model_fields = ('clip_model', 'hidden_dim', 'num_tokens', 'num_heads', 'num_layers', 'dropout', 'merge_tokens',
                    'merge_type', 'max_token_len', 'clip_prompt', 'vib_logvar_bias', 'vib_logvar_clip_max', 'pseudo_norm')
    diff = _deep_equal(new['mapper_state'], dict(src['mapper_state']))
    diff += [f'config/{k}' for k in model_fields if new['config'].get(k) != src['config'].get(k)]
    diff += [k for k in ('mapper_type', 'mapper_kwargs', 'merger_state', 'logit_scale') if new[k] != src[k]]
    return not diff, diff


def _place_without_overwrite(tmp, dest):
    """Move tmp to dest unless dest exists (then it must be byte-identical); a hard link or O_EXCL never overwrites."""
    if dest.exists():
        same = R.sha256_file(dest) == R.sha256_file(tmp)
        tmp.unlink()
        if not same:
            raise RuntimeError(f'{dest} exists and differs from the rebuilt copy: it was not overwritten')
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(tmp, dest)
    except FileExistsError:
        return _place_without_overwrite(tmp, dest)
    except OSError:
        fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, 'wb') as fh:
            fh.write(tmp.read_bytes())
            fh.flush()
            os.fsync(fh.fileno())
    tmp.unlink()
    return True


def xn_e5_build(args, odir):
    import tempfile
    import torch
    pairs = []
    work = odir / 'build'
    work.mkdir(parents=True, exist_ok=True)
    for rel_label, src_label, vib_beta in RELEASE_PAIRS:
        existing = ck_path(rel_label)
        built = release_copy(torch.load(ck_path(src_label), map_location='cpu'), vib_beta)
        with tempfile.TemporaryDirectory(dir=work) as td:
            p = Path(td) / existing.name                       # same file name: torch.save names the archive after it
            torch.save(built, p)
            cmp = compare_checkpoint_files(p, existing)
        pairs.append(dict(cmp, release=XCK[rel_label]['rel'], source=ck_info(src_label)['rel'], vib_beta_set=vib_beta))
        R.log(f'  rebuild   : {XCK[rel_label]["rel"]} from {ck_info(src_label)["rel"]}: bytes identical '
              f'{cmp["bytes_identical"]}, records differing {cmp["members_differing"] or "none"}, values equal '
              f'{cmp["deep_equal"]}  {"ok" if cmp["equivalent"] else "DIFFERS"}')
    new = {'file': NEW_RELEASE_REL, 'created': False}
    if all(p['equivalent'] for p in pairs):
        src = torch.load(ck_path('run2'), map_location='cpu')
        if src['config'].get('vib_beta') != 0.1024:
            raise RuntimeError(f'run 2 stores vib_beta {src["config"].get("vib_beta")}, expected 0.1024')
        built = release_copy(src)
        ok, diff = _loader_fields_equal(built, src)
        dest = R.EXPERIMENT / NEW_RELEASE_REL
        tmp = work / dest.name
        if tmp.exists():
            tmp.unlink()
        torch.save(built, tmp)
        created = _place_without_overwrite(tmp, dest)
        cfg_diff = {k: (src['config'].get(k), built['config'].get(k)) for k in src['config']
                    if src['config'].get(k) != built['config'].get(k)}
        new.update(created=created, sha256=R.sha256_file(dest), bytes=dest.stat().st_size, loader_fields_equal=ok,
                   loader_fields_differing=diff, config_differing={k: list(v) for k, v in cfg_diff.items()},
                   keys=list(built), dropped=[k for k in src if k not in built])
    shutil.rmtree(work, ignore_errors=True)
    return {'pairs': pairs, 'new': new}


def val_quick(q, sh, gal):
    """final_run.nx_quick_metrics for the val rows, which all lie in its first scoring batch (rows 0..BATCH-1), with the
    per-query AP@10 and reference rank."""
    from evaluation.eval_circo_v2 import _compute_query_ap_at_k
    g, img_ids, id2idx = gal['emb'], gal['img_ids'], gal['id2idx']
    start, n = sh.splits['val']
    scores = q[slice(0, min(F.BATCH, len(sh.queries)))].to(g.device) @ g.T
    ap10, ranks = [], []
    for i in range(start, start + n):
        a, row = sh.queries[i], scores[i]
        ref_idx = id2idx[str(a['ref_img_id'])]
        ranks.append(int((row > row[ref_idx]).sum().item() + 1))
        row[ref_idx] = float('-inf')
        top = [int(img_ids[j]) for j in row.topk(50).indices.cpu().tolist()]
        ap10.append(_compute_query_ap_at_k(top, a['gt_img_ids'], 10))
    return {'val_map10': F.mean(ap10), 'val_ref10': F.mean([int(r <= 10) for r in ranks]), 'ap10': ap10, 'rank_ref': ranks}


def xn_e6(label, sigma2, ndir, args, sh):
    """S1's tokens, eps, composition and scoring (nx_s1) with the noise scaled by a."""
    import numpy as np
    import torch
    dev = args.device
    model = F.nx_load_model(label, dev)
    start, n = sh.splits['val']
    b = min(F.BATCH, len(sh.queries))
    if start != 0 or n > b:
        raise RuntimeError('the CIRCO val rows must lie in the first scoring batch')
    caps, rows = [q['caption'] for q in sh.queries], [q['ref_row'] for q in sh.queries]
    with torch.no_grad(), R.cudnn_settings(deterministic=False):
        v = sh.vraw.to(dev)
        if sigma2 is None:
            if not hasattr(model.mapper, 'fc_logvar'):
                raise RuntimeError(f'{label} is not a VIB model')
            mu, lv = (torch.cat(x) for x in zip(*[F.vib_mu_logvar(model, v[s]) for s in F.batched(v.shape[0], F.BATCH)]))
            base, scale = mu, (0.5 * lv).exp()
            det = F.tokens_batched(model, v)[0]
            kind = 'z = mu + a * exp(0.5 * clamp(logvar)) * eps, then merger and pseudo_norm (as S1)'
            scale_info = {'sigma2_mean': float((scale ** 2).mean())}
        else:
            base = det = F.tokens_batched(model, v)[0]
            scale = torch.full_like(base, math.sqrt(sigma2))
            kind = 'z = z_det + a * s * eps, s^2 = mean sigma^2 of VIB e01s1500 (as the S1 control)'
            scale_info = {'s': math.sqrt(sigma2), 's2': sigma2}
        eps = F.load_array(DUMP / 'shared' / 's1_eps.npy')
        if tuple(eps.shape) != (F.S1_SAMPLES,) + tuple(base.shape):
            raise RuntimeError(f's1_eps.npy has shape {tuple(eps.shape)}, tokens {tuple(base.shape)}')
        gal = F.nx_gallery(dev)
        qd = F.composed(model, det[rows], caps, dev)            # every query, for the check against nx_quick_metrics
        full = F.nx_quick_metrics(qd, sh, gal)
        mine = val_quick(qd[:b], sh, gal)
        quick_check = {'equal': full['val_map10'] == mine['val_map10'] and full['val_ref10'] == mine['val_ref10'],
                       'nx_quick_metrics': [full['val_map10'], full['val_ref10']],
                       'val_quick': [mine['val_map10'], mine['val_ref10']]}
        curve, ap_arr, rk_arr, tn_arr = [], [], [], []
        for a in E6_ALPHAS:
            samples = [0] if a == 0.0 else range(F.S1_SAMPLES)
            per = []
            for k in samples:
                zk = base + (scale * a) * eps[k].to(dev)
                if sigma2 is None:
                    if model.merger is not None:
                        zk = model.merger(zk)
                    zk = F.project(model, zk)
                m = val_quick(F.composed(model, zk[rows[:b]], caps[:b], dev), sh, gal)
                m['token_norm'] = float(zk.norm(dim=-1).reshape(zk.shape[0], -1).mean(dim=1).mean())
                per.append(m)
            if a == 0.0:
                per = per * F.S1_SAMPLES
            ap_arr.append([m['ap10'] for m in per])
            rk_arr.append([m['rank_ref'] for m in per])
            tn_arr.append([m['token_norm'] for m in per])
            maps, refs = [m['val_map10'] for m in per], [m['val_ref10'] for m in per]
            curve.append({'a': a, 'map10': maps, 'ref10': refs, 'map10_mean': F.mean(maps),
                          'map10_sd': statistics.pstdev(maps), 'ref10_mean': F.mean(refs), 'ref10_sd': statistics.pstdev(refs),
                          'token_norm_mean': F.mean([m['token_norm'] for m in per]),
                          'per_query_ap10': [m['ap10'] for m in per], 'per_query_rank_ref': [m['rank_ref'] for m in per]})
    arrays = [F.save_array(ndir / 'ap10.npy', np.array(ap_arr, dtype=np.float64), 'float64'),
              F.save_array(ndir / 'rank_ref.npy', np.array(rk_arr, dtype=np.int32), 'int32'),
              F.save_array(ndir / 'token_norm.npy', np.array(tn_arr, dtype=np.float64), 'float64')]
    ap_back, rk_back = np.load(ndir / 'ap10.npy'), np.load(ndir / 'rank_ref.npy')
    recomputed = all(F.mean(ap_back[i, k].tolist()) == c['map10'][k]
                     and F.mean([int(r <= 10) for r in rk_back[i, k].tolist()]) == c['ref10'][k]
                     for i, c in enumerate(curve) for k in range(F.S1_SAMPLES))
    R.write_json(ndir / 'info.json', {'label': label, 'kind': kind, 'alphas': list(E6_ALPHAS), 'scale': scale_info,
                                      'arrays': arrays, 'axes': '[alpha, sample, val query]',
                                      'val_query_ids': [q['query_id'] for q in sh.queries[start:start + n]],
                                      'eps': 'dump/shared/s1_eps.npy (S1)', 'note': 'a = 0 computed once, repeated'})
    del model
    F.free_memory()
    return {'kind': kind, 'scale': scale_info, 'curve': curve, 'arrays': arrays, 'quick_check': quick_check,
            'recomputed_equal': recomputed}


if __name__ == '__main__':
    main()

"""Reference-kept CIRCO run for the arXiv version (eval only, nothing is trained): the GPU inputs for scoring every CIRCO
validation number of the paper under two rules, with the reference image removed from each ranked list (the paper's
rule) and with it kept (the rule of SEARLE's and LinCIR's released CIRCO code). Run once on the A6000 machine of the
earlier runs (evaluation/reeval.py, reeval_addon.py, final_run.py, extra_run.py), in the same environment.

Why
    Every CIRCO number of the paper was computed with the reference image removed: its score is set to -inf before
    topk(50) (eval_circo_v2.py, eval_circo_test.py). The published CIRCO code of SEARLE and LinCIR keeps it, so every
    CIRCO validation number gets a reference-kept counterpart. What can be computed from saved features is done on the
    Mac; this run makes the rest. It saves the composed-query features of everything, and all final scoring under both
    rules is done on the Mac by one code, evaluation/circo_ref_protocol.py. The numbers printed here (both-rule mAP@10
    and Ref@10 per item, and the self-checks) are a summary for the log.

Items (in this order; --only selects a subset; each one is resumable)
    ours         every checkpoint of refkept_bundle/bundle_info.json (every checkpoint of every run behind a CIRCO val
                 number of the paper): the 220 CIRCO val composed queries, one query at a time through the canonical
                 path of eval_circo_v2.eval_circo (reference image -> CLIP preprocess -> model.encode_query), the model
                 built by eval_circo_v2.load_model, the loader of every earlier run (all mapper types, token mergers,
                 pseudo_norm, old configs), after final_run's checkpoint checks. Each query is scored as eval_circo_v2
                 scores it (p @ gallery.T on the GPU, one query; REMOVED sets the reference to -inf before topk(50), KEPT
                 takes topk(50) of the same scores; AP@K by eval_circo_v2._compute_query_ap_at_k). Self-checks: where
                 reference/known_val.json holds the checkpoint (an earlier GPU evaluation), the REMOVED per-query AP@10
                 must equal it exactly; where the checkpoint is a paper model (dump_label), max |V1 -
                 reference/val_V1/<dump_label>.npy| (the final run's features) is recorded, and counts as exact only
                 when it is 0.0 (expected on the GPU type and software of the final run)
    e6           the extra run's E6 sampling-noise curve, z = mu + a * sigma * eps (vib, run2, beta02) and z = z_det + a *
                 s * eps (the baseline control, s^2 = vib's mean sigma^2, from reference/e6/baseline_control/info.json),
                 replaying extra_run.xn_e6 step by step (final_run's vib_mu_logvar, tokens_batched, project, composed and
                 extra_run.val_quick) on the CIRCO part of final_run's dump/shared, rebuilt here exactly as nx_shared
                 builds it (queries val then test, references in first-use order, v_raw one image at a time);
                 eps = reference/s1_eps.npy. Per draw: the val query features, AP@10 under both rules (KEPT: the same
                 scores without the mask) and the reference rank. Self-check: the REMOVED AP@10 and the reference ranks
                 equal reference/e6/<model>/ exactly, and so does the noise scale
    qualitative  find_qualitative.py's selection, its own main() (gallery load, per_query_ranks, filter, order) fed with
                 the ours step's composed queries of the paper's baseline and VIB (dump labels baseline and vib), under
                 both rules (KEPT: per_query_ranks with the reference left in the gallery). find_qualitative leaves the
                 final pick to the author, and its order puts query 150 first; the paper shows query 60. The pick is the
                 paper's query whenever it qualifies under the rule, otherwise the first in find_qualitative's order.
                 Self-check (REMOVED): 13 qualifying val queries, query 60 among them with baseline rank 46 and VIB rank
                 1 (the fig:qual caption), and the whole candidate list equal to metrics/qualitative_candidates.json,
                 which the current figure was drawn from
    qualitative_figures (part of --only qualitative) both figures drawn by render_qualitative.py's main(); gallery
                 images they need are downloaded like prepare_eval_data.py does (extra_run.xn_e4_fetch: the URL of the
                 COCO image list, each file decoded in full), each listed in qualitative/downloaded_files.txt
    externals    evaluation/ref_kept_externals.py's run_externals (the published methods). Its outputs go to externals/;
                 the status, numbers and self-check it returns per method go into the summary. A failure there is
                 recorded, not fatal

Exact checks
    CIRCO scores one query at a time on the GPU and has reproduced bit for bit across sessions and hosts of the same GPU
    type (the earlier runs' known models, the final run's dump self-check, the extra run's gate), so every self-check is
    exact: per-query AP@10 and reference ranks equal, features equal to the last bit.

Outputs (results/ref_kept_2026_11/)
    ours/<label>/circo_val_V1.npy     220 x 768 fp32, val-annotation order
    ours/<label>/meta.json            tau, mapper type, tokens, merger, pseudo_norm, checkpoint sha256, seconds; the
                                      per-query reference rank and AP@10 under both rules; numbers and self-check
    shared/                           circo_vraw.npy (1,017 x 768 fp32) and circo_index.json, the E6 input
    e6/<model>/                       ap10_removed.npy, ap10_kept.npy, rank_ref.npy (6 alphas x 8 samples x 220),
                                      token_norm.npy (6 x 8), circo_val_queries.npy (6 x 8 x 220 x 768 fp32), info.json
    qualitative/                      candidates_{removed,kept}.json (find_qualitative's output), selection_{removed,
                                      kept}.json (qualifying queries and the pick), fig_{removed,kept}.{pdf,png},
                                      downloaded_files.txt
    externals/                        written by ref_kept_externals.py
    entries/<item>.json               one record per item: status, numbers, self-check, outputs with their sha256
    summary.json, preflight.json, run_info.json, run_log.txt (stdout, every line stamped with UTC time),
    MANIFEST.sha256                   every file under the output folder except itself and run_log.txt
    The log's last line is "EXIT CODE n". summary.json covers the items of the last invocation.

Exit codes: 0 everything done and every self-check exact; 1 preflight failed (nothing was run); 2 done, but at least one
self-check mismatched (listed); 3 at least one item failed with an exception (listed, with any mismatches). Every item
runs even if an earlier one fails. Rerunning the same command resumes: an item is skipped when its record says done, its
parameters are the same and every output it lists exists with its stored sha256; anything else runs again (also a done
item with a mismatched self-check is skipped: delete its entries/<item>.json, or pass --no_resume, to measure it again).

--max_queries N is for TESTS ONLY: it keeps the first N queries of each CIRCO split, so nothing it writes is a paper
number; the ours self-checks then compare those N queries, and the E6 and qualitative paper checks are not applied.

Usage (repo at /notebooks/experiment/codes; paperspace_bundle.zip and paperspace_refkept_bundle.zip unpacked into
/notebooks/experiment; evaluation/prepare_eval_data.py run; matplotlib installed for the figures; the external methods'
inputs in /notebooks/experiment/refkept_assets)
    cd /notebooks/experiment/codes
    PYTHONPATH=/notebooks/experiment/codes ~/venv/bin/python -u evaluation/ref_kept_run.py --preflight_only
    PYTHONPATH=/notebooks/experiment/codes ~/venv/bin/python -u evaluation/ref_kept_run.py
"""

import argparse
import importlib
import inspect
import math
import os
import shutil
import statistics
import sys
import time
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import reeval as R          # noqa: E402  paths, helpers, cuDNN settings of the canonical CIRCO runs
from evaluation import reeval_addon as A    # noqa: E402  checkpoint metadata (load_meta)
from evaluation import final_run as F       # noqa: E402  gallery, tokens, composition, arrays, versions
from evaluation import extra_run as X       # noqa: E402  E6 scoring (val_quick), image fetching, manifest, disk quota


# ---------------------------------------------------------------------------
# Fixed inputs
# ---------------------------------------------------------------------------
ITEMS = ('ours', 'e6', 'qualitative', 'externals')          # run order; --only selects a subset
RULES = ('removed', 'kept')
RULE_TEXT = {'removed': "the reference's score set to -inf before topk(50), the paper's rule (eval_circo_v2.py)",
             'kept': "topk(50) over the whole gallery, the reference included (SEARLE's and LinCIR's CIRCO code)"}
N_VAL, N_TEST, N_REFS = 220, 800, 1017                       # CIRCO val / test queries, their unique reference images
KS = (5, 10, 25, 50)
TOP = 50
MAP_TOL = X.CIRCO_OFFICIAL_TOL      # mean of identical per-query AP@10 vs the stored (official) mAP@10: summation order
EXPECTED_SHA256 = {                 # the bundled copies the earlier runs read (paperspace_bundle.zip)
    'annotations/val.json': '6b31a1dbc3c85627501d9a250f87708f24a27d74d6a65360b6ddd2537a154d04',
    'gallery_cache_ViT-L-14.pt': '437d997bc71df5f15ceaeb4c9505fd7ffe099a0d5bda0cc1e98ebd787db62604'}
E6_MODELS = X.E6_MODELS + ('baseline_control',)              # reference/e6/<model>
E6_DUMP_LABEL = {'vib': 'vib', 'run2': 'run2', 'beta02': 'beta02', 'baseline_control': 'baseline'}
E6_ALPHAS = X.E6_ALPHAS
QUAL_MODELS = ('baseline', 'vib')                            # dump labels of the figure's two rows
QUAL_ARGS = {'base_gt_min': 10, 'vib_gt_max': 5, 'ref_max': 5, 'topk': 5}   # find_qualitative.py's defaults
QUAL_SHOW = 15
QUAL_PAPER = {'n_qualifying': 13, 'pick': 60, 'baseline_gt_rank': 46, 'vib_gt_rank': 1}   # fig:qual
QUAL_STORED = REPO_ROOT / 'metrics' / 'qualitative_candidates.json'                       # the current figure's list
MATPLOTLIB_INSTALL = '~/venv/bin/pip install matplotlib==3.9.4'
EXT_DONE = ('done', 'ok', 'complete', 'completed', 'passed')
EXT_SELFCHECK_KEYS = ('selfcheck_ok', 'selfcheck', 'self_check', 'checks', 'check')   # ref_kept_externals: selfcheck_ok
MIN_FREE_GB = 2.0                   # ours ~0.25 GB, e6 ~0.13 GB, the externals and the tarball
EST_SECONDS = {'ours': 14, 'e6_shared': 45, 'e6': 25, 'qualitative': 60}   # A6000, from the earlier runs' timings


def _self_check_tables():
    assert E6_MODELS == ('vib', 'run2', 'beta02', 'baseline_control') and set(E6_DUMP_LABEL) == set(E6_MODELS)
    assert E6_ALPHAS[0] == 0.0 and 1.0 in E6_ALPHAS
    assert set(QUAL_MODELS) <= set(E6_DUMP_LABEL.values())


_self_check_tables()


# ---------------------------------------------------------------------------
# Log: stdout with a UTC time stamp on every line, mirrored into <out>/run_log.txt
# ---------------------------------------------------------------------------
class _Log:
    def __init__(self, stream, path):
        self.stream, self.file, self.bol = stream, (open(path, 'a') if path else None), True   # noqa: SIM115

    def write(self, s):
        if not s:
            return 0
        parts = []
        for piece in s.splitlines(keepends=True):
            if self.bol:
                parts.append(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S  '))
            parts.append(piece)
            self.bol = piece.endswith(('\n', '\r'))
        self._out(''.join(parts))
        return len(s)

    def raw(self, s):
        """Unstamped text (the final EXIT CODE line)."""
        self._out(s if self.bol else '\n' + s)
        self.bol = s.endswith('\n')
        self.flush()

    def _out(self, text):
        self.stream.write(text)
        if self.file:
            self.file.write(text)

    def flush(self):
        self.stream.flush()
        if self.file:
            self.file.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self.stream.fileno()


def start_log(out):
    """Stamp stdout; mirror it into out/run_log.txt unless stdout is that file already (a nohup redirect into it)."""
    path = out / 'run_log.txt'
    try:
        same = path.exists() and os.path.samestat(os.fstat(sys.stdout.fileno()), os.stat(path))
    except (OSError, ValueError, AttributeError):         # io.UnsupportedOperation is both an OSError and a ValueError
        same = False
    old = sys.stdout
    log = _Log(old, None if same else path)
    sys.stdout = log

    def restore():
        log.flush()
        if log.file:
            log.file.close()
        sys.stdout = old
    return log, restore


# ---------------------------------------------------------------------------
# Context, records, resume
# ---------------------------------------------------------------------------
class Ctx:
    def __init__(self, args):
        self.args, self.out, self.bundle_dir = args, args.out, args.bundle_dir
        self.resume, self.max_q = not args.no_resume, args.max_queries
        self.items = tuple(i for i in ITEMS if not args.only or i in args.only)
        self.recs, self.cache = {}, {}
        self.info, self.entries, self.by_label, self.known_val = None, [], {}, {}
        self.n_val = self.n_test = None

    def load_bundle(self):
        self.info = R.read_json(self.bundle_dir / 'bundle_info.json')
        self.entries = list(self.info['entries'])
        self.by_label = {e['label']: e for e in self.entries}
        kv = self.bundle_dir / 'reference' / 'known_val.json'
        self.known_val = R.read_json(kv) if kv.exists() else {}

    def entry_for(self, dump_label):
        hits = [e for e in self.entries if e.get('dump_label') == dump_label]
        if len(hits) != 1:
            raise RuntimeError(f'bundle_info.json has {len(hits)} entries with dump_label {dump_label!r}, expected 1')
        return hits[0]

    def ours_entries(self):
        if not self.args.labels:
            return list(self.entries)
        want = set(self.args.labels)
        return [e for e in self.entries if e['label'] in want]

    def rel(self, path):
        return Path(path).relative_to(self.out).as_posix()


def entry_file(out, item):
    return out / 'entries' / f'{item}.json'


def outputs_intact(out, rec):
    for rel, sha in (rec.get('outputs') or {}).items():
        p = out / rel
        if not p.exists() or R.sha256_file(p) != sha:
            return False, rel
    return True, None


def verdict(rec):
    if rec.get('status') != 'done':
        return 'FAILED'
    ok = (rec.get('selfcheck') or {}).get('ok')
    return {True: 'PASS (self-check exact)', False: 'SELF-CHECK MISMATCH', None: 'done (nothing to check)'}[ok]


def run_item(ctx, item, title, fn, params=None, resume=None):
    """Run one item, or skip it when its record says done with the same parameters and intact outputs. A failure is
    recorded (the run goes on) and retried on the next invocation."""
    out, path = ctx.out, entry_file(ctx.out, item)
    resume = ctx.resume if resume is None else resume
    if resume and path.exists():
        try:
            rec = R.read_json(path)
        except (OSError, ValueError):
            rec = None
        if rec and rec.get('status') == 'done':
            if rec.get('params') != params:
                R.log(f'  {item}: done before with other parameters ({rec.get("params")}): measuring again')
            else:
                intact, bad = outputs_intact(out, rec)
                if intact:
                    R.log(f'  {item}: already done ({verdict(rec)})')
                    ctx.recs[item] = rec
                    return rec
                R.log(f'  {item}: output {bad} is missing or differs from its stored sha256: measuring again')
    R.log()
    R.log(f'--- {item}: {title}   [{R.now()}]')
    t0 = time.time()
    try:
        rec = fn()
        rec.update(item=item, title=title, status='done', params=params, seconds=round(time.time() - t0, 1),
                   finished_at=R.now())
    except (Exception, SystemExit):                           # noqa: BLE001  (SystemExit: a reused main() that gave up)
        R.log(traceback.format_exc())
        R.log(f'  FAILED: {item} (continuing with the others; rerun the same command to retry it)')
        rec = {'item': item, 'title': title, 'status': 'failed', 'params': params,
               'error': traceback.format_exc(limit=8), 'finished_at': R.now()}
    F.free_memory()
    R.write_json(path, rec)
    R.log(f'  => {item}: {verdict(rec)}  ({rec.get("seconds", "-")} s)')
    ctx.recs[item] = rec
    return rec


def jsonable(o):
    """A JSON-safe copy (numpy / torch scalars and arrays, Paths, tuples) of what another module returned."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, 'tolist'):
        return jsonable(o.tolist())
    if hasattr(o, 'item'):
        return o.item()
    return repr(o)


# ---------------------------------------------------------------------------
# Checkpoints: the canonical loader with final_run's checks
# ---------------------------------------------------------------------------
def expected_pseudo_norm(source):
    """1.0 for the unit-norm run (pnorm1.0), 0.0 (or absent) everywhere else: the scan of every checkpoint."""
    return 1.0 if '_pnorm1.0' in source else 0.0


def meta_problems(e, meta):
    """final_run.checkpoint_problems' rules for one bundle entry (reeval_addon.load_meta of the file)."""
    p = []
    if not meta['has_mapper_state'] or not meta['mapper_type']:
        p.append('no mapper_state / mapper_type entry')
    if e.get('mapper_type') is not None and e['mapper_type'] != meta['mapper_type']:
        p.append(f'mapper_type {meta["mapper_type"]!r}, bundle_info.json says {e["mapper_type"]!r}')
    wants = meta['merge_tokens'] is not None and meta['merge_tokens'] < meta['num_tokens']
    if wants != meta['has_merger_state']:
        p.append(f'merge_tokens={meta["merge_tokens"]} but merger_state '
                 f'{"present" if meta["has_merger_state"] else "missing"}')
    if e.get('has_merger') is not None and e['has_merger'] != meta['has_merger_state']:
        p.append(f'merger_state {"present" if meta["has_merger_state"] else "missing"}, bundle_info.json says '
                 f'has_merger={e["has_merger"]}')
    if meta['pseudo_norm'] != expected_pseudo_norm(e['source']):
        p.append(f'pseudo_norm={meta["pseudo_norm"]}, expected {expected_pseudo_norm(e["source"])} for this run')
    return p


def load_entry_model(e, bundle_dir, device):
    """The model exactly as eval_circo_v2.load_model builds it (the loader of reeval, reeval_addon and final_run: every
    mapper type, mergers, pseudo_norm, old configs with absent keys), checked against the checkpoint's own config."""
    from evaluation import eval_circo_v2
    path = Path(bundle_dir) / e['file']
    meta = A.load_meta(path)
    problems = meta_problems(e, meta)
    if problems:
        raise RuntimeError(f'{e["label"]}: ' + '; '.join(problems))
    model, clip_name = eval_circo_v2.load_model(path, device)
    if clip_name != 'ViT-L/14':
        raise RuntimeError(f'{e["label"]}: backbone {clip_name}; the CIRCO gallery cache is ViT-L/14')
    if float(model.pseudo_norm) != meta['pseudo_norm']:
        raise RuntimeError(f'{e["label"]}: model.pseudo_norm={model.pseudo_norm}, the checkpoint says '
                           f'{meta["pseudo_norm"]}')
    if (model.merger is not None) != meta['has_merger_state']:
        raise RuntimeError(f'{e["label"]}: the model {"has" if model.merger is not None else "has no"} merger, the '
                           f'checkpoint {"has" if meta["has_merger_state"] else "has no"} merger_state')
    if int(model.mapper.num_tokens) != meta['num_tokens']:
        raise RuntimeError(f'{e["label"]}: {model.mapper.num_tokens} tokens, the checkpoint config says '
                           f'{meta["num_tokens"]}')
    info = {'mapper_type': meta['mapper_type'], 'num_tokens': int(model.mapper.num_tokens),
            'merger': None if model.merger is None else {'type': type(model.merger).__name__,
                                                         'merge_tokens': int(model.merger.merge_tokens)},
            'merge_tokens_config': meta['merge_tokens'], 'pseudo_norm': float(model.pseudo_norm),
            'logit_scale': float(model.clip.logit_scale.item()), 'tau': float(model.clip.logit_scale.exp().item()),
            'clip_model': clip_name, 'use_multilayer': bool(model.use_multilayer)}
    return model, info


# ---------------------------------------------------------------------------
# Scoring (one query, as eval_circo_v2.eval_circo)
# ---------------------------------------------------------------------------
def score_one(p, ref_idx, gt_img_ids, target_img_id, gal):
    """One composed query p (1 x 768, on the gallery's device) scored as eval_circo_v2.eval_circo scores it, under both
    rules. Reference rank: 1 + the number of gallery scores strictly above the reference's."""
    from evaluation.eval_circo_v2 import _compute_query_ap_at_k
    g, img_ids = gal['emb'], gal['img_ids']
    scores = (p @ g.T).squeeze(0)
    sim_ref = scores[ref_idx].item()
    rank_ref = (scores > scores[ref_idx]).sum().item() + 1
    kept = [int(img_ids[i]) for i in scores.topk(TOP).indices.cpu().tolist()]
    masked = scores.clone()
    masked[ref_idx] = float('-inf')
    removed = [int(img_ids[i]) for i in masked.topk(TOP).indices.cpu().tolist()]
    rec = {'rank_ref': int(rank_ref), 'sim_ref': float(sim_ref)}
    for rule, top in (('removed', removed), ('kept', kept)):
        rec[rule] = {'ap': {k: _compute_query_ap_at_k(top, gt_img_ids, k) for k in KS},
                     'recall': {k: float(target_img_id is not None and int(target_img_id) in top[:k]) for k in KS}}
    return rec


def aggregate(rows):
    """mAP@K and Recall@K (CIRCO evaluation.py: the target alone for recall) under both rules, Ref@K as counts."""
    import numpy as np
    ranks = [r['rank_ref'] for r in rows]
    out = {'n_queries': len(rows), **{f'ref{k}': sum(x <= k for x in ranks) for k in (1, 5, 10, 50)},
           'median_rank_ref': float(np.median(ranks)) if ranks else None}
    for rule in RULES:
        out[rule] = {'map': {str(k): float(np.mean([r[rule]['ap'][k] for r in rows])) for k in KS},
                     'recall': {str(k): float(np.mean([r[rule]['recall'][k] for r in rows])) for k in KS}}
    return out


def val_kept(q, sh, gal):
    """extra_run.val_quick with the reference kept: the same scores (the first scoring batch), topk(50) of all of them."""
    from evaluation.eval_circo_v2 import _compute_query_ap_at_k
    g, img_ids = gal['emb'], gal['img_ids']
    start, n = sh.splits['val']
    scores = q[slice(0, min(F.BATCH, len(sh.queries)))].to(g.device) @ g.T
    ap10 = []
    for i in range(start, start + n):
        top = [int(img_ids[j]) for j in scores[i].topk(TOP).indices.cpu().tolist()]
        ap10.append(_compute_query_ap_at_k(top, sh.queries[i]['gt_img_ids'], 10))
    return {'val_map10': F.mean(ap10), 'ap10': ap10}


# ---------------------------------------------------------------------------
# Step 1: preflight
# ---------------------------------------------------------------------------
def coco_info_path():
    return R.CIRCO_DIR / 'COCO2017_unlabeled' / 'annotations' / 'image_info_unlabeled2017.json'


def ref_order(anns):
    """Unique reference image ids in order of first use (final_run.nx_shared's row order)."""
    return list(dict.fromkeys(int(a['reference_img_id']) for a in anns))


def env_problems(device):
    """reeval / final_run's environment checks without their Fashion-IQ data."""
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
        problems.append(f'This clone is at {R.REPO_ROOT}, but the loaders hardcode {R.CODES_ROOT}. Clone it there.')
    if not R.CONFIG_YAML.exists():
        problems.append(f'Missing {R.CONFIG_YAML}')
    return problems


def read_manifest(path):
    out = {}
    for ln in path.read_text().splitlines():
        if ln.strip():
            sha, rel = ln.split(None, 1)
            out[rel.strip().lstrip('*')] = sha.strip()
    return out


def bundle_problems(ctx):
    """MANIFEST.sha256 verified file by file; bundle_info.json consistent with it; the reference files present."""
    bd, problems, warnings, info = ctx.bundle_dir, [], [], {}
    man = bd / 'MANIFEST.sha256'
    if not bd.is_dir() or not man.exists():
        return [f'Missing {man}: unzip paperspace_refkept_bundle.zip into {R.EXPERIMENT}'], warnings, info
    manifest = read_manifest(man)
    missing = [rel for rel in manifest if not (bd / rel).exists()]
    differ = [rel for rel in manifest if rel not in missing and R.sha256_file(bd / rel) != manifest[rel]]
    info.update(manifest_files=len(manifest), manifest_sha256=R.sha256_file(man), missing=missing[:20],
                differing=differ[:20])
    problems += [f'bundle file missing: {rel}' for rel in missing[:10]]
    problems += [f'bundle file differs from MANIFEST.sha256: {rel}' for rel in differ[:10]]
    if len(missing) + len(differ) > 20:
        problems.append(f'{len(missing)} bundle files missing and {len(differ)} differing in all')
    if 'bundle_info.json' not in manifest:
        problems.append('bundle_info.json is not in MANIFEST.sha256')
    try:
        ctx.load_bundle()
    except Exception as e:                                    # noqa: BLE001
        problems.append(f'{bd / "bundle_info.json"} cannot be read: {e}')
        return problems, warnings, info
    labels = [e['label'] for e in ctx.entries]
    if len(labels) != len(set(labels)):
        problems.append('bundle_info.json: labels are not unique')
    for e in ctx.entries:
        if manifest.get(e['file']) != e['sha256']:
            problems.append(f'{e["label"]}: {e["file"]} is not in MANIFEST.sha256 with the sha256 of bundle_info.json')
        if e.get('group') not in ('4M', 'tau', '1p5m'):
            warnings.append(f'{e["label"]}: unknown group {e.get("group")!r}')
    unknown = sorted(set(ctx.known_val) - {e['source'] for e in ctx.entries})
    if unknown:
        warnings.append(f'reference/known_val.json holds {len(unknown)} sources that are not bundle entries')
    if ctx.args.labels:
        bad = [x for x in ctx.args.labels if x not in ctx.by_label]
        if bad:
            problems.append(f'--labels: not in bundle_info.json: {bad[:10]}')
    need = ['reference/known_val.json']
    need += [f'reference/val_V1/{e["dump_label"]}.npy' for e in ctx.entries if e.get('dump_label')]
    if 'e6' in ctx.items:
        need += ['reference/s1_eps.npy'] + [f'reference/e6/{m}/{f}' for m in E6_MODELS
                                            for f in ('ap10.npy', 'rank_ref.npy', 'info.json')]
    problems += [f'bundle reference file missing from MANIFEST.sha256: {rel}' for rel in need if rel not in manifest]
    dls = set()
    if 'e6' in ctx.items:
        dls |= set(E6_DUMP_LABEL.values())
    if 'qualitative' in ctx.items:
        dls |= set(QUAL_MODELS)
    for dl in sorted(dls):
        try:
            ctx.entry_for(dl)
        except RuntimeError as ex:
            problems.append(str(ex))
    info.update(entries=len(ctx.entries), runs=len({e['run'] for e in ctx.entries}),
                groups={g: sum(e.get('group') == g for e in ctx.entries) for g in ('4M', 'tau', '1p5m')},
                dump_labels=sorted(e['dump_label'] for e in ctx.entries if e.get('dump_label')),
                known_val=sum(e['source'] in ctx.known_val for e in ctx.entries),
                excluded_runs=len(ctx.info.get('excluded_runs', [])))
    return problems, warnings, info


def needed_entries(ctx):
    labels = set()
    if 'ours' in ctx.items:
        labels |= {e['label'] for e in ctx.ours_entries()}
    dls = (set(E6_DUMP_LABEL.values()) if 'e6' in ctx.items else set()) | \
          (set(QUAL_MODELS) if 'qualitative' in ctx.items else set())
    labels |= {e['label'] for e in ctx.entries if e.get('dump_label') in dls}
    return [e for e in ctx.entries if e['label'] in labels]


def checkpoint_problems(ctx, entries):
    rows, problems = {}, []
    for e in entries:
        p = ctx.bundle_dir / e['file']
        row = rows[e['label']] = {'run': e['run'], 'group': e.get('group'), 'file': e['file']}
        if not p.exists():
            problems.append(f'{e["label"]}: missing checkpoint {p}')
            continue
        try:
            meta = A.load_meta(p)
        except Exception as ex:                               # noqa: BLE001
            problems.append(f'{e["label"]}: torch.load failed: {ex}')
            continue
        row.update(meta)
        problems += [f'{e["label"]}: {x}' for x in meta_problems(e, meta)]
    return rows, problems


def data_problems(ctx):
    """CIRCO annotations, the ViT-L/14 gallery cache, the COCO image list, CIRCO's code and the reference images."""
    problems, warnings, info = [], [], {}
    circo = R.CIRCO_DIR
    hint = ': unzip paperspace_bundle.zip into /notebooks/experiment and run evaluation/prepare_eval_data.py'
    need = [circo / 'annotations' / 'val.json', circo / 'annotations' / 'test.json', R.GALLERY, coco_info_path(),
            circo / 'circo_repo' / 'src' / 'dataset.py', circo / 'circo_repo' / 'src' / 'evaluation.py']
    problems += [f'Missing data: {p}{hint}' for p in need if not p.exists()]
    if not all(p.exists() for p in need[:2]):
        return problems, warnings, info
    val, test = (R.read_json(circo / 'annotations' / f'{s}.json') for s in ('val', 'test'))
    if ctx.max_q is None:
        ctx.n_val, ctx.n_test = len(val), len(test)
        if (len(val), len(test)) != (N_VAL, N_TEST):
            problems.append(f'CIRCO has {len(val)} val / {len(test)} test queries, expected {N_VAL} / {N_TEST}')
    else:
        ctx.n_val, ctx.n_test = min(ctx.max_q, len(val)), min(ctx.max_q, len(test))
        warnings.append(f'--max_queries {ctx.max_q} (TESTS ONLY): {ctx.n_val} val and {ctx.n_test} test queries')
    used = val[:ctx.n_val] + test[:ctx.n_test]
    refs = ref_order(used)
    if ctx.max_q is None and len(refs) != N_REFS:
        problems.append(f'the CIRCO queries have {len(refs)} reference images, expected {N_REFS}')
    sha = R.sha256_file(circo / 'annotations' / 'val.json')
    if sha != EXPECTED_SHA256['annotations/val.json']:
        warnings.append(f'annotations/val.json has sha256 {sha[:16]}..., the bundled copy of the earlier runs '
                        f'{EXPECTED_SHA256["annotations/val.json"][:16]}...')
    info.update(n_val=ctx.n_val, n_test=ctx.n_test, n_refs=len(refs), val_sha256=sha)
    if coco_info_path().exists():
        names = {int(i['id']): i['file_name'] for i in R.read_json(coco_info_path())['images']}
        lacking = [r for r in refs if r not in names
                   or not (circo / 'COCO2017_unlabeled' / 'unlabeled2017' / names[r]).exists()]
        if lacking:
            problems.append(f'{len(lacking)} of the {len(refs)} reference images are missing (e.g. {lacking[:5]}): run '
                            'evaluation/prepare_eval_data.py')
        info['reference_images_present'] = len(refs) - len(lacking)
    if R.GALLERY.exists():
        import torch
        cache = torch.load(R.GALLERY, map_location='cpu')
        emb, id2idx = cache['embeddings'], cache['img_id_to_idx']
        if cache.get('clip_model') != 'ViT-L/14' or emb.ndim != 2 or emb.shape[1] != 768:
            problems.append(f'{R.GALLERY}: clip_model {cache.get("clip_model")}, embeddings {tuple(emb.shape)}')
        gts = {int(g) for a in val[:ctx.n_val] for g in a['gt_img_ids']}
        absent = [x for x in refs + sorted(gts) if str(x) not in id2idx]
        if absent:
            problems.append(f'{len(absent)} reference / ground-truth ids are not in the gallery cache, e.g. {absent[:5]}')
        dev = float((emb.float().norm(dim=1) - 1).abs().max())
        if dev > 1e-4:
            problems.append(f'{R.GALLERY}: embeddings are not L2-normalized (max |norm - 1| {dev:.2e})')
        gsha = R.sha256_file(R.GALLERY)
        if gsha != EXPECTED_SHA256['gallery_cache_ViT-L-14.pt']:
            warnings.append(f'{R.GALLERY.name} has sha256 {gsha[:16]}..., the bundled copy of the earlier runs '
                            f'{EXPECTED_SHA256["gallery_cache_ViT-L-14.pt"][:16]}...')
        info.update(gallery_size=int(emb.shape[0]), gallery_sha256=gsha)
        del cache, emb
    return problems, warnings, info


def externals_preflight(ctx):
    """ref_kept_externals.preflight(circo_dir, assets_dir) (its preflight_report when present, which adds warnings),
    called with the arguments it takes; its problems block the run."""
    try:
        mod = importlib.import_module('evaluation.ref_kept_externals')
    except Exception as e:                                    # noqa: BLE001
        return [f'evaluation/ref_kept_externals.py cannot be imported: {e}'], [], {}
    fn = getattr(mod, 'preflight_report', None) or getattr(mod, 'preflight', None)
    if fn is None:
        return ['evaluation/ref_kept_externals.py has no preflight()'], [], {}
    offer = {'out_dir': ctx.out / 'externals', 'circo_dir': R.CIRCO_DIR, 'assets_dir': ctx.args.assets_dir,
             'device': ctx.args.device, 'log': R.log, 'max_queries': ctx.max_q, 'resume': ctx.resume}
    try:
        params = inspect.signature(fn).parameters
        takes_all = any(p.kind == p.VAR_KEYWORD for p in params.values())
        res = fn(**{k: v for k, v in offer.items() if takes_all or k in params})
    except Exception:                                         # noqa: BLE001
        return [f'ref_kept_externals.preflight raised: {traceback.format_exc(limit=3)}'], [], {}
    problems, warnings = [], []
    if isinstance(res, dict):
        problems, warnings = list(res.get('problems') or []), list(res.get('warnings') or [])
    elif isinstance(res, tuple) and len(res) == 2 and all(isinstance(x, (list, tuple)) for x in res):
        problems, warnings = list(res[0]), list(res[1])
    elif isinstance(res, (list, tuple)):
        problems = list(res)
    elif res not in (None, True):
        problems = [f'ref_kept_externals.preflight returned {res!r}']
    tag = lambda s: str(s) if str(s).startswith('externals:') else f'externals: {s}'      # noqa: E731
    return [tag(p) for p in problems], [tag(w) for w in warnings], jsonable(res) if isinstance(res, dict) else {}


def disk_problems(out):
    problems, warnings = [], []
    free_gb = shutil.disk_usage(out).free / 1e9
    info = {'free_gb': round(free_gb, 1)}
    if free_gb < MIN_FREE_GB:
        problems.append(f'Only {free_gb:.1f} GB free under {out}; this run needs about {MIN_FREE_GB:.0f} GB')
    ceph = X.cephfs_usage(out)
    if ceph:
        qdir, quota, used = ceph
        info.update(quota_dir=qdir, quota_gb=round(quota / 1e9, 2), used_gb=round(used / 1e9, 2))
        if (quota - used) / 1e9 < MIN_FREE_GB:
            problems.append(f'The storage quota of {qdir} leaves {(quota - used) / 1e9:.1f} GB; this run needs about '
                            f'{MIN_FREE_GB:.0f} GB')
    else:
        warnings.append(f'the volume reports {free_gb:,.0f} GB free; on a network mount that need not be your quota: '
                        f'check that about {MIN_FREE_GB:.0f} GB of it are left')
    return problems, warnings, info


def preflight(ctx):
    problems, warnings, report = [], [], {}
    v = F.versions()
    problems += env_problems(ctx.args.device)
    warnings += F.version_warnings(v)
    p, w, report['bundle'] = bundle_problems(ctx)
    problems, warnings = problems + p, warnings + w
    p, w, report['data'] = data_problems(ctx)
    problems, warnings = problems + p, warnings + w
    if ctx.info is not None:
        rows, p = checkpoint_problems(ctx, needed_entries(ctx))
        report['checkpoints'] = rows
        problems += p
    if 'qualitative' in ctx.items:
        try:
            import matplotlib                                 # noqa: F401  render_qualitative.py draws with it
        except Exception as e:                                # noqa: BLE001
            problems.append(f'matplotlib is not importable ({e}); the qualitative figures need it: {MATPLOTLIB_INSTALL}')
        if not QUAL_STORED.exists():
            problems.append(f'Missing {QUAL_STORED} (the current figure\'s candidate list, the qualitative self-check)')
    if 'externals' in ctx.items:
        p, w, report['externals'] = externals_preflight(ctx)
        problems, warnings = problems + p, warnings + w
    p, w, report['disk'] = disk_problems(ctx.out)
    return problems + p, warnings + w, report, v


def log_preflight(ctx, report, v, warnings):
    b = report.get('bundle', {})
    if b.get('entries'):
        groups = ', '.join(f'{g} {n}' for g, n in b['groups'].items())
        R.log(f'  bundle  : {b["entries"]} checkpoints in {b["runs"]} runs ({groups}); '
              f'{len(b["dump_labels"])} paper models; {b["known_val"]} with an exact earlier val result; '
              f'{b["excluded_runs"]} runs left out; MANIFEST {b["manifest_files"]} files')
    rows = report.get('checkpoints', {})
    by_run = {}
    for e in ctx.entries:
        if e['label'] in rows:
            by_run.setdefault(e['run'], []).append((e, rows[e['label']]))
    for run, items in by_run.items():
        metas = [m for _, m in items if 'mapper_type' in m]
        taus = [m['tau'] for m in metas if m.get('tau')]
        merger = sorted({f'{m["merge_tokens"]}/{m["merge_type"]}' for m in metas if m['has_merger_state']}) or ['-']
        R.log(f'  {items[0][0].get("group", "?"):4s} {run:26s} {len(items):3d} checkpoints  '
              f'{"/".join(sorted({m["mapper_type"] for m in metas})):4s} tokens '
              f'{"/".join(str(t) for t in sorted({m["num_tokens"] for m in metas}))}  merger {",".join(merger):10s} '
              f'pseudo_norm {"/".join(str(x) for x in sorted({m["pseudo_norm"] for m in metas}))}  tau '
              f'{(f"{min(taus):.2f}-{max(taus):.2f}" if taus else "-"):13s} paper models '
              f'{sum(bool(e.get("dump_label")) for e, _ in items)}  known val '
              f'{sum(e["source"] in ctx.known_val for e, _ in items)}')
    d = report.get('data', {})
    if d:
        R.log(f'  CIRCO   : {d.get("n_val")} val / {d.get("n_test")} test queries, {d.get("n_refs")} reference images '
              f'({d.get("reference_images_present", "-")} present), gallery {d.get("gallery_size", "-")} images')
    R.log('  versions: ' + ', '.join(f'{k} {v.get(k)}' for k in ('python', 'torch', 'torchvision', 'numpy', 'pillow',
                                                                 'clip_commit', 'gpu')))
    for w in warnings:
        R.log(f'  note: {w}')


def log_estimate(ctx):
    steps = []
    if 'ours' in ctx.items:
        n = len(ctx.ours_entries())
        steps.append((f'ours: {n} checkpoints x {ctx.n_val} val queries', n * EST_SECONDS['ours']))
    if 'e6' in ctx.items:
        steps.append((f'e6: v_raw of the reference images, {len(E6_MODELS)} models x 41 draws',
                      EST_SECONDS['e6_shared'] + len(E6_MODELS) * EST_SECONDS['e6']))
    if 'qualitative' in ctx.items:
        steps.append(('qualitative: selection under both rules, image downloads, two figures', EST_SECONDS['qualitative']))
    R.log('  Runtime estimate (A6000; finished items are skipped on a rerun):')
    for name, sec in steps:
        R.log(f'    {name:72s} ~{sec / 60:5.1f} min')
    if 'externals' in ctx.items:
        R.log(f'    {"externals: see ref_kept_externals.py":72s}     ? min')


# ---------------------------------------------------------------------------
# Step 2: ours
# ---------------------------------------------------------------------------
def val_dataset(ctx):
    """The CIRCO val dataset and CLIP preprocess, as eval_circo_v2.eval_circo builds them (built once)."""
    if 'val' not in ctx.cache:
        import clip
        from evaluation import eval_circo_v2
        circo_ds, _ = eval_circo_v2._load_circo_official(R.CIRCO_DIR)
        _, preprocess = clip.load('ViT-L/14', device='cpu')
        ctx.cache['val'] = (circo_ds(data_path=str(R.CIRCO_DIR), split='val', mode='relative', preprocess=preprocess),
                            preprocess)
    return ctx.cache['val']


def ours_params(ctx, e):
    return {'checkpoint_sha256': e['sha256'], 'n_queries': ctx.n_val, 'max_queries': ctx.max_q}


def ours_selfcheck(ctx, e, rows, qids, v1):
    import numpy as np
    n, checks = len(rows), {}
    kv = ctx.known_val.get(e['source'])
    if kv is not None:
        now = [r['removed']['ap'][10] for r in rows]
        stored = list(kv['ap10_per_query'])[:n]
        diff = [i for i, (a, b) in enumerate(zip(now, stored)) if a != b]
        full = n == len(kv['ap10_per_query'])
        ref10, map10 = sum(r['rank_ref'] <= 10 for r in rows), float(np.mean(now))
        ok = len(stored) == n and not diff
        if full:
            ok = ok and ref10 == kv['ref10'] and abs(map10 - kv['map10']) <= MAP_TOL
        checks['known_val'] = {'ok': ok, 'origin': kv.get('origin'), 'n_compared': len(stored), 'partial': not full,
                               'ap10_identical': n - len(diff), 'differing_query_ids': [qids[i] for i in diff][:20],
                               'map10_now': map10, 'map10_stored': kv['map10'], 'map10_abs_diff': abs(map10 - kv['map10']),
                               'ref10_now': ref10, 'ref10_stored': kv['ref10']}
    dl = e.get('dump_label')
    if dl:
        rp = ctx.bundle_dir / 'reference' / 'val_V1' / f'{dl}.npy'
        ref = np.load(rp, allow_pickle=False)[:n]
        if ref.shape != v1.shape:
            checks['dump_V1'] = {'ok': False, 'reason': f'shape {ref.shape} vs {v1.shape}', 'reference': f'val_V1/{dl}.npy'}
        else:
            d = np.abs(v1.astype(np.float64) - ref.astype(np.float64))
            md = float(d.max()) if d.size else 0.0
            checks['dump_V1'] = {'ok': md == 0.0, 'exact': md == 0.0, 'max_abs_diff': md,
                                 'rows_differing': int((d > 0).any(axis=1).sum()), 'n_rows': n,
                                 'reference': f'reference/val_V1/{dl}.npy (final run, dump/models/{dl}/circo_V1.npy)'}
    return {'ok': None if not checks else all(c['ok'] for c in checks.values()), 'checks': checks}


def fmt_selfcheck(sc):
    c = sc['checks']
    if not c:
        return 'no reference (no earlier exact val result, not a paper model)'
    out = []
    if 'known_val' in c:
        k = c['known_val']
        out.append(f'known val: AP@10 {k["ap10_identical"]}/{k["n_compared"]} identical'
                   + ('' if k['partial'] else f', Ref@10 {k["ref10_now"]} vs {k["ref10_stored"]}')
                   + (' (partial)' if k['partial'] else '') + f' {"exact" if k["ok"] else "MISMATCH"}')
    if 'dump_V1' in c:
        d = c['dump_V1']
        out.append(f'dump V1: max |d| {d.get("max_abs_diff", "-")} {"exact" if d["ok"] else "NOT EXACT"}')
    return '; '.join(out)


def ours_item(ctx, e):
    import clip
    import PIL.Image
    import torch
    t0, dev, n = time.time(), ctx.args.device, ctx.n_val
    odir = ctx.out / 'ours' / e['label']
    model, minfo = load_entry_model(e, ctx.bundle_dir, dev)
    ds, preprocess = val_dataset(ctx)
    gal = F.nx_gallery(dev)
    feats, rows, qids = [], [], []
    with torch.no_grad(), R.cudnn_settings(deterministic=False):          # eval_circo_v2.eval_circo, as reeval runs it
        for q_idx in range(n):
            ann = ds.annotations[q_idx]
            ref_path = ds.img_paths[ds.img_ids_indexes_map[str(ann['reference_img_id'])]]
            ref_img = preprocess(PIL.Image.open(ref_path).convert('RGB')).unsqueeze(0).to(dev)
            txt_tok = clip.tokenize([ann['relative_caption']], truncate=True).to(dev)
            p = model.encode_query(ref_img, txt_tok)                      # (1, 768), normalized
            rows.append(score_one(p, gal['id2idx'][str(ann['reference_img_id'])], ann['gt_img_ids'],
                                  ann.get('target_img_id'), gal))
            feats.append(p.float().cpu())
            qids.append(ann['id'])
    del model
    v1 = torch.cat(feats)
    arr = F.save_array(odir / 'circo_val_V1.npy', v1, 'float32')
    numbers = aggregate(rows)
    sc = ours_selfcheck(ctx, e, rows, qids, v1.numpy())
    meta = {'label': e['label'], 'run': e['run'], 'group': e.get('group'), 'source': e['source'], 'file': e['file'],
            'checkpoint_sha256': e['sha256'], 'source_sha256': e.get('source_sha256'), 'dump_label': e.get('dump_label'),
            'epoch': e.get('epoch'), 'step': e.get('step'), 'is_final': e.get('is_final'), **minfo,
            'n_queries': n, 'max_queries': ctx.max_q, 'query_ids': qids, 'features': arr,
            'path': 'eval_circo_v2.eval_circo: one query at a time, CLIP preprocess of the reference image, '
                    'model.encode_query(image, clip.tokenize([caption], truncate=True)); scores p @ gallery.T on the GPU',
            'rules': RULE_TEXT, 'numbers': numbers,
            'per_query': {'rank_ref': [r['rank_ref'] for r in rows], 'sim_ref': [r['sim_ref'] for r in rows],
                          **{f'ap10_{rule}': [r[rule]['ap'][10] for r in rows] for rule in RULES}},
            'selfcheck': sc, 'seconds': round(time.time() - t0, 1), 'created_at': R.now()}
    R.write_json(odir / 'meta.json', meta)
    m_r, m_k = numbers['removed']['map']['10'], numbers['kept']['map']['10']
    merger = minfo['merger']['type'] if minfo['merger'] else '-'
    R.log(f'  mAP@10 removed {100 * m_r:.4f}  kept {100 * m_k:.4f} ({100 * (m_k - m_r):+.4f})  Ref@10 '
          f'{numbers["ref10"]}/{n}  tau {minfo["tau"]:.2f}  {minfo["mapper_type"]}, {minfo["num_tokens"]} token(s), '
          f'merger {merger}, pseudo_norm {minfo["pseudo_norm"]}')
    R.log(f'  self-check : {fmt_selfcheck(sc)}')
    return {'label': e['label'], 'run': e['run'], 'group': e.get('group'), 'dump_label': e.get('dump_label'),
            'numbers': numbers, 'model': minfo, 'selfcheck': sc,
            'outputs': {ctx.rel(odir / 'circo_val_V1.npy'): arr['sha256'],
                        ctx.rel(odir / 'meta.json'): R.sha256_file(odir / 'meta.json')}}


def ensure_ours(ctx, e):
    item = f'ours__{e["label"]}'
    if item in ctx.recs:
        return ctx.recs[item]
    return run_item(ctx, item, f'{e["source"]} (needed by the qualitative figure)', lambda: ours_item(ctx, e),
                    params=ours_params(ctx, e))


def ours_step(ctx):
    todo = ctx.ours_entries()
    durations = []
    for i, e in enumerate(todo, 1):
        t0 = time.time()
        rec = run_item(ctx, f'ours__{e["label"]}', f'[{i}/{len(todo)}] {e["source"]}', lambda e=e: ours_item(ctx, e),
                       params=ours_params(ctx, e))
        if rec.get('finished_at') and time.time() - t0 > 1:
            durations.append(time.time() - t0)
        left = len(todo) - i
        if durations and left and i % 10 == 0:
            R.log(f'  progress   : {i}/{len(todo)} checkpoints, about {left * statistics.mean(durations) / 60:.0f} min '
                  f'left in this step')


# ---------------------------------------------------------------------------
# Step 3: E6
# ---------------------------------------------------------------------------
class CircoShared:
    """The CIRCO part of final_run's dump/shared (circo_index.json, circo_vraw.npy) loaded back; fp32 round-trips
    exactly. Has what extra_run.xn_e6 / val_quick read: queries, refs, splits, vraw."""
    def __init__(self, sdir):
        ci = R.read_json(sdir / 'circo_index.json')
        self.queries, self.refs, self.splits = ci['queries'], ci['refs'], ci['splits']
        self.vraw = F.load_array(sdir / 'circo_vraw.npy')
        self.dir = sdir


def shared_item(ctx):
    """final_run.nx_shared's CIRCO part: queries in annotation order (val then test), unique references in order of
    first use, v_raw of each reference one image at a time with the frozen CLIP of the baseline's model."""
    import PIL.Image
    import torch
    from evaluation import eval_circo_v2
    dev = ctx.args.device
    model, _ = load_entry_model(ctx.entry_for('baseline'), ctx.bundle_dir, dev)
    preprocess = model.clip.preprocess                       # = clip.load(...)[1] of the canonical loaders
    circo_ds, _ = eval_circo_v2._load_circo_official(R.CIRCO_DIR)
    queries, refs, ref_row, splits = [], [], {}, {}
    for split, n in (('val', ctx.n_val), ('test', ctx.n_test)):
        ds = circo_ds(data_path=str(R.CIRCO_DIR), split=split, mode='relative', preprocess=preprocess)
        splits[split] = [len(queries), n]
        for qi, ann in enumerate(ds.annotations[:n]):
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
    del model
    sdir = ctx.out / 'shared'
    arr = F.save_array(sdir / 'circo_vraw.npy', vr, 'float32')
    R.write_json(sdir / 'circo_index.json', {'queries': queries, 'refs': refs, 'splits': splits,
                                             'rows': 'circo_vraw rows = refs; query rows = queries (val then test)'})
    R.log(f'  shared     : {len(queries)} queries ({splits}), {len(refs)} reference images, v_raw {tuple(vr.shape)}')
    return {'counts': {'queries': len(queries), 'refs': len(refs)}, 'splits': splits,
            'outputs': {ctx.rel(sdir / 'circo_vraw.npy'): arr['sha256'],
                        ctx.rel(sdir / 'circo_index.json'): R.sha256_file(sdir / 'circo_index.json')}}


def get_shared(ctx):
    rec = ctx.recs.get('e6_shared')
    if not rec or rec.get('status') != 'done':
        raise RuntimeError('the shared encodes (item e6_shared) are not available')
    if 'shared' not in ctx.cache:
        ctx.cache['shared'] = CircoShared(ctx.out / 'shared')
    return ctx.cache['shared']


def e6_eps(ctx, sh):
    """reference/s1_eps.npy ([8, 1017, T, 768]); under --max_queries its rows of the references used (by id)."""
    eps = F.load_array(ctx.bundle_dir / 'reference' / 's1_eps.npy')
    if ctx.max_q is None:
        return eps, 'all rows'
    full = {r: i for i, r in enumerate(ref_order(R.read_json(R.CIRCO_DIR / 'annotations' / 'val.json')
                                                 + R.read_json(R.CIRCO_DIR / 'annotations' / 'test.json')))}
    rows = [full[int(r['ref_img_id'])] for r in sh.refs]
    return eps[:, rows], rows


def e6_params(ctx, name):
    return {'checkpoint_sha256': ctx.entry_for(E6_DUMP_LABEL[name])['sha256'], 'max_queries': ctx.max_q,
            'shared': (ctx.recs.get('e6_shared') or {}).get('outputs')}


def e6_item(ctx, name):
    """extra_run.xn_e6, step by step, with the KEPT rule and the query features of every draw added."""
    import numpy as np
    import torch
    sh = get_shared(ctx)
    dev, control = ctx.args.device, name == 'baseline_control'
    e = ctx.entry_for(E6_DUMP_LABEL[name])
    ref_dir = ctx.bundle_dir / 'reference' / 'e6' / name
    ref_info = R.read_json(ref_dir / 'info.json')
    sigma2 = float(ref_info['scale']['s2']) if control else None     # the extra run's control: s1_vib's sigma2_mean
    model, minfo = load_entry_model(e, ctx.bundle_dir, dev)
    if model.use_multilayer:
        raise RuntimeError(f'{name}: multilayer mappers are not supported here')
    start, n = sh.splits['val']
    b = min(F.BATCH, len(sh.queries))
    if start != 0 or n > b:
        raise RuntimeError('the CIRCO val rows must lie in the first scoring batch')
    caps, rows = [q['caption'] for q in sh.queries], [q['ref_row'] for q in sh.queries]
    with torch.no_grad(), R.cudnn_settings(deterministic=False):
        v = sh.vraw.to(dev)
        if sigma2 is None:
            if not hasattr(model.mapper, 'fc_logvar'):
                raise RuntimeError(f'{name} is not a VIB model')
            mu, lv = (torch.cat(x) for x in zip(*[F.vib_mu_logvar(model, v[s]) for s in F.batched(v.shape[0], F.BATCH)]))
            base, scale = mu, (0.5 * lv).exp()
            kind = 'z = mu + a * exp(0.5 * clamp(logvar)) * eps, then merger and pseudo_norm (as S1)'
            scale_info = {'sigma2_mean': float((scale ** 2).mean())}
        else:
            base = F.tokens_batched(model, v)[0]
            scale = torch.full_like(base, math.sqrt(sigma2))
            kind = 'z = z_det + a * s * eps, s^2 = mean sigma^2 of VIB e01s1500 (as the S1 control)'
            scale_info = {'s': math.sqrt(sigma2), 's2': sigma2}
        eps, eps_rows = e6_eps(ctx, sh)
        if tuple(eps.shape) != (F.S1_SAMPLES,) + tuple(base.shape):
            raise RuntimeError(f's1_eps.npy has shape {tuple(eps.shape)}, tokens {tuple(base.shape)}')
        gal = F.nx_gallery(dev)
        curve, ap_r, ap_k, rk, tn, qf = [], [], [], [], [], []
        for a in E6_ALPHAS:
            per = []
            for k in ([0] if a == 0.0 else range(F.S1_SAMPLES)):
                zk = base + (scale * a) * eps[k].to(dev)
                if sigma2 is None:
                    if model.merger is not None:
                        zk = model.merger(zk)
                    zk = F.project(model, zk)
                q = F.composed(model, zk[rows[:b]], caps[:b], dev)
                m = X.val_quick(q, sh, gal)
                m['ap10_kept'] = val_kept(q, sh, gal)['ap10']
                m['token_norm'] = float(zk.norm(dim=-1).reshape(zk.shape[0], -1).mean(dim=1).mean())
                m['q'] = q[start:start + n].clone()
                per.append(m)
            if a == 0.0:
                per = per * F.S1_SAMPLES
            ap_r.append([m['ap10'] for m in per])
            ap_k.append([m['ap10_kept'] for m in per])
            rk.append([m['rank_ref'] for m in per])
            tn.append([m['token_norm'] for m in per])
            qf.append(torch.stack([m['q'] for m in per]))
            mr, mk = [F.mean(m['ap10']) for m in per], [F.mean(m['ap10_kept']) for m in per]
            r10 = [sum(x <= 10 for x in m['rank_ref']) for m in per]
            curve.append({'a': a, 'map10_removed': mr, 'map10_kept': mk, 'ref10': r10,
                          'map10_removed_mean': F.mean(mr), 'map10_removed_sd': statistics.pstdev(mr),
                          'map10_kept_mean': F.mean(mk), 'map10_kept_sd': statistics.pstdev(mk),
                          'ref10_mean': F.mean(r10), 'ref10_sd': statistics.pstdev(r10),
                          'token_norm_mean': F.mean([m['token_norm'] for m in per])})
    del model
    edir = ctx.out / 'e6' / name
    now_ap, now_rk = np.array(ap_r, dtype=np.float64), np.array(rk, dtype=np.int32)
    arrays = [F.save_array(edir / 'ap10_removed.npy', now_ap, 'float64'),
              F.save_array(edir / 'ap10_kept.npy', np.array(ap_k, dtype=np.float64), 'float64'),
              F.save_array(edir / 'rank_ref.npy', now_rk, 'int32'),
              F.save_array(edir / 'token_norm.npy', np.array(tn, dtype=np.float64), 'float64'),
              F.save_array(edir / 'circo_val_queries.npy', torch.stack(qf), 'float32')]
    sc = e6_selfcheck(ctx, ref_dir, ref_info, now_ap, now_rk, scale_info, arrays)
    info = {'label': name, 'dump_label': E6_DUMP_LABEL[name], 'bundle_label': e['label'], 'checkpoint': e['source'],
            'checkpoint_sha256': e['sha256'], 'kind': kind, 'alphas': list(E6_ALPHAS), 'samples': F.S1_SAMPLES,
            'scale': scale_info, 'eps': 'reference/s1_eps.npy (the final dump\'s shared/s1_eps.npy)',
            'eps_rows': eps_rows, 'arrays': arrays,
            'axes': '[alpha, sample, val query] (circo_val_queries: [alpha, sample, val query, 768])',
            'val_query_ids': [q['query_id'] for q in sh.queries[start:start + n]], 'max_queries': ctx.max_q,
            'scoring': "S1's (extra_run.val_quick): the first 256 queries composed in one batch and scored as one "
                       'matmul; removed masks the reference; kept takes topk(50) of the same scores; a = 0 computed once',
            'curve': curve, 'selfcheck': sc, 'model': minfo, 'created_at': R.now()}
    R.write_json(edir / 'info.json', info)
    for c in curve:
        R.log(f'  a = {c["a"]:4.2f}: mAP@10 removed {100 * c["map10_removed_mean"]:6.2f} +- '
              f'{100 * c["map10_removed_sd"]:4.2f}  kept {100 * c["map10_kept_mean"]:6.2f} +- '
              f'{100 * c["map10_kept_sd"]:4.2f}'
              f'  Ref@10 {c["ref10_mean"]:6.1f}/{n}  mean ||z|| {c["token_norm_mean"]:.3f}')
    fi = sc['checks']['files_identical']
    R.log('  self-check : ' + ', '.join(f'{k} {"exact" if v["ok"] else ("not compared" if v["ok"] is None else "MISMATCH")}'
                                        for k, v in sc['checks'].items() if k != 'files_identical')
          + "; .npy files byte-identical to the extra run's: "
          + ', '.join(f'{k} {"yes" if fi[k] else "no"}' for k in ('ap10', 'rank_ref', 'token_norm')))
    outputs = {ctx.rel(edir / a['file']): a['sha256'] for a in arrays}
    outputs[ctx.rel(edir / 'info.json')] = R.sha256_file(edir / 'info.json')
    return {'label': name, 'checkpoint': e['source'], 'scale': scale_info,
            'curve': [{k: v for k, v in c.items()} for c in curve], 'selfcheck': sc, 'outputs': outputs}


def e6_selfcheck(ctx, ref_dir, ref_info, now_ap, now_rk, scale_info, arrays):
    import numpy as np
    n = now_ap.shape[-1]
    ref_ap = np.load(ref_dir / 'ap10.npy', allow_pickle=False)[:, :, :n]
    ref_rk = np.load(ref_dir / 'rank_ref.npy', allow_pickle=False)[:, :, :n]
    partial = ctx.max_q is not None

    def cmp(now, ref, what):
        same_shape = now.shape == ref.shape
        eq = same_shape and np.array_equal(now, ref)
        c = {'ok': None if partial else bool(eq), 'equal': bool(eq), 'shape': list(now.shape),
             'reference_shape': list(ref.shape), 'reference': f'reference/e6/{ref_dir.name}/{what}',
             'n_differing': int((now != ref).sum()) if same_shape else None}
        if partial:
            c['reason'] = '--max_queries changes the composed batch: not comparable'
        return c
    checks = {'ap10_removed': cmp(now_ap, ref_ap, 'ap10.npy'), 'rank_ref': cmp(now_rk, ref_rk, 'rank_ref.npy'),
              'scale': {'ok': None if partial else scale_info == ref_info.get('scale'), 'now': scale_info,
                        'reference': ref_info.get('scale')}}
    files = {a['file']: a['sha256'] for a in ref_info.get('arrays', [])}   # info.json lists the extra run's sha256s
    mine = {a['file']: a['sha256'] for a in arrays}
    checks['files_identical'] = {'ok': None, 'note': 'information: the sha256 of the extra run\'s files (its info.json)',
                                 'ap10': mine['ap10_removed.npy'] == files.get('ap10.npy'),
                                 'rank_ref': mine['rank_ref.npy'] == files.get('rank_ref.npy'),
                                 'token_norm': mine['token_norm.npy'] == files.get('token_norm.npy')}
    states = [c['ok'] for c in checks.values() if c['ok'] is not None]
    return {'ok': (all(states) if states else None), 'checks': checks}


def e6_step(ctx):
    run_item(ctx, 'e6_shared', f'v_raw of the CIRCO reference images (final_run.nx_shared, CIRCO part)',
             lambda: shared_item(ctx), params={'n_val': ctx.n_val, 'n_test': ctx.n_test, 'max_queries': ctx.max_q})
    for i, name in enumerate(E6_MODELS, 1):
        try:
            params = e6_params(ctx, name)
        except RuntimeError:
            params = None
        what = 'z_det + a * s * eps (baseline)' if name == 'baseline_control' else 'z = mu + a * sigma * eps'
        run_item(ctx, f'e6__{name}', f'[{i}/{len(E6_MODELS)}] {name}: {what}', lambda name=name: e6_item(ctx, name),
                 params=params)
    vib, ctl = ctx.recs.get('e6__vib', {}), ctx.recs.get('e6__baseline_control', {})
    if vib.get('status') == 'done' and ctl.get('status') == 'done':
        same = vib['scale'].get('sigma2_mean') == ctl['scale'].get('s2')
        R.log(f'  control s^2 = vib mean sigma^2 of this run: {"yes" if same else "NO"} '
              f'({ctl["scale"].get("s2")} vs {vib["scale"].get("sigma2_mean")})')


# ---------------------------------------------------------------------------
# Step 4: qualitative figure
# ---------------------------------------------------------------------------
def per_query_ranks_kept(Q, gallery, img_ids, id2idx, anns, topk=5):
    """find_qualitative.per_query_ranks with the reference KEPT in the gallery: target ranks and top-k over all images
    (the reference can be one of them). The reference rank is unmasked under both rules."""
    import torch
    out = []
    with torch.no_grad():
        for q, a in enumerate(anns):
            p = Q[q].to(gallery.device)
            scores = (p @ gallery.T).squeeze(0)
            ref_idx = id2idx[str(a['reference_img_id'])]
            ref_rank = int((scores > scores[ref_idx]).sum().item()) + 1
            gt_idx = [id2idx[str(g)] for g in a['gt_img_ids'] if str(g) in id2idx]
            gt_ranks = [int((scores > scores[gi]).sum().item()) + 1 for gi in gt_idx]
            best_gt = min(gt_ranks) if gt_ranks else None
            top = scores.topk(topk).indices
            top_ids = [int(img_ids[i]) for i in top.cpu().tolist()]
            topk_sim_to_ref = float((gallery[top] @ gallery[ref_idx]).mean().item())
            out.append(dict(best_gt=best_gt, ref_rank=ref_rank, top_ids=top_ids,
                            gt_ids=[int(g) for g in a['gt_img_ids']], topk_sim_to_ref=topk_sim_to_ref))
    return out


def circo_val_anns(n):
    """The val annotations as stickiness_compare.load_circo_split returns them (what find_qualitative reads)."""
    from evaluation.stickiness_compare import load_circo_split
    _, anns = load_circo_split(R.CIRCO_DIR, (lambda x: x), 'val')
    return anns[:n]


def qual_select(rule, qb, qv, anns, out_json, device, ckpts):
    """find_qualitative.main() itself (its gallery load, per_query_ranks, filter, order and JSON), with queries_ours
    returning the given composed queries; under KEPT its per_query_ranks is per_query_ranks_kept."""
    from evaluation import find_qualitative as FQ
    feats = {str(ckpts['baseline']): qb, str(ckpts['vib']): qv}

    def queries(ckpt_path, circo_data_path, dev, split):
        if split != 'val':
            raise RuntimeError(f'unexpected split {split}')
        return feats[str(ckpt_path)].clone(), [dict(a) for a in anns], None, 'openai-ViT-L/14'

    saved = (FQ.queries_ours, FQ.per_query_ranks, sys.argv)
    FQ.queries_ours = queries
    if rule == 'kept':
        FQ.per_query_ranks = per_query_ranks_kept
    sys.argv = ['find_qualitative.py', '--baseline_ckpt', str(ckpts['baseline']), '--vib_ckpt', str(ckpts['vib']),
                '--circo_data_path', str(R.CIRCO_DIR), '--openai_gallery', str(R.GALLERY), '--device', device,
                '--base_gt_min', str(QUAL_ARGS['base_gt_min']), '--vib_gt_max', str(QUAL_ARGS['vib_gt_max']),
                '--ref_max', str(QUAL_ARGS['ref_max']), '--topk', str(QUAL_ARGS['topk']), '--show', str(QUAL_SHOW),
                '--out', str(out_json)]
    try:
        FQ.main()
    finally:
        FQ.queries_ours, FQ.per_query_ranks, sys.argv = saved
    return R.read_json(out_json)


def make_selection(rule, cands, n):
    qids = [c['query_id'] for c in cands]
    paper = QUAL_PAPER['pick']
    if paper in qids:
        pick, how = paper, (f"the paper's query {paper}: find_qualitative leaves the final choice to the author (its "
                            f'order puts {qids[0]} first)')
    elif qids:
        pick, how = qids[0], (f"the paper's query {paper} does not qualify under this rule: the first in "
                              "find_qualitative's order")
    else:
        pick, how = None, 'no query qualifies'
    entry = next((c for c in cands if c['query_id'] == pick), None)
    sel = {'rule': rule, 'rule_text': RULE_TEXT[rule], 'n_val_queries': n,
           'filter': (f'baseline best-GT rank > {QUAL_ARGS["base_gt_min"]}, VIB best-GT rank <= {QUAL_ARGS["vib_gt_max"]}, '
                      f'baseline reference rank (unmasked) <= {QUAL_ARGS["ref_max"]}; GT ranks and top-{QUAL_ARGS["topk"]} '
                      + ('with the reference removed' if rule == 'removed' else 'with the reference kept')),
           'order': "find_qualitative: CARD / SPAT captions first, then the largest baseline -> VIB rank gain, then the "
                    "baseline top-5's mean cosine to the reference",
           'n_qualifying': len(cands), 'qualifying': qids, 'first_in_order': qids[0] if qids else None,
           'paper_pick': paper, 'paper_pick_qualifies': paper in qids, 'pick': pick, 'pick_rule': how,
           'pick_entry': entry, 'candidates': cands}
    if entry is not None:
        ref = entry['reference_img_id']
        sel['reference_in_top5'] = {m: (entry[f'{m}_top5'].index(ref) + 1 if ref in entry[f'{m}_top5'] else None)
                                    for m in ('baseline', 'vib')}
    return sel


def qual_checks(ctx, sel):
    """The REMOVED selection against the paper and the current figure's candidate list."""
    if ctx.max_q is not None:
        return {'ok': None, 'reason': '--max_queries: the paper checks need every val query', 'checks': {}}
    e = sel['pick_entry'] or {}
    stored = R.read_json(QUAL_STORED) if QUAL_STORED.exists() else None
    checks = {'n_qualifying': {'ok': sel['n_qualifying'] == QUAL_PAPER['n_qualifying'], 'now': sel['n_qualifying'],
                               'paper': QUAL_PAPER['n_qualifying']},
              'paper_pick_qualifies': {'ok': sel['paper_pick_qualifies'], 'pick': sel['pick'], 'paper': QUAL_PAPER['pick']},
              'paper_caption_ranks': {'ok': (e.get('baseline_gt_rank'), e.get('vib_gt_rank'))
                                      == (QUAL_PAPER['baseline_gt_rank'], QUAL_PAPER['vib_gt_rank']),
                                      'now': [e.get('baseline_gt_rank'), e.get('vib_gt_rank')],
                                      'paper': [QUAL_PAPER['baseline_gt_rank'], QUAL_PAPER['vib_gt_rank']]}}
    if stored is None:
        checks['stored_candidates'] = {'ok': False, 'reason': f'{QUAL_STORED} missing'}
    else:
        by_q = {c['query_id']: c for c in stored}
        diff = {}
        for c in sel['candidates']:
            s = by_q.get(c['query_id'])
            fields = sorted(k for k in set(c) | set(s or {}) if s is None or c.get(k) != s.get(k))
            if fields:
                diff[str(c['query_id'])] = fields
        checks['stored_candidates'] = {
            'ok': stored == sel['candidates'], 'file': str(QUAL_STORED), 'n_stored': len(stored),
            'same_order': [c['query_id'] for c in stored] == sel['qualifying'],
            'only_now': sorted(set(sel['qualifying']) - set(by_q)),
            'only_stored': sorted(set(by_q) - set(sel['qualifying'])),
            'differing_fields': diff}
        s = by_q.get(sel['pick'], {})
        checks['figure_lists'] = {'ok': bool(e) and all(e.get(k) == s.get(k) for k in
                                                        ('baseline_top5', 'vib_top5', 'gt_ids', 'reference_img_id')),
                                  'query_id': sel['pick']}
    return {'ok': all(c['ok'] for c in checks.values()), 'checks': checks}


def qualitative_item(ctx):
    import numpy as np
    import torch
    qdir = ctx.out / 'qualitative'
    v1, ckpts, deps = {}, {}, {}
    for dl in QUAL_MODELS:
        e = ctx.entry_for(dl)
        rec = ensure_ours(ctx, e)
        if rec.get('status') != 'done':
            raise RuntimeError(f'the ours item of {e["label"]} ({dl}) did not finish')
        v1[dl] = torch.from_numpy(np.load(ctx.out / 'ours' / e['label'] / 'circo_val_V1.npy', allow_pickle=False))
        ckpts[dl] = ctx.bundle_dir / e['file']
        deps[dl] = {'label': e['label'], 'features_sha256': rec['outputs'][f'ours/{e["label"]}/circo_val_V1.npy']}
    anns = circo_val_anns(ctx.n_val)
    if [a['query_id'] for a in anns] != R.read_json(ctx.out / 'ours' / deps['vib']['label'] / 'meta.json')['query_ids']:
        raise RuntimeError('the val annotations and the ours features are not in the same order')
    sel, outputs = {}, {}
    for rule in RULES:
        R.log(f'\n  find_qualitative.py, reference {rule}:')
        cand_path = qdir / f'candidates_{rule}.json'
        cands = qual_select(rule, v1['baseline'], v1['vib'], anns, cand_path, ctx.args.device, ckpts)
        sel[rule] = make_selection(rule, cands, len(anns))
        R.write_json(qdir / f'selection_{rule}.json', sel[rule])
        for p in (cand_path, qdir / f'selection_{rule}.json'):
            outputs[ctx.rel(p)] = R.sha256_file(p)
    sc = qual_checks(ctx, sel['removed'])
    for rule in RULES:
        s, e = sel[rule], sel[rule]['pick_entry'] or {}
        R.log(f'  {rule:8s}: {s["n_qualifying"]} qualifying val queries {s["qualifying"]}; pick {s["pick"]} '
              f'(baseline rank {e.get("baseline_gt_rank")} -> VIB rank {e.get("vib_gt_rank")}; {s["pick_rule"]})'
              + (f'; reference in the top 5 at {s["reference_in_top5"]}' if rule == 'kept' and e else ''))
    R.log('  self-check : ' + (', '.join(f'{k} {"ok" if v["ok"] else "MISMATCH"}' for k, v in sc['checks'].items())
                               or sc.get('reason', '-')))
    return {'selection': {r: {k: sel[r][k] for k in ('n_qualifying', 'qualifying', 'first_in_order', 'pick', 'pick_rule',
                                                       'paper_pick_qualifies')}
                          | {'pick_entry': sel[r]['pick_entry']} for r in RULES},
            'inputs': deps, 'selfcheck': sc, 'outputs': outputs}


def qualitative_figures_item(ctx):
    """render_qualitative.main() for each rule's pick, after fetching the gallery images it shows."""
    import matplotlib.pyplot as plt
    from evaluation import render_qualitative as RQ
    qdir = ctx.out / 'qualitative'
    sel = {r: R.read_json(qdir / f'selection_{r}.json') for r in RULES}
    ids = set()
    for s in sel.values():
        if s['pick_entry']:
            e = s['pick_entry']
            ids |= {int(e['reference_img_id'])} | {int(x) for x in e['baseline_top5'] + e['vib_top5']}
    by_id = {int(i['id']): i for i in R.read_json(coco_info_path())['images']}
    unknown = sorted(i for i in ids if i not in by_id)
    if unknown:
        raise RuntimeError(f'{len(unknown)} figure images are not in the COCO image list: {unknown[:5]}')
    fetch = X.xn_e4_fetch([by_id[i] for i in sorted(ids)], types.SimpleNamespace(download_workers=4, no_download=False),
                          qdir)
    R.log(f'  images     : {fetch["needed"]} needed, {fetch["present_before"]} present, {fetch["downloaded"]} downloaded')
    if fetch['missing'] or fetch['failed']:
        raise RuntimeError(f'figure images could not be fetched: {fetch["missing"] or fetch["failed"]}')
    outputs, figs = {}, {}
    for rule in RULES:
        if sel[rule]['pick'] is None:
            R.log(f'  {rule}: no query qualifies, no figure')
            continue
        dest = qdir / f'fig_{rule}'
        saved = sys.argv
        sys.argv = ['render_qualitative.py', '--query_id', str(sel[rule]['pick']), '--candidates',
                    str(qdir / f'candidates_{rule}.json'), '--circo_data_path', str(R.CIRCO_DIR), '--out', str(dest)]
        try:
            RQ.main()
        finally:
            sys.argv = saved
            plt.close('all')
        for ext in ('.pdf', '.png'):
            p = dest.with_suffix(ext)
            outputs[ctx.rel(p)] = R.sha256_file(p)
        figs[rule] = {'query_id': sel[rule]['pick'], 'files': [dest.with_suffix(x).name for x in ('.pdf', '.png')]}
    ledger = qdir / X.E4_LEDGER
    if ledger.exists():
        outputs[ctx.rel(ledger)] = R.sha256_file(ledger)
    return {'figures': figs, 'fetch': {k: v for k, v in fetch.items() if k != 'downloaded_files'},
            'downloaded_files': fetch.get('downloaded_files', []), 'outputs': outputs}


def qualitative_step(ctx):
    for dl in QUAL_MODELS:                       # the inputs first, so that the selection's parameters can name them
        try:
            ensure_ours(ctx, ctx.entry_for(dl))
        except RuntimeError as ex:               # a missing dump label: the qualitative item fails below with it
            R.log(f'  {ex}')
    rec = run_item(ctx, 'qualitative', "find_qualitative.py's selection under both rules (paper baseline and VIB)",
                   lambda: qualitative_item(ctx), params={'max_queries': ctx.max_q, 'filter': QUAL_ARGS,
                                                          'paper': QUAL_PAPER, 'inputs': qual_inputs(ctx)})

    def figures():
        if rec.get('status') != 'done':
            raise RuntimeError('the qualitative selection did not finish')
        return qualitative_figures_item(ctx)
    run_item(ctx, 'qualitative_figures', "render_qualitative.py for each rule's pick", figures,
             params={'selection': rec.get('outputs')})


def qual_inputs(ctx):
    """The sha256 of the two feature files the selection reads (a changed input measures the selection again)."""
    out = {}
    for dl in QUAL_MODELS:
        try:
            e = ctx.entry_for(dl)
        except RuntimeError:
            continue
        rec = ctx.recs.get(f'ours__{e["label"]}') or {}
        out[dl] = (rec.get('outputs') or {}).get(f'ours/{e["label"]}/circo_val_V1.npy')
    return out


# ---------------------------------------------------------------------------
# Step 5: external methods
# ---------------------------------------------------------------------------
def selfcheck_state(v):
    """True / False / None from the self-check an external method reports (bool, string, or a dict of checks)."""
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ('ok', 'pass', 'passed', 'exact', 'true', 'yes')
    if isinstance(v, dict):
        for key in ('ok', 'exact', 'passed', 'pass', 'status'):
            if key in v:
                return selfcheck_state(v[key])
        states = [selfcheck_state(x) for x in v.values() if isinstance(x, (bool, dict))]
        return False if False in states else (None if not states or None in states else True)
    return None


def find_number(d, include, exclude=(), path='', depth=0):
    """First numeric leaf of a nested dict whose key path contains every string of include and none of exclude."""
    if not isinstance(d, dict) or depth > 3:
        return None
    for k, v in d.items():
        p = f'{path}/{str(k).lower()}'
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if all(s in p for s in include) and not any(s in p for s in exclude):
                return v
        elif isinstance(v, dict):
            hit = find_number(v, include, exclude, p, depth + 1)
            if hit is not None:
                return hit
    return None


def ext_summary(name, d):
    """One method's record of run_externals: status, val mAP@10 under both rules (in % as the module reports them),
    Ref@10 over the val queries and the self-check (selfcheck_ok: True / False / None for a partial run)."""
    status = str(d.get('status', 'unknown'))
    sc_key = next((k for k in EXT_SELFCHECK_KEYS if k in d), None)

    def num(key, include, exclude=()):
        v = d.get(key)
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else find_number(d, include, exclude)
    return {'method': name, 'status': status, 'done': status.strip().lower() in EXT_DONE,
            'map10_removed': num('map10_removed', ('map', '10', 'remov'), ('kept',)),
            'map10_kept': num('map10_kept', ('map', '10', 'kept')),
            'ref10': num('ref10_val', ('ref', '10'), ('map',)), 'n_val': d.get('n_val'),
            'selfcheck': selfcheck_state(d.get(sc_key)) if sc_key else None, 'selfcheck_key': sc_key,
            'mismatches': d.get('mismatches'), 'error': d.get('error')}


def fmt_ext(s):
    f = lambda x: '-' if x is None else (f'{x:.4f}' if isinstance(x, float) else str(x))   # noqa: E731
    ref = f(s['ref10']) + (f'/{s["n_val"]}' if s.get('n_val') is not None and s['ref10'] is not None else '')
    return (f'{s["method"]:12s} {s["status"]:7s} mAP@10 removed {f(s["map10_removed"]):>8s}  kept '
            f'{f(s["map10_kept"]):>8s}  Ref@10 val {ref:>8s}  self-check {_state(s["selfcheck"])}')


def externals_item(ctx):
    mod = importlib.import_module('evaluation.ref_kept_externals')
    odir = ctx.out / 'externals'
    res = mod.run_externals(out_dir=odir, circo_dir=R.CIRCO_DIR, assets_dir=ctx.args.assets_dir, device=ctx.args.device,
                            log=R.log, resume=ctx.resume, max_queries=ctx.max_q)
    methods = res.get('methods') if isinstance(res, dict) and isinstance(res.get('methods'), dict) else res
    if not isinstance(methods, dict):
        raise RuntimeError(f'run_externals returned {type(res).__name__}, expected a dict per method')
    summ = {str(k): ext_summary(str(k), v) for k, v in methods.items() if isinstance(v, dict)}
    if not summ:
        raise RuntimeError('run_externals returned no method')
    for s in summ.values():
        R.log(f'  {fmt_ext(s)}')
        for mm in s.get('mismatches') or []:
            R.log(f'  {"":12s} mismatch: {mm}')
    states = [s['selfcheck'] for s in summ.values() if s['done']]
    files = sorted(p for p in odir.rglob('*') if p.is_file() and not p.name.endswith('.tmp')) if odir.exists() else []
    return {'methods': summ, 'returned': jsonable(res),
            'selfcheck': {'ok': False if False in states else (True if states and all(states) else None),
                          'checks': {k: {'ok': s['selfcheck']} for k, s in summ.items()}},
            'files': {ctx.rel(p): R.sha256_file(p) for p in files}}


def _state(x):
    return {True: 'ok', False: 'MISMATCH', None: 'not checked'}[x]


# ---------------------------------------------------------------------------
# Step 6: summary
# ---------------------------------------------------------------------------
def duplicate_check(ctx):
    """Byte-identical checkpoint files (each run's final.pt) must give identical features."""
    groups = {}
    for e in ctx.entries:
        r = ctx.recs.get(f'ours__{e["label"]}')
        if r and r.get('status') == 'done':
            groups.setdefault(e['sha256'], []).append((e['label'], r['outputs'][f'ours/{e["label"]}/circo_val_V1.npy']))
    rows = [{'labels': [l for l, _ in g], 'identical_features': len({s for _, s in g}) == 1}
            for g in groups.values() if len(g) > 1]
    return {'ok': all(r['identical_features'] for r in rows) if rows else None, 'groups': rows}


def item_row(rec):
    n = rec.get('numbers') or {}
    return {'status': rec.get('status'), 'map10_removed': (n.get('removed') or {}).get('map', {}).get('10'),
            'map10_kept': (n.get('kept') or {}).get('map', {}).get('10'), 'ref10': n.get('ref10'),
            'n_queries': n.get('n_queries'), 'selfcheck': rec.get('selfcheck'), 'error': rec.get('error')}


def log_tables(ctx, summ):
    pc = lambda x, d=4: '-' if x is None else f'{100 * x:.{d}f}'     # noqa: E731
    R.banner('Results: CIRCO val mAP@10 (percent) with the reference removed / kept, Ref@10 = queries whose reference '
             'ranks in the top 10')
    runs = {}
    for e in ctx.entries:
        if f'ours__{e["label"]}' in ctx.recs:
            runs.setdefault(e['run'], []).append(e)
    for run, es in runs.items():
        R.log(f'\n  {run}  ({es[0].get("group")}, {len(es)} checkpoints)')
        R.log(f'  {"label":36s} {"removed":>8s} {"kept":>8s} {"change":>8s} {"Ref@10":>8s} {"paper model":>15s}  '
              'self-check')
        for e in es:
            r = item_row(ctx.recs[f'ours__{e["label"]}'])
            if r['status'] != 'done':
                R.log(f'  {e["label"]:36s} FAILED')
                continue
            ch = '-' if r['map10_kept'] is None else f'{100 * (r["map10_kept"] - r["map10_removed"]):+.4f}'
            ref10 = f'{r["ref10"]}/{r["n_queries"]}'
            R.log(f'  {e["label"]:36s} {pc(r["map10_removed"]):>8s} {pc(r["map10_kept"]):>8s} {ch:>8s} {ref10:>8s} '
                  f'{e.get("dump_label") or "":>15s}  {fmt_selfcheck(r["selfcheck"])}')
    if summ.get('e6'):
        R.log('\n  E6 noise curve: mAP@10 removed / kept (mean over 8 samples, percent), Ref@10 (mean count)')
        R.log(f'  {"model":18s} ' + ' '.join(f'{"a=" + str(a):>20s}' for a in E6_ALPHAS))
        for name, r in summ['e6'].items():
            if r['status'] != 'done':
                R.log(f'  {name:18s} FAILED')
                continue
            R.log(f'  {name:18s} ' + ' '.join(f'{pc(c["map10_removed_mean"], 2):>6s}/{pc(c["map10_kept_mean"], 2):>6s}/'
                                              f'{c["ref10_mean"]:6.1f}' for c in r['curve'])
                  + f'   self-check {_state(r["selfcheck"]["ok"]) if r["selfcheck"] else "-"}')
    q = summ.get('qualitative')
    if q and q.get('status') == 'done':
        R.log('\n  Qualitative figure (find_qualitative.py rule)')
        for rule, s in q['selection'].items():
            e = s['pick_entry'] or {}
            R.log(f'  {rule:8s}: {s["n_qualifying"]:3d} qualifying, pick {s["pick"]} (baseline rank '
                  f'{e.get("baseline_gt_rank")} -> VIB rank {e.get("vib_gt_rank")})')
    if summ.get('externals'):
        R.log('\n  External methods (as ref_kept_externals.run_externals reports them; mAP@10 in its units, %)')
        for s in summ['externals'].values():
            R.log(f'  {fmt_ext(s)}')


def summarize(ctx, t_start, versions):
    R.banner('Step 6/6  Summary')
    recs = ctx.recs
    failed = [k for k, r in recs.items() if r.get('status') != 'done']
    mismatched = [k for k, r in recs.items() if r.get('status') == 'done' and (r.get('selfcheck') or {}).get('ok') is False]
    ext = (recs.get('externals') or {}).get('methods') or {}
    for name, s in ext.items():
        if not s['done']:
            failed.append(f'externals:{name}')
        elif s['selfcheck'] is False:
            mismatched.append(f'externals:{name}')
    mismatched = [m for m in mismatched if m != 'externals']
    dup = duplicate_check(ctx)
    if dup['ok'] is False:
        mismatched.append('ours_duplicate_files')
    code = 3 if failed else 2 if mismatched else 0
    summ = {
        'finished_at': R.now(), 'exit_code': code, 'minutes': round((time.time() - t_start) / 60, 1),
        'argv': sys.argv, 'code_commit': R.git_commit(), 'versions': versions, 'host': X.host_fingerprint(),
        'test_mode_max_queries': ctx.max_q, 'items_selected': list(ctx.items), 'labels': ctx.args.labels,
        'bundle': str(ctx.bundle_dir), 'rules': RULE_TEXT, 'failed': failed, 'mismatched': mismatched,
        'ours': {k[len('ours__'):]: item_row(r) for k, r in recs.items() if k.startswith('ours__')},
        'ours_duplicate_files': dup,
        'e6': {k[len('e6__'):]: {'status': r.get('status'), 'curve': r.get('curve'), 'selfcheck': r.get('selfcheck'),
                                 'error': r.get('error')} for k, r in recs.items() if k.startswith('e6__')},
        'e6_shared': item_row(recs['e6_shared']) if 'e6_shared' in recs else None,
        'qualitative': ({'status': recs['qualitative'].get('status'), 'selection': recs['qualitative'].get('selection'),
                         'selfcheck': recs['qualitative'].get('selfcheck'), 'error': recs['qualitative'].get('error')}
                        if 'qualitative' in recs else None),
        'qualitative_figures': ({k: recs['qualitative_figures'].get(k) for k in ('status', 'figures', 'error')}
                                if 'qualitative_figures' in recs else None),
        'externals': ext, 'externals_status': (recs.get('externals') or {}).get('status'),
        'entries': recs}
    R.write_json(ctx.out / 'summary.json', summ)
    man = X.write_manifest(ctx.out)
    try:
        log_tables(ctx, summ)
    except Exception:                                         # noqa: BLE001  (the files above are complete)
        R.log(traceback.format_exc())
        R.log('  the result tables could not be printed (traceback above); summary.json is complete')
    R.log(f'\n  summary.json and MANIFEST.sha256 ({man["n_files"]} files, {man["bytes"] / 1e6:.1f} MB) written')
    if ctx.max_q is not None:
        R.log(f'  TEST MODE (--max_queries {ctx.max_q}): none of these numbers is a paper number')
    verdict_text = {0: 'DONE: every item finished and every self-check is exact',
                    2: f'DONE, SELF-CHECK MISMATCH: {mismatched}',
                    3: f'ITEMS FAILED: {failed}' + (f'; self-check mismatch: {mismatched}' if mismatched else '')}[code]
    R.banner(f'{verdict_text}   {(time.time() - t_start) / 60:.1f} min   {R.now()}')
    R.log(f'  results: {ctx.out}  (summary.json, entries/, ours/, e6/, qualitative/, externals/)')
    if code:
        R.log('  rerun the same command to retry failed items; finished ones are skipped')
    return code


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--bundle_dir', type=Path, default=None,
                   help='refkept_bundle/ of paperspace_refkept_bundle.zip (default /notebooks/experiment/refkept_bundle)')
    p.add_argument('--assets_dir', type=Path, default=None,
                   help="the external methods' inputs, passed to ref_kept_externals (default "
                        '/notebooks/experiment/refkept_assets)')
    p.add_argument('--out', type=Path, default=None, help='output folder (default <repo>/results/ref_kept_2026_11)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--only', action='append', choices=ITEMS,
                   help=f'run only this item (repeatable; any of {", ".join(ITEMS)}); default: all, in this order')
    p.add_argument('--labels', default=None, help='comma-separated subset of the bundle labels for the ours item')
    p.add_argument('--preflight_only', action='store_true', help='check everything, evaluate nothing')
    p.add_argument('--no_resume', action='store_true', help='measure every item again')
    p.add_argument('--max_queries', type=int, default=None,
                   help='TESTS ONLY: keep the first N queries of each CIRCO split (nothing written is a paper number)')
    args = p.parse_args(argv)
    args.bundle_dir = Path(args.bundle_dir or R.EXPERIMENT / 'refkept_bundle')
    args.assets_dir = Path(args.assets_dir or R.EXPERIMENT / 'refkept_assets')
    args.out = Path(args.out or REPO_ROOT / 'results' / 'ref_kept_2026_11')
    args.labels = [x.strip() for x in args.labels.split(',') if x.strip()] if args.labels else None
    if args.max_queries is not None and args.max_queries < 1:
        p.error('--max_queries must be at least 1')
    return args


def main(argv=None):
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    log, restore = start_log(args.out)
    code = 3
    try:
        code = _main(args)
    except Exception:                                         # noqa: BLE001  (outside the items: a bug of this file)
        R.log(traceback.format_exc())
        R.log('UNEXPECTED ERROR outside the items (traceback above)')
        code = 3
    finally:
        log.raw(f'EXIT CODE {code}\n')
        restore()
    sys.exit(code)


def _main(args):
    ctx = Ctx(args)
    t_start = time.time()
    R.banner(f'Reference-kept CIRCO run (eval only, nothing trained)   {R.now()}')
    R.log(f'  code   : {R.REPO_ROOT}  (commit {R.git_commit()})')
    R.log(f'  bundle : {ctx.bundle_dir}')
    R.log(f'  assets : {args.assets_dir}')
    R.log(f'  output : {ctx.out}')
    R.log(f'  items  : {", ".join(ctx.items)}' + (f'; ours labels: {len(args.labels)} given' if args.labels else '')
          + ('' if ctx.resume else '; --no_resume'))
    if ctx.max_q is not None:
        R.log(f'  --max_queries {ctx.max_q}: TEST MODE, nothing written is a paper number')

    R.banner('Step 1/6  Preflight')
    problems, warnings, report, versions = preflight(ctx)
    log_preflight(ctx, report, versions, warnings)
    info = dict(R.environment(args), versions=versions, host=X.host_fingerprint())
    R.write_json(ctx.out / 'preflight.json', {'checked_at': R.now(), 'environment': info, 'problems': problems,
                                              'warnings': warnings, 'report': report})
    history = R.read_json(ctx.out / 'run_info.json') if (ctx.out / 'run_info.json').exists() else []
    R.write_json(ctx.out / 'run_info.json', history + [info])
    if problems:
        R.log('\nPREFLIGHT FAILED (nothing was run):')
        for pr in problems:
            R.log(f'  - {pr}')
        return 1
    R.log(f'\n  Preflight passed: {len(needed_entries(ctx))} checkpoints to load; torch {versions.get("torch")}, '
          f'GPU {versions.get("gpu")}')
    log_estimate(ctx)
    if args.preflight_only:
        return 0

    R.banner(f'Step 2/6  ours: {len(ctx.ours_entries())} checkpoints, {ctx.n_val} CIRCO val queries each, one at a time'
             if 'ours' in ctx.items else 'Step 2/6  ours: skipped (--only)')
    if 'ours' in ctx.items:
        ours_step(ctx)
    R.banner(f'Step 3/6  E6 noise curve: a in {list(E6_ALPHAS)}, {F.S1_SAMPLES} samples, {", ".join(E6_MODELS)}'
             if 'e6' in ctx.items else 'Step 3/6  E6: skipped (--only)')
    if 'e6' in ctx.items:
        e6_step(ctx)
    R.banner("Step 4/6  Qualitative figure: find_qualitative.py's rule under both rules, render_qualitative.py"
             if 'qualitative' in ctx.items else 'Step 4/6  qualitative: skipped (--only)')
    if 'qualitative' in ctx.items:
        qualitative_step(ctx)
    R.banner('Step 5/6  External methods: evaluation/ref_kept_externals.py'
             if 'externals' in ctx.items else 'Step 5/6  externals: skipped (--only)')
    if 'externals' in ctx.items:
        F._CACHE.pop('gallery', None)                         # the externals load their own gallery
        ctx.cache.clear()
        F.free_memory()
        run_item(ctx, 'externals', 'ref_kept_externals.run_externals', lambda: externals_item(ctx),
                 params={'max_queries': ctx.max_q}, resume=False)      # the module resumes its own methods
    try:
        return summarize(ctx, t_start, versions)
    except Exception:                                         # noqa: BLE001
        R.log(traceback.format_exc())
        R.banner('SUMMARY FAILED: summary.json or MANIFEST.sha256 could not be written (traceback above); every item '
                 'record is in entries/. Rerun the same command (finished items are skipped).')
        return 3


if __name__ == '__main__':
    main()

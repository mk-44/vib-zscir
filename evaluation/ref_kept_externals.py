"""External methods for the reference-kept CIRCO run: the composed queries of SEARLE-XL, Pic2Word and LinCIR.

Why
    Every CIRCO number of the paper removes the reference image from the ranked list; the published CIRCO code of
    SEARLE and LinCIR keeps it. The same-protocol comparison with the released external mappers
    (metrics/stickiness_{searle,pic2word,lincir}.json, June 2026: CIRCO-val mAP@10 13.39 / 9.41 / 13.97 with the
    reference removed) is therefore also reported with it kept. The final scoring under both rules happens on the Mac
    with evaluation/circo_ref_protocol.py, from the query features saved here.

What it runs
    For each method, the query builder of evaluation/stickiness_compare.py that made the June numbers, imported and
    called unchanged (queries_searle, queries_pic2word, queries_lincir), with the June settings: split 'val+test' (the
    220 val then the 800 test queries, each in annotation order), template argument 'a photo of $ that {caption}'
    (queries_pic2word ignores it and uses Pic2Word's fixed 'a photo of * , {caption}'; queries_lincir hard-codes
    'a photo of $ that {caption}'), and the inputs pinned and verified by evaluation/prepare_ref_kept.py (repos,
    checkpoints, the torch.hub copy of SEARLE, the Hugging Face cache, loaded offline). The queries are L2-normalized as
    stickiness_compare.main does (F.normalize(Q.float(), dim=-1)), and exactly that tensor is saved and scored.

Gallery
    All three methods are scored against the OpenAI ViT-L/14 gallery cache (eval_data/circo/gallery_cache_ViT-L-14.pt),
    loaded with stickiness_compare.load_gallery. For LinCIR this is stickiness_compare's --reuse_openai_gallery:
    LinCIR's HF openai/clip-vit-large-patch14 has the OpenAI weights (queries_lincir's docstring), and no HF gallery
    cache was ever built: build_circo_gallery.py has no HF option, no backup holds one, and on 2026-07-02, a week after
    the June runs, eval_data/circo held only gallery_cache_ViT-L-14.pt and gallery_cache_ViT-H-14.pt (keep_rest.zip
    file_inventory.txt and layout.txt). Without the flag the June LinCIR run would have stopped at torch.load of the
    missing HF cache, so it used this gallery. Its 'space' "hf-ViT-L/14" is the builder's tag, set whichever gallery
    is used. The self-check below tests all of this.

On-machine summary and self-check (one query at a time, scores = p @ gallery.T, as stickiness_compare)
    REMOVED rule (the June runs): stickiness_compare.compute_stickiness and CIRCO's official compute_metrics on the
    new features, compared with the June JSON:
        mAP@10 over the 220 val queries, rounded to 2 dp            equal
        Ref@10 count over the 1,020 queries (June r10_ref x 1020)   equal
        median_rank_ref                                             equal
        mean_rank_ref, sim_ref_mean, sim_gt_mean                    within 1e-6 relative
    sim_delta, best_gt_rank_median, p_gtset_mean, n_queries, n_gt and space are compared for information. A mismatch is
    reported, never fatal. Both rules: val mAP@{5,10,25,50} with the reference removed (its score set to -inf before
    topk(50)) and kept (topk(50) over the whole gallery), CIRCO's AP@K; Ref@10 = queries whose reference ranks in the
    top 10 (rank = 1 + gallery scores strictly above it) over val (220), test (800) and all. Internal checks: the
    removed-rule lists equal compute_stickiness's and this file's mAP equals compute_metrics'.

Outputs (<out_dir>/externals/<method>/; with max_queries N, <method>__first<N>/, so full outputs are never replaced)
    circo_V1_all.npy   (1020, 768) float32: the scored query features; rows = 220 val then 800 test queries
    meta.json          builder, template, embedding space, gallery (file, sha256, normalization, why), tau, assets
                       (pinned commits and revisions, checksums), versions, numerics settings, query ids per row, the
                       summary, per-query reference ranks and AP@K under both rules, the self-check
    Resume: a method whose circo_V1_all.npy exists and matches the sha256 in its meta.json (same configuration) is not
    encoded again; it is rescored from the saved features (seconds).

Interface (the reference-kept driver imports this module)
    preflight(circo_dir, assets_dir) -> list[str]
        Problems; empty means ready. Each starts with 'searle:', 'pic2word:', 'lincir:' or 'externals:' (shared
        inputs). preflight_report(...) also returns warnings (e.g. a package version other than the pin).
    run_externals(out_dir, circo_dir, assets_dir, device, log, resume=True, methods=METHODS, max_queries=None) -> dict
        out_dir is the run's results folder (files go to out_dir/externals/...; an out_dir named 'externals' is used
        as is). Returns {method: record} with: status ('done' | 'failed'), resumed, map10_removed, map10_kept (val
        mAP@10 in %, unrounded), map_removed, map_kept ({K: %}), ref10_val, ref10_test, ref10_all (counts), n_val,
        n_test, selfcheck_ok (True / False; None for a partial run), selfcheck (the comparisons), mismatches (str),
        consistency_ok, features, features_sha256, meta (paths), seconds, error (traceback or None).

Usage (standalone; repo at /notebooks/experiment/codes, assets prepared with evaluation/prepare_ref_kept.py)
    PYTHONPATH=/notebooks/experiment/codes python -u evaluation/ref_kept_externals.py --preflight_only
    PYTHONPATH=/notebooks/experiment/codes python -u evaluation/ref_kept_externals.py
    Exit codes: 0 all done and every self-check exact, 1 preflight failed, 2 a self-check mismatched, 3 a method
    failed. The last line printed is 'EXIT CODE n'.
"""

import argparse
import contextlib
import gc
import json
import os
import platform
import sys
import time
import traceback
import urllib.error
import warnings
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Third-party checkpoints are full pickles; keep torch.load's pre-2.6 default (no effect on torch 2.1.1).
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

from evaluation import prepare_ref_kept as P   # noqa: E402  pins, asset layout and verification (stdlib only)

METHODS = ('searle', 'pic2word', 'lincir')
SPLIT = 'val+test'
TEMPLATE = 'a photo of $ that {caption}'         # args.template of the June runs (stickiness_*.json "template")
N_QUERIES = {'val': 220, 'test': 800}
GALLERY_FILE = 'gallery_cache_ViT-L-14.pt'
KS = (5, 10, 25, 50)
TOP = 50
REL_TOL = 1e-6
FEATURES = 'circo_V1_all.npy'
META = 'meta.json'
JUNE_JSON = {m: REPO_ROOT / 'metrics' / f'stickiness_{m}.json' for m in METHODS}
DEFAULT_OUT = REPO_ROOT / 'results' / 'ref_kept_2026_11'
DEFAULT_CIRCO = Path('/notebooks/experiment/eval_data/circo')
ROW_DEFINITION = ('F.normalize(Q.float(), dim=-1) of the builder output Q, as stickiness_compare.main does before '
                  'compute_stickiness: exactly the tensor scored here')
GALLERY_LOAD = 'stickiness_compare.load_gallery: F.normalize(cache["embeddings"].float(), dim=-1), on the device'
SPECS = {
    'searle': {
        'name': 'SEARLE-XL', 'builder': 'queries_searle', 'space': 'openai-ViT-L/14',
        'template_used': TEMPLATE,
        'how': "torch.hub.load('miccunifi/SEARLE', 'searle', source='github', backbone='ViT-L/14') (phi) with OpenAI "
               "CLIP ViT-L/14 from clip.load, in fp32; the pinned SEARLE copy is the torch.hub cache under "
               "<assets>/torch_hub",
        'gallery_note': 'OpenAI CLIP ViT-L/14 queries; the OpenAI ViT-L/14 gallery, as in June'},
    'pic2word': {
        'name': 'Pic2Word', 'builder': 'queries_pic2word', 'space': 'openai-ViT-L/14',
        'template_used': 'a photo of * , {caption}',
        'how': "google-research/composed_image_retrieval: model/clip.py load('ViT-L/14', jit=False) and IM2TEXT, both "
               "state dicts from pic2word_model.pt; the template is fixed inside queries_pic2word (the template "
               "argument is ignored)",
        'gallery_note': 'OpenAI CLIP ViT-L/14 queries; the OpenAI ViT-L/14 gallery, as in June'},
    'lincir': {
        'name': 'LinCIR', 'builder': 'queries_lincir', 'space': 'hf-ViT-L/14',
        'template_used': TEMPLATE,
        'how': "navervision/lincir build_text_encoder('large'): HF openai/clip-vit-large-patch14 (vision + text, fp32) "
               "and the SDXL tokenizer_2 with '[$]' (49408 -> 259); Phi from lincir_large.pt; the template is "
               "hard-coded inside queries_lincir",
        'gallery_note': ("Scored against the OpenAI ViT-L/14 gallery cache, i.e. stickiness_compare "
                         "--reuse_openai_gallery: LinCIR's HF openai/clip-vit-large-patch14 has the same weights, and "
                         "no HF gallery cache ever existed (build_circo_gallery.py has no HF option; eval_data/circo "
                         "held only the ViT-L-14 and ViT-H-14 caches on 2026-07-02, keep_rest.zip "
                         "file_inventory.txt), so the June run must have used this gallery. The 'hf-ViT-L/14' space "
                         "in stickiness_lincir.json is the builder's tag, set whichever gallery is used.")},
}


# ---------------------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------------------
def _now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')


def _stickiness():
    from evaluation import stickiness_compare as S
    return S


def _method_dir(ext_root, method, max_queries):
    return ext_root / (method if max_queries is None else f'{method}__first{max_queries}')


def _config(method, max_queries):
    return {'method': method, 'builder': f'stickiness_compare.{SPECS[method]["builder"]}', 'split': SPLIT,
            'template_argument': TEMPLATE, 'max_queries': max_queries, 'rows': ROW_DEFINITION,
            'pins_resolved': P.RESOLVED}


def _save_npy(path, arr):
    import numpy as np
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        np.save(f, a, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return {'file': path.name, 'shape': list(a.shape), 'dtype': str(a.dtype), 'bytes': path.stat().st_size,
            'sha256': P.sha256_file(path)}


def _versions():
    import importlib.metadata as md
    v = {'python': platform.python_version()}
    for dist in ('torch', 'torchvision', 'numpy', 'Pillow', 'transformers', 'huggingface-hub', 'tokenizers',
                 'safetensors', 'gdown', 'setuptools'):
        try:
            v[dist] = md.version(dist)
        except Exception:                                     # noqa: BLE001
            v[dist] = None
    try:
        d = json.loads(md.distribution('clip').read_text('direct_url.json') or '{}')
        v['clip'] = (d.get('vcs_info') or {}).get('commit_id') or d.get('url')
    except Exception:                                         # noqa: BLE001
        v['clip'] = None
    try:
        import torch
        v.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None)
    except Exception as e:                                    # noqa: BLE001
        v['torch_error'] = str(e)
    return v


def _version_warnings(v, methods):
    pins = {n: ver for n, ver, _ in P.PIP_PINS}
    need = ['transformers', 'huggingface-hub', 'tokenizers', 'safetensors'] if 'lincir' in methods else []
    return [f'{n} is {v.get(n)}, pinned {pins[n]} (the version of the June runs)' for n in need if v.get(n) != pins[n]]


@contextlib.contextmanager
def _env(values):
    """Set (value str) or unset (value None) environment variables for the duration, then restore them.
    huggingface_hub / transformers read them once, at import."""
    old = {k: os.environ.get(k) for k in values}
    for k, val in values.items():
        if val is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = val
    try:
        yield
    finally:
        for k, val in old.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val


@contextlib.contextmanager
def _hub_dir(path):
    """queries_searle's unpinned torch.hub.load('miccunifi/SEARLE', ...) finds the pinned copy under path."""
    import torch
    old = getattr(torch.hub, '_hub_dir', None)
    torch.hub.set_dir(str(path))
    try:
        yield
    finally:
        torch.hub._hub_dir = old


@contextlib.contextmanager
def _library_numerics():
    """PyTorch's defaults, as in the June standalone runs of stickiness_compare.py; the caller's are restored after."""
    import torch
    old = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
           torch.backends.cudnn.allow_tf32, torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    try:
        yield {'float32_matmul_precision': 'highest', 'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': True,
               'cudnn_deterministic': False, 'cudnn_benchmark': False}
    finally:
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.deterministic,
         torch.backends.cudnn.benchmark) = old[1:]
        torch.set_float32_matmul_precision(old[0])       # last: setting matmul.allow_tf32 also sets the precision


@contextlib.contextmanager
def _first_queries(S, n):
    """max_queries: the builders see only the first n queries of each split (load_circo_split wrapped, then restored)."""
    if n is None:
        yield
        return
    orig = S.load_circo_split

    def limited(circo_data_path, preprocess, split='val'):
        ds, anns = orig(circo_data_path, preprocess, split)
        seen, keep = {}, []
        for a in anns:
            seen[a['split']] = seen.get(a['split'], 0) + 1
            if seen[a['split']] <= n:
                keep.append(a)
        return ds, keep

    S.load_circo_split = limited
    try:
        yield
    finally:
        S.load_circo_split = orig


def _free():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                         # noqa: BLE001
        pass


def _annotations(circo_dir):
    return {s: json.loads((Path(circo_dir) / 'annotations' / f'{s}.json').read_text()) for s in N_QUERIES}


def _check_rows(Q, anns, ann_files, max_queries):
    """The builder's rows must be the val then the test queries, each in annotation order, finite, one per query."""
    import torch
    want = [(s, a) for s in ('val', 'test') for a in ann_files[s][:max_queries]]
    if len(anns) != len(want):
        raise RuntimeError(f'builder returned {len(anns)} queries, expected {len(want)}')
    for i, (a, (split, ref)) in enumerate(zip(anns, want)):
        if (a['split'] != split or int(a['raw_id']) != int(ref['id'])
                or int(a['reference_img_id']) != int(ref['reference_img_id'])
                or a['relative_caption'] != ref['relative_caption']):
            raise RuntimeError(f'row {i}: builder query {a.get("query_id")} is not {split}_{ref["id"]} of the '
                               f'annotation files')
    if Q.ndim != 2 or Q.shape[0] != len(want):
        raise RuntimeError(f'builder returned features of shape {tuple(Q.shape)} for {len(want)} queries')
    if not torch.isfinite(Q).all():
        raise RuntimeError('builder returned non-finite features')


def _gallery(S, circo_dir, device, cache):
    """The OpenAI ViT-L/14 gallery cache, loaded once per run with stickiness_compare.load_gallery."""
    if 'gallery' not in cache:
        import numpy as np
        path = Path(circo_dir) / GALLERY_FILE
        emb, img_ids, id2idx = S.load_gallery(path, device)
        cache['gallery'] = {'emb': emb, 'img_ids': img_ids, 'id2idx': id2idx,
                            'ids_np': np.asarray([int(x) for x in img_ids], dtype=np.int64),
                            'record': {'file': str(path), 'bytes': path.stat().st_size,
                                       'sha256': P.sha256_file(path), 'n_images': int(emb.shape[0]),
                                       'space': 'openai-ViT-L/14', 'loaded_with': GALLERY_LOAD}}
    return cache['gallery']


def _ap_at_k(pred_ids, gt_ids):
    """CIRCO evaluation.py: AP@K over the ground-truth set, normalized by min(|GT|, K)."""
    import numpy as np
    labels = np.isin(pred_ids, np.asarray(gt_ids, dtype=np.int64))
    precisions = np.cumsum(labels, axis=0) * labels / np.arange(1, labels.shape[0] + 1)
    return {k: float(np.sum(precisions[:k]) / min(len(gt_ids), k)) for k in KS}


def _rank_both(Qn, anns, g):
    """Per query: the reference's rank, the top-50 with the reference removed and kept, AP@K of both (val)."""
    import torch
    emb, id2idx, ids_np = g['emb'], g['id2idx'], g['ids_np']
    out = {'query_id': [], 'rank_ref': [], 'ref_in_kept_top10': [], 'removed_top50': {},
           'ap_removed': {f'@{k}': [] for k in KS}, 'ap_kept': {f'@{k}': [] for k in KS}}
    with torch.no_grad():
        for q, a in enumerate(anns):
            p = Qn[q].to(emb.device)
            scores = (p @ emb.T).squeeze(0)                    # the expression of compute_stickiness
            ref_idx = id2idx[str(a['reference_img_id'])]
            out['query_id'].append(str(a['query_id']))
            out['rank_ref'].append(int((scores > scores[ref_idx]).sum().item()) + 1)
            kept = scores.topk(TOP).indices.cpu().numpy()
            masked = scores.clone()
            masked[ref_idx] = float('-inf')
            removed = masked.topk(TOP).indices.cpu().numpy()
            out['ref_in_kept_top10'].append(bool(int(ref_idx) in kept[:10].tolist()))
            out['removed_top50'][str(a['query_id'])] = [int(x) for x in ids_np[removed]]
            if a['split'] == 'val':
                for rule, idx in (('removed', removed), ('kept', kept)):
                    ap = _ap_at_k(ids_np[idx], a['gt_img_ids'])
                    for k in KS:
                        out[f'ap_{rule}'][f'@{k}'].append(ap[k])
    return out


def _june_stats(S, Qn, anns, tau, g, circo_dir):
    """The June computation: compute_stickiness, then CIRCO's compute_metrics on the val lists (stickiness_compare.main)."""
    s, preds = S.compute_stickiness(Qn, g['emb'], g['img_ids'], g['id2idx'], anns, tau=tau)
    val_preds = {str(a['raw_id']): preds[str(a['query_id'])] for a in anns if a['split'] == 'val'}
    if val_preds:
        compute_metrics = S.load_official_compute_metrics(Path(circo_dir))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)   # empty semantic aspects in a partial run
            map_atk, _, _ = compute_metrics(Path(circo_dir), val_preds, list(KS))
        s['map@10'] = round(float(map_atk[10]) * 100, 2)
        s['map_official'] = {f'@{k}': float(map_atk[k]) * 100 for k in KS}
        s['map_n'] = len(val_preds)
    return s, preds


def _summary(ranks, anns):
    import numpy as np
    split = [a['split'] for a in anns]
    rr = ranks['rank_ref']
    val = [r for r, s in zip(rr, split) if s == 'val']
    test = [r for r, s in zip(rr, split) if s == 'test']
    m_removed = {k: 100 * float(np.mean(v)) for k, v in ranks['ap_removed'].items()} if val else {}
    m_kept = {k: 100 * float(np.mean(v)) for k, v in ranks['ap_kept'].items()} if val else {}
    return {'n_val': len(val), 'n_test': len(test), 'n_all': len(rr),
            'map_removed': m_removed, 'map_kept': m_kept,
            'map10_removed': m_removed.get('@10'), 'map10_kept': m_kept.get('@10'),
            'ref10_val': sum(r <= 10 for r in val), 'ref10_test': sum(r <= 10 for r in test),
            'ref10_all': sum(r <= 10 for r in rr),
            'ref_in_kept_top10_val': sum(t for t, s in zip(ranks['ref_in_kept_top10'], split) if s == 'val'),
            'median_rank_ref_all': int(np.median(rr)), 'mean_rank_ref_all': float(np.mean(rr))}


def _consistency(s, preds, ranks, summ):
    import numpy as np
    rr = ranks['rank_ref']
    differ = sum(preds.get(q) != lst for q, lst in ranks['removed_top50'].items())
    checks = [
        {'name': 'removed-rule top-50 lists equal compute_stickiness predictions', 'differ': differ, 'ok': differ == 0},
        {'name': 'Ref@10 fraction equals compute_stickiness r10_ref',
         'ok': sum(r <= 10 for r in rr) / len(rr) == s['r10_ref']},
        {'name': 'median / mean reference rank equal compute_stickiness',
         'ok': int(np.median(rr)) == s['median_rank_ref'] and float(np.mean(rr)) == s['mean_rank_ref']}]
    if 'map_official' in s:
        diff = max(abs(summ['map_removed'][k] - s['map_official'][k]) for k in s['map_official'])
        checks.append({'name': 'removed-rule mAP@K equals CIRCO compute_metrics', 'max_abs_diff': diff,
                       'ok': diff <= 1e-9})
    return {'ok': all(c['ok'] for c in checks), 'checks': checks}


def _selfcheck(method, s, summ, partial, space):
    """REMOVED-rule numbers vs the June JSON of this method (space: the builder's tag)."""
    path = Path(JUNE_JSON[method])
    out = {'reference': str(path), 'rule': 'reference removed (as the June runs)', 'checks': [], 'info': []}
    if partial:
        out.update(ok=None, skipped='partial run (max_queries): the June values cover all 1,020 queries')
        return out
    try:
        june = json.loads(path.read_text())['rows'][method]
    except Exception as e:                                    # noqa: BLE001
        out.update(ok=False, error=f'cannot read the June reference {path}: {e}')
        return out

    def exact(bucket, name, ref, got):
        out[bucket].append({'name': name, 'june': ref, 'now': got, 'rule': 'equal',
                            'ok': ref is not None and ref == got})

    def rel(bucket, name, ref, got):
        ok = ref is not None and got is not None
        d = abs(got - ref) / max(abs(ref), 1e-300) if ok else None
        out[bucket].append({'name': name, 'june': ref, 'now': got, 'rule': f'rel <= {REL_TOL:g}', 'rel_diff': d,
                            'ok': ok and d <= REL_TOL})

    jr = june.get('r10_ref')
    jcount = None
    if jr is not None:
        jcount = round(jr * 1020)
        if abs(jr * 1020 - jcount) > 1e-6:
            out['note'] = f'June r10_ref x 1020 = {jr * 1020!r} is not an integer'
    exact('checks', 'mAP@10 val, 2 dp', june.get('map@10'), s.get('map@10'))
    exact('checks', 'Ref@10 count over 1020', jcount, summ['ref10_all'])
    exact('checks', 'median_rank_ref', june.get('median_rank_ref'), s.get('median_rank_ref'))
    rel('checks', 'mean_rank_ref', june.get('mean_rank_ref'), s.get('mean_rank_ref'))
    rel('checks', 'sim_ref_mean', june.get('sim_ref_mean'), s.get('sim_ref_mean'))
    rel('checks', 'sim_gt_mean', june.get('sim_gt_mean'), s.get('sim_gt_mean'))
    rel('info', 'sim_delta', june.get('sim_delta'), s.get('sim_delta'))
    exact('info', 'best_gt_rank_median', june.get('best_gt_rank_median'), s.get('best_gt_rank_median'))
    if 'p_gtset_mean' in june or 'p_gtset_mean' in s:
        rel('info', 'p_gtset_mean', june.get('p_gtset_mean'), s.get('p_gtset_mean'))
    exact('info', 'n_queries', june.get('n_queries'), s.get('n_queries'))
    exact('info', 'n_gt', june.get('n_gt'), s.get('n_gt'))
    exact('info', 'space', june.get('space'), space)
    out['ok'] = all(c['ok'] for c in out['checks'])
    return out


def _fmt(v):
    if isinstance(v, float):
        return f'{v:.10g}'
    return str(v)


def _log_selfcheck(log, sc, cons):
    log(f'  self-check vs {sc["reference"]} ({sc["rule"]}):')
    if sc.get('skipped'):
        log(f'    skipped: {sc["skipped"]}')
    if sc.get('error'):
        log(f'    MISMATCH: {sc["error"]}')
    for bucket in ('checks', 'info'):
        for c in sc[bucket]:
            tag = ('ok  ' if c['ok'] else 'FAIL') if bucket == 'checks' else ('info' if c['ok'] else 'info*')
            extra = f'  (rel diff {c["rel_diff"]:.1e})' if c.get('rel_diff') is not None else ''
            log(f'    {tag:5s} {c["name"]:24s} June {_fmt(c["june"]):>16s}   now {_fmt(c["now"]):>16s}{extra}')
    n = len(sc['checks'])
    good = sum(c['ok'] for c in sc['checks'])
    verdict = 'skipped' if sc['ok'] is None else ('EXACT' if sc['ok'] else 'MISMATCH')
    log(f'  self-check: {verdict} ({good}/{n} required comparisons ok); internal consistency: '
        f'{"ok" if cons["ok"] else "FAILED " + str([c["name"] for c in cons["checks"] if not c["ok"]])}')
    if sc.get('note'):
        log(f'  note: {sc["note"]}')


# ---------------------------------------------------------------------------------------------------------------------
# One method
# ---------------------------------------------------------------------------------------------------------------------
def _build(S, method, circo_dir, assets_dir, device):
    a = Path(assets_dir)
    if method == 'searle':
        return S.queries_searle(circo_dir, device, TEMPLATE, SPLIT)
    if method == 'pic2word':
        return S.queries_pic2word(a / P.REPOS['pic2word']['dir'], a / P.PIC2WORD_CKPT, circo_dir, device, TEMPLATE,
                                  SPLIT)
    if method == 'lincir':
        return S.queries_lincir(a / P.REPOS['lincir']['dir'], circo_dir, device, TEMPLATE, SPLIT)
    raise ValueError(f'unknown method {method}')


def _encode(S, method, circo_dir, assets_dir, device, log):
    """The builder, retried on network errors for SEARLE (torch.hub asks github.com for the default branch)."""
    attempts = 3 if method == 'searle' else 1
    for i in range(1, attempts + 1):
        try:
            return _build(S, method, circo_dir, assets_dir, device)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if i == attempts:
                raise
            log(f'  network error ({type(e).__name__}: {e}); retry {i}/{attempts - 1} in 20 s')
            time.sleep(20)


def _run_method(S, method, mdir, circo_dir, assets_dir, device, log, resume, max_queries, cache, numerics):
    import numpy as np
    import torch
    import torch.nn.functional as F
    t0 = time.time()
    config = _config(method, max_queries)
    ann_files = _annotations(circo_dir)
    feat_path, meta_path = mdir / FEATURES, mdir / META
    meta, why = None, 'resume disabled'
    if resume:
        meta, why = _resumable(mdir, config)
    resumed = meta is not None
    if resumed:
        log(f'  resume: {feat_path} matches the sha256 in meta.json; not encoding again (rescoring)')
        Qn = torch.from_numpy(np.load(feat_path, allow_pickle=False))
        with _first_queries(S, max_queries):
            _, anns = S.load_circo_split(Path(circo_dir), (lambda x: x), SPLIT)
        _check_rows(Qn, anns, ann_files, max_queries)
        if [str(a['query_id']) for a in anns] != meta['rows']['query_ids']:
            raise RuntimeError(f'{meta_path}: rows do not match the annotation files')
        tau = meta.get('tau')
    else:
        if resume:
            log(f'  encoding ({why})')
        records, problems = P.verify_assets(assets_dir, methods=(method,))
        for p in problems[method]:
            log(f'  WARNING asset: {p}')
        log(f'  builder stickiness_compare.{SPECS[method]["builder"]}(split={SPLIT!r}, template={TEMPLATE!r}) '
            f'on {device}' + (f', first {max_queries} queries of each split' if max_queries else ''))
        te = time.time()
        with _first_queries(S, max_queries):
            Q, anns, tau, space = _encode(S, method, Path(circo_dir), assets_dir, device, log)
        seconds_features = time.time() - te
        Q = Q.detach().cpu()
        _check_rows(Q, anns, ann_files, max_queries)
        if space != SPECS[method]['space']:
            log(f'  WARNING: builder space tag {space!r}, expected {SPECS[method]["space"]!r}')
        Qn = F.normalize(Q.float(), dim=-1)                   # stickiness_compare.main, before compute_stickiness
        mdir.mkdir(parents=True, exist_ok=True)
        feats = _save_npy(feat_path, Qn.numpy())
        meta = {
            'method': method, 'name': SPECS[method]['name'], 'status': 'features_saved', 'config': config,
            'features': feats, 'row_definition': ROW_DEFINITION,
            'rows': {'order': '220 CIRCO val queries then 800 test queries, each in annotation order '
                              '(stickiness_compare.load_circo_split, split val+test)',
                     'n_val': sum(a['split'] == 'val' for a in anns),
                     'n_test': sum(a['split'] == 'test' for a in anns),
                     'query_ids': [str(a['query_id']) for a in anns],
                     'reference_img_ids': [int(a['reference_img_id']) for a in anns]},
            'builder': {'function': f'stickiness_compare.{SPECS[method]["builder"]}', 'how': SPECS[method]['how'],
                        'template_argument': TEMPLATE, 'template_used': SPECS[method]['template_used'],
                        'split': SPLIT, 'max_queries': max_queries},
            'embedding_space': {'queries': space, 'gallery': 'openai-ViT-L/14'},
            'tau': tau, 'tau_note': 'logit_scale.exp() of the CLIP model the builder used; only p_gtset_mean uses it',
            'max_abs_change_from_normalize': float((Qn - Q.float()).abs().max()),
            'assets': {'records': records[method], 'problems': problems[method]},
            'versions': _versions(), 'numerics': numerics, 'device': device,
            'seconds_features': round(seconds_features, 1), 'features_saved_at': _now()}
        P.write_json(meta_path, meta)
        log(f'  features {tuple(Qn.shape)} float32 -> {feat_path}  (sha256 {feats["sha256"][:16]}..., '
            f'{seconds_features:.1f} s)')

    # ---- scoring (always from the saved tensor) ---------------------------------------------------------------------
    ts = time.time()
    g = _gallery(S, circo_dir, device, cache)
    ranks = _rank_both(Qn, anns, g)
    summ = _summary(ranks, anns)
    s, preds = _june_stats(S, Qn, anns, tau, g, circo_dir)
    cons = _consistency(s, preds, ranks, summ)
    partial = max_queries is not None
    sc = _selfcheck(method, s, summ, partial, meta['embedding_space']['queries'])
    mismatches = [f'{c["name"]}: June {_fmt(c["june"])}, now {_fmt(c["now"])}' for c in sc['checks'] if not c['ok']]
    if sc.get('error'):
        mismatches.append(sc['error'])
    if not cons['ok']:
        mismatches += [f'internal: {c["name"]}' for c in cons['checks'] if not c['ok']]
    stats = {k: v for k, v in s.items() if k != 'map_official'}
    meta.update(status='done', gallery=dict(g['record'], note=SPECS[method]['gallery_note']),
                summary=summ, june_rule_stats=stats, map_official_removed=s.get('map_official'),
                selfcheck=sc, consistency=cons, scored_at=_now(), seconds_scoring=round(time.time() - ts, 1),
                per_query={'query_id': ranks['query_id'], 'rank_ref': ranks['rank_ref'],
                           'ref_in_kept_top10': ranks['ref_in_kept_top10'],
                           'ap_removed': ranks['ap_removed'], 'ap_kept': ranks['ap_kept']})
    P.write_json(meta_path, meta)

    log(f'  scored against {GALLERY_FILE} ({g["record"]["n_images"]:,} images, {GALLERY_LOAD.split(":")[0]})')
    if summ['n_val']:
        mr, mk = summ['map_removed'], summ['map_kept']
        log(f'  REMOVED  val mAP@10 {mr["@10"]:6.2f}  ({mr["@10"]:.4f};  @5 {mr["@5"]:.2f}  @25 {mr["@25"]:.2f}  '
            f'@50 {mr["@50"]:.2f})')
        log(f'  KEPT     val mAP@10 {mk["@10"]:6.2f}  ({mk["@10"]:.4f};  @5 {mk["@5"]:.2f}  @25 {mk["@25"]:.2f}  '
            f'@50 {mk["@50"]:.2f})')
    log(f'  Ref@10   val {summ["ref10_val"]}/{summ["n_val"]}   test {summ["ref10_test"]}/{summ["n_test"]}   '
        f'all {summ["ref10_all"]}/{summ["n_all"]}   (reference in the kept top 10 on val: '
        f'{summ["ref_in_kept_top10_val"]})')
    _log_selfcheck(log, sc, cons)
    selfcheck_ok = None if sc['ok'] is None else bool(sc['ok'] and cons['ok'])
    return {'method': method, 'status': 'done', 'resumed': resumed,
            'map10_removed': summ['map10_removed'], 'map10_kept': summ['map10_kept'],
            'map_removed': summ['map_removed'], 'map_kept': summ['map_kept'],
            'ref10_val': summ['ref10_val'], 'ref10_test': summ['ref10_test'], 'ref10_all': summ['ref10_all'],
            'n_val': summ['n_val'], 'n_test': summ['n_test'], 'selfcheck_ok': selfcheck_ok, 'selfcheck': sc,
            'consistency_ok': cons['ok'], 'mismatches': mismatches, 'features': str(feat_path),
            'features_sha256': meta['features']['sha256'], 'meta': str(meta_path),
            'seconds': round(time.time() - t0, 1), 'error': None}


def _resumable(mdir, config):
    meta_p, feat_p = mdir / META, mdir / FEATURES
    if not (meta_p.is_file() and feat_p.is_file()):
        return None, 'no saved features'
    try:
        meta = json.loads(meta_p.read_text())
    except Exception as e:                                    # noqa: BLE001
        return None, f'unreadable {meta_p}: {e}'
    if meta.get('config') != config:
        return None, 'the saved features were made with another configuration'
    if meta.get('status') not in ('features_saved', 'done'):
        return None, f'saved status {meta.get("status")!r}'
    if P.sha256_file(feat_p) != (meta.get('features') or {}).get('sha256'):
        return None, f'{feat_p.name} does not match the sha256 in meta.json'
    return meta, 'ok'


# ---------------------------------------------------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------------------------------------------------
def preflight_report(circo_dir, assets_dir, methods=METHODS):
    """(problems, warnings) for the given methods. Sets the Hugging Face cache variables of this process (they must be
    in place before anything imports huggingface_hub)."""
    import importlib.util
    circo_dir, assets_dir = Path(circo_dir), Path(assets_dir)
    problems, warns = [], []
    if 'lincir' in methods:
        problems += [f'lincir: {p}' for p in P.apply_hf_env(assets_dir / P.HF_HOME, offline=True)]

    # ---- shared inputs --------------------------------------------------------------------------------------------
    need = [circo_dir / 'annotations' / 'val.json', circo_dir / 'annotations' / 'test.json', circo_dir / GALLERY_FILE,
            circo_dir / 'circo_repo' / 'src' / 'dataset.py', circo_dir / 'circo_repo' / 'src' / 'evaluation.py',
            circo_dir / 'COCO2017_unlabeled' / 'annotations' / 'image_info_unlabeled2017.json']
    problems += [f'externals: missing {p}' for p in need if not p.exists()]
    if all(p.exists() for p in need):
        try:
            anns = _annotations(circo_dir)
            for s, n in N_QUERIES.items():
                if len(anns[s]) != n:
                    problems.append(f'externals: CIRCO {s} has {len(anns[s])} queries, expected {n}')
            info = json.loads(need[-1].read_text())
            names = {i['id']: i['file_name'] for i in info['images']}
            images = circo_dir / 'COCO2017_unlabeled' / 'unlabeled2017'
            missing = sum(names.get(q['reference_img_id']) is None
                          or not (images / names[q['reference_img_id']]).exists()
                          for s in N_QUERIES for q in anns[s])
            if missing:
                problems.append(f'externals: {missing} CIRCO queries lack their reference image in {images}')
            import torch
            cache = torch.load(circo_dir / GALLERY_FILE, map_location='cpu')
            ids = set(cache['img_id_to_idx'])
            want = ({str(q['reference_img_id']) for s in N_QUERIES for q in anns[s]}
                    | {str(g) for q in anns['val'] for g in q['gt_img_ids']})
            if cache.get('clip_model') != 'ViT-L/14' or want - ids:
                problems.append(f'externals: {GALLERY_FILE} ({cache.get("clip_model")}) lacks {len(want - ids)} '
                                f'reference / ground-truth ids')
            del cache
        except Exception as e:                                # noqa: BLE001
            problems.append(f'externals: cannot check the CIRCO inputs: {type(e).__name__}: {e}')
    try:
        S = _stickiness()
        for fn in ('queries_searle', 'queries_pic2word', 'queries_lincir', 'load_gallery', 'load_circo_split',
                   'compute_stickiness', 'load_official_compute_metrics'):
            if not callable(getattr(S, fn, None)):
                problems.append(f'externals: stickiness_compare.{fn} not found')
    except Exception as e:                                    # noqa: BLE001
        problems.append(f'externals: evaluation/stickiness_compare.py does not import: {type(e).__name__}: {e}')
    try:
        import clip
        if 'ViT-L/14' not in clip.available_models():
            problems.append('externals: the clip package does not offer ViT-L/14')
    except Exception as e:                                    # noqa: BLE001
        problems.append(f'externals: OpenAI CLIP is not importable ({e})')
    cw = Path(os.path.expanduser(P.CLIP_WEIGHTS['path']))
    if not cw.is_file() or cw.stat().st_size != P.CLIP_WEIGHTS['bytes']:
        warns.append(f'{cw} missing or incomplete: clip.load will download it (932 MB, needs network)')

    # ---- per method -------------------------------------------------------------------------------------------------
    for m in methods:
        try:
            row = json.loads(Path(JUNE_JSON[m]).read_text())['rows'][m]
            lack = [k for k in ('map@10', 'r10_ref', 'median_rank_ref', 'mean_rank_ref', 'sim_ref_mean', 'sim_gt_mean')
                    if k not in row]
            if lack:
                problems.append(f'{m}: June reference {JUNE_JSON[m]} lacks {lack}')
        except Exception as e:                                # noqa: BLE001
            problems.append(f'{m}: June reference {JUNE_JSON[m]} unreadable: {e}')
    if 'lincir' in methods:
        for mod in ('transformers', 'huggingface_hub', 'tokenizers', 'safetensors'):
            if importlib.util.find_spec(mod) is None:
                problems.append(f'lincir: Python package {mod} is not installed: {P.PIP_LINE}')
    if 'pic2word' in methods:
        pk = P.pkg_resources_packaging_problem()
        if pk:
            problems.append(f'pic2word: {pk}')
    warns += _version_warnings(_versions(), methods)
    _, per_method = P.verify_assets(assets_dir, methods=tuple(methods))
    for m in methods:
        problems += [p if p.startswith(m + ':') else f'{m}: {p}' for p in per_method[m]]
    return problems, warns


def preflight(circo_dir, assets_dir):
    """Problems that would stop the external methods; empty means ready."""
    return preflight_report(circo_dir, assets_dir)[0]


def run_externals(out_dir, circo_dir, assets_dir, device, log, resume=True, methods=METHODS, max_queries=None):
    """Composed queries of each method, saved, scored under both rules and self-checked; see the module docstring."""
    out_dir, circo_dir, assets_dir = Path(out_dir), Path(circo_dir), Path(assets_dir)
    ext_root = out_dir if out_dir.name == 'externals' else out_dir / 'externals'
    if max_queries is not None and int(max_queries) < 1:
        raise ValueError('max_queries must be a positive integer or None')
    max_queries = None if max_queries is None else int(max_queries)
    log(f'External methods (SEARLE-XL, Pic2Word, LinCIR): composed CIRCO queries   [{_now()}]')
    log(f'  output {ext_root}   assets {assets_dir}   circo {circo_dir}   device {device}   resume {resume}'
        + (f'   max_queries {max_queries}' if max_queries else ''))
    results, cache = {}, {}
    hf = dict(P.hf_env(assets_dir / P.HF_HOME, offline=True))
    hf.update({k: None for k in ('HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE', 'PYTORCH_TRANSFORMERS_CACHE',
                                 'PYTORCH_PRETRAINED_BERT_CACHE')})
    with _env(hf):
        for p in P.apply_hf_env(assets_dir / P.HF_HOME, offline=True):
            log(f'  WARNING: {p}')
        for w in _version_warnings(_versions(), methods):
            log(f'  WARNING: {w}')
        try:
            S = _stickiness()                                 # imports torch, numpy, PIL: fail here, not per method
        except Exception:                                     # noqa: BLE001
            tb = traceback.format_exc()
            log(tb)
            return {m: {'method': m, 'status': 'failed', 'error': tb, 'selfcheck_ok': None} for m in methods}
        with _hub_dir(assets_dir / P.TORCH_HUB), _library_numerics() as numerics:
            for m in methods:
                t0 = time.time()
                log('')
                log(f'--- externals/{m}: {SPECS.get(m, {}).get("name", m)}   [{_now()}]')
                mdir = _method_dir(ext_root, m, max_queries)
                try:
                    if m not in SPECS:
                        raise ValueError(f'unknown method {m!r}; choose from {METHODS}')
                    rec = _run_method(S, m, mdir, circo_dir, assets_dir, device, log, resume, max_queries, cache,
                                      numerics)
                except Exception:                             # noqa: BLE001
                    tb = traceback.format_exc()
                    log(tb.rstrip())
                    log(f'  FAILED: {m} (continuing with the others; rerun to retry it)')
                    rec = {'method': m, 'status': 'failed', 'resumed': False, 'selfcheck_ok': None,
                           'map10_removed': None, 'map10_kept': None, 'ref10_val': None, 'ref10_test': None,
                           'ref10_all': None, 'mismatches': [], 'error': tb, 'seconds': round(time.time() - t0, 1),
                           'features': str(mdir / FEATURES), 'meta': str(mdir / META)}
                    meta_p = mdir / META
                    try:                                      # keep saved features; note the failure otherwise
                        old = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
                        if old.get('status') not in ('features_saved', 'done'):
                            P.write_json(meta_p, {'method': m, 'status': 'failed', 'error': tb, 'failed_at': _now(),
                                                  'config': _config(m, max_queries) if m in SPECS else None})
                    except Exception:                         # noqa: BLE001
                        pass
                finally:
                    _free()
                results[m] = rec
                verdict = {True: 'self-check exact', False: 'SELF-CHECK MISMATCH', None: 'self-check skipped'}
                log(f'  => {m}: {rec["status"].upper()}'
                    + (f', {verdict[rec["selfcheck_ok"]]}' if rec['status'] == 'done' else '')
                    + f'  ({rec.get("seconds", "-")} s)')
        cache.clear()
        _free()

    log('')
    log(f'External methods, CIRCO val ({"first " + str(max_queries) + " queries per split" if max_queries else "220 queries"})')
    log(f'  {"method":9s} {"status":7s} {"mAP@10 removed":>15s} {"mAP@10 kept":>12s} {"Ref@10 val":>11s} '
        f'{"Ref@10 all":>11s}  self-check')
    def num(v, w):
        return f'{v:{w}.2f}' if isinstance(v, float) else f'{"-":>{w}s}'

    for m, r in results.items():
        if r['status'] == 'done':
            sc = {True: 'exact', False: 'MISMATCH', None: 'skipped'}[r['selfcheck_ok']]
            log(f'  {m:9s} {"done":7s} {num(r["map10_removed"], 15)} {num(r["map10_kept"], 12)} '
                f'{str(r["ref10_val"]) + "/" + str(r["n_val"]):>11s} '
                f'{str(r["ref10_all"]) + "/" + str(r["n_val"] + r["n_test"]):>11s}  {sc}')
            for mm in r['mismatches']:
                log(f'  {"":9s}   mismatch: {mm}')
        else:
            log(f'  {m:9s} {"FAILED":7s} {"-":>15s} {"-":>12s} {"-":>11s} {"-":>11s}  -')
    return results


# ---------------------------------------------------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Composed CIRCO queries of SEARLE-XL, Pic2Word and LinCIR, '
                                             'scored with the reference removed and kept.')
    ap.add_argument('--out_dir', type=Path, default=DEFAULT_OUT, help='results folder (writes <out_dir>/externals)')
    ap.add_argument('--circo_dir', type=Path, default=DEFAULT_CIRCO)
    ap.add_argument('--assets_dir', type=Path, default=P.DEFAULT_ASSETS)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    ap.add_argument('--max_queries', type=int, default=None, help='quick check: first N queries of each split')
    ap.add_argument('--no_resume', action='store_true', help='encode again even if valid features exist')
    ap.add_argument('--preflight_only', action='store_true')
    ap.add_argument('--june_dir', type=Path, default=None,
                    help='folder with stickiness_<method>.json (default: <repo>/metrics)')
    args = ap.parse_args()
    if args.june_dir is not None:
        JUNE_JSON.update({m: args.june_dir / f'stickiness_{m}.json' for m in METHODS})

    def log(msg=''):
        for line in str(msg).splitlines() or ['']:
            print(f'[{datetime.now(timezone.utc):%H:%M:%S}] {line}', flush=True)

    log(f'ref_kept_externals.py   {_now()}   methods {args.methods}')
    problems, warns = preflight_report(args.circo_dir, args.assets_dir, methods=tuple(args.methods))
    if args.device.startswith('cuda'):
        import torch
        if not torch.cuda.is_available():
            problems.append('externals: no CUDA GPU is visible to PyTorch')
    for w in warns:
        log(f'warning: {w}')
    if problems:
        log(f'PREFLIGHT FAILED ({len(problems)} problems):')
        for p in problems:
            log(f'  - {p}')
        log('EXIT CODE 1')
        sys.exit(1)
    log('preflight ok')
    if args.preflight_only:
        log('EXIT CODE 0')
        return
    res = run_externals(args.out_dir, args.circo_dir, args.assets_dir, args.device, log,
                        resume=not args.no_resume, methods=tuple(args.methods), max_queries=args.max_queries)
    ext_root = args.out_dir if args.out_dir.name == 'externals' else args.out_dir / 'externals'
    P.write_json(ext_root / ('summary_standalone.json' if args.max_queries is None
                             else f'summary_standalone__first{args.max_queries}.json'),
                 {'finished_at': _now(), 'methods': list(args.methods), 'max_queries': args.max_queries,
                  'results': res})
    failed = [m for m, r in res.items() if r['status'] != 'done']
    mismatched = [m for m, r in res.items() if r['status'] == 'done' and r['selfcheck_ok'] is False]
    code = 3 if failed else 2 if mismatched else 0
    if failed:
        log(f'failed: {failed}')
    if mismatched:
        log(f'self-check mismatched: {mismatched}')
    log(f'EXIT CODE {code}')
    sys.exit(code)


if __name__ == '__main__':
    main()

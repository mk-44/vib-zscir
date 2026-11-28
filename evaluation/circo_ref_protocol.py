#!/usr/bin/env python3
"""CIRCO scored twice: with the reference image removed from each ranked list, and with it kept.

Our CIRCO evaluation (eval_circo_v2.py on val, eval_circo_test.py on test) removes the reference image from the
gallery before it takes the top 50. CIRCO's evaluation.py scores each list as it is given, and the CIRCO README
points to SEARLE's generate_test_submission.py as the example submission code. That code takes the top 50 of the
whole gallery, so the reference stays in. Published CIRCO scores computed with it can therefore come from lists
that contain the reference. This script scores the same composed queries of every model under both rules.

It needs no GPU, no CLIP weights and no images. For each model, it reads the composed query features that
final_run.py saved: models/<model>/circo_V1.npy holds all 1,020 CIRCO queries, val then test, each encoded on its
own through the canonical encode_query path. It ranks the CIRCO gallery features (gallery_cache_ViT-L-14.pt) one
query at a time, as eval_circo_v2.py does. The metrics follow CIRCO's evaluation.py.

Usage (CIRCO val):
    python circo_ref_protocol.py val --dump <dump folder, or a .tar.gz that contains it> \
        --circo <eval_data/circo folder, or a .zip that contains eval_data/circo/> --out <new folder>
Archives are read in memory; nothing is extracted to disk.

Outputs in --out:
    summary.json   per model and rule: mAP@{5,10,25,50}, Recall@{5,10,25,50} and per-aspect mAP@10 (all in %);
                   how often the kept reference ranks in the top 1/5/10/50; the checks below; provenance.
    summary.tsv    the main numbers, one row per model.
    per_query/<model>.json
                   per query: the reference's rank in the full gallery, and AP@K and Recall@K under both rules.
    lists/<model>__reference_removed.json, lists/<model>__reference_kept.json
                   the top-50 lists in CIRCO submission format (query id -> 50 image ids).

Checks:
    - Every circo_V1.npy matches its sha256 in the dump's MANIFEST.sha256. The run stops if not.
    - The val annotations match the dump's query index (reference, target, ground truths, caption). The run stops
      if not.
    - With the reference removed, mAP@10, and the count of queries whose reference ranks in the top 10, equal the
      GPU values that final_run.py recorded in models/<model>/info.json. A mismatch is reported, and the script
      then exits with status 1.
"""
import argparse
import hashlib
import io
import json
import os
import platform
import re
import sys
import tarfile
import time
import zipfile
from datetime import datetime, timezone

import numpy as np
import torch

KS = (5, 10, 25, 50)
TOP = 50
RULES = ('reference_removed', 'reference_kept')
ASPECTS = ('cardinality', 'addition', 'negation', 'direct_addressing', 'compare_change', 'comparative_statement',
           'statement_with_conjunction', 'spatial_relations_background', 'viewpoint')
GALLERY = 'gallery_cache_ViT-L-14.pt'
N_QUERIES = 1020                       # CIRCO val (220) then test (800), as in the dump's shared/circo_index.json
DUMP_FILE = re.compile(r'(?:^|/)dump/(dump_info\.json|MANIFEST\.sha256|shared/circo_index\.json|'
                       r'models/[^/]+/(?:circo_V1\.npy|info\.json))$')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_dump(path):
    """Return {path relative to dump/: bytes} for the dump files this script uses, read in memory."""
    files = {}
    if os.path.isdir(path):
        for root, _, names in os.walk(path):
            for name in names:
                full = os.path.join(root, name)
                match = DUMP_FILE.search('dump/' + os.path.relpath(full, path).replace(os.sep, '/'))
                if match:
                    with open(full, 'rb') as f:
                        files[match.group(1)] = f.read()
    else:
        with tarfile.open(path, 'r|*') as tar:                      # one streaming pass
            for member in tar:
                match = DUMP_FILE.search(member.name)
                if match and member.isfile():
                    if match.group(1) in files:
                        sys.exit(f'{path}: more than one dump holds {match.group(1)}')
                    files[match.group(1)] = tar.extractfile(member).read()
    for need in ('dump_info.json', 'MANIFEST.sha256', 'shared/circo_index.json'):
        if need not in files:
            sys.exit(f'{path}: {need} not found; pass the final_run.py dump folder or an archive of it')
    return files


def read_circo(path, split):
    """Return (annotation bytes of the split, gallery cache bytes) from eval_data/circo or a .zip holding it."""
    wanted = (f'annotations/{split}.json', GALLERY)
    if os.path.isdir(path):
        out = []
        for rel in wanted:
            with open(os.path.join(path, rel), 'rb') as f:
                out.append(f.read())
        return tuple(out)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        out = []
        for rel in wanted:
            hits = [n for n in names if n == f'eval_data/circo/{rel}' or n.endswith(f'/eval_data/circo/{rel}')]
            if len(hits) != 1:
                sys.exit(f'{path}: expected one eval_data/circo/{rel}, found {len(hits)}')
            out.append(z.read(hits[0]))
        return tuple(out)


def load_gallery(data):
    cache = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    emb = cache['embeddings'].float().contiguous()                  # (N, 768), L2-normalized
    norms = emb.norm(dim=1)
    if float((norms - 1).abs().max()) > 1e-4:
        sys.exit('gallery embeddings are not L2-normalized; this script assumes the eval_circo cache')
    img_ids = np.asarray([int(i) for i in cache['img_ids']], dtype=np.int64)
    id_to_idx = {int(k): int(v) for k, v in cache['img_id_to_idx'].items()}
    return emb, img_ids, id_to_idx


def rank_query(query_feature, gallery, ref_idx):
    """Rank the gallery for one query: the reference's rank, then the top 50 with the reference kept and removed.

    Mirrors eval_circo_v2.py and eval_circo_test.py: scores = p @ gallery.T for a single query, the rank of the
    reference is 1 + the number of gallery images scored strictly above it, and the reference is removed by
    setting its score to -inf before topk(50).
    """
    scores = query_feature @ gallery.T
    ref_rank = int((scores > scores[ref_idx]).sum()) + 1
    kept = scores.topk(TOP).indices.numpy()
    scores[ref_idx] = float('-inf')
    removed = scores.topk(TOP).indices.numpy()
    return ref_rank, kept, removed


def circo_query_metrics(pred_ids, gt_ids, target_id):
    """CIRCO evaluation.py for one query: AP@K over the ground-truth set and Recall@K of the single target."""
    pred = np.asarray(pred_ids, dtype=np.int64)
    if len(set(pred.tolist())) != len(pred):
        raise ValueError('duplicate predictions')
    labels = np.isin(pred, np.asarray(gt_ids, dtype=np.int64))
    precisions = np.cumsum(labels, axis=0) * labels / np.arange(1, labels.shape[0] + 1)
    ap = {k: float(np.sum(precisions[:k]) / min(len(gt_ids), k)) for k in KS}
    hits = pred == target_id
    recall = {k: float(np.sum(hits[:k])) for k in KS}
    return ap, recall


def check_manifest(files, models):
    manifest = {}
    for line in files['MANIFEST.sha256'].decode().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            manifest[rel.strip().lstrip('*')] = digest
    for name in models:
        rel = f'models/{name}/circo_V1.npy'
        if rel not in files:
            sys.exit(f'{rel} is missing from the dump')
        if manifest.get(rel) != sha256(files[rel]):
            sys.exit(f'{rel} does not match MANIFEST.sha256')


def run_val(args):
    t0 = time.time()
    files = read_dump(args.dump)
    info = json.loads(files['dump_info.json'])
    all_models = list(info['models']['core']) + list(info['models']['extra'])
    models = [m for m in all_models if not args.models or m in args.models.split(',')]
    if args.models and len(models) != len(set(args.models.split(','))):
        sys.exit(f'unknown model in --models; the dump has: {", ".join(all_models)}')
    check_manifest(files, models)
    index = json.loads(files['shared/circo_index.json'])
    queries = index['queries']
    start, count = index['splits']['val']
    rows = list(range(start, start + count))
    if len(queries) != N_QUERIES or any(queries[r]['split'] != 'val' for r in rows):
        sys.exit('unexpected layout of shared/circo_index.json')

    ann_bytes, gallery_bytes = read_circo(args.circo, 'val')
    annotations = {int(a['id']): a for a in json.loads(ann_bytes)}
    for r in rows:
        q, a = queries[r], annotations.get(int(queries[r]['query_id']))
        if a is None or (int(a['reference_img_id']), int(a['target_img_id']), [int(g) for g in a['gt_img_ids']],
                         a['relative_caption']) != (int(q['ref_img_id']), int(q['target_img_id']),
                                                    [int(g) for g in q['gt_img_ids']], q['caption']):
            sys.exit(f'val query {q["query_id"]}: the annotations and the dump index disagree')
    if len(annotations) != count:
        sys.exit(f'val annotations hold {len(annotations)} queries, the dump index {count}')
    gallery_sha = sha256(gallery_bytes)
    gallery, img_ids, id_to_idx = load_gallery(gallery_bytes)
    del gallery_bytes
    print(f'loaded {len(models)} models, {count} val queries, gallery {tuple(gallery.shape)} '
          f'({time.time() - t0:.0f}s)', flush=True)

    out = args.out
    if os.path.isdir(out) and os.listdir(out) and not args.overwrite:
        sys.exit(f'{out} is not empty; choose a new folder or pass --overwrite')
    for sub in ('per_query', 'lists'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    summary_models, failures = {}, []
    for name in models:
        feats = np.load(io.BytesIO(files[f'models/{name}/circo_V1.npy']), allow_pickle=False)
        if feats.shape != (N_QUERIES, gallery.shape[1]):
            sys.exit(f'{name}: circo_V1.npy has shape {feats.shape}')
        feats = torch.from_numpy(feats).float()
        per_query = []
        lists = {rule: {} for rule in RULES}
        ap = {rule: {k: [] for k in KS} for rule in RULES}
        rec = {rule: {k: [] for k in KS} for rule in RULES}
        aspect_ap10 = {rule: {a: [] for a in ASPECTS} for rule in RULES}
        ref_ranks = []
        for r in rows:
            q = queries[r]
            a = annotations[int(q['query_id'])]
            ref_rank, kept, removed = rank_query(feats[r], gallery, id_to_idx[int(q['ref_img_id'])])
            ref_ranks.append(ref_rank)
            record = {'query_id': int(q['query_id']), 'reference_img_id': int(q['ref_img_id']),
                      'reference_rank': ref_rank}
            for rule, idx in (('reference_removed', removed), ('reference_kept', kept)):
                pred = img_ids[idx]
                lists[rule][str(q['query_id'])] = [int(i) for i in pred]
                q_ap, q_rec = circo_query_metrics(pred, a['gt_img_ids'], int(a['target_img_id']))
                for k in KS:
                    ap[rule][k].append(q_ap[k])
                    rec[rule][k].append(q_rec[k])
                for aspect in a['semantic_aspects']:
                    aspect_ap10[rule][aspect].append(q_ap[10])
                record[rule] = {**{f'AP@{k}': q_ap[k] for k in KS}, **{f'R@{k}': q_rec[k] for k in KS}}
            per_query.append(record)

        ranks = np.asarray(ref_ranks)
        entry = {'title': info['models']['titles'].get(name, name)}
        model_info = json.loads(files[f'models/{name}/info.json']) if f'models/{name}/info.json' in files else {}
        entry['checkpoint'] = model_info.get('checkpoint')
        entry['checkpoint_sha256'] = model_info.get('sha256')
        entry['circo_V1_sha256'] = sha256(files[f'models/{name}/circo_V1.npy'])
        for rule in RULES:
            entry[rule] = {**{f'mAP@{k}': 100 * float(np.mean(ap[rule][k])) for k in KS},
                           **{f'Recall@{k}': 100 * float(np.mean(rec[rule][k])) for k in KS},
                           'aspect_mAP@10': {x: 100 * float(np.mean(v)) for x, v in aspect_ap10[rule].items() if v}}
        entry['reference_in_top'] = {str(k): int((ranks <= k).sum()) for k in (1, 5, 10, 50)}
        entry['reference_rank_median'] = float(np.median(ranks))
        entry['n_queries'] = len(rows)

        gpu = model_info.get('selfcheck', {}).get('checks', {}).get('circo_val', {})
        cpu_map10, cpu_ref10 = float(np.mean(ap['reference_removed'][10])), int((ranks <= 10).sum())
        if 'map10_now' in gpu:
            exact = abs(cpu_map10 - gpu['map10_now']) < 1e-12 and cpu_ref10 == gpu.get('ref10_now')
            entry['check_against_gpu'] = {'gpu_mAP@10_removed': 100 * gpu['map10_now'],
                                          'cpu_mAP@10_removed': 100 * cpu_map10,
                                          'gpu_reference_in_top10': gpu.get('ref10_now'),
                                          'cpu_reference_in_top10': cpu_ref10, 'exact': exact}
            if not exact:
                failures.append(name)
        else:
            entry['check_against_gpu'] = None
        summary_models[name] = entry

        with open(os.path.join(out, 'per_query', f'{name}.json'), 'w') as f:
            json.dump(per_query, f, indent=1)
        for rule in RULES:
            with open(os.path.join(out, 'lists', f'{name}__{rule}.json'), 'w') as f:
                json.dump(lists[rule], f)
        check = entry['check_against_gpu']
        print(f'{name:16} mAP@10 removed {entry["reference_removed"]["mAP@10"]:6.2f}   kept '
              f'{entry["reference_kept"]["mAP@10"]:6.2f}   reference in top-10 {cpu_ref10:3d}/{len(rows)}   '
              f'GPU check: {"exact" if check and check["exact"] else ("MISMATCH" if check else "none")}', flush=True)

    summary = {
        'split': 'val',
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'command': ' '.join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
        'versions': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                     'platform': platform.platform()},
        'inputs': {'dump': os.path.basename(os.path.normpath(args.dump)),
                   'dump_written_at': info.get('written_at'), 'dump_code_commit': info.get('code_commit'),
                   'circo': os.path.basename(os.path.normpath(args.circo)),
                   'val_annotations_sha256': sha256(ann_bytes), 'gallery_cache_sha256': gallery_sha,
                   'gallery_size': int(gallery.shape[0])},
        'rules': {'reference_removed': 'the reference image is removed before topk(50), as in eval_circo_v2.py '
                                       'and eval_circo_test.py (the paper\'s CIRCO protocol)',
                  'reference_kept': 'topk(50) over the whole gallery, as in SEARLE\'s and LinCIR\'s released '
                                    'CIRCO code'},
        'metrics': 'CIRCO evaluation.py: AP@K = sum of precision at the ground-truth hits in the top K / '
                   'min(#ground truths, K); Recall@K for the single target image; aspect mAP@10 averages AP@10 '
                   'over the queries tagged with that aspect. All values in %.',
        'models': summary_models,
        'checks_failed': failures,
    }
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    head = (['model', 'title'] + [f'{rule}_mAP@{k}' for rule in RULES for k in KS] +
            [f'{rule}_Recall@{k}' for rule in RULES for k in KS] +
            ['change_mAP@10', 'reference_in_top1', 'reference_in_top10', 'reference_in_top50', 'gpu_check'])
    with open(os.path.join(out, 'summary.tsv'), 'w') as f:
        f.write('\t'.join(head) + '\n')
        for name, e in summary_models.items():
            check = e['check_against_gpu']
            row = ([name, e['title']] + [f'{e[rule][f"mAP@{k}"]:.2f}' for rule in RULES for k in KS] +
                   [f'{e[rule][f"Recall@{k}"]:.2f}' for rule in RULES for k in KS] +
                   [f'{e["reference_kept"]["mAP@10"] - e["reference_removed"]["mAP@10"]:+.2f}',
                    e['reference_in_top']['1'], e['reference_in_top']['10'], e['reference_in_top']['50'],
                    'exact' if check and check['exact'] else ('MISMATCH' if check else 'none')])
            f.write('\t'.join(str(x) for x in row) + '\n')
    print(f'\nwrote {out} in {time.time() - t0:.0f}s')
    if failures:
        print(f'CHECK FAILED: the reference-removed results differ from the GPU evaluation for {", ".join(failures)}')
        sys.exit(1)
    print('all reference-removed results equal the GPU evaluation')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='split', required=True)
    val = sub.add_parser('val', help='score CIRCO val with the reference removed and kept, for every dump model')
    val.add_argument('--dump', required=True, help='final_run.py dump folder, or a .tar.gz that contains it')
    val.add_argument('--circo', required=True, help='eval_data/circo folder, or a .zip that contains it')
    val.add_argument('--out', required=True, help='output folder (new or empty)')
    val.add_argument('--models', default='', help='comma-separated subset of the dump models (default: all)')
    val.add_argument('--overwrite', action='store_true', help='allow a non-empty output folder')
    args = parser.parse_args()
    if args.split == 'val':
        run_val(args)


if __name__ == '__main__':
    main()

"""Prepare /notebooks/experiment on a fresh machine for evaluation/reeval.py.

Run it after
    unzip -q paperspace_bundle.zip -d /notebooks/experiment      checkpoints, CLIP feature caches, annotations
    git clone https://github.com/miccunifi/CIRCO.git /notebooks/experiment/eval_data/circo/circo_repo

It then
  1. verifies every bundled file against MANIFEST.sha256;
  2. checks CIRCO's code is present and compares its annotations with the bundled copy (the
     evaluation reads the bundled copy, which is the one the original runs used);
  3. downloads the COCO 2017 unlabeled image list and the reference images of the 220 val and
     800 test CIRCO queries (the 123k-image gallery is read from the cached CLIP features);
  4. downloads the Fashion-IQ images from the dataset the original runs used and keeps the
     val-split images;
  5. checks completeness: every CIRCO query has its reference image, every Fashion-IQ val query has
     both images (the original runs skipped none), and the caches cover what the queries need;
  6. downloads the CLIP ViT-L/14 weights (the clip package verifies their checksum).
Safe to rerun: finished steps are skipped. Writes eval_data/prepared.json and exits non-zero if
anything is missing.

Usage:
    python evaluation/prepare_eval_data.py
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT   = Path('/notebooks/experiment')
MANIFEST     = EXPERIMENT / 'MANIFEST.sha256'
EVAL         = EXPERIMENT / 'eval_data'
CIRCO        = EVAL / 'circo'
CIRCO_REPO   = CIRCO / 'circo_repo'
COCO         = CIRCO / 'COCO2017_unlabeled'
COCO_INFO    = COCO / 'annotations' / 'image_info_unlabeled2017.json'
COCO_IMAGES  = COCO / 'unlabeled2017'
COCO_INFO_URL = 'http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip'
COCO_IMG_URL  = 'http://images.cocodataset.org/unlabeled2017/{}'
FIQ          = EVAL / 'fashion_iq'
FIQ_IMAGES   = FIQ / 'images' / 'images'                 # the layout eval_fashion.py reads
FIQ_ZIP      = FIQ / 'images.zip'
FIQ_ZIP_URL  = ('https://huggingface.co/datasets/BUAADreamer/cir_dataset/resolve/main/'
                'fashionIQ_dataset/images.zip')
FIQ_ZIP_BYTES = 992_974_388
CATS         = ('dress', 'shirt', 'toptee')
# Query counts of the original evaluations (CIRCO annotations; Fashion-IQ "Valid queries" in logs/).
EXPECTED = {'circo val': 220, 'circo test': 800, 'dress': 2017, 'shirt': 2038, 'toptee': 1961}
UA = {'User-Agent': 'eval-data-prep/1.0'}


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')


def log(msg=''):
    print(msg, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, expected_bytes=None, retries=6, timeout=120, report_every=0):
    """Download to dest via a .part file; resumes a partial .part with an HTTP Range request."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    for attempt in range(1, retries + 1):
        try:
            start = part.stat().st_size if part.exists() else 0
            headers = dict(UA)
            if start:
                headers['Range'] = f'bytes={start}-'
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
                if start and r.status != 206:      # server ignored the Range header: start over
                    start = 0
                with open(part, 'ab' if start else 'wb') as f:
                    done, next_report = start, start + report_every
                    for chunk in iter(lambda: r.read(1 << 20), b''):
                        f.write(chunk)
                        done += len(chunk)
                        if report_every and done >= next_report:
                            total = f' / {expected_bytes / 1e6:.0f}' if expected_bytes else ''
                            log(f'    {done / 1e6:.0f}{total} MB')
                            next_report += report_every
            if expected_bytes is not None and part.stat().st_size != expected_bytes:
                raise IOError(f'got {part.stat().st_size} bytes, expected {expected_bytes}')
            os.replace(part, dest)
            return
        except Exception as e:                                 # noqa: BLE001
            if attempt == retries:
                raise RuntimeError(f'download failed after {retries} attempts: {url}: {e}') from e
            log(f'    retry {attempt}/{retries - 1} for {url.rsplit("/", 1)[-1]}: {e}')
            time.sleep(3 * attempt)


def image_ok(path):
    from PIL import Image
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:                                          # noqa: BLE001
        return False


def git_head(repo):
    try:
        head = (repo / '.git' / 'HEAD').read_text().strip()
        if not head.startswith('ref:'):
            return head
        ref = head.split(' ', 1)[1]
        if (repo / '.git' / ref).exists():
            return (repo / '.git' / ref).read_text().strip()
        for line in (repo / '.git' / 'packed-refs').read_text().splitlines():
            if line.endswith(' ' + ref):
                return line.split(' ', 1)[0]
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def verify_bundle(report, problems):
    log('\n[1/6] Bundle files vs MANIFEST.sha256')
    if not MANIFEST.exists():
        problems.append(f'{MANIFEST} not found: unzip paperspace_bundle.zip into {EXPERIMENT} first')
        return
    entries = [line.split('  ', 1) for line in MANIFEST.read_text().splitlines() if line.strip()]
    bad = []
    for sha, rel in entries:
        p = EXPERIMENT / rel
        if not p.exists():
            bad.append(f'missing {rel}')
        elif sha256_file(p) != sha:
            bad.append(f'sha256 mismatch {rel}')
    report['bundle'] = {'files': len(entries), 'problems': bad}
    log(f'  {len(entries) - len(bad)}/{len(entries)} files verified')
    problems.extend(bad)


def check_circo_code(report, problems, notes):
    log('\n[2/6] CIRCO code and annotations')
    for f in ('src/dataset.py', 'src/evaluation.py'):
        if not (CIRCO_REPO / f).exists():
            problems.append(f'missing {CIRCO_REPO / f}: git clone https://github.com/miccunifi/CIRCO.git {CIRCO_REPO}')
    same = {}
    for split in ('val', 'test'):
        a, b = CIRCO / 'annotations' / f'{split}.json', CIRCO_REPO / 'annotations' / f'{split}.json'
        if not a.exists():
            problems.append(f'missing bundled annotations {a}')
            continue
        same[split] = b.exists() and sha256_file(a) == sha256_file(b)
        if not same[split]:
            notes.append(f'CIRCO {split}.json on GitHub differs from the bundled copy; the bundled one (used '
                         'originally) is what the evaluation reads')
    report['circo_code'] = {'repo_commit': git_head(CIRCO_REPO), 'annotations_identical_to_github': same}
    log(f'  repo commit {report["circo_code"]["repo_commit"]}; annotations identical to GitHub: {same}')


def prepare_coco(report, problems, workers):
    log('\n[3/6] COCO 2017 unlabeled: image list + CIRCO reference images')
    if not COCO_INFO.exists():
        tmp_zip = COCO / 'image_info_unlabeled2017.zip'
        log(f'  downloading {COCO_INFO_URL}')
        download(COCO_INFO_URL, tmp_zip)
        with zipfile.ZipFile(tmp_zip) as z:
            member = next(n for n in z.namelist() if n.endswith('image_info_unlabeled2017.json'))
            COCO_INFO.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as fin, open(COCO_INFO.with_suffix('.json.part'), 'wb') as fout:
                shutil.copyfileobj(fin, fout)
        os.replace(COCO_INFO.with_suffix('.json.part'), COCO_INFO)
        tmp_zip.unlink()
    info = json.loads(COCO_INFO.read_text())
    by_id = {img['id']: img for img in info['images']}
    queries = {s: json.loads((CIRCO / 'annotations' / f'{s}.json').read_text()) for s in ('val', 'test')}
    refs = sorted({q['reference_img_id'] for s in queries for q in queries[s]})
    missing_info = [r for r in refs if r not in by_id]
    if missing_info:
        problems.append(f'{len(missing_info)} CIRCO reference ids not in the COCO image list, e.g. {missing_info[:5]}')
    todo = [by_id[r] for r in refs if r in by_id and not (COCO_IMAGES / by_id[r]['file_name']).exists()]
    log(f'  {len(refs)} reference images needed, {len(refs) - len(todo)} already present, downloading {len(todo)}')

    def fetch(img):
        dest = COCO_IMAGES / img['file_name']
        for _ in range(3):
            download(img.get('coco_url') or COCO_IMG_URL.format(img['file_name']), dest)
            if image_ok(dest):
                return None
            dest.unlink()
        return img['file_name']

    with ThreadPoolExecutor(max_workers=workers) as pool:
        failed = [f for f in pool.map(fetch, todo) if f]
    if failed:
        problems.append(f'{len(failed)} COCO images could not be downloaded intact, e.g. {failed[:5]}')
    present = [r for r in refs if r in by_id and (COCO_IMAGES / by_id[r]['file_name']).exists()]
    report['coco'] = {'image_list_entries': len(by_id), 'image_list_sha256': sha256_file(COCO_INFO),
                      'reference_images_needed': len(refs), 'reference_images_present': len(present),
                      'queries': {s: len(q) for s, q in queries.items()}}
    for s, n in (('val', EXPECTED['circo val']), ('test', EXPECTED['circo test'])):
        if len(queries[s]) != n:
            problems.append(f'CIRCO {s} has {len(queries[s])} queries, expected {n}')
    log(f'  reference images present: {len(present)}/{len(refs)}')
    return queries


def prepare_fashion(report, problems, keep_zip):
    log('\n[4/6] Fashion-IQ images')
    caps = {c: json.loads((FIQ / 'captions' / f'cap.{c}.val.json').read_text()) for c in CATS}
    splits = {c: json.loads((FIQ / 'image_splits' / f'split.{c}.val.json').read_text()) for c in CATS}
    needed = sorted(set().union(*splits.values()))
    referenced = sorted({n for c in CATS for a in caps[c] for n in (a['candidate'], a['target'])})
    missing = [n for n in needed if not (FIQ_IMAGES / f'{n}.png').exists()]
    if missing:
        if not FIQ_ZIP.exists():
            log(f'  downloading {FIQ_ZIP_URL} ({FIQ_ZIP_BYTES / 1e6:.0f} MB)')
            download(FIQ_ZIP_URL, FIQ_ZIP, expected_bytes=FIQ_ZIP_BYTES, report_every=100_000_000)
        want = set(missing)
        FIQ_IMAGES.mkdir(parents=True, exist_ok=True)
        n_out = 0
        with zipfile.ZipFile(FIQ_ZIP) as z:
            for m in z.infolist():
                stem = Path(m.filename).stem
                if m.filename.endswith('.png') and stem in want:
                    part = FIQ_IMAGES / f'{stem}.png.part'
                    with z.open(m) as fin, open(part, 'wb') as fout:
                        shutil.copyfileobj(fin, fout)
                    os.replace(part, FIQ_IMAGES / f'{stem}.png')
                    want.discard(stem)
                    n_out += 1
        log(f'  extracted {n_out} val images')
    still = [n for n in needed if not (FIQ_IMAGES / f'{n}.png').exists()]
    broken = [n for n in referenced if (FIQ_IMAGES / f'{n}.png').exists() and not image_ok(FIQ_IMAGES / f'{n}.png')]
    per_cat = {}
    for c in CATS:
        ok = sum((FIQ_IMAGES / f'{a["candidate"]}.png').exists() and (FIQ_IMAGES / f'{a["target"]}.png').exists()
                 for a in caps[c])
        per_cat[c] = {'queries': len(caps[c]), 'queries_with_both_images': ok, 'gallery': len(splits[c])}
        if len(caps[c]) != EXPECTED[c] or ok != len(caps[c]):
            problems.append(f'Fashion-IQ {c}: {ok}/{len(caps[c])} queries have both images '
                            f'(original runs: {EXPECTED[c]} valid, 0 skipped)')
    if still:
        problems.append(f'{len(still)} Fashion-IQ val images missing after extraction, e.g. {still[:5]}')
    if broken:
        problems.append(f'{len(broken)} Fashion-IQ images do not open, e.g. {broken[:5]}')
    report['fashion_iq'] = {'val_images_needed': len(needed), 'val_images_present': len(needed) - len(still),
                            'images_referenced_by_queries': len(referenced), 'unreadable': len(broken),
                            'per_category': per_cat, 'source': FIQ_ZIP_URL}
    log(f'  val images present: {len(needed) - len(still)}/{len(needed)}; '
        + ', '.join(f'{c} {v["queries_with_both_images"]}/{v["queries"]} queries' for c, v in per_cat.items()))
    if FIQ_ZIP.exists() and not keep_zip and not still and not broken:
        FIQ_ZIP.unlink()
        log('  removed images.zip (all val images extracted)')
    return caps, splits


def check_caches(report, problems, notes, queries, splits):
    log('\n[5/6] CLIP feature caches cover the queries')
    try:
        import torch
    except Exception as e:                                     # noqa: BLE001
        notes.append(f'torch not importable, cache coverage not checked: {e}')
        return
    cache = torch.load(CIRCO / 'gallery_cache_ViT-L-14.pt', map_location='cpu')
    ids = set(cache['img_id_to_idx'])
    refs = {str(q['reference_img_id']) for s in queries for q in queries[s]}
    gts = {str(g) for q in queries['val'] for g in q['gt_img_ids']}
    circo = {'clip_model': cache['clip_model'], 'n_gallery': int(cache['embeddings'].shape[0]),
             'missing_reference_ids': len(refs - ids), 'missing_val_gt_ids': len(gts - ids)}
    if cache['clip_model'] != 'ViT-L/14' or refs - ids or gts - ids:
        problems.append(f'CIRCO gallery cache does not cover the queries: {circo}')
    del cache
    fashion = {}
    for c in CATS:
        feats = torch.load(FIQ / f'gallery_cache_{c}_ViT-L-14.pt', map_location='cpu')
        fashion[c] = list(feats.shape)
        if feats.shape[0] != len(splits[c]):
            problems.append(f'Fashion-IQ {c} cache has {feats.shape[0]} rows, split has {len(splits[c])} images')
    report['caches'] = {'circo': circo, 'fashion_iq_shapes': fashion}
    log(f'  CIRCO gallery {circo["n_gallery"]} images, all query ids covered: '
        f'{not circo["missing_reference_ids"] and not circo["missing_val_gt_ids"]}; Fashion-IQ {fashion}')


def fetch_clip(report, problems):
    log('\n[6/6] CLIP ViT-L/14 weights')
    try:
        import clip
        clip.load('ViT-L/14', device='cpu')
        report['clip'] = 'ViT-L/14 weights present (checksum verified by clip)'
        log('  ok')
    except Exception as e:                                     # noqa: BLE001
        problems.append(f'CLIP ViT-L/14 could not be loaded: {e}')


def main():
    p = argparse.ArgumentParser(description='Prepare the evaluation inputs on a fresh machine.')
    p.add_argument('--workers', type=int, default=16, help='parallel COCO image downloads')
    p.add_argument('--keep_zip', action='store_true', help='keep fashion_iq/images.zip after extraction')
    args = p.parse_args()

    log(f'Preparing {EXPERIMENT}   {now()}')
    report, problems, notes = {'started_at': now()}, [], []
    verify_bundle(report, problems)
    check_circo_code(report, problems, notes)
    queries = prepare_coco(report, problems, args.workers)
    _, splits = prepare_fashion(report, problems, args.keep_zip)
    check_caches(report, problems, notes, queries, splits)
    fetch_clip(report, problems)
    report.update(finished_at=now(), problems=problems, notes=notes)
    out = EVAL / 'prepared.json'
    out.write_text(json.dumps(report, indent=2))
    log()
    for n in notes:
        log(f'note: {n}')
    if problems:
        log(f'NOT READY ({len(problems)} problems):')
        for pr in problems:
            log(f'  - {pr}')
        sys.exit(1)
    log(f'READY. Report: {out}')


if __name__ == '__main__':
    main()

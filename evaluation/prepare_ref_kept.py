"""Download and verify everything the external CIRCO methods need for the reference-kept run.

evaluation/ref_kept_externals.py recomputes the composed queries of the released SEARLE-XL, Pic2Word and LinCIR models
with the query builders of evaluation/stickiness_compare.py (the code behind metrics/stickiness_{searle,pic2word,lincir}
.json, June 2026). This script fetches what those builders load, pinned, into one folder (--assets_dir), verifies every
file, and proves that the Hugging Face cache loads offline and that the unpinned torch.hub call of
stickiness_compare.queries_searle resolves to the pinned SEARLE copy. Run it once on the machine, after the pip line
(python evaluation/prepare_ref_kept.py --print_pip) and before the run. Safe to rerun: verified files are kept, missing
or bad ones are fetched again. Exits 1 if anything is missing.

Pins (resolved on 2026-11-28 through the GitHub and Hugging Face APIs; every one predates the June 2026 runs except the
LinCIR README-only commit, so the June runs used the same code and weights):
  Pic2Word code   google-research/composed_image_retrieval @ 8c053297 (main, 2023-05-10)
  LinCIR code     navervision/lincir @ 1dec42d1 (master, 2026-01-05; README only after 2024-07-25's loader.py)
  SEARLE code     miccunifi/SEARLE @ a9c314ba (main, 2025-07-31; README only since 2023-10-17). torch.hub accepts
                  'owner/repo:ref', but queries_searle calls torch.hub.load('miccunifi/SEARLE', ...) without one, which
                  resolves to <hub dir>/miccunifi_SEARLE_main; the pinned commit is placed there, so that call loads it.
  SEARLE weights  release 'weights' SEARLE_ViT-L14.pt (what hubconf.searle downloads), sha256 pinned
  LinCIR (HF)     navervision/zeroshot-cir-models @ 5118df46: lincir_large.pt
                  openai/clip-vit-large-patch14 @ 32bd6428: config.json, model.safetensors (LinCIR's build_text_encoder
                  loads CLIPVisionModelWithProjection and CLIPTextModelWithProjection from it; transformers 4.35.2 with
                  safetensors installed reads model.safetensors)
                  stabilityai/stable-diffusion-xl-base-1.0 @ 46216598: tokenizer_2/ (CLIPTokenizer; build_text_encoder
                  adds '[$]' = id 49408 itself)
  Pic2Word ckpt   pic2word_model.pt. The README's Google Drive file (1IxRi2Cj...) is gone (HTTP 404 on 2026-11-28; repo
                  issue #30, open since 2024-08). Sources tried in order: the README id, the Drive re-upload linked in
                  issue #30 (1Bza7SFA..., by the context-i2w author, 2025-04-20), the Hugging Face mirror
                  flashingtt/pic2word @ 7e70a868 (uploaded 2024-01-19). Checked by HTTP range requests on 2026-11-28:
                  both mirrors serve 1,723,251,477 bytes and their first 4 MB, which hold the whole 92,012-byte
                  data.pkl, are byte-identical; that pickle (read without unpickling) has Pic2Word's training keys
                  epoch / name / state_dict / state_dict_img2text / optimizer with 'module.' prefixes. So the file is
                  pinned to the mirror's LFS sha256 bde01cfd...; a source whose file differs is rejected and the next
                  one tried. The June run's copy was 1.7G (du -h, keep_rest.zip file_inventory.txt of 2026-07-02),
                  which fits this size; its source is not recorded, so the run's self-check against the June numbers
                  is the final test. The sha256 is printed.
  OpenAI CLIP     ~/.cache/clip/ViT-L-14.pt (SEARLE's clip.load and Pic2Word's own model/clip.py read it there)

Layout of --assets_dir (default /notebooks/experiment/refkept_assets):
  repos/composed_image_retrieval/      Pic2Word code (queries_pic2word's `repo`)
  repos/lincir/                        LinCIR code (queries_lincir's `repo`)
  pic2word/pic2word_model.pt           Pic2Word checkpoint (queries_pic2word's `ckpt_path`)
  torch_hub/miccunifi_SEARLE_main/     torch.hub.set_dir(<assets>/torch_hub) makes queries_searle use it
  torch_hub/checkpoints/SEARLE_ViT-L14.pt
  hf_home/                             HF_HOME (hub cache under hf_home/hub)
  requirements_ref_kept.txt            the pip pins and the install line
  prepared_ref_kept.json               the verification report

Usage (repo at /notebooks/experiment/codes, in the venv of the earlier runs):
    python evaluation/prepare_ref_kept.py --print_pip            # prints the pip line: run it first
    PYTHONPATH=/notebooks/experiment/codes python -u evaluation/prepare_ref_kept.py
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import pickletools
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = Path('/notebooks/experiment/refkept_assets')
RESOLVED = '2026-11-28'
UA = {'User-Agent': 'refkept-prepare/1.0'}

# ---------------------------------------------------------------------------------------------------------------------
# Pins (resolved on RESOLVED; blob = git blob SHA-1 of the file at that commit, from the GitHub tree API)
# ---------------------------------------------------------------------------------------------------------------------
REPOS = {
    'pic2word': {
        'github': 'google-research/composed_image_retrieval', 'branch': 'main',
        'commit': '8c053297c2fae9cd17ddcded48445a4f47208dbd', 'commit_date': '2023-05-10',
        'dir': 'repos/composed_image_retrieval',
        'blobs': {  # every file queries_pic2word imports or reads
            'model/clip.py': '02625b9f734f8803d5f52ec8f3434f41706cd5d9',
            'model/model.py': '7525fc029ce7141dc7bf1eba6c698b39618a16d7',
            'third_party/open_clip/clip.py': '2e44431c5744c61e6d7d2d9e6992975ed1bf6d10',
            'third_party/open_clip/model.py': 'cda8161982f94cc0e8577ca4a018d5aa5233515d',
            'third_party/open_clip/simple_tokenizer.py': '876f08f4d5bbe3ab430af2fd5d822828affbbd4b',
            'third_party/open_clip/bpe_simple_vocab_16e6.txt.gz': '7b5088a527f720063f044eb928eee315f63b2fc0',
            'README.md': '01e212deb18f013286a23e46b595769e718cd0bc',
        }},
    'lincir': {
        'github': 'navervision/lincir', 'branch': 'master',
        'commit': '1dec42d118da816be2a43fd43cc0746e05f63881', 'commit_date': '2026-01-05',
        'dir': 'repos/lincir',
        'blobs': {
            'models.py': 'a465bde1c721fc9a9656d2433b4c64b6501c066f',
            'encode_with_pseudo_tokens.py': '81a2af1e99a97fa1e522c13b640761782bb48550',
        }},
    'searle': {
        'github': 'miccunifi/SEARLE', 'branch': 'main',
        'commit': 'a9c314ba4b6e6e14be5a9c3fdf6b66e6a0c23e37', 'commit_date': '2025-07-31',
        'dir': 'torch_hub/miccunifi_SEARLE_main',   # torch.hub's cache name for 'miccunifi/SEARLE' (default branch main)
        'blobs': {
            'hubconf.py': '318e7f246718f525cabdda8f22b4ea1a3bf2005d',
            'src/phi.py': 'd245c820a32f7c0619f7ab69ef37df51b757f843',
            'src/encode_with_pseudo_tokens.py': '840ab88d2fce2cb2e0a7807f6f567bd878f16fdb',
        }},
}
TORCH_HUB = 'torch_hub'
SEARLE_WEIGHTS = {'url': 'https://github.com/miccunifi/SEARLE/releases/download/weights/SEARLE_ViT-L14.pt',
                  'file': 'torch_hub/checkpoints/SEARLE_ViT-L14.pt', 'bytes': 56_652_823,
                  'sha256': '0ce31fb8eecaad95cdf60287dbad2ac9f968d79b74559f11b462701ce4693b95',
                  'phi_params': 14_162_688}
HF_HOME = 'hf_home'
# (repo, pinned commit, file, bytes, digest kind, digest). sha256 = the LFS object id, gitsha1 = the git blob id; both
# are also the blob names huggingface_hub uses in its cache.
HF_FILES = (
    ('navervision/zeroshot-cir-models', '5118df4683de6efa09f8e5336d86ca626ed21c44', 'lincir_large.pt',
     56_653_151, 'sha256', '5ba98d52db7a5e9a78f9ba7436991235477b89544caf246e4659642bf9b409bd'),
    ('openai/clip-vit-large-patch14', '32bd64288804d66eefd0ccbe215aa642df71cc41', 'config.json',
     4_519, 'gitsha1', '2c19f6666e0e163c7954df66cb901353fcad088e'),
    ('openai/clip-vit-large-patch14', '32bd64288804d66eefd0ccbe215aa642df71cc41', 'model.safetensors',
     1_710_540_580, 'sha256', 'a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb'),
    ('stabilityai/stable-diffusion-xl-base-1.0', '462165984030d82259a11f4367a4eed129e94a7b', 'tokenizer_2/vocab.json',
     1_059_962, 'gitsha1', '469be27c5c010538f845f518c4f5e8574c78f7c8'),
    ('stabilityai/stable-diffusion-xl-base-1.0', '462165984030d82259a11f4367a4eed129e94a7b', 'tokenizer_2/merges.txt',
     524_619, 'gitsha1', '76e821f1b6f0a9709293c3b6b51ed90980b3166b'),
    ('stabilityai/stable-diffusion-xl-base-1.0', '462165984030d82259a11f4367a4eed129e94a7b',
     'tokenizer_2/special_tokens_map.json', 460, 'gitsha1', 'ae0c5be6f35217e51c4c000fd325d8de0294e99c'),
    ('stabilityai/stable-diffusion-xl-base-1.0', '462165984030d82259a11f4367a4eed129e94a7b',
     'tokenizer_2/tokenizer_config.json', 725, 'gitsha1', 'a8438e020c4497a429240d6b89e0bf9a6e2ffa92'),
)
PIC2WORD_CKPT = 'pic2word/pic2word_model.pt'
PIC2WORD_SOURCES = (
    ('readme', 'drive', '1IxRi2Cj81RxMu0ViT4q4nkfyjbSHm1dF',
     "the Google Drive link in the repo README (gone since 2024: issue #30)"),
    ('drive_mirror', 'drive', '1Bza7SFAnokXy-dOgxkL346lDXXz8DxMy',
     "the Drive re-upload linked in issue #30 (github.com/Pter61/context-i2w/issues/16#issuecomment-2817104557)"),
    ('hf_mirror', 'url', 'https://huggingface.co/flashingtt/pic2word/resolve/7e70a868a843bd4f4c136aa0f9d9139aed1d46c8/'
     'pic2word_model.pt', "the Hugging Face mirror flashingtt/pic2word (uploaded 2024-01-19)"),
)
# The file both mirrors serve (HF LFS object id; the Drive re-upload has the same size and identical first 4 MB).
PIC2WORD_HF_MIRROR = {'bytes': 1_723_251_477,
                      'sha256': 'bde01cfd53f642c58011b44590a26e14a0fd7d94efb3a16653e660e342aeab7e'}
PIC2WORD_JUNE_DU = '1.7G'          # keep_rest.zip file_inventory.txt (2026-07-02): 1.7G ./pic2word/pic2word_model.pt
CLIP_WEIGHTS = {'path': '~/.cache/clip/ViT-L-14.pt', 'bytes': 932_768_134,
                'sha256': 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'}
# Extra packages beyond the machine's pins (torch 2.1.1+cu121, torchvision 0.16.1, numpy 1.26.3, Pillow 9.5.0, CLIP @
# d05afc4, PyYAML, regex, ftfy, tqdm, requests, filelock, fsspec, packaging, typing_extensions; logs/08_pip_freeze.txt).
# transformers / huggingface-hub / tokenizers / safetensors are the versions of the June 2026 machine
# (keep_rest.zip requirements_frozen.txt), so LinCIR runs the code that made the June numbers. All have Python 3.11
# wheels and need nothing newer than what the venv has; installing with --no-deps leaves every existing pin untouched.
PIP_PINS = (
    ('transformers', '4.35.2', "LinCIR models.py / encode_with_pseudo_tokens.py; June 2026 version"),
    ('huggingface-hub', '0.20.3', "hf_hub_download in queries_lincir and transformers; June 2026 version"),
    ('tokenizers', '0.15.1', "transformers 4.35.2 needs >=0.14,<0.19; June 2026 version"),
    ('safetensors', '0.4.0', "transformers reads CLIP-L from model.safetensors; June 2026 version"),
    ('gdown', '6.4.2', "the Pic2Word checkpoint from Google Drive (6.x raises on failure; Python >=3.10)"),
    ('beautifulsoup4', '4.12.3', "gdown"),
    ('soupsieve', '2.5', "beautifulsoup4"),
    ('PySocks', '1.7.1', "requests[socks], needed by gdown"),
    ('setuptools', '69.5.1', "Pic2Word's third_party/open_clip/clip.py does `from pkg_resources import packaging`, "
                             "which setuptools 70.0 removed"),
)
PIP_LINE = ('python -m pip install --no-deps ' + ' '.join(f'{n}=={v}' for n, v, _ in PIP_PINS)
            + ' && python -m pip check')
MIN_FREE_GB = 4.0
# Pickle globals a checkpoint may reference without running foreign code (everything torch.save writes for tensors and
# plain containers, plus numpy arrays and argparse namespaces that training scripts sometimes store).
SAFE_GLOBALS = {
    'collections.OrderedDict', 'torch._utils._rebuild_tensor_v2', 'torch._utils._rebuild_parameter',
    'torch._utils._rebuild_parameter_with_state', 'torch._tensor._rebuild_from_type_v2', 'torch.Size', 'torch.device',
    'torch.FloatStorage', 'torch.HalfStorage', 'torch.DoubleStorage', 'torch.BFloat16Storage', 'torch.LongStorage',
    'torch.IntStorage', 'torch.ShortStorage', 'torch.CharStorage', 'torch.ByteStorage', 'torch.BoolStorage',
    'torch.storage.UntypedStorage', 'torch.float32', 'torch.float16', 'torch.bfloat16', 'torch.int64',
    'builtins.set', 'builtins.frozenset', 'builtins.slice', 'builtins.complex', 'builtins.bytearray',
    '_codecs.encode', 'numpy.core.multiarray._reconstruct', 'numpy.core.multiarray.scalar', 'numpy.ndarray',
    'numpy.dtype', 'argparse.Namespace'}
STEPS = ('pip', 'repos', 'searle', 'lincir_hf', 'pic2word', 'clip', 'verify')


# ---------------------------------------------------------------------------------------------------------------------
# Small helpers (stdlib only, so ref_kept_externals.py can import this module anywhere)
# ---------------------------------------------------------------------------------------------------------------------
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


def git_blob_sha1(path):
    """The git blob id of a file (what the GitHub tree API and the Hugging Face API report for non-LFS files)."""
    h = hashlib.sha1(b'blob %d\0' % Path(path).stat().st_size)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(path, kind):
    return sha256_file(path) if kind == 'sha256' else git_blob_sha1(path)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def human(n):
    return f'{n / 1e6:,.1f} MB' if n is not None else '? MB'


def download(url, dest, expected_bytes=None, retries=6, timeout=120, report_every=0):
    """Download to dest via a .part file; resumes a partial .part with an HTTP Range request."""
    dest = Path(dest)
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
                            log(f'    {done / 1e6:,.0f} / {expected_bytes / 1e6:,.0f} MB' if expected_bytes
                                else f'    {done / 1e6:,.0f} MB')
                            next_report += report_every
            if expected_bytes is not None and part.stat().st_size != expected_bytes:
                got = part.stat().st_size
                if got > expected_bytes:
                    part.unlink()
                raise IOError(f'got {got} bytes, expected {expected_bytes}')
            os.replace(part, dest)
            return
        except Exception as e:                                 # noqa: BLE001
            if attempt == retries:
                raise RuntimeError(f'download failed after {retries} attempts: {url}: {e}') from e
            log(f'    retry {attempt}/{retries - 1} for {url.rsplit("/", 1)[-1]}: {e}')
            time.sleep(3 * attempt)


def free_gb(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def pickle_globals(path):
    """Every global a torch checkpoint's pickle references, found without unpickling (pickletools only)."""
    def scan(stream, found):
        strings, memo = [], {}
        for op, arg, _ in pickletools.genops(stream):
            name = op.name
            if name in ('GLOBAL', 'INST'):
                found.add(arg.replace(' ', '.'))
            elif name in ('SHORT_BINUNICODE', 'BINUNICODE', 'BINUNICODE8', 'UNICODE', 'SHORT_BINSTRING', 'BINSTRING',
                          'STRING'):
                strings.append(arg)
            elif name in ('BINPUT', 'LONG_BINPUT', 'PUT') and strings:
                memo[arg] = strings[-1]
            elif name == 'MEMOIZE' and strings:
                memo[len(memo)] = strings[-1]
            elif name in ('BINGET', 'LONG_BINGET', 'GET') and arg in memo:
                strings.append(memo[arg])
            elif name == 'STACK_GLOBAL':
                found.add(f'{strings[-2]}.{strings[-1]}' if len(strings) >= 2 else '<unresolved STACK_GLOBAL>')
    found = set()
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            pkl = [n for n in z.namelist() if n == 'data.pkl' or n.endswith('/data.pkl')]
            if len(pkl) != 1:
                raise ValueError(f'{path}: expected one data.pkl in the torch zip archive, found {len(pkl)}')
            scan(io.BytesIO(z.read(pkl[0])), found)
    else:                                   # legacy torch.save format: magic, protocol, sys_info, then the object
        with open(path, 'rb') as f:
            for _ in range(4):
                scan(f, found)
    return found


def unsafe_globals(path):
    return sorted(g for g in pickle_globals(path) if g not in SAFE_GLOBALS)


def git_commit_of(repo_dir):
    marker = Path(repo_dir) / '.pinned_commit'
    return marker.read_text().strip() if marker.exists() else None


def hf_storage(hf_home, repo):
    return Path(hf_home) / 'hub' / ('models--' + repo.replace('/', '--'))


def hf_env(hf_home, offline):
    """Environment for huggingface_hub / transformers: they read it once, at import."""
    hf_home = Path(hf_home)
    env = {'HF_HOME': str(hf_home), 'HF_HUB_CACHE': str(hf_home / 'hub'), 'HF_HUB_DISABLE_TELEMETRY': '1',
           'HF_HUB_DISABLE_PROGRESS_BARS': '1'}
    if offline:
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env


def apply_hf_env(hf_home, offline):
    """Point this process at hf_home before huggingface_hub is imported. Returns a problem if it already was."""
    for legacy in ('HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE', 'PYTORCH_TRANSFORMERS_CACHE',
                   'PYTORCH_PRETRAINED_BERT_CACHE'):
        os.environ.pop(legacy, None)
    os.environ.update(hf_env(hf_home, offline))
    if 'huggingface_hub' in sys.modules:
        try:
            from huggingface_hub import constants
            want = str(Path(hf_home) / 'hub')
            if os.path.abspath(constants.HF_HUB_CACHE) != os.path.abspath(want) or (
                    offline and not constants.HF_HUB_OFFLINE):
                return [f'huggingface_hub was imported before its cache was set (cache {constants.HF_HUB_CACHE}, '
                        f'offline {constants.HF_HUB_OFFLINE}); expected {want}, offline {offline}']
        except Exception as e:                               # noqa: BLE001
            return [f'cannot read huggingface_hub.constants: {e}']
    return []


def installed_version(dist):
    try:
        import importlib.metadata as md
        return md.version(dist)
    except Exception:                                         # noqa: BLE001
        return None


def pkg_resources_packaging_problem():
    """Pic2Word's third_party/open_clip/clip.py runs `from pkg_resources import packaging` at import."""
    code = ('import warnings; warnings.simplefilter("ignore"); from pkg_resources import packaging; '
            'assert packaging.version.parse("2.1.1+cu121") >= packaging.version.parse("1.8.0")')
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    if r.returncode:
        tail = (r.stderr.strip().splitlines() or ['?'])[-1]
        return f'`from pkg_resources import packaging` fails ({tail}); install setuptools==69.5.1'
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Verification (shared with ref_kept_externals.preflight)
# ---------------------------------------------------------------------------------------------------------------------
def verify_repo(assets_dir, key):
    spec = REPOS[key]
    root = Path(assets_dir) / spec['dir']
    rec = {'dir': str(root), 'github': spec['github'], 'commit': spec['commit']}
    if not root.is_dir():
        return rec, [f'{key}: missing {root} (run evaluation/prepare_ref_kept.py)']
    problems = []
    got = git_commit_of(root)
    if got != spec['commit']:
        problems.append(f'{key}: {root} holds commit {got}, pinned {spec["commit"]}')
    bad = []
    for rel, blob in spec['blobs'].items():
        p = root / rel
        if not p.is_file():
            bad.append(f'missing {rel}')
        elif git_blob_sha1(p) != blob:
            bad.append(f'{rel} differs from the pinned commit')
    if bad:
        problems.append(f'{key}: {root}: ' + '; '.join(bad))
    rec['files_verified'] = len(spec['blobs']) - len(bad)
    return rec, problems


def verify_file(path, expected_bytes, kind, expected_digest, label, full_hash=True):
    path = Path(path)
    rec = {'path': str(path), 'bytes_expected': expected_bytes, kind + '_expected': expected_digest}
    if not path.is_file():
        return rec, [f'{label}: missing {path} (run evaluation/prepare_ref_kept.py)']
    rec['bytes'] = path.stat().st_size
    if expected_bytes is not None and rec['bytes'] != expected_bytes:
        return rec, [f'{label}: {path} has {rec["bytes"]} bytes, expected {expected_bytes}']
    if full_hash and expected_digest:
        rec[kind] = digest(path, kind)
        if rec[kind] != expected_digest:
            return rec, [f'{label}: {path} {kind} {rec[kind][:16]}... differs from the pinned {expected_digest[:16]}...']
    return rec, []


def verify_searle(assets_dir, full_hash=True):
    rec, problems = verify_repo(assets_dir, 'searle')
    w = SEARLE_WEIGHTS
    wrec, wp = verify_file(Path(assets_dir) / w['file'], w['bytes'], 'sha256', w['sha256'], 'searle', full_hash)
    rec['weights'] = wrec
    return rec, problems + wp


def verify_lincir(assets_dir, full_hash=True):
    rec, problems = verify_repo(assets_dir, 'lincir')
    hf_home = Path(assets_dir) / HF_HOME
    rec['hf_home'] = str(hf_home)
    rec['hf_files'] = []
    for repo, commit, rel, nbytes, kind, dig in HF_FILES:
        storage = hf_storage(hf_home, repo)
        ref = storage / 'refs' / 'main'
        main = ref.read_text().strip() if ref.is_file() else None
        frec, fp = verify_file(storage / 'snapshots' / commit / rel, nbytes, kind, dig, 'lincir', full_hash)
        frec.update(repo=repo, file=rel, commit=commit, refs_main=main)
        if not fp and main != commit:
            fp.append(f'lincir: {storage}/refs/main is {main}, pinned {commit} (offline loading resolves main)')
        rec['hf_files'].append(frec)
        problems += fp
    return rec, problems


def read_report(assets_dir):
    p = Path(assets_dir) / 'prepared_ref_kept.json'
    try:
        return json.loads(p.read_text())
    except Exception:                                         # noqa: BLE001
        return {}


def verify_pic2word(assets_dir, full_hash=True, prepared=None):
    """The code, and the checkpoint against the pinned file. `prepared`: the checkpoint record prepare made (default:
    read from prepared_ref_kept.json), for its source."""
    rec, problems = verify_repo(assets_dir, 'pic2word')
    ck = Path(assets_dir) / PIC2WORD_CKPT
    pin = PIC2WORD_HF_MIRROR
    crec, bad = verify_file(ck, pin['bytes'], 'sha256', pin['sha256'], 'pic2word', full_hash)
    if prepared is None:
        prepared = read_report(assets_dir).get('assets', {}).get('pic2word', {}).get('checkpoint', {})
    crec['source'] = prepared.get('source')
    rec['checkpoint'] = crec
    problems += bad
    if not bad:
        try:
            crec['unsafe_pickle_globals'] = unsafe_globals(ck)
            if crec['unsafe_pickle_globals']:
                problems.append(f'pic2word: {ck} references unexpected pickle globals {crec["unsafe_pickle_globals"]}')
        except Exception as e:                                # noqa: BLE001
            problems.append(f'pic2word: cannot scan {ck}: {e}')
    return rec, problems


def verify_assets(assets_dir, methods=('searle', 'pic2word', 'lincir'), full_hash=True, pic2word_prepared=None):
    """Per-method records and problems for the assets of `methods` (no network, no imports beyond the stdlib)."""
    records, problems = {}, {}
    for m in methods:
        if m == 'pic2word':
            records[m], problems[m] = verify_pic2word(assets_dir, full_hash, pic2word_prepared)
        else:
            records[m], problems[m] = {'searle': verify_searle, 'lincir': verify_lincir}[m](assets_dir, full_hash)
    return records, problems


# ---------------------------------------------------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------------------------------------------------
def step_pip(assets_dir, report, problems):
    log('\n[pip] Extra Python packages')
    req = Path(assets_dir) / 'requirements_ref_kept.txt'
    req.parent.mkdir(parents=True, exist_ok=True)
    req.write_text(f'# Extra packages for evaluation/ref_kept_externals.py (pins resolved {RESOLVED}).\n'
                   f'# Install into the venv of the earlier runs, without touching its other pins:\n#   {PIP_LINE}\n'
                   + ''.join(f'{n}=={v}    # {why}\n' for n, v, why in PIP_PINS))
    found, off = {}, []
    for name, want, _ in PIP_PINS:
        got = installed_version(name)
        found[name] = got
        if got != want and name != 'setuptools':        # setuptools: what matters is the import checked below
            off.append(f'{name} {got or "not installed"} (pin {want})')
    if off:
        problems.append(f'packages not at their pins: {", ".join(off)}; run  {PIP_LINE}')
    pk = pkg_resources_packaging_problem()
    if pk:
        problems.append(pk)
    report['pip'] = {'pins': {n: v for n, v, _ in PIP_PINS}, 'installed': found, 'pip_line': PIP_LINE,
                     'requirements_file': str(req), 'pkg_resources_packaging': pk is None}
    log(f'  pip line: {PIP_LINE}')
    log('  installed: ' + ', '.join(f'{n} {v}' for n, v in found.items()))
    log(f'  written: {req}')


def fetch_github_archive(assets_dir, key):
    """The repo at its pinned commit, from GitHub's zip archive (its comment is the commit id; no git needed)."""
    spec = REPOS[key]
    dest = Path(assets_dir) / spec['dir']
    rec, problems = verify_repo(assets_dir, key)
    if not problems:
        log(f'  {key}: {dest} already at {spec["commit"][:8]}, {rec["files_verified"]} files verified')
        return
    tmp_zip = dest.parent / f'.{dest.name}.{spec["commit"][:8]}.zip'
    url = f'https://github.com/{spec["github"]}/zipball/{spec["commit"]}'
    log(f'  {key}: downloading {url}')
    download(url, tmp_zip)
    staging = dest.parent / f'.{dest.name}.staging'
    shutil.rmtree(staging, ignore_errors=True)
    with zipfile.ZipFile(tmp_zip) as z:
        comment = z.comment.decode(errors='replace').strip()
        if comment != spec['commit']:
            raise RuntimeError(f'{url}: archive is for commit {comment!r}, expected {spec["commit"]}')
        names = z.namelist()
        top = names[0].split('/')[0]
        for n in names:
            if n.startswith('/') or '..' in Path(n).parts or not n.startswith(top + '/'):
                raise RuntimeError(f'{url}: unexpected archive member {n!r}')
        z.extractall(staging)
    shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging / top, dest)
    shutil.rmtree(staging, ignore_errors=True)
    tmp_zip.unlink()
    (dest / '.pinned_commit').write_text(spec['commit'] + '\n')
    log(f'  {key}: unpacked to {dest}')


def step_repos(assets_dir, report, problems):
    log('\n[repos] Pic2Word and LinCIR code at the pinned commits')
    for key in ('pic2word', 'lincir'):
        try:
            fetch_github_archive(assets_dir, key)
        except Exception as e:                                # noqa: BLE001
            problems.append(f'{key}: code download failed: {e}')


def step_searle(assets_dir, report, problems, max_bytes):
    log('\n[searle] SEARLE code in the torch.hub cache, and its ViT-L/14 weights')
    try:
        fetch_github_archive(assets_dir, 'searle')
    except Exception as e:                                    # noqa: BLE001
        problems.append(f'searle: code download failed: {e}')
    w = SEARLE_WEIGHTS
    dest = Path(assets_dir) / w['file']
    _, bad = verify_file(dest, w['bytes'], 'sha256', w['sha256'], 'searle')
    if bad:
        if max_bytes is not None and w['bytes'] > max_bytes:
            problems.append(f'searle: weights skipped ({human(w["bytes"])} > --max_download_mb)')
            return
        log(f'  downloading {w["url"]} ({human(w["bytes"])})')
        try:
            download(w['url'], dest, expected_bytes=w['bytes'])
        except Exception as e:                                # noqa: BLE001
            problems.append(f'searle: weights download failed: {e}')
            return
        _, bad = verify_file(dest, w['bytes'], 'sha256', w['sha256'], 'searle')
        problems.extend(bad)
    log(f'  weights {dest}: {"ok" if not bad else bad}')


def _hf_manual(hf_home, repo, commit, rel, nbytes, kind, dig):
    """Fallback: write one file into the hub cache layout huggingface_hub uses (blob named by its etag)."""
    storage = hf_storage(hf_home, repo)
    blob = storage / 'blobs' / dig
    download(f'https://huggingface.co/{repo}/resolve/{commit}/{rel}', blob, expected_bytes=nbytes,
             report_every=200_000_000 if nbytes > 200_000_000 else 0)
    link = storage / 'snapshots' / commit / rel
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(os.path.relpath(blob, link.parent), link)


def mark_hub_cache(hf_home):
    """transformers 4.35 migrates a non-empty hub cache without version.txt (and, offline, warns that every
    from_pretrained will fail). This cache has the current layout, so mark it as transformers itself would."""
    v = Path(hf_home) / 'hub' / 'version.txt'
    if not v.is_file():
        v.parent.mkdir(parents=True, exist_ok=True)
        v.write_text('1')


def step_lincir_hf(assets_dir, report, problems, max_bytes):
    log('\n[lincir_hf] LinCIR weights and Hugging Face files into HF_HOME')
    hf_home = Path(assets_dir) / HF_HOME
    mark_hub_cache(hf_home)
    env_problem = apply_hf_env(hf_home, offline=False)
    if env_problem:
        problems.extend(f'lincir: {p}' for p in env_problem)
        return
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:                                    # noqa: BLE001
        problems.append(f'lincir: huggingface_hub not importable ({e}): run  {PIP_LINE}')
        return
    for repo, commit, rel, nbytes, kind, dig in HF_FILES:
        storage = hf_storage(hf_home, repo)
        target = storage / 'snapshots' / commit / rel
        _, bad = verify_file(target, nbytes, kind, dig, 'lincir')
        if bad:
            if max_bytes is not None and nbytes > max_bytes:
                problems.append(f'lincir: {repo}/{rel} skipped ({human(nbytes)} > --max_download_mb)')
                continue
            log(f'  {repo} @ {commit[:8]}: {rel} ({human(nbytes)})')
            err = None
            for attempt in range(1, 4):
                try:
                    hf_hub_download(repo, rel, revision=commit, resume_download=True)
                    err = None
                    break
                except Exception as e:                        # noqa: BLE001
                    err = e
                    log(f'    hf_hub_download attempt {attempt}/3 failed: {type(e).__name__}: {e}')
                    time.sleep(5 * attempt)
            if err is not None:
                log('    falling back to a direct download into the cache layout')
                try:
                    _hf_manual(hf_home, repo, commit, rel, nbytes, kind, dig)
                except Exception as e:                        # noqa: BLE001
                    problems.append(f'lincir: {repo}/{rel}: download failed: {e}')
                    continue
            _, bad = verify_file(target, nbytes, kind, dig, 'lincir')
            if bad:
                problems.extend(bad)
                continue
        ref = storage / 'refs' / 'main'
        if not ref.is_file() or ref.read_text().strip() != commit:
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(commit)                            # builders call from_pretrained / hf_hub_download on main
        log(f'  ok  {repo}/{rel}')


def pic2word_check_load(path):
    """The keys queries_pic2word reads. Loaded with weights_only=True (tensors and plain containers only); a plain
    load is used only when that refuses a global the non-executing scan already found harmless (SAFE_GLOBALS)."""
    import torch
    try:
        ck = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:                                         # noqa: BLE001
        if unsafe_globals(path):
            raise
        ck = torch.load(path, map_location='cpu')
    out = {'keys': sorted(ck.keys()) if isinstance(ck, dict) else str(type(ck))}
    if not isinstance(ck, dict) or 'state_dict' not in ck or 'state_dict_img2text' not in ck:
        raise RuntimeError(f'no state_dict / state_dict_img2text in {path}: keys {out["keys"]}')
    sd, sdi = ck['state_dict'], ck['state_dict_img2text']
    strip = lambda d: {k[len('module.'):] if k.startswith('module.') else k: v for k, v in d.items()}  # noqa: E731
    sd, sdi = strip(sd), strip(sdi)
    for k in ('visual.conv1.weight', 'token_embedding.weight', 'text_projection', 'logit_scale'):
        if k not in sd:
            raise RuntimeError(f'state_dict lacks {k}')
    if 'fc_out.weight' not in sdi:
        raise RuntimeError(f'state_dict_img2text lacks fc_out.weight: {sorted(sdi)[:6]}')
    out.update(n_clip_tensors=len(sd), img2text={k: list(v.shape) for k, v in sdi.items()},
               clip_dtype=str(sd['visual.conv1.weight'].dtype), epoch=ck.get('epoch'))
    return out


def gdown_fetch(file_id, dest):
    import inspect
    import gdown
    part = Path(str(dest) + '.gdown')
    if part.exists():
        part.unlink()
    kw = {}
    params = inspect.signature(gdown.download).parameters
    if 'timeout' in params and 'retries' in params:          # gdown >= 6.3
        kw.update(timeout=(30, 300), retries=3)
    out = gdown.download(id=file_id, output=str(part), quiet=False, **kw)   # gdown 6 raises on failure, 5 returns None
    if not out or not part.is_file() or part.stat().st_size < 100_000_000:
        size = part.stat().st_size if part.is_file() else 0
        if part.exists():
            part.unlink()
        raise RuntimeError(f'gdown returned {out!r} ({size} bytes)')
    os.replace(part, dest)


def step_pic2word(assets_dir, report, problems, max_bytes, source, given):
    log('\n[pic2word] Pic2Word checkpoint')
    pin = PIC2WORD_HF_MIRROR
    dest = Path(assets_dir) / PIC2WORD_CKPT
    dest.parent.mkdir(parents=True, exist_ok=True)
    prev = read_report(assets_dir).get('assets', {}).get('pic2word', {}).get('checkpoint', {})
    rec = {'path': str(dest), 'tried': []}

    def pinned():
        r, bad = verify_file(dest, pin['bytes'], 'sha256', pin['sha256'], 'pic2word')
        return not bad, r

    if given:
        given = Path(given).resolve()
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        os.symlink(given, dest)
        rec['source'] = f'--pic2word_ckpt {given}'
        log(f'  using {given} (symlinked as {dest})')
        ok, r = pinned()
        if not ok:
            problems.append(f'pic2word: --pic2word_ckpt {given} is not the pinned file ({r.get("bytes")} bytes, '
                            f'sha256 {str(r.get("sha256"))[:16]}...; pinned {pin["bytes"]} bytes, {pin["sha256"][:16]}...)')
    elif dest.is_file() and pinned()[0]:
        rec['source'] = prev.get('source') or 'present before this run'
        log(f'  present and verified (from {rec["source"]})')
    else:
        if dest.is_symlink() or dest.exists():
            log(f'  {dest} is not the pinned file; fetching it again')
            dest.unlink()
        if max_bytes is not None and pin['bytes'] > max_bytes:
            problems.append(f'pic2word: checkpoint skipped ({human(pin["bytes"])} > --max_download_mb)')
            report.setdefault('assets', {})['pic2word'] = {'checkpoint': rec}
            return
        for name, kind, ref, why in [s for s in PIC2WORD_SOURCES if source in ('auto', s[0])]:
            log(f'  trying {name}: {why}')
            try:
                if kind == 'drive':
                    gdown_fetch(ref, dest)
                else:
                    download(ref, dest, expected_bytes=pin['bytes'], report_every=200_000_000)
                ok, r = pinned()
                if not ok:
                    dest.unlink()                             # 1.7 GB; its size and checksum are kept in the report
                    raise RuntimeError(f'not the pinned file: {r.get("bytes")} bytes, sha256 {r.get("sha256")}')
                rec['source'] = f'{name}: {ref}'
                rec['tried'].append({'source': name, 'ok': True})
                break
            except Exception as e:                            # noqa: BLE001
                rec['tried'].append({'source': name, 'ok': False, 'error': f'{type(e).__name__}: {e}'[:400]})
                log(f'    failed: {type(e).__name__}: {str(e)[:300]}')
        if not dest.is_file():
            problems.append('pic2word: no source delivered the pinned checkpoint: ' + json.dumps(rec['tried']) +
                            '; download pic2word_model.pt by hand and rerun with --pic2word_ckpt <path>')
            report.setdefault('assets', {})['pic2word'] = {'checkpoint': rec}
            return
    rec['bytes'] = dest.stat().st_size
    rec['sha256'] = sha256_file(dest)
    rec['same_as_hf_mirror'] = rec['sha256'] == pin['sha256']
    try:
        rec['unsafe_pickle_globals'] = unsafe_globals(dest)
    except Exception as e:                                    # noqa: BLE001
        rec['unsafe_pickle_globals'] = [f'<scan failed: {e}>']
    if rec['unsafe_pickle_globals']:
        problems.append(f'pic2word: {dest} references unexpected pickle globals {rec["unsafe_pickle_globals"]}')
    else:
        try:
            rec['load_check'] = pic2word_check_load(dest)
        except Exception as e:                                # noqa: BLE001
            problems.append(f'pic2word: {dest} does not load as a Pic2Word checkpoint: {e}')
    report.setdefault('assets', {})['pic2word'] = {'checkpoint': rec}
    log(f'  {dest}')
    log(f'    source  {rec["source"]}')
    log(f'    bytes   {rec["bytes"]:,}   (June copy: {PIC2WORD_JUNE_DU} by du -h; HF mirror '
        f'{PIC2WORD_HF_MIRROR["bytes"]:,})')
    log(f'    sha256  {rec["sha256"]}   ({"identical to" if rec["same_as_hf_mirror"] else "DIFFERENT from"} the HF '
        f'mirror flashingtt/pic2word {PIC2WORD_HF_MIRROR["sha256"][:12]}...)')
    if rec.get('load_check'):
        lc = rec['load_check']
        log(f'    loads: keys {lc["keys"]}, {lc["n_clip_tensors"]} CLIP tensors ({lc["clip_dtype"]}), '
            f'img2text {lc["img2text"]}')


def step_clip(report, problems, max_bytes):
    log('\n[clip] OpenAI CLIP ViT-L/14 weights (~/.cache/clip)')
    path = Path(os.path.expanduser(CLIP_WEIGHTS['path']))
    ok = path.is_file() and path.stat().st_size == CLIP_WEIGHTS['bytes']
    if not ok:
        if max_bytes is not None and CLIP_WEIGHTS['bytes'] > max_bytes:
            problems.append(f'clip: {path} missing; download skipped ({human(CLIP_WEIGHTS["bytes"])} > --max_download_mb)')
            return
        try:
            import clip
            clip.load('ViT-L/14', device='cpu')          # downloads to ~/.cache/clip and verifies the checksum
        except Exception as e:                                # noqa: BLE001
            problems.append(f'CLIP ViT-L/14 could not be loaded: {e}')
            return
    rec, bad = verify_file(path, CLIP_WEIGHTS['bytes'], 'sha256', CLIP_WEIGHTS['sha256'], 'clip')
    problems.extend(bad)
    report['clip'] = rec
    log(f'  {path}: {"ok" if not bad else bad}')


OFFLINE_CHECK = r'''
import json, sys, torch
sys.path.insert(0, sys.argv[1])
import models, encode_with_pseudo_tokens
from huggingface_hub import hf_hub_download
class A:
    clip_model_name = 'large'
    mixed_precision = 'fp32'
    cache_dir = None
vision, preprocess, text, tok = models.build_text_encoder(A())
phi = models.Phi(input_dim=text.config.projection_dim, hidden_dim=text.config.projection_dim * 4,
                 output_dim=text.config.hidden_size, dropout=0)
phi.load_state_dict(torch.load(hf_hub_download('navervision/zeroshot-cir-models', 'lincir_large.pt'),
                               map_location='cpu')['Phi'])
ids = tok(text='a photo of [$] that is red', return_tensors='pt', padding='max_length', truncation=True)['input_ids']
print(json.dumps({'dollar_id': tok.convert_tokens_to_ids('[$]'), 'has_49408': bool((ids == 49408).any()),
                  'context': int(ids.shape[1]), 'vision_dtype': str(vision.dtype), 'text_dtype': str(text.dtype),
                  'projection_dim': text.config.projection_dim, 'hidden_size': text.config.hidden_size,
                  'phi_params': sum(p.numel() for p in phi.parameters())}))
'''


def offline_lincir_check(assets_dir):
    """LinCIR's own build_text_encoder + Phi, in a fresh process with HF_HUB_OFFLINE=1: proves the cache is complete."""
    env = dict(os.environ)
    env.update(hf_env(Path(assets_dir) / HF_HOME, offline=True))
    r = subprocess.run([sys.executable, '-c', OFFLINE_CHECK, str(Path(assets_dir) / REPOS['lincir']['dir'])],
                       capture_output=True, text=True, env=env, timeout=1800)
    if r.returncode:
        raise RuntimeError('offline LinCIR load failed: ' + ' | '.join(r.stderr.strip().splitlines()[-6:]))
    out = json.loads(r.stdout.strip().splitlines()[-1])
    if out['dollar_id'] != 49408 or not out['has_49408'] or out['context'] != 77:
        raise RuntimeError(f'LinCIR tokenizer is not the one the builder expects: {out}')
    return out


def searle_hub_check(assets_dir):
    """The exact call of stickiness_compare.queries_searle, with the hub dir the run sets: must use the pinned copy."""
    import torch
    w = Path(assets_dir) / SEARLE_WEIGHTS['file']
    before = (w.stat().st_mtime_ns, w.stat().st_size)
    old = torch.hub.get_dir()
    err = io.StringIO()
    torch.hub.set_dir(str(Path(assets_dir) / TORCH_HUB))
    try:
        with contextlib.redirect_stderr(err):
            phi, enc = torch.hub.load('miccunifi/SEARLE', 'searle', source='github', backbone='ViT-L/14')
    finally:
        torch.hub.set_dir(old)
    msg = err.getvalue()
    repo_dir = str(Path(assets_dir) / REPOS['searle']['dir'])
    n = sum(p.numel() for p in phi.parameters())
    if 'Using cache found in ' + repo_dir not in msg or 'Downloading' in msg:
        raise RuntimeError(f'torch.hub did not use the pinned copy {repo_dir}: {msg.strip()[:400]}')
    if (w.stat().st_mtime_ns, w.stat().st_size) != before:
        raise RuntimeError('torch.hub re-downloaded the SEARLE weights')
    if n != SEARLE_WEIGHTS['phi_params']:
        raise RuntimeError(f'SEARLE phi has {n} parameters, expected {SEARLE_WEIGHTS["phi_params"]}')
    used = next(line for line in msg.splitlines() if line.startswith('Using cache found in '))
    return {'hub_message': used, 'phi_params': n, 'encode_fn': f'{enc.__module__}.{enc.__name__}'}


def step_verify(assets_dir, report, problems, offline_check=True):
    log('\n[verify] Every asset against its pin')
    pic = report.get('assets', {}).get('pic2word', {}).get('checkpoint')
    records, per_method = verify_assets(assets_dir, pic2word_prepared=pic)
    report['assets'] = records
    if pic:                                                   # keep what step_pic2word learned (source, load check)
        records['pic2word']['checkpoint'].update({k: v for k, v in pic.items() if k not in records['pic2word']
                                                  ['checkpoint']})
    for m, ps in per_method.items():
        problems.extend(p for p in ps if p not in problems)
        log(f'  {m:9s} {"ok" if not ps else f"{len(ps)} problem(s)"}')
    if not per_method['searle']:
        try:
            report['searle_hub_check'] = searle_hub_check(assets_dir)
            log(f'  searle    torch.hub.load(\'miccunifi/SEARLE\', ...) -> {report["searle_hub_check"]["hub_message"]}')
        except Exception as e:                                # noqa: BLE001
            problems.append(f'searle: {e}')
    if not per_method['lincir'] and offline_check:
        mark_hub_cache(Path(assets_dir) / HF_HOME)
        log('  lincir    loading LinCIR offline in a fresh process (CLIP-L, ~1 min) ...')
        try:
            report['lincir_offline_check'] = offline_lincir_check(assets_dir)
            log(f'  lincir    offline load ok: {report["lincir_offline_check"]}')
        except Exception as e:                                # noqa: BLE001
            problems.append(f'lincir: {e}')


# ---------------------------------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description='Fetch and verify the assets of the external CIRCO methods.')
    p.add_argument('--assets_dir', type=Path, default=DEFAULT_ASSETS)
    p.add_argument('--print_pip', action='store_true', help='print the pip line and exit')
    p.add_argument('--steps', nargs='+', choices=STEPS, default=list(STEPS))
    p.add_argument('--pic2word_source', choices=('auto',) + tuple(s[0] for s in PIC2WORD_SOURCES), default='auto')
    p.add_argument('--pic2word_ckpt', type=Path, help='use this pic2word_model.pt instead of downloading one')
    p.add_argument('--max_download_mb', type=float, default=None, help='skip any single download larger than this')
    p.add_argument('--skip_offline_check', action='store_true', help='verify: do not load LinCIR offline')
    args = p.parse_args()
    if args.print_pip:
        print(PIP_LINE)
        return
    assets = args.assets_dir.resolve()
    max_bytes = None if args.max_download_mb is None else args.max_download_mb * 1e6
    log(f'Preparing {assets} for evaluation/ref_kept_externals.py   {now()}   (pins resolved {RESOLVED})')
    assets.mkdir(parents=True, exist_ok=True)
    report = {'started_at': now(), 'assets_dir': str(assets), 'pins_resolved': RESOLVED, 'python': sys.version,
              'steps': args.steps}
    problems, notes = [], []
    free = free_gb(assets)
    log(f'  free space at {assets}: {free:.1f} GB (needs about 3.6 GB; plus 0.9 GB in ~/.cache/clip if missing)')
    if free < MIN_FREE_GB and any(s in args.steps for s in ('lincir_hf', 'pic2word')):
        problems.append(f'only {free:.1f} GB free at {assets}; need {MIN_FREE_GB} GB (use --assets_dir elsewhere)')
    if 'pip' in args.steps:
        step_pip(assets, report, problems)
    if 'repos' in args.steps:
        step_repos(assets, report, problems)
    if 'searle' in args.steps:
        step_searle(assets, report, problems, max_bytes)
    if 'lincir_hf' in args.steps:
        step_lincir_hf(assets, report, problems, max_bytes)
    if 'pic2word' in args.steps:
        step_pic2word(assets, report, problems, max_bytes, args.pic2word_source, args.pic2word_ckpt)
    if 'clip' in args.steps:
        step_clip(report, problems, max_bytes)
    if 'verify' in args.steps:
        step_verify(assets, report, problems, offline_check=not args.skip_offline_check)
    else:
        notes.append('verify step not run')
    report.update(finished_at=now(), problems=problems, notes=notes)
    out = assets / 'prepared_ref_kept.json'
    write_json(out, report)

    log('\n' + '=' * 78)
    log('Verification report')
    log('=' * 78)
    a = report.get('assets', {})
    for key in ('pic2word', 'lincir', 'searle'):
        r = a.get(key, {})
        log(f'  {key:9s} code {REPOS[key]["github"]} @ {REPOS[key]["commit"][:8]} ({REPOS[key]["commit_date"]}): '
            f'{r.get("files_verified", 0)}/{len(REPOS[key]["blobs"])} files verified')
    def size(r):
        return 'MISSING' if r.get('bytes') is None else f'{r["bytes"]:,} bytes'

    if 'searle' in a:
        wr = a['searle'].get('weights', {})
        log(f'  searle    weights {wr.get("path")}: {size(wr)}, sha256 {wr.get("sha256", "-")}')
    for f in a.get('lincir', {}).get('hf_files', []):
        dk = 'sha256' if 'sha256_expected' in f else 'gitsha1'
        log(f'  lincir    {f["repo"]}/{f["file"]} @ {f["commit"][:8]}: {size(f)}, {dk} {f.get(dk, "-")}')
    ck = a.get('pic2word', {}).get('checkpoint', {})
    if ck:
        log(f'  pic2word  checkpoint {ck.get("path")}: {size(ck)}, sha256 {ck.get("sha256", "-")}, '
            f'source {ck.get("source")}')
    if report.get('clip'):
        log(f'  clip      {report["clip"]["path"]}: {report["clip"].get("bytes")} bytes')
    log(f'  pip line  {PIP_LINE}')
    for n in notes:
        log(f'note: {n}')
    if problems:
        log(f'\nNOT READY ({len(problems)} problems):')
        for pr in problems:
            log(f'  - {pr}')
        log(f'Report: {out}')
        sys.exit(1)
    log(f'\nREADY. Report: {out}')


if __name__ == '__main__':
    main()

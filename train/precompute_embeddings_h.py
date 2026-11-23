"""
precompute_embeddings_h.py   (parallel-decode version)

One-time precompute of frozen ViT-H image embeddings (v_raw) for the cc12m
training shards, so train_h.py can train the mapper off a cache instead of
re-encoding 1024 images through the 986M-param frozen ViT-H on every step.

What is cached
--------------
For every training image we store
    v_raw = CLIPWrapper.encoder_image_unnorm(img)
i.e. the UNNORMALISED ViT-H visual features (1024-dim). This is exactly the
tensor the training step consumes in Pic2Word.forward: it is both the mapper
input and, after F.normalize, the contrastive target. Nothing else from the
image side is needed for the core Pic2Word loss (use_captions is False for
this run, so captions are not cached).

Why this version is faster
--------------------------
A single .tar shard is read by exactly ONE DataLoader worker, so a
one-shard-per-loader design leaves the other workers idle and becomes
JPEG-decode-bound (~50s/shard) while the GPU could encode it in ~20s.

Here we feed --shards_per_group shards (default = num_workers) into ONE
loader, so all workers decode different shards in parallel. WebDataset splits
shards across workers disjointly (verified: no duplication), so the total set
is unchanged. Each sample carries its __url__, and after the (large, GPU-fed)
encode we scatter rows back into the correct per-shard buffer. Output format
is byte-for-byte the same as the serial version, so the two are interchangeable
and a partially-finished cache resumes cleanly.

Output layout (under --out_dir)
--------------------------------
    vith_emb_0000.npy        float16/float32 array  [N_0, 1024]
    vith_emb_0000.keys.json  list[str] of WebDataset __key__, length N_0
    ...                      (one pair per shard)
    manifest.json            run metadata + per-shard image counts

The .npy rows and the .keys.json entries are 1:1 and in on-disk shard (tar)
order. Shuffling is OFF here on purpose; shuffling happens at train time.

Run (from codes/)
-----------------
    cd /notebooks/experiment/codes
    PYTHONPATH=. python train/precompute_embeddings_h.py \
        --out_dir /notebooks/experiment/vith_emb_cache \
        --batch_size 2048

Notes
-----
* Inference only (no autograd graph), so --batch_size can be large (2048-4096).
* Resumable: a shard whose .npy already exists is skipped unless --overwrite;
  a group with no work left is skipped without building a loader.
* clip_model / clip_backend / clip_pretrained come from config_h defaults
  (ViT-H-14 / open_clip / laion2b_s32b_b79k), matching the training run.
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
from torch.cuda.amp import autocast
from PIL import ImageFile

from config.config_h import Config, load_from_yaml
from model.model_h import Pic2Word

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("precompute_h")

_NP_DTYPE = {"float16": np.float16, "float32": np.float32}
_TORCH_DTYPE = {"float16": torch.float16, "float32": torch.float32}

_SHARD_RE = re.compile(r"cc12m-train-(\d{4})\.tar")


# ---------------------------------------------------------------------------
# Shard URLs  (mirrors dataset.py exactly: local-first, HuggingFace fallback)
# ---------------------------------------------------------------------------

def _make_url(shard_idx: int, train_data_dir: str, hf_base_url: str,
              hf_token: str = None) -> str:
    local_path = Path(train_data_dir) / f"cc12m-train-{shard_idx:04d}.tar"
    if local_path.exists():
        return str(local_path)
    logger.warning(f"LOCAL MISS: {local_path} not found — falling back to HuggingFace download!")
    file_url = f"{hf_base_url}/cc12m-train-{shard_idx:04d}.tar"
    if hf_token:
        return f"pipe:curl -s -L -H 'Authorization: Bearer {hf_token}' {file_url}"
    return f"pipe:curl -s -L {file_url}"


def _shard_idx_from_url(url: str):
    m = _SHARD_RE.search(url)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# WebDataset pipeline  (same validity rule + preprocess as training, but the
# __url__ and __key__ are kept and shuffle is OFF). Multiple shards per loader
# so workers decode in parallel.
# ---------------------------------------------------------------------------

def _is_valid(data: dict) -> bool:
    return "jpg" in data and data["jpg"] is not None


def _make_transform(clip_preprocess):
    def transform(data: dict):
        try:
            img = clip_preprocess(data["jpg"].convert("RGB"))
            return (data["__url__"], data["__key__"], img)
        except Exception:
            return None
    return transform


def _collate(samples):
    urls = [s[0] for s in samples]
    keys = [s[1] for s in samples]
    imgs = torch.stack([s[2] for s in samples], dim=0)
    return urls, keys, imgs


def _build_group_loader(shard_urls, clip_preprocess, batch_size, num_workers):
    transform = _make_transform(clip_preprocess)
    ds = (
        wds.WebDataset(
            shard_urls,
            shardshuffle=False,
            nodesplitter=wds.shardlists.single_node_only,
            empty_check=False,
            handler=wds.warn_and_continue,
        )
        .decode("pil")
        .select(_is_valid)
        .map(transform)
        .select(lambda x: x is not None)
        .batched(batch_size, collation_fn=_collate, partial=True)
    )
    return wds.WebLoader(ds, batch_size=None, num_workers=num_workers, pin_memory=True)


# ---------------------------------------------------------------------------
# Encode a group of shards, scattering rows back to their shard of origin
# ---------------------------------------------------------------------------

def _encode_group(model, shard_urls, todo_shards, clip_preprocess, batch_size,
                  num_workers, device, amp_enabled, torch_dtype, embed_dim):
    # Pre-seed buffers so a shard with zero valid samples still gets written.
    buffers = {si: {"keys": [], "embs": []} for si in todo_shards}
    n_batches = 0
    loader = _build_group_loader(shard_urls, clip_preprocess, batch_size, num_workers)
    for urls, keys, imgs in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.no_grad(), autocast(enabled=amp_enabled):
            v = model.clip.encoder_image_unnorm(imgs)        # [B, embed_dim], unnormalised
        v = v.to(torch_dtype).cpu().numpy()
        n_batches += 1
        # Scatter this batch's rows into per-shard buffers. Order within a shard
        # is preserved: a shard is read by one worker, in tar order, and idx is
        # ascending, so a shard's rows accrue in tar order across batches.
        for u in dict.fromkeys(urls):                        # unique urls, stable order
            si = _shard_idx_from_url(u)
            if si not in buffers:                            # safety; should not happen
                buffers[si] = {"keys": [], "embs": []}
            idx = [i for i, x in enumerate(urls) if x == u]
            buffers[si]["keys"].extend(keys[i] for i in idx)
            buffers[si]["embs"].append(v[idx])
    out = {}
    for si, buf in buffers.items():
        if buf["embs"]:
            emb = np.concatenate(buf["embs"], axis=0)
        else:
            emb = np.empty((0, embed_dim), dtype=_NP_DTYPE["float16"])
        out[si] = (buf["keys"], emb)
    return out, n_batches


# ---------------------------------------------------------------------------
# Save / resume / manifest
# ---------------------------------------------------------------------------

def _save_shard(out_dir: Path, shard_idx: int, keys, emb):
    np.save(out_dir / f"vith_emb_{shard_idx:04d}.npy", emb)
    (out_dir / f"vith_emb_{shard_idx:04d}.keys.json").write_text(json.dumps(keys))


def _shard_done(out_dir: Path, shard_idx: int) -> bool:
    return (out_dir / f"vith_emb_{shard_idx:04d}.npy").exists() and \
           (out_dir / f"vith_emb_{shard_idx:04d}.keys.json").exists()


def _load_manifest(out_dir: Path) -> dict:
    p = out_dir / "manifest.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _write_manifest(out_dir: Path, cfg: Config, dtype_str: str, embed_dim, shard_counts: dict):
    manifest = {
        "clip_model": cfg.clip_model,
        "clip_backend": cfg.clip_backend,
        "clip_pretrained": cfg.clip_pretrained,
        "embed_dim": int(embed_dim) if embed_dim else None,
        "dtype": dtype_str,
        "normalized": False,
        "source": "encoder_image_unnorm (unnormalised ViT-H visual features)",
        "file_pattern": "vith_emb_{idx:04d}.npy  +  vith_emb_{idx:04d}.keys.json",
        "shards": {f"{k:04d}": int(v) for k, v in sorted(shard_counts.items())},
        "total_images": int(sum(shard_counts.values())),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Precompute frozen ViT-H image embeddings (v_raw) for the cc12m shards "
                    "(parallel decode across workers).")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Directory to write the embedding cache into (created if absent). REQUIRED.")
    p.add_argument("--config", type=str, default=None,
                   help="Optional config_h yaml; otherwise config_h defaults are used.")
    # shard range
    p.add_argument("--start", type=int, default=0,
                   help="First train shard index (inclusive). Default 0.")
    p.add_argument("--end", type=int, default=None,
                   help="Last train shard index (EXCLUSIVE). Default = cfg.train_shards (793).")
    p.add_argument("--also_val", dest="also_val", action="store_true", default=True,
                   help="Also cache the held-out val shard (cfg.val_shard). On by default.")
    p.add_argument("--no_val", dest="also_val", action="store_false",
                   help="Skip the val shard.")
    # runtime
    p.add_argument("--batch_size", type=int, default=None,
                   help="Encode batch size. Default = cfg.batch_size. Inference only, so "
                        "2048-4096 is fine and faster.")
    p.add_argument("--num_workers", type=int, default=None,
                   help="Decode workers. Default = cfg.num_workers. With grouping, all of them "
                        "decode in parallel.")
    p.add_argument("--shards_per_group", type=int, default=None,
                   help="Shards fed to one loader at a time. Default = num_workers (1 shard per "
                        "worker, fully parallel decode). Larger uses more RAM for buffers.")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"],
                   help="Stored embedding precision. float16 (default) matches AMP training and "
                        "halves disk; float32 is bit-faithful to a non-AMP encode.")
    p.add_argument("--device", type=str, default=None, help="Default = cfg.device.")
    p.add_argument("--train_data_dir", type=str, default=None,
                   help="Override cfg.train_data_dir if the shards live elsewhere.")
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="Re-encode shards even if their .npy already exists.")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = load_from_yaml(args.config) if args.config else Config()
    for k in ("batch_size", "num_workers", "device", "train_data_dir", "hf_token"):
        v = getattr(args, k)
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)

    device      = cfg.device
    amp_enabled = bool(cfg.amp) and str(device).startswith("cuda")
    torch_dtype = _TORCH_DTYPE[args.dtype]
    spg         = args.shards_per_group if args.shards_per_group is not None else max(1, cfg.num_workers)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = args.start
    end   = args.end if args.end is not None else cfg.train_shards
    shard_ids = list(range(start, end))
    if args.also_val:
        shard_ids.append(cfg.val_shard)

    logger.info("=" * 60)
    logger.info("  ViT-H embedding precompute  (parallel decode)")
    logger.info("=" * 60)
    logger.info(f"  clip_model      : {cfg.clip_model}  ({cfg.clip_backend}, {cfg.clip_pretrained})")
    logger.info(f"  out_dir         : {out_dir}")
    logger.info(f"  train shards    : {start:04d} → {end - 1:04d}  ({end - start} shards)")
    logger.info(f"  also val shard  : {cfg.val_shard:04d}" if args.also_val else "  also val shard  : no")
    logger.info(f"  batch_size      : {cfg.batch_size}")
    logger.info(f"  num_workers     : {cfg.num_workers}")
    logger.info(f"  shards_per_group: {spg}")
    logger.info(f"  store dtype     : {args.dtype}")
    logger.info(f"  device / amp    : {device} / {amp_enabled}")
    logger.info(f"  train_data_dir  : {cfg.train_data_dir}")

    model = Pic2Word(cfg).to(device)
    model.eval()
    clip_preprocess = model.clip.preprocess
    embed_dim = model.clip.visual_dim
    logger.info(f"  embed_dim       : {embed_dim}")
    logger.info("=" * 60)

    manifest = _load_manifest(out_dir)
    shard_counts = {int(k): int(v) for k, v in manifest.get("shards", {}).items()}

    groups = list(_chunks(shard_ids, spg))
    t0 = time.time()
    for gi, group in enumerate(groups, 1):
        todo = [si for si in group if args.overwrite or not _shard_done(out_dir, si)]
        done = [si for si in group if si not in todo]
        for si in done:                                       # keep counts current for skips
            keys = json.loads((out_dir / f"vith_emb_{si:04d}.keys.json").read_text())
            shard_counts[si] = len(keys)
        gtag = f"[group {gi}/{len(groups)}] shards {group[0]:04d}-{group[-1]:04d}"
        if not todo:
            logger.info(f"{gtag}: all cached — skip")
            continue

        urls = [_make_url(si, cfg.train_data_dir, cfg.hf_base_url, cfg.hf_token) for si in todo]
        t_s = time.time()
        out, n_batches = _encode_group(
            model, urls, todo, clip_preprocess, cfg.batch_size, cfg.num_workers,
            device, amp_enabled, torch_dtype, embed_dim)

        n_imgs = 0
        for si, (keys, emb) in sorted(out.items()):
            _save_shard(out_dir, si, keys, emb)
            shard_counts[si] = len(keys)
            n_imgs += len(keys)
        dt = time.time() - t_s
        _write_manifest(out_dir, cfg, args.dtype, embed_dim, shard_counts)
        rate = n_imgs / dt if dt > 0 else 0.0
        logger.info(f"{gtag}: {len(todo)} shards | {n_imgs:>6d} imgs | {n_batches} batches | "
                    f"{dt:.1f}s | {rate:.0f} img/s")

    manifest = _write_manifest(out_dir, cfg, args.dtype, embed_dim, shard_counts)
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"  DONE. {manifest['total_images']:,} images across "
                f"{len(shard_counts)} shards in {elapsed/60:.1f} min.")
    logger.info(f"  cache: {out_dir}")
    logger.info(f"  dim={embed_dim}  dtype={args.dtype}  normalized=False")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
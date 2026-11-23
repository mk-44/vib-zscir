"""
train/generate_vision_encoder_l_14_cache.py

One-time precompute of frozen ViT-L/14 image embeddings (v_raw) for the cc12m
training shards, saved as a SINGLE .pt file. train.py can then train the mapper
off the cache (--emb_cache_pt / cached=True) instead of re-encoding every image
through the frozen ViT-L visual tower on every step.

What is cached
--------------
For every valid training image:
    v_raw = CLIPWrapper.encoder_image_unnorm(img)   # [768], UNNORMALIZED
stored as fp16. This is exactly the tensor Pic2Word.forward consumes: it is the
mapper input AND, after F.normalize, the contrastive target. Because the task is
image-self-supervised (use_captions=False), each cached row is fully
self-contained -- no captions or external labels are needed, and rows may be
shuffled freely at train time. The model upcasts fp16 -> fp32 on the cached path
so numerics match the live path.

The encode is run under autocast (amp), matching the live training regime, so
the cached v_raw is the same tensor the live step would have produced.

Output (single file)
--------------------
    torch.save({
        "v"          : HalfTensor[N, 768],   # encounter order (rows shuffled at train time)
        "keys"       : list[str] length N,   # WebDataset __key__, 1:1 with rows (for verification)
        "clip_model" : "ViT-L/14",
        "dim"        : 768,
        "normalized" : False,
        "n"          : N,
    }, out_path)

Run (from codes/)
-----------------
    cd /notebooks/experiment/codes
    PYTHONPATH=/notebooks/experiment/codes python train/generate_vision_encoder_l_14_cache.py \
        --out /notebooks/experiment/vitl_vraw_cache.pt \
        --batch_size 1024

Notes
-----
* Inference only (no autograd), so --batch_size can be raised (1024-2048).
* Shuffle is OFF on purpose: shuffling happens at train time. keys[i] pairs with
  v[i] regardless of worker interleaving, because pairs are appended together.
* Local-first shard resolution mirrors dataset.py. A missing-local shard is
  warned and skipped (run on the fully-local shard set the live run reported).
"""

import argparse
import logging
import time
from pathlib import Path

import torch
import webdataset as wds
from torch.cuda.amp import autocast
from PIL import ImageFile
import clip

from config.config import load_from_yaml, Config
from model.model import CLIPWrapper

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("gen_vitl_cache")


# ---------------------------------------------------------------------------
# Shard URLs (local-first; mirrors dataset.py). Missing-local -> warn + skip.
# ---------------------------------------------------------------------------

def _local_shard_urls(cfg: Config) -> list:
    urls, missing = [], []
    for i in range(cfg.train_shards):
        p = Path(cfg.train_data_dir) / f"cc12m-train-{i:04d}.tar"
        if p.exists():
            urls.append(str(p))
        else:
            missing.append(i)
    if missing:
        logger.warning(f"{len(missing)} shard(s) NOT local and will be SKIPPED "
                       f"(first few: {missing[:5]}). Cache will cover local shards only.")
    return urls


# ---------------------------------------------------------------------------
# WebDataset pipeline: same preprocess + validity rule as training, shuffle OFF,
# __key__ kept. Multiple workers decode different shards in parallel.
# ---------------------------------------------------------------------------

def _is_valid(d: dict) -> bool:
    return "jpg" in d and d["jpg"] is not None


def _make_transform(preprocess):
    def transform(d: dict):
        try:
            return (d["__key__"], preprocess(d["jpg"].convert("RGB")))
        except Exception:
            return None
    return transform


def _collate(samples):
    keys = [s[0] for s in samples]
    imgs = torch.stack([s[1] for s in samples], dim=0)
    return keys, imgs


def _build_loader(urls, preprocess, batch_size, num_workers):
    transform = _make_transform(preprocess)
    ds = (
        wds.WebDataset(
            urls,
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
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",         type=str, required=True,
                    help="Output .pt path, e.g. /notebooks/experiment/vitl_vraw_cache.pt")
    ap.add_argument("--config",      type=str, default="config/config.yaml")
    ap.add_argument("--batch_size",  type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    cfg = load_from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Same preprocess the training loader uses.
    _, preprocess = clip.load(cfg.clip_model, device="cpu")

    clipw = CLIPWrapper(model_name=cfg.clip_model, device=device)
    clipw.eval()
    dim = clipw.visual_dim
    logger.info(f"clip_model={cfg.clip_model} | visual_dim={dim} | device={device} "
                f"| batch_size={args.batch_size} | workers={args.num_workers}")

    urls = _local_shard_urls(cfg)
    logger.info(f"{len(urls)} local shard(s) -> encoding all valid images")
    loader = _build_loader(urls, preprocess, args.batch_size, args.num_workers)

    embs, keys = [], []
    n = 0
    t0 = time.time()
    for bi, (bkeys, imgs) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        with torch.no_grad(), autocast(enabled=True):
            v = clipw.encoder_image_unnorm(imgs)        # [B, dim], UNNORMALIZED
        embs.append(v.half().cpu())                     # store fp16
        keys.extend(bkeys)
        n += len(bkeys)
        if bi % 50 == 0:
            rate = n / max(1e-9, time.time() - t0)
            logger.info(f"batch {bi:6d} | {n:>9,} imgs | {rate:7.0f} img/s")

    if n == 0:
        raise RuntimeError("No images encoded -- check shard paths / preprocess.")

    V = torch.cat(embs, dim=0)                          # [N, dim] fp16
    del embs
    assert V.shape[0] == len(keys) == n, \
        f"row/key/count mismatch: V={V.shape[0]} keys={len(keys)} n={n}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "v":          V,
            "keys":       keys,
            "clip_model": cfg.clip_model,
            "dim":        int(dim),
            "normalized": False,
            "n":          int(n),
        },
        out,
    )
    gb = V.numel() * 2 / 1e9
    logger.info(f"DONE: saved v_raw cache {tuple(V.shape)} fp16 (~{gb:.1f} GB) -> {out}")
    logger.info(f"elapsed: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
"""
dataset.py
"""
import logging
from pathlib import Path
from typing import Callable, Optional, Tuple
import torch
import webdataset as wds
from PIL import ImageFile

from config.config import Config

logger = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _hf_url(shard_idx     : int,
            hf_base_url    : str,
            hf_token       : Optional[str] = None) -> str:
    file_url = f"{hf_base_url}/cc12m-train-{shard_idx:04d}.tar"
    if hf_token:
        return f"pipe:curl -s -L -H 'Authorization: Bearer {hf_token}' {file_url}"
    return f"pipe:curl -s -L {file_url}"


def _build_shard_urls(start          : int,
                      end            : int,
                      train_data_dir : str,
                      hf_base_url    : str,
                      hf_token       : Optional[str] = None) -> Tuple[list, list]:
    """
    Build shard URLs for the half-open range [start, end).

    A shard that exists locally resolves to its local path; one that does not
    falls back to a HuggingFace streaming URL (and is recorded as a miss).

    Returns (urls, miss_ids) where miss_ids are the shard indices NOT found
    locally — i.e. the ones that will be streamed.
    """
    urls, miss_ids = [], []
    for i in range(start, end):
        local_path = Path(train_data_dir) / f"cc12m-train-{i:04d}.tar"
        if local_path.exists():
            urls.append(str(local_path))
        else:
            miss_ids.append(i)
            logger.warning(f"LOCAL MISS — streaming from HF: cc12m-train-{i:04d}.tar")
            urls.append(_hf_url(i, hf_base_url, hf_token))
    return urls, miss_ids


def _log_data_source(miss_ids       : list,
                     total          : int,
                     train_data_dir : str,
                     split_name     : str) -> bool:
    """
    Print a loud, summarized data-source banner so an unintended HF fallback
    cannot scroll past unnoticed at startup.

    Returns True if ANY shard for this split is being streamed from HF.
    """
    if not miss_ids:
        logger.info(f"DATA SOURCE OK [{split_name}]: all {total} shard(s) LOCAL ({train_data_dir})")
        return False
    n       = len(miss_ids)
    preview = miss_ids[:20]
    logger.warning("=" * 70)
    logger.warning(f"  !!! DATA SOURCE WARNING [{split_name}]: {n}/{total} shard(s) NOT LOCAL !!!")
    logger.warning(f"  These will be STREAMED from HuggingFace (slow, and samples may be")
    logger.warning(f"  dropped silently on connection errors via warn_and_continue).")
    logger.warning(f"  Missing shard ids: {preview}{' ...' if n > 20 else ''}")
    logger.warning(f"  train_data_dir = {train_data_dir}")
    logger.warning(f"  >>> If this is UNINTENDED, Ctrl-C NOW and fix train_data_dir. <<<")
    logger.warning("=" * 70)
    return True


def _is_valid(data: dict) -> bool:
    return "jpg" in data and data["jpg"] is not None


def _make_transform(clip_preprocess: Callable) -> Callable:
    def transform(data: dict):
        try:
            img = clip_preprocess(data["jpg"].convert("RGB"))
            txt = data.get("txt", "")
            return (img, txt)
        except Exception:
            return None
    return transform


# def _build_wds(shard_urls      : list,
#                clip_preprocess : Callable,
#                shuffle         : bool = True,
#                shuffle_buf     : int  = 2_000) -> wds.WebDataset:
#     transform    = _make_transform(clip_preprocess)
#     shardshuffle = len(shard_urls) if shuffle else False
#     dataset = (
#         wds.WebDataset(shard_urls,
#                        shardshuffle = shardshuffle,
#                        nodesplitter = wds.shardlists.single_node_only,
#                        empty_check = False, 
#                        handler = wds.warn_and_continue)
#         .decode("pil")
#         .select(_is_valid)
#         .map(transform)
#         .select(lambda x: x is not None)
#     )
#     if shuffle:
#         dataset = dataset.shuffle(shuffle_buf)
#     return dataset


def _build_wds(shard_urls      : list,
               clip_preprocess : Callable,
               shuffle         : bool = True,
               shuffle_buf     : int  = 2_000, 
               seed : Optional[int] = None) -> wds.WebDataset:
    transform    = _make_transform(clip_preprocess)
    shardshuffle = len(shard_urls) if shuffle else False
    dataset = (
        wds.WebDataset(shard_urls,
                       shardshuffle = shardshuffle,
                       detshuffle   = (seed is not None),
                       seed         = seed if seed is not None else -1,
                       nodesplitter = wds.shardlists.single_node_only,
                       empty_check = False, 
                       handler = wds.warn_and_continue)
        .decode("pil")
        .select(_is_valid)
        .map(transform)
        .select(lambda x: x is not None)
    )
    if shuffle:
        dataset = dataset.shuffle(shuffle_buf, seed=seed)
    return dataset
    

def build_loaders(cfg             : Config,
                  clip_preprocess : Callable,
                  pin_memory      : bool = True) -> Tuple:
    """
    Build train/val WebLoaders.

    Returns (train_loader, val_loader, streaming_active, n_missing):
      streaming_active : True if ANY shard (train or val) is being streamed
                         from HuggingFace instead of read locally.
      n_missing        : total number of shards being streamed.
    The training loop re-warns periodically while streaming_active is True.
    """

    # --- Train shards ---
    train_urls, train_miss = _build_shard_urls(
        start          = 0,
        end            = cfg.train_shards,
        train_data_dir = cfg.train_data_dir,
        hf_base_url    = cfg.hf_base_url,
        hf_token       = cfg.hf_token,
    )
    streaming_train = _log_data_source(train_miss, cfg.train_shards, cfg.train_data_dir, "train")

#     train_ds = _build_wds(
#         shard_urls      = train_urls,
#         clip_preprocess = clip_preprocess,
#         shuffle         = True,
#         shuffle_buf     = cfg.shuffle_buf,
#     ).batched(cfg.batch_size, partial=False)

#     train_loader = wds.WebLoader(
#         train_ds,
#         batch_size  = None,
#         num_workers = cfg.num_workers,
#         pin_memory  = pin_memory,
#     )

    train_ds = _build_wds(
        shard_urls      = train_urls,
        clip_preprocess = clip_preprocess,
        shuffle         = True,
        shuffle_buf     = cfg.shuffle_buf,
        seed            = cfg.seed,
    ).batched(cfg.batch_size, partial=False)

    def _worker_init(worker_id):
        if cfg.seed is not None:
            import random as _random, numpy as _np
            s = cfg.seed + worker_id
            _random.seed(s)
            _np.random.seed(s)
            torch.manual_seed(s)

    train_loader = wds.WebLoader(
        train_ds,
        batch_size     = None,
        num_workers    = cfg.num_workers,
        pin_memory     = pin_memory,
        worker_init_fn = _worker_init if cfg.seed is not None else None,
    )

    # --- Val shard ---
    val_urls, val_miss = _build_shard_urls(
        start          = cfg.val_shard,
        end            = cfg.val_shard + 1,
        train_data_dir = cfg.train_data_dir,
        hf_base_url    = cfg.hf_base_url,
        hf_token       = cfg.hf_token,
    )
    streaming_val = _log_data_source(val_miss, 1, cfg.train_data_dir, "val")

    val_ds = _build_wds(
        shard_urls      = val_urls,
        clip_preprocess = clip_preprocess,
        shuffle         = False,
    ).batched(cfg.batch_size, partial=True)

    val_loader = wds.WebLoader(
        val_ds,
        batch_size  = None,
        num_workers = 1,
        pin_memory  = pin_memory,
    )

    logger.info(f"Train shards : 0000 → {cfg.train_shards - 1:04d}  (~{cfg.approx_train_samples:,} images)")
    logger.info(f"Val shard    : {cfg.val_shard:04d}            (~{cfg.approx_val_samples:,} images)")
    logger.info(f"Batch size   : {cfg.batch_size}")
    logger.info(f"Workers      : train={cfg.num_workers}, val=1")

    streaming_active = streaming_train or streaming_val
    n_missing        = len(train_miss) + len(val_miss)
    return train_loader, val_loader, streaming_active, n_missing
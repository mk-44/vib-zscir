"""
dataset/emb_cache_dataset.py

Iterable dataset that streams precomputed frozen-ViT-H image embeddings
(v_raw, UNNORMALIZED, 1024-d, fp16) from the cache built by
train/precompute_embeddings_h.py, so the mapper can be trained without
re-encoding images through the ViT-H visual tower every step.

Cache layout (per shard XXXX in [0, train_shards)):
    <cache_dir>/vith_emb_XXXX.npy        float16 [N, D]   (tar order)
    <cache_dir>/vith_emb_XXXX.keys.json  list[str]        (1:1 with rows)
    <cache_dir>/manifest.json            metadata

Yields (per sample):
    (v_raw,)   where v_raw is a torch.float16 tensor of shape [D]

The 1-tuple keeps `batch[0]` semantics identical to the live image loader
(which yields (img, caption, ...)), so the training step is unchanged apart
from passing the cached tensor as v_raw.

The training step normalizes the contrastive target itself
(v = F.normalize(v_raw.detach())) and feeds raw v_raw to the mapper, exactly
as the live path does, so the recipe is unchanged. That is why the cache must
be normalized=False.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

logger = logging.getLogger(__name__)


def _shard_npy(cache_dir: Path, idx: int) -> Path:
    return cache_dir / f"vith_emb_{idx:04d}.npy"


class EmbCacheIterableDataset(IterableDataset):
    """
    Streams cached v_raw rows with shard-order + within-buffer shuffling that
    mirrors the WebDataset pipeline (shard shuffle + shuffle buffer).

    Shuffling
      - shard order is permuted every epoch  (seed = base_seed + epoch)
      - shards are processed in buffers of `shards_per_buffer`; all rows in a
        buffer are concatenated and permuted before being yielded, giving
        cross-shard negative diversity within each contrastive batch
    Worker sharding
      - shards are split disjointly across DataLoader workers (no duplication)

    Call set_epoch(e) before each epoch so the shuffle reseeds. Use
    persistent_workers=False so fresh workers pick up the new epoch.
    """

    def __init__(self,
                 cache_dir: str,
                 start_shard: int,
                 end_shard: int,                 # exclusive
                 shards_per_buffer: int = 8,
                 base_seed: int = 0):
        super().__init__()
        self.cache_dir         = Path(cache_dir)
        self.shard_ids         = list(range(start_shard, end_shard))
        self.shards_per_buffer = max(1, shards_per_buffer)
        self.base_seed         = base_seed
        self.epoch             = 0

        # Fail fast if any shard is missing, rather than mid-epoch.
        missing = [i for i in self.shard_ids if not _shard_npy(self.cache_dir, i).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} cache shard(s) missing under {self.cache_dir}; "
                f"first few: {missing[:5]}"
            )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _assigned_shards(self):
        """Per-epoch shuffled shard list, split disjointly across workers."""
        rng   = np.random.default_rng(self.base_seed + self.epoch)
        order = list(self.shard_ids)
        rng.shuffle(order)
        info = get_worker_info()
        if info is None:
            return order
        return order[info.id::info.num_workers]

    def __iter__(self):
        shards = self._assigned_shards()
        info   = get_worker_info()
        wid    = 0 if info is None else info.id
        rng    = np.random.default_rng(self.base_seed + self.epoch * 100003 + wid)

        for b in range(0, len(shards), self.shards_per_buffer):
            group = shards[b:b + self.shards_per_buffer]
            mats  = [np.asarray(np.load(_shard_npy(self.cache_dir, s), mmap_mode="r"))
                     for s in group]
            buf   = np.concatenate(mats, axis=0)          # fp16 [sum_N, D]
            for j in rng.permutation(buf.shape[0]):
                # copy the row so the underlying buffer can be freed each group
                yield (torch.from_numpy(np.array(buf[j], dtype=np.float16)),)


def _manifest_check(cache_dir: Path, cfg):
    mpath = cache_dir / "manifest.json"
    if not mpath.exists():
        logger.warning(f"No manifest.json in {cache_dir} - skipping consistency check.")
        return
    m = json.load(open(mpath))
    emb_dim, cm, norm, total = (m.get("embed_dim"), m.get("clip_model"),
                                m.get("normalized"), m.get("total"))
    logger.info(f"  cache manifest  : clip_model={cm} dim={emb_dim} "
                f"normalized={norm} total={total}")
    exp_dim = getattr(cfg, "visual_dim", emb_dim)
    if emb_dim is not None and exp_dim is not None and emb_dim != exp_dim:
        raise ValueError(
            f"Cache embed_dim {emb_dim} != cfg.visual_dim {exp_dim}. Wrong cache for this model."
        )
    if norm is True:
        raise ValueError(
            "Cache is normalized=True, but training expects raw unnormalized v_raw "
            "(the model normalizes the target itself). Rebuild with normalized=False."
        )
    exp_model = getattr(cfg, "clip_model", None)
    if cm is not None and exp_model is not None and cm != exp_model:
        logger.warning(f"  cache clip_model={cm} != cfg.clip_model={exp_model} "
                       f"- verify this is the intended cache.")


def build_emb_cache_loader(cfg):
    """
    Build the cached-embedding TRAIN loader. Validation is unchanged (live
    images via build_loaders), so only the train loader is returned here.

    Reads:
      cfg.emb_cache              cache dir
      cfg.train_shards           train shard count (uses shards 0 .. train_shards-1)
      cfg.batch_size, cfg.num_workers, cfg.seed
      cfg.emb_shards_per_buffer  (optional, default 8)

    Returns:
      (loader, dataset)  - keep `dataset` to call set_epoch() each epoch.
    """
    cache_dir = Path(cfg.emb_cache)
    if not cache_dir.exists():
        raise FileNotFoundError(f"--emb_cache dir not found: {cache_dir}")

    _manifest_check(cache_dir, cfg)

    spb = int(getattr(cfg, "emb_shards_per_buffer", 8) or 8)
    ds  = EmbCacheIterableDataset(
        cache_dir         = str(cache_dir),
        start_shard       = 0,
        end_shard         = cfg.train_shards,          # exclusive -> 0..train_shards-1
        shards_per_buffer = spb,
        base_seed         = cfg.seed if cfg.seed is not None else 0,
    )

    loader = DataLoader(
        ds,
        batch_size         = cfg.batch_size,
        num_workers        = cfg.num_workers,
        pin_memory         = True,
        drop_last          = True,
        persistent_workers = False,                    # so set_epoch reaches fresh workers
    )
    logger.info(f"  emb-cache train loader: shards 0..{cfg.train_shards - 1} "
                f"| buffer {spb} shards | bs {cfg.batch_size} | workers {cfg.num_workers}")
    return loader, ds
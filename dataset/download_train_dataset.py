"""
dataset/download_train_dataset.py — Download CC12M training dataset locally
"""

import argparse
import logging
import urllib.request
from pathlib import Path
from tqdm import tqdm
from config.config import Config, load_from_yaml

logging.basicConfig(
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt = "%H:%M:%S",
    level   = logging.INFO,
)
logger = logging.getLogger(__name__)

SAMPLES_PER_SHARD = 5041


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_shard(shard_idx : int,
                   out_path  : Path,
                   hf_base_url : str,
                   hf_token  : str = None):
    url      = f"{hf_base_url}/cc12m-train-{shard_idx:04d}.tar"
    filename = f"cc12m-train-{shard_idx:04d}.tar"

    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    req = urllib.request.Request(url, headers=headers)

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=filename, leave=True) as pbar:
        with urllib.request.urlopen(req) as response:
            pbar.total = int(response.headers.get("Content-Length", 0))
            chunk_size = 1024 * 1024
            with open(out_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))


def download(cfg : Config, start_shard : int = 0):
    train_dir = Path(cfg.train_data_dir)
    train_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving shards to  : {train_dir.resolve()}")
    logger.info(f"Target samples    : ~{cfg.approx_train_samples:,}")
    logger.info(f"Total shards      : {cfg.train_shards} (0000 → {cfg.train_shards - 1:04d})")

    for i in range(start_shard, cfg.train_shards):
        out_path = train_dir / f"cc12m-train-{i:04d}.tar"
        if out_path.exists():
            logger.info(f"Shard {i:04d} already exists — skipping")
            continue
        logger.info(f"\nDownloading shard {i:04d}/{cfg.train_shards - 1:04d}...")
        try:
            download_shard(i, out_path, cfg.hf_base_url, cfg.hf_token)
            logger.info(f"Shard {i:04d} done → {out_path}")
        except Exception as e:
            logger.error(f"Shard {i:04d} failed: {e}")
            if out_path.exists():
                out_path.unlink()

    val_path = train_dir / f"cc12m-train-{cfg.val_shard:04d}.tar"
    if not val_path.exists():
        logger.info(f"Downloading val shard {cfg.val_shard:04d}...")
        download_shard(cfg.val_shard, val_path, cfg.hf_base_url, cfg.hf_token)

    _verify(cfg)


def _verify(cfg : Config):
    train_dir = Path(cfg.train_data_dir)
    logger.info("\n-- Verifying downloads --")
    missing = []

    for i in range(cfg.train_shards):
        path = train_dir / f"cc12m-train-{i:04d}.tar"
        if path.exists():
            size_gb = path.stat().st_size / 1e9
            logger.info(f"  YES  cc12m-train-{i:04d}.tar  ({size_gb:.2f} GB)")
        else:
            logger.warning(f"  NO   cc12m-train-{i:04d}.tar  MISSING")
            missing.append(i)

    if not missing:
        logger.info(f"\nAll {cfg.train_shards} shards downloaded. {cfg.train_data_dir}/ is ready.")
    else:
        logger.warning(f"\nMissing shards: {missing}. Re-run to retry.")

    val_path = train_dir / f"cc12m-train-{cfg.val_shard:04d}.tar"
    if val_path.exists():
        size_gb = val_path.stat().st_size / 1e9
        logger.info(f"  YES  cc12m-train-{cfg.val_shard:04d}.tar (val)  ({size_gb:.2f} GB)")
    else:
        logger.warning(f"  NO   cc12m-train-{cfg.val_shard:04d}.tar (val)  MISSING")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",    type=str, default=None)
    p.add_argument("--start_shard", type=int, default=0)
    p.add_argument("--config",      type=str, default="config/config.yaml", help="Path to YAML config file")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_from_yaml(args.config)

    if args.hf_token:
        cfg.hf_token = args.hf_token

    download(cfg=cfg, start_shard=args.start_shard)
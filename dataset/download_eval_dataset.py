"""
dataset/download_eval_dataset.py — Download Fashion-IQ eval dataset locally
"""

import argparse
import logging
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

from config.config import Config, load_from_yaml

logging.basicConfig(
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt = "%H:%M:%S",
    level   = logging.INFO,
)
logger = logging.getLogger(__name__)


def download(cfg: Config):
    eval_dir = Path(cfg.eval_data_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {cfg.hf_eval_repo_id} -> {eval_dir.resolve()}")

    tmp_dir = Path("_hf_tmp_cir")
    snapshot_download(
        repo_id   = cfg.hf_eval_repo_id,
        repo_type = "dataset",
        local_dir = str(tmp_dir),
        token     = cfg.hf_token,
    )
    logger.info("Download complete. Reorganising into eval_data/...")

    fashion_src = tmp_dir / "fashionIQ_dataset"
    fashion_dst = eval_dir / "fashion_iq"
    if fashion_src.exists():
        if fashion_dst.exists():
            shutil.rmtree(fashion_dst)
        shutil.copytree(fashion_src, fashion_dst)
        logger.info(f"Fashion-IQ -> {fashion_dst}")
    else:
        logger.warning(f"Fashion-IQ source not found at {fashion_src}")

    shutil.rmtree(tmp_dir)
    logger.info("Cleaned up temp files.")

    zip_path = fashion_dst / "images.zip"
    img_path = fashion_dst / "images"
    if zip_path.exists():
        logger.info("Extracting images.zip...")
        img_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(img_path)
        zip_path.unlink()
        logger.info(f"Images extracted -> {img_path}")
    else:
        logger.warning(f"images.zip not found at {zip_path}")

    _verify(cfg)


def _verify(cfg: Config):
    eval_dir = Path(cfg.eval_data_dir)
    logger.info("\n-- Verifying structure --")
    checks = [
        eval_dir / "fashion_iq" / "captions"     / "cap.dress.val.json",
        eval_dir / "fashion_iq" / "captions"     / "cap.shirt.val.json",
        eval_dir / "fashion_iq" / "captions"     / "cap.toptee.val.json",
        eval_dir / "fashion_iq" / "image_splits" / "split.dress.val.json",
        eval_dir / "fashion_iq" / "image_splits" / "split.shirt.val.json",
        eval_dir / "fashion_iq" / "image_splits" / "split.toptee.val.json",
        eval_dir / "fashion_iq" / "images",
    ]
    all_ok = True
    for path in checks:
        status = "Y" if path.exists() else "X MISSING"
        logger.info(f"  {status}  {path}")
        if not path.exists():
            all_ok = False

    if all_ok:
        logger.info(f"\nAll files present. {cfg.eval_data_dir}/ is ready.")
    else:
        logger.warning("\nSome files missing — check HF repo structure.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--config",   type=str, default="config/config.yaml",
                   help="Path to YAML config file")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_from_yaml(args.config)

    # CLI token overrides yaml/default
    if args.hf_token:
        cfg.hf_token = args.hf_token

    download(cfg=cfg)
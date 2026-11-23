"""
evaluation/eval_fashion.py
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from config.config_h import Config, load_from_yaml, get_config
from model.model_h import Pic2Word
from evaluation.retrieval import extract_features, recall_at_k, log_recalls

logging.basicConfig(
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt = "%H:%M:%S",
    level   = logging.INFO,
)
logger = logging.getLogger(__name__)

CATEGORIES = ["dress", "shirt", "toptee"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FashionIQImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], transform):
        self.image_paths = image_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return (self.transform(img),)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_annotations(category: str, split: str, eval_dir: Path) -> List[dict]:
    path = eval_dir / "captions" / f"cap.{category}.{split}.json"
    with open(path) as f:
        return json.load(f)


def load_split(category: str, split: str, eval_dir: Path) -> List[str]:
    path = eval_dir / "image_splits" / f"split.{category}.{split}.json"
    with open(path) as f:
        return json.load(f)


def build_gallery(img_names  : List[str],
                  images_dir : Path,
                  transform,
                  batch_size  : int = 512,
                  num_workers : int = 4):
    img_paths = [images_dir / f"{name}.png" for name in img_names]
    name2idx  = {name: i for i, name in enumerate(img_names)}
    dataset   = FashionIQImageDataset(img_paths, transform)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    return loader, name2idx


def build_queries(annotations : List[dict],
                  name_to_idx : Dict[str, int],
                  images_dir  : Path,
                  transform,
                  batch_size  : int = 512,
                  num_workers : int = 4,
                  verbose     : bool = True) -> Tuple[DataLoader, List[str], torch.Tensor]:
    ref_paths, modifiers, gt_indices = [], [], []
    skipped = 0

    for ann in annotations:
        ref_name    = ann["candidate"]
        target_name = ann["target"]

        if ref_name not in name_to_idx or target_name not in name_to_idx:
            skipped += 1
            continue

        ref_path    = images_dir / f"{ref_name}.png"
        target_path = images_dir / f"{target_name}.png"
        if not ref_path.exists() or not target_path.exists():
            skipped += 1
            continue

        ref_paths.append(ref_path)
        modifiers.append(ann["captions"][0] + " and " + ann["captions"][1])
        gt_indices.append(name_to_idx[target_name])

    if verbose:
        print(f"Annotations skipped : {skipped}")

    dataset   = FashionIQImageDataset(ref_paths, transform)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    gt_tensor = torch.tensor(gt_indices, dtype=torch.long)
    return loader, modifiers, gt_tensor


# ---------------------------------------------------------------------------
# Gallery cache
# ---------------------------------------------------------------------------

def get_cache_path(category: str, clip_model: str, eval_dir: Path) -> Path:
    model_tag = clip_model.replace("/", "-")
    return eval_dir / f"gallery_cache_{category}_{model_tag}.pt"


def load_gallery_cache(category: str, clip_model: str, eval_dir: Path) -> torch.Tensor:
    path  = get_cache_path(category, clip_model, eval_dir)
    feats = torch.load(path, map_location="cpu")
    logger.info(f"Gallery cache loaded from {path} shape={feats.shape}")
    return feats


def save_gallery_cache(feats: torch.Tensor, category: str,
                       clip_model: str, eval_dir: Path):
    path = get_cache_path(category, clip_model, eval_dir)
    torch.save(feats, path)
    logger.info(f"Gallery cache saved -> {path}")


# ---------------------------------------------------------------------------
# Per-category eval
# ---------------------------------------------------------------------------

def eval_category(model       : Pic2Word,
                  category    : str,
                  cfg         : Config,
                  transform,
                  split       : str       = "val",
                  use_cache   : bool      = True,
                  k_values    : List[int] = [1, 5, 10, 50]) -> Dict[int, float]:

    eval_dir   = Path(cfg.eval_data_dir) / "fashion_iq"
    images_dir = eval_dir / "images" / "images"

    logger.info(f"\n-- Fashion-IQ | {category.upper()} --")

    annotations = load_annotations(category, split, eval_dir)
    image_names = load_split(category, split, eval_dir)
    logger.info(f"Gallery : {len(image_names)} images")
    logger.info(f"Queries : {len(annotations)}")

    gallery_loader, name_to_idx = build_gallery(
        image_names, images_dir, transform, cfg.batch_size, cfg.num_workers)

    cache_path = get_cache_path(category, cfg.clip_model, eval_dir)
    if use_cache and cache_path.exists():
        gallery_feats = load_gallery_cache(category, cfg.clip_model, eval_dir)
    else:
        logger.info("Computing gallery features")
        gallery_feats = extract_features(model, gallery_loader, cfg.device, text_list=None)
        if use_cache:
            save_gallery_cache(gallery_feats, category, cfg.clip_model, eval_dir)

    query_loader, modifiers, gt_indices = build_queries(
        annotations, name_to_idx, images_dir, transform,
        cfg.batch_size, cfg.num_workers)

    logger.info(f"Valid queries: {len(modifiers)}")
    logger.info("Computing query features")
    query_feats = extract_features(model, query_loader, cfg.device, text_list=modifiers)

    sims    = query_feats @ gallery_feats.T
    recalls = recall_at_k(sims, gt_indices, k_values=k_values)
    log_recalls(recalls, prefix=f"FashionIQ/{category} | ")

    return recalls


# ---------------------------------------------------------------------------
# Main evaluate
# ---------------------------------------------------------------------------

def evaluate(checkpoint : str,
             cfg        : Config,
             use_cache  : bool      = True,
             k_values   : List[int] = [1, 5, 10, 50]):

    ckpt          = torch.load(checkpoint, map_location="cpu")
    ckpt_cfg      = ckpt.get("config", {})

    # Build a model config from checkpoint — overrides only model-relevant fields
    model_cfg = get_config(
        clip_model      = ckpt_cfg.get("clip_model",      cfg.clip_model),
        mapper_type     = ckpt["mapper_type"],
        merge_tokens    = ckpt_cfg.get("merge_tokens",    None),
        merge_type      = ckpt_cfg.get("merge_type",      "learned"),
        max_token_len   = ckpt_cfg.get("max_token_len",   cfg.max_token_len),
        clip_prompt     = ckpt_cfg.get("clip_prompt",     cfg.clip_prompt),
        # pass through eval settings from cfg
        device          = cfg.device,
        hidden_dim      = ckpt_cfg.get("hidden_dim",      cfg.hidden_dim),
        num_tokens      = ckpt_cfg.get("num_tokens",      cfg.num_tokens),
        num_heads       = ckpt_cfg.get("num_heads",       cfg.num_heads),
        num_layers      = ckpt_cfg.get("num_layers",      cfg.num_layers),
        dropout         = ckpt_cfg.get("dropout",         cfg.dropout),
        vib_logvar_bias = ckpt_cfg.get("vib_logvar_bias", -2.0),
        vib_logvar_clip_max = ckpt_cfg.get("vib_logvar_clip_max",  2.0),
        clip_backend    = ckpt_cfg.get("clip_backend",    "open_clip"),
        clip_pretrained = ckpt_cfg.get("clip_pretrained", None),
    )

    model = Pic2Word(model_cfg).to(cfg.device)
    model.mapper.load_state_dict(ckpt["mapper_state"])
    if ckpt.get("merger_state") is not None and model.merger is not None:
        model.merger.load_state_dict(ckpt["merger_state"])
    if "logit_scale" in ckpt:
        with torch.no_grad():
            model.clip.logit_scale.fill_(ckpt["logit_scale"])
    model.eval()

    logger.info(f"Loaded checkpoint: {checkpoint}")

    transform = model.clip.preprocess   # open_clip ViT-H preprocess (clip.load cannot load ViT-H-14)

    all_recalls = {}
    for category in CATEGORIES:
        all_recalls[category] = eval_category(
            model     = model,
            category  = category,
            cfg       = cfg,
            transform = transform,
            use_cache = use_cache,
            k_values  = k_values,
        )

    logger.info(f"\n{'='*50}")
    logger.info(f"Fashion-IQ Summary")
    logger.info(f"{'='*50}")
    for cat in CATEGORIES:
        log_recalls(all_recalls[cat], prefix=f" {cat:<8} | ")
    for k in k_values:
        avg = sum(all_recalls[c][k] for c in CATEGORIES) / len(CATEGORIES)
        logger.info(f"  {'avg':<8} | R@{k}={avg*100:.1f}%")
    logger.info(f"{'='*50}")

    return all_recalls


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--config",      type=str, default="config/config_h.yaml")
    p.add_argument("--clip_model",  type=str, default=None)
    p.add_argument("--batch_size",  type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--no_cache",    action="store_true", default=False)
    p.add_argument("--device",      type=str, default=None)
    p.add_argument("--eval_data_dir", type=str, default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_from_yaml(args.config)

    if args.clip_model  : cfg.clip_model  = args.clip_model
    if args.batch_size  : cfg.batch_size  = args.batch_size
    if args.num_workers : cfg.num_workers = args.num_workers
    if args.device      : cfg.device      = args.device
    if args.eval_data_dir : cfg.eval_data_dir = args.eval_data_dir


    evaluate(
        checkpoint = args.checkpoint,
        cfg        = cfg,
        use_cache  = not args.no_cache,
    )
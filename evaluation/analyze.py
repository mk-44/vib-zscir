"""
evaluation/analyze.py
Diagnostic script to detect identity collapse and stickiness.

Two checks:
  1. Fashion-IQ val — reference image stickiness vs GT:
       - avg sim(p, v_ref)              [should be ~0.35-0.40]
       - avg sim(p, gt)                 [should be > sim(p, v_ref) if model works]
       - R@10(v_ref) in gallery         [should be LOW ~30-46%]
       - mean / median rank of v_ref    [stickiness signal]
       - R@10(GT) in gallery            [actual task performance]
       - mean / median rank of GT       [task signal]

  2. CC12M val shard — reconstruction quality:
       - avg sim(p, v_raw)              [should be ~0.35-0.40, not 0.95+]
       - R@1 / R@10 reconstruction recall

Usage:
  python evaluation/analyze.py --checkpoint path/to/epoch_XX.pt
  python evaluation/analyze.py --checkpoint path/to/epoch_XX.pt --skip_cc12m
"""

import argparse
import json
import logging
from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from config.config import Config, get_config
from model.model import Pic2Word, FusedWeightedMLPMapper
from dataset.dataset import build_loaders

logging.basicConfig(
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt = "%H:%M:%S",
    level   = logging.INFO,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str):
    ckpt     = torch.load(checkpoint_path, map_location="cpu")
    ckpt_cfg = ckpt.get("config", {})

    cfg = get_config(
        clip_model      = ckpt_cfg.get("clip_model",      "ViT-L/14"),
        mapper_type     = ckpt["mapper_type"],
        hidden_dim      = ckpt_cfg.get("hidden_dim",      256),
        num_tokens      = ckpt_cfg.get("num_tokens",      1),
        num_heads       = ckpt_cfg.get("num_heads",       8),
        num_layers      = ckpt_cfg.get("num_layers",      1),
        dropout         = ckpt_cfg.get("dropout",         0.1),
        merge_tokens    = ckpt_cfg.get("merge_tokens",    None),
        merge_type      = ckpt_cfg.get("merge_type",      "learned"),
        max_token_len   = ckpt_cfg.get("max_token_len",   77),
        clip_prompt     = ckpt_cfg.get("clip_prompt",     "a photo of"),
        vib_logvar_bias = ckpt_cfg.get("vib_logvar_bias", -2.0),
        vib_logvar_clip_max = ckpt_cfg.get("vib_logvar_clip_max",  2.0),
        device          = device,
    )

    model = Pic2Word(cfg).to(device)
    model.mapper.load_state_dict(ckpt["mapper_state"])
    if ckpt.get("merger_state") is not None and model.merger is not None:
        model.merger.load_state_dict(ckpt["merger_state"])
    if "logit_scale" in ckpt:
        with torch.no_grad():
            model.clip.logit_scale.fill_(ckpt["logit_scale"])

    logger.info(f"Loaded  : {checkpoint_path}")
    logger.info(f"Epoch   : {ckpt['epoch']}  |  logit_scale = {model.clip.logit_scale.exp().item():.2f}")
    model.eval()
    return model, cfg.clip_model


# ---------------------------------------------------------------------------
# Check 1 — Fashion-IQ stickiness vs GT
# ---------------------------------------------------------------------------

@torch.no_grad()
def check_fashioniq_stickiness(model      : Pic2Word,
                                clip_model : str,
                                device     : str,
                                data_root  : str = "eval_data/fashion_iq"):
    """
    For every Fashion-IQ query encodes ref image + modifier text → p,
    then ranks the full gallery by sim(p, gallery).

    Reports THREE signals side-by-side:
      sim(p, v_ref) : cosine sim to reference image  → stickiness signal
      sim(p, gt)    : cosine sim to target image      → task signal
      rank(v_ref)   : where ref image lands in ranked gallery
      rank(GT)      : where target image lands in ranked gallery

    Healthy  : sim(p,gt) > sim(p,v_ref), R@10(GT) ~20-25%, R@10(v_ref) ~30-46%
    Collapsed: sim(p,v_ref) >> sim(p,gt), R@10(v_ref) ~80%+, R@10(GT) ~3-5%
    """
    logger.info("\n" + "="*75)
    logger.info("CHECK 1 — Fashion-IQ  |  sim(p,v_ref)  sim(p,gt)  rank(v_ref)  rank(GT)")
    logger.info("="*75)

    _, preprocess = clip.load(clip_model, device="cpu")
    categories    = ["dress", "shirt", "toptee"]

    all_ref_sims, all_gt_sims                = [], []
    all_ref_ranks, all_ref_r10               = [], []
    all_gt_ranks,  all_gt_r10                = [], []

    for cat in categories:
        ann_path     = Path(data_root) / "captions"     / f"cap.{cat}.val.json"
        gallery_path = Path(data_root) / f"gallery_cache_{cat}_{clip_model.replace('/', '-')}.pt"
        img_root     = Path(data_root) / "images" / "images"
        id_list_path = Path(data_root) / "image_splits" / f"split.{cat}.val.json"

        if not ann_path.exists():
            logger.warning(f"Skipping {cat} — annotations not found at {ann_path}")
            continue

        annotations  = json.load(open(ann_path))
        gallery_data = torch.load(gallery_path, map_location=device)
        gallery_embs = gallery_data if isinstance(gallery_data, torch.Tensor) else gallery_data["embeddings"]
        gallery_embs = F.normalize(gallery_embs.to(device), dim=-1)

        id_list   = json.load(open(id_list_path))
        id_to_idx = {img_id: i for i, img_id in enumerate(id_list)}

        ref_sims, gt_sims        = [], []
        ref_ranks, ref_r10       = [], []
        gt_ranks,  gt_r10        = [], []

        for ann in tqdm(annotations, desc=f"  {cat}", leave=False):
            ref_id    = ann["candidate"]
            target_id = ann["target"]

            ref_img_path = img_root / f"{ref_id}.png"
            if not ref_img_path.exists():
                ref_img_path = img_root / f"{ref_id}.jpg"
            if not ref_img_path.exists():
                continue

            ref_img  = preprocess(Image.open(ref_img_path).convert("RGB")).unsqueeze(0).to(device)
            modifier = ann["captions"][0] + " and " + ann["captions"][1]
            txt_tok  = clip.tokenize([modifier], truncate=True).to(device)

            p     = model.encode_query(ref_img, txt_tok)        # (1, D)
            v_ref = model.clip.encode_image(ref_img)            # (1, D) normalised

            # --- sim(p, v_ref) ---
            ref_sims.append((p @ v_ref.T).item())

            # --- sim(p, gt) ---
            gt_img_path = img_root / f"{target_id}.png"
            if not gt_img_path.exists():
                gt_img_path = img_root / f"{target_id}.jpg"
            if gt_img_path.exists():
                v_gt = model.clip.encode_image(
                    preprocess(Image.open(gt_img_path).convert("RGB")).unsqueeze(0).to(device)
                )
                gt_sims.append((p @ v_gt.T).item())

            # --- gallery ranks ---
            sim_gallery = (p @ gallery_embs.T).squeeze(0)       # (N,)

            ref_idx = id_to_idx.get(ref_id)
            if ref_idx is not None:
                rank_ref = (sim_gallery > sim_gallery[ref_idx]).sum().item() + 1
                ref_ranks.append(rank_ref)
                ref_r10.append(1 if rank_ref <= 10 else 0)

            gt_idx = id_to_idx.get(target_id)
            if gt_idx is not None:
                rank_gt = (sim_gallery > sim_gallery[gt_idx]).sum().item() + 1
                gt_ranks.append(rank_gt)
                gt_r10.append(1 if rank_gt <= 10 else 0)

        def _mean(lst):   return sum(lst) / max(1, len(lst))
        def _median(lst): return sorted(lst)[len(lst) // 2] if lst else 0

        logger.info(
            f"  {cat:8s} | "
            f"sim(p,v_ref)={_mean(ref_sims):.4f}  sim(p,gt)={_mean(gt_sims):.4f} | "
            f"R@10(v_ref)={100*_mean(ref_r10):.1f}%  mean={_mean(ref_ranks):.0f}  med={_median(ref_ranks)} | "
            f"R@10(GT)={100*_mean(gt_r10):.1f}%  mean={_mean(gt_ranks):.0f}  med={_median(gt_ranks)}"
        )

        all_ref_sims.extend(ref_sims)
        all_gt_sims.extend(gt_sims)
        all_ref_ranks.extend(ref_ranks)
        all_ref_r10.extend(ref_r10)
        all_gt_ranks.extend(gt_ranks)
        all_gt_r10.extend(gt_r10)

    if all_ref_sims:
        def _mean(lst):   return sum(lst) / max(1, len(lst))
        def _median(lst): return sorted(lst)[len(lst) // 2] if lst else 0

        logger.info(
            f"  {'OVERALL':8s} | "
            f"sim(p,v_ref)={_mean(all_ref_sims):.4f}  sim(p,gt)={_mean(all_gt_sims):.4f} | "
            f"R@10(v_ref)={100*_mean(all_ref_r10):.1f}%  mean={_mean(all_ref_ranks):.0f}  med={_median(all_ref_ranks)} | "
            f"R@10(GT)={100*_mean(all_gt_r10):.1f}%  mean={_mean(all_gt_ranks):.0f}  med={_median(all_gt_ranks)}"
        )


# ---------------------------------------------------------------------------
# Check 2 — CC12M val shard reconstruction
# ---------------------------------------------------------------------------

@torch.no_grad()
def check_cc12m_reconstruction(model      : Pic2Word,
                                clip_model : str,
                                device     : str):
    """
    On the CC12M val shard:
      p     : pseudo-token query embedding (no modifier text)
      v_raw : normalised image embedding  (GT for reconstruction)

    Healthy  : avg sim ~0.35-0.40,  R@1 ~24%+
    Collapsed: avg sim ~0.95+,      R@1 ~99%+
    """
    logger.info("\n" + "="*75)
    logger.info("CHECK 2 — CC12M Val Shard  |  p  vs  v_raw (GT)")
    logger.info("="*75)

    _, preprocess = clip.load(clip_model, device="cpu")

    cfg = get_config(
        clip_model  = clip_model,
        batch_size  = 256,
        num_workers = 2,
        shuffle_buf = 1,
        device      = device,
    )
    _, val_loader, *_ = build_loaders(cfg=cfg, clip_preprocess=preprocess)

    all_p, all_v, all_sims = [], [], []

    for batch in tqdm(val_loader, desc="  cc12m val", leave=False):
        imgs = batch[0].to(device)

        if model.use_multilayer:
            v_raw, intermediate = model.clip.encode_image_multilayer(imgs)
            pseudo_tokens = (model.mapper(v_raw.detach(), intermediate)
                             if isinstance(model.mapper, FusedWeightedMLPMapper)
                             else model.mapper(intermediate))
        else:
            v_raw         = model.clip.encoder_image_unnorm(imgs)
            pseudo_tokens = model.mapper(v_raw)

        if model.merger is not None:
            pseudo_tokens = model.merger(pseudo_tokens)

        v       = F.normalize(v_raw, dim=-1)
        tok_emb, eot_pos = model._build_token_sequence(pseudo_tokens)
        p       = model.clip.forward_text_transform(tok_emb, eot_pos)

        all_sims.extend((p * v).sum(dim=-1).cpu().tolist())
        all_p.append(p.cpu())
        all_v.append(v.cpu())

    P          = torch.cat(all_p, dim=0)
    V          = torch.cat(all_v, dim=0)
    M          = P.shape[0]
    sim_matrix = P @ V.T

    r1  = sum(sim_matrix.topk(1,  dim=-1).indices[i].item() == i for i in range(M)) / M
    r10 = sum(any(j.item() == i for j in sim_matrix.topk(10, dim=-1).indices[i]) for i in range(M)) / M

    avg_sim = sum(all_sims) / len(all_sims)
    logger.info(f"  Samples           : {M}")
    logger.info(f"  avg sim(p, v_raw) : {avg_sim:.4f}   ← ~0.35-0.40 healthy | 0.90+ = collapsed")
    logger.info(f"  R@1  (p vs v_raw) : {r1*100:.1f}%")
    logger.info(f"  R@10 (p vs v_raw) : {r10*100:.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     type=str, required=True)
    p.add_argument("--device",         type=str, default="cuda")
    p.add_argument("--data_root",      type=str, default="eval_data/fashion_iq")
    p.add_argument("--skip_fashioniq", action="store_true", default=False)
    p.add_argument("--skip_cc12m",     action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    args              = parse_args()
    model, clip_model = load_model(args.checkpoint, args.device)

    if not args.skip_fashioniq:
        check_fashioniq_stickiness(model, clip_model, args.device, args.data_root)

    if not args.skip_cc12m:
        check_cc12m_reconstruction(model, clip_model, args.device)

    logger.info("\nDone.")
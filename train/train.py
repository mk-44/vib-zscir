"""
train/train.py
"""

import argparse
import json
import math
import logging
import shutil
import time
from pathlib import Path
from tqdm import tqdm
import random

import clip
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW

from config.config import Config, get_config, load_from_yaml
from dataset.dataset import build_loaders
from model.model import Pic2Word, FusedWeightedMLPMapper

logging.basicConfig(
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt = "%H:%M:%S",
    level   = logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0):
        self.opt          = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr       = min_lr
        self.base_lrs     = [g["lr"] for g in optimizer.param_groups]
        self._step        = 0

    def step(self):
        self._step += 1
        if self._step <= self.warmup_steps:
            scale = self._step / max(1.0, self.warmup_steps)
        else:
            progress = (self._step - self.warmup_steps) / max(1.0, self.total_steps - self.warmup_steps)
            scale    = self.min_lr + 0.5 * (1 - self.min_lr) * (1 + math.cos(math.pi * progress))
        for g, base_lr in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = base_lr * scale

    def get_lr(self):
        return self.opt.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Reconstruction eval (CC12M val shard — fast, used for resume tracking)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_reconstruction(model: Pic2Word, val_loader, device: str, k_vals: list = [1, 5]) -> dict:
    model.eval()
    all_p, all_v = [], []
    for batch in val_loader:
        imgs = batch[0].to(device)
        if model.use_multilayer:
            v_raw, intermediate = model.clip.encode_image_multilayer(imgs)
            pseudo_tokens = (model.mapper(v_raw, intermediate)
                             if isinstance(model.mapper, FusedWeightedMLPMapper)
                             else model.mapper(intermediate))
        else:
            v_raw         = model.clip.encoder_image_unnorm(imgs)
            pseudo_tokens = model.mapper(v_raw)
        if model.merger is not None:
            pseudo_tokens = model.merger(pseudo_tokens)
        v                = F.normalize(v_raw, dim=-1)
        tok_emb, eot_pos = model._build_token_sequence(pseudo_tokens)
        p                = model.clip.forward_text_transform(tok_emb, eot_pos)
        all_p.append(p.cpu())
        all_v.append(v.cpu())

    P      = torch.cat(all_p, dim=0)
    V      = torch.cat(all_v, dim=0)
    sims   = P @ V.T
    M      = P.shape[0]
    labels = torch.arange(M)
    recalls = {}
    for k in k_vals:
        top_k      = sims.topk(k, dim=-1).indices
        hits       = sum(labels[i].item() in top_k[i].tolist() for i in range(M))
        recalls[k] = hits / M
    return recalls


# ---------------------------------------------------------------------------
# Fashion-IQ eval (task metric — used for best model selection)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_fashion_eval(model: Pic2Word, cfg: Config, preprocess) -> dict:
    """
    Run full Fashion-IQ eval on the in-memory model.
    Returns {category: {k: recall}, "avg": {k: recall}} or {} if eval data missing.
    """
    try:
        from evaluation.eval_fashion import eval_category, CATEGORIES
    except ImportError:
        logger.warning("evaluation.eval_fashion not importable — skipping fashion eval")
        return {}

    eval_dir = Path(cfg.eval_data_dir) / "fashion_iq"
    if not eval_dir.exists():
        logger.warning(f"Fashion-IQ eval dir not found at {eval_dir} — skipping")
        return {}

    model.eval()
    k_values    = [1, 5, 10, 50]
    all_recalls = {}

    for category in CATEGORIES:
        try:
            recalls = eval_category(
                model     = model,
                category  = category,
                cfg       = cfg,
                transform = preprocess,
                use_cache = True,
                k_values  = k_values,
            )
            all_recalls[category] = {str(k): round(float(v), 6) for k, v in recalls.items()}
        except Exception as e:
            logger.warning(f"Fashion eval failed for {category}: {e}")

    if all_recalls:
        avg = {}
        for k in k_values:
            vals = [all_recalls[c][str(k)] for c in CATEGORIES
                    if c in all_recalls and str(k) in all_recalls[c]]
            if vals:
                avg[str(k)] = round(float(sum(vals) / len(vals)), 6)
        all_recalls["avg"] = avg

    return all_recalls


# ---------------------------------------------------------------------------
# Analysis eval (stickiness + reconstruction — epoch end only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_analysis(model: Pic2Word, cfg: Config, preprocess, val_loader) -> dict:
    """
    Run stickiness check (Fashion-IQ) + CC12M reconstruction analysis.
    Returns nested dict or {} if eval data missing.
    """
    results = {}
    model.eval()
    # --- CC12M reconstruction ---
    try:
        all_p, all_v, all_sims = [], [], []
        for batch in val_loader:
            imgs = batch[0].to(cfg.device)
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
            v                = F.normalize(v_raw, dim=-1)
            tok_emb, eot_pos = model._build_token_sequence(pseudo_tokens)
            p                = model.clip.forward_text_transform(tok_emb, eot_pos)
            all_sims.extend((p * v).sum(dim=-1).cpu().tolist())
            all_p.append(p.cpu())
            all_v.append(v.cpu())

        P          = torch.cat(all_p, dim=0)
        V          = torch.cat(all_v, dim=0)
        M          = P.shape[0]
        sim_matrix = P @ V.T
        r1  = sum(sim_matrix.topk(1,  dim=-1).indices[i].item() == i for i in range(M)) / M
        r10 = sum(any(j.item() == i for j in sim_matrix.topk(10, dim=-1).indices[i]) for i in range(M)) / M

        results["reconstruction"] = {
            "n_samples"   : M,
            "avg_sim_pv"  : round(float(sum(all_sims) / len(all_sims)), 6),
            "r1"          : round(float(r1),  6),
            "r10"         : round(float(r10), 6),
        }
    except Exception as e:
        logger.warning(f"CC12M reconstruction analysis failed: {e}")

    # --- Fashion-IQ stickiness ---
    try:
        from PIL import Image
        eval_dir  = Path(cfg.eval_data_dir) / "fashion_iq"
        img_root  = eval_dir / "images" / "images"
        if not eval_dir.exists():
            raise FileNotFoundError(f"eval dir not found: {eval_dir}")

        clip_preprocess    = preprocess
        categories         = ["dress", "shirt", "toptee"]
        stickiness         = {}

        for cat in categories:
            ann_path     = eval_dir / "captions"     / f"cap.{cat}.val.json"
            gallery_path = eval_dir / f"gallery_cache_{cat}_{cfg.clip_model.replace('/', '-')}.pt"
            id_list_path = eval_dir / "image_splits" / f"split.{cat}.val.json"

            if not ann_path.exists() or not gallery_path.exists():
                continue

            annotations  = json.load(open(ann_path))
            gallery_data = torch.load(gallery_path, map_location=cfg.device)
            gallery_embs = gallery_data if isinstance(gallery_data, torch.Tensor) else gallery_data["embeddings"]
            gallery_embs = F.normalize(gallery_embs.to(cfg.device), dim=-1)
            id_list      = json.load(open(id_list_path))
            id_to_idx    = {img_id: i for i, img_id in enumerate(id_list)}

            ref_sims, gt_sims        = [], []
            ref_ranks, ref_r10       = [], []
            gt_ranks,  gt_r10        = [], []

            for ann in annotations:
                ref_id    = ann["candidate"]
                target_id = ann["target"]

                ref_img_path = img_root / f"{ref_id}.png"
                if not ref_img_path.exists():
                    ref_img_path = img_root / f"{ref_id}.jpg"
                if not ref_img_path.exists():
                    continue

                ref_img  = clip_preprocess(
                    Image.open(ref_img_path).convert("RGB")
                ).unsqueeze(0).to(cfg.device)
                modifier = ann["captions"][0] + " and " + ann["captions"][1]
                txt_tok  = clip.tokenize([modifier], truncate=True).to(cfg.device)

                p        = model.encode_query(ref_img, txt_tok)
                v_ref    = model.clip.encode_image(ref_img)
                ref_sims.append((p @ v_ref.T).item())

                gt_img_path = img_root / f"{target_id}.png"
                if not gt_img_path.exists():
                    gt_img_path = img_root / f"{target_id}.jpg"
                if gt_img_path.exists():
                    v_gt = model.clip.encode_image(
                        clip_preprocess(Image.open(gt_img_path).convert("RGB")).unsqueeze(0).to(cfg.device)
                    )
                    gt_sims.append((p @ v_gt.T).item())

                sim_gallery = (p @ gallery_embs.T).squeeze(0)

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

            def _mean(lst):   return round(float(sum(lst) / max(1, len(lst))), 6)
            def _median(lst): return int(sorted(lst)[len(lst) // 2]) if lst else 0

            stickiness[cat] = {
                "avg_sim_ref"      : _mean(ref_sims),
                "avg_sim_gt"       : _mean(gt_sims),
                "r10_ref"          : _mean(ref_r10),
                "r10_gt"           : _mean(gt_r10),
                "mean_rank_ref"    : _mean(ref_ranks),
                "mean_rank_gt"     : _mean(gt_ranks),
                "median_rank_ref"  : _median(ref_ranks),
                "median_rank_gt"   : _median(gt_ranks),
            }

        if stickiness:
            def _avg_key(key):
                vals = [stickiness[c][key] for c in stickiness]
                return round(float(sum(vals) / len(vals)), 6)

            stickiness["overall"] = {k: _avg_key(k) for k in list(stickiness[next(iter(stickiness))].keys())}
            results["stickiness"] = stickiness

    except Exception as e:
        logger.warning(f"Stickiness analysis failed: {e}")

    return results


# ---------------------------------------------------------------------------
# JSON / log helpers
# ---------------------------------------------------------------------------

def _save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved -> {path}")


def _append_eval_log(fashion_results: dict, path: Path, epoch: int, step: int, global_step: int):
    """Append a human-readable line to eval_log.txt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    avg = fashion_results.get("avg", {})
    r10 = avg.get("10", 0.0)
    cats = ["dress", "shirt", "toptee"]
    cat_str = "  ".join(
        f"{c} R@10={fashion_results.get(c, {}).get('10', 0.0)*100:.1f}%"
        for c in cats if c in fashion_results
    )
    line = (
        f"[Epoch {epoch:02d} | Step {step:05d} | Global {global_step:06d}]  "
        f"{cat_str}  |  avg R@10={r10*100:.1f}%\n"
    )
    with open(path, "a") as f:
        f.write(line)


def _append_training_stats(path: Path, header: list, row: list):
    """
    Append one row of training stats to a TSV file.
    Writes the header automatically the first time the file is opened.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a") as f:
        if write_header:
            f.write("\t".join(header) + "\n")
        f.write("\t".join(str(x) for x in row) + "\n")        

# ---------------------------------------------------------------------------
# Margin curriculum
# ---------------------------------------------------------------------------

def _compute_lambda_m(global_step: int, total_steps: int,
                      margin_weight: float, margin_delay: float) -> float:
    if margin_weight <= 0.0:
        return 0.0
    delay_steps = int(margin_delay * total_steps)
    if global_step < delay_steps:
        return 0.0
    ramp = min(1.0, (global_step - delay_steps) / max(1, total_steps - delay_steps))
    return margin_weight * ramp


# ---------------------------------------------------------------------------
# VIB beta schedule — linear warmup from beta_initial to beta_max
# ---------------------------------------------------------------------------
def _compute_vib_beta(global_step  : int,
                      total_steps  : int,
                      beta_max     : float,
                      warmup_frac  : float,
                      delay_frac   : float = 0.0,
                      beta_initial : float = 0.0) -> float:
    """
    Phases:
      [0, delay_steps)                        : β = 0      (pure contrastive, no KL)
      [delay_steps, delay_steps + warmup)     : β linearly ramps β_initial → β_max
      [delay_steps + warmup, total_steps)     : β = β_max  (steady state)
    """
    if beta_max <= 0.0:
        return 0.0
    delay_steps = int(delay_frac * total_steps)
    if global_step < delay_steps:
        return 0.0                                            # pure contrastive phase
    warmup_steps = int(warmup_frac * total_steps)
    if warmup_steps <= 0:
        return beta_max
    steps_since_delay = global_step - delay_steps
    if steps_since_delay >= warmup_steps:
        return beta_max
    alpha = steps_since_delay / warmup_steps                  # in [0, 1)
    return beta_initial + alpha * (beta_max - beta_initial)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model            : Pic2Word,
                    loader,
                    optimizer        : AdamW,
                    scheduler        : WarmupCosineScheduler,
                    scaler           : GradScaler,
                    epoch            : int,
                    cfg              : Config,
                    out_dir          : Path,
                    total_steps      : int,
                    start_step       : int = 0,
                    save_every_steps : int = 500,
                    preprocess                 = None,
                    step_eval_dir    : Path    = None,
                    streaming_active : bool    = False,
                    n_missing        : int     = 0):

    model.train()
    total_loss = 0.0
    n_batches  = 0
    t0         = time.time()
    steps_per_epoch = cfg.steps_per_epoch
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}", dynamic_ncols=True,
                leave=True, total=steps_per_epoch)

    for step, batch in enumerate(pbar):

        # Mid-epoch resume — skip completed steps, keep scheduler in sync
        if step < start_step:
            scheduler.step()
            continue

        global_step = epoch * steps_per_epoch + step

        lambda_m = _compute_lambda_m(
            global_step   = global_step,
            total_steps   = total_steps,
            margin_weight = cfg.margin_weight,
            margin_delay  = cfg.margin_delay,
        )

        current_beta = _compute_vib_beta(
            global_step  = global_step,
            total_steps  = total_steps,
            beta_max     = cfg.vib_beta,
            warmup_frac  = cfg.vib_beta_warmup,
            delay_frac   = cfg.vib_beta_delay,
            beta_initial = cfg.vib_beta_initial,
        )

        # L2-on-token weight (deterministic VIB ablation) — same schedule as beta
        current_l2 = _compute_vib_beta(
            global_step  = global_step,
            total_steps  = total_steps,
            beta_max     = cfg.l2_token_weight,
            warmup_frac  = cfg.vib_beta_warmup,
            delay_frac   = cfg.vib_beta_delay,
            beta_initial = 0.0,
        )

        in_delay = (cfg.vib_delay > 0) and (global_step < int(cfg.vib_delay * total_steps))
        imgs       = batch[0].to(cfg.device)
        cap_tokens = None
        if cfg.use_captions and random.random() < 0.5:
            cap_tokens = clip.tokenize(batch[1], truncate=True).to(cfg.device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=cfg.amp):
            out  = model(imgs, captions=cap_tokens, margin=cfg.margin,
                         margin_weight=lambda_m, vib_beta=current_beta, vib_delay=in_delay,
                         vib_kl_mu_weight=cfg.vib_kl_mu_weight,
                         vib_kl_var_weight=cfg.vib_kl_var_weight,
                         l2_token_weight=current_l2)
            loss = out["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if hasattr(model.mapper, 'query_tokens'):
            torch.nn.utils.clip_grad_norm_([model.mapper.query_tokens], cfg.grad_clip)
            other = [p for n, p in model.mapper.named_parameters() if 'query_tokens' not in n]
            if model.clip.logit_scale.requires_grad:
                other += [model.clip.logit_scale]
            torch.nn.utils.clip_grad_norm_(other, cfg.grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.grad_clip)

        scaler.step(optimizer)

        if cfg.ls_max > 0.0 and model.clip.logit_scale.requires_grad:
            with torch.no_grad():
                model.clip.logit_scale.clamp_(0.0, math.log(cfg.ls_max))

        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

        # --- Progress bar ---
        postfix = {
            "loss"  : f"{loss.item():.4f}",
            "l_t2i" : f"{out['loss_t2i'].item():.4f}",
            "l_i2t" : f"{out['loss_i2t'].item():.4f}",
            "lr"    : f"{scheduler.get_lr():.2e}",
            "ls"    : f"{out['logit_scale'].item():.2f}",
            "lm"    : f"{lambda_m:.3f}",
        }
        # Show VIB stats only when active
        if cfg.vib_beta > 0:
            kl_val     = out["kl_loss"].item() if isinstance(out["kl_loss"], torch.Tensor) else out["kl_loss"]
            kl_mu_val  = out["kl_mu"].item()  if isinstance(out["kl_mu"],  torch.Tensor) else out["kl_mu"]
            kl_var_val = out["kl_var"].item() if isinstance(out["kl_var"], torch.Tensor) else out["kl_var"]
            postfix["kl"]    = f"{kl_val:.1f}"
            postfix["klμ"]   = f"{kl_mu_val:.1f}"
            postfix["klσ"]   = f"{kl_var_val:.1f}"
            postfix["β"]     = f"{current_beta:.1e}"
            # mu_norm / std_mean are present only when sampling is on (not in vib_delay phase)
            if "mu_norm" in out:
                postfix["μ"] = f"{out['mu_norm'].item():.2f}"
                postfix["σ"] = f"{out['std_mean'].item():.3f}"
        
        pbar.set_postfix(postfix, refresh=True)

        if step % 100 == 0:
            # Persistent data-source reminder: if any shard is being streamed
            # from HF, re-warn every 100 steps so it cannot be missed mid-run.
            if streaming_active:
                tqdm.write(
                    f"  [DATA WARNING] step {step:05d}: STILL STREAMING {n_missing} "
                    f"shard(s) from HuggingFace — data may be incomplete. Ctrl-C to stop."
                )

            elap = time.time() - t0
            kl_str = ""
            
            if cfg.vib_beta > 0:
                kl_val     = out["kl_loss"].item() if isinstance(out["kl_loss"], torch.Tensor) else out["kl_loss"]
                kl_mu_val  = out["kl_mu"].item()  if isinstance(out["kl_mu"],  torch.Tensor) else out["kl_mu"]
                kl_var_val = out["kl_var"].item() if isinstance(out["kl_var"], torch.Tensor) else out["kl_var"]
                kl_str = (
                    f" | kl {kl_val:.1f}"
                    f" (μ={kl_mu_val:.1f} σ={kl_var_val:.1f})"
                    f" | β {current_beta:.1e}"
                )
                if "mu_norm" in out:
                    kl_str += (
                        f" | μ_norm {out['mu_norm'].item():.3f}"
                        f" | logvar {out['logvar_mean'].item():.3f}"
                        f" | σ {out['std_mean'].item():.4f}"
                    )
            
            
            # === NEW: dump structured stats to TSV ===
            stats_row = [
                f"{epoch:02d}",
                f"{step:05d}",
                f"{global_step:06d}",
                f"{loss.item():.6f}",
                f"{out['loss_t2i'].item():.6f}",
                f"{out['loss_i2t'].item():.6f}",
                f"{scheduler.get_lr():.3e}",
                f"{out['logit_scale'].item():.4f}",
                f"{lambda_m:.4f}",
            ]
            stats_header = [
                "epoch", "step", "global_step",
                "loss", "l_t2i", "l_i2t",
                "lr", "ls", "lambda_m",
            ]
            
            if cfg.vib_beta > 0:
                kl_val     = out["kl_loss"].item() if isinstance(out["kl_loss"], torch.Tensor) else out["kl_loss"]
                kl_mu_val  = out["kl_mu"].item()  if isinstance(out["kl_mu"],  torch.Tensor) else out["kl_mu"]
                kl_var_val = out["kl_var"].item() if isinstance(out["kl_var"], torch.Tensor) else out["kl_var"]
                kl_mu_w_val  = out["kl_mu_weighted"].item()  if isinstance(out["kl_mu_weighted"],  torch.Tensor) else out["kl_mu_weighted"]
                kl_var_w_val = out["kl_var_weighted"].item() if isinstance(out["kl_var_weighted"], torch.Tensor) else out["kl_var_weighted"]
                stats_row.extend([
                    f"{kl_val:.4f}",
                    f"{kl_mu_val:.4f}",
                    f"{kl_var_val:.4f}",
                    f"{kl_mu_w_val:.4f}",
                    f"{kl_var_w_val:.4f}",
                    f"{current_beta:.3e}",
                ])
                stats_header.extend(["kl", "kl_mu", "kl_var", "kl_mu_weighted", "kl_var_weighted", "beta"])
                if "mu_norm" in out:
                    stats_row.extend([
                        f"{out['mu_norm'].item():.4f}",
                        f"{out['logvar_mean'].item():.4f}",
                        f"{out['std_mean'].item():.4f}",
                    ])
                    stats_header.extend(["mu_norm", "logvar_mean", "std_mean"])
            _append_training_stats(out_dir / "training_stats.tsv", stats_header, stats_row)

            
            
            
            tqdm.write(
                f"Epoch {epoch:02d} | step {step:05d}"
                f" | loss {loss.item():.4f}"
                f" | l_t2i {out['loss_t2i'].item():.4f}"
                f" | l_i2t {out['loss_i2t'].item():.4f}"
                f" | lr {scheduler.get_lr():.2e}"
                f" | ls {out['logit_scale'].item():.2f}"
                f" | lm {lambda_m:.3f}"
                f"{kl_str}"
                f" | {elap:.1f}s"
            )
            t0 = time.time()

        # --- Mid-epoch checkpoint ---
        if save_every_steps > 0 and step > 0 and step % save_every_steps == 0:
            mid_path = out_dir / f"epoch_{epoch:02d}_step_{step:05d}.pt"
            save_checkpoint(model, optimizer, epoch, {"step": step},
                            mid_path, cfg, total_steps, step=step)
            tqdm.write(f"Mid-epoch checkpoint -> {mid_path}")

        # --- Step-level fashion eval ---
        if (cfg.eval_every_steps > 0
                and step > 0
                and step % cfg.eval_every_steps == 0
                and preprocess is not None
                and step_eval_dir is not None):

            tqdm.write(f"Running fashion eval at epoch {epoch:02d} step {step:05d}...")
            fashion = _run_fashion_eval(model, cfg, preprocess)
            model.train()  # restore train mode after eval

            if fashion:
                payload = {
                    "epoch"       : epoch,
                    "step"        : step,
                    "global_step" : global_step,
                    "fashion_eval": fashion,
                }
                json_path = step_eval_dir / f"step_{step:05d}_epoch_{epoch:02d}.json"
                _save_json(payload, json_path)
                _append_eval_log(fashion, step_eval_dir / "eval_log.txt",
                                 epoch, step, global_step)
                avg_r10 = fashion.get("avg", {}).get("10", 0.0)
                tqdm.write(f"Step eval avg R@10={avg_r10*100:.1f}%")

    return {"loss": total_loss / max(1, n_batches)}


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(model       : Pic2Word,
                    optimizer   : AdamW,
                    epoch       : int,
                    metrics     : dict,
                    path        : Path,
                    cfg         : Config,
                    total_steps : int,
                    step        : int = None):
    """
    step=None → end-of-epoch checkpoint (resumes from next epoch)
    step=int  → mid-epoch checkpoint    (resumes from step+1 in same epoch)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch"        : epoch,
        "step"         : step,
        "mapper_state" : model.mapper.state_dict(),
        "optimizer"    : optimizer.state_dict(),
        "metrics"      : metrics,
        "mapper_type"  : cfg.mapper_type,
        "mapper_kwargs": cfg.mapper_kwargs,
        "config"       : {
            **{k: v for k, v in cfg.__dict__.items() if not callable(v)},
            "warmup_steps": cfg.warmup_steps,
        },
        "total_steps"  : total_steps,
        "logit_scale"  : model.clip.logit_scale.item(),
        "merger_state" : model.merger.state_dict() if model.merger is not None else None,
    }, path)
    logger.info(f"Checkpoint saved -> {path}")


def load_checkpoint(model: Pic2Word, optimizer: AdamW, path: Path):
    ckpt = torch.load(path, map_location="cpu")
    model.mapper.load_state_dict(ckpt["mapper_state"])
    if ckpt.get("merger_state") is not None and model.merger is not None:
        model.merger.load_state_dict(ckpt["merger_state"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "logit_scale" in ckpt:
        with torch.no_grad():
            model.clip.logit_scale.fill_(ckpt["logit_scale"])

    epoch        = ckpt["epoch"]
    step         = ckpt.get("step", None)
    is_mid_epoch = step is not None

    if is_mid_epoch:
        start_epoch = epoch
        start_step  = step + 1
        logger.info(f"Resumed from {path} (mid-epoch: epoch {epoch}, step {step})")
    else:
        start_epoch = epoch + 1
        start_step  = 0
        logger.info(f"Resumed from {path} (epoch {epoch} complete)")

    return (
        start_epoch,
        start_step,
        ckpt.get("total_steps", None),
        ckpt.get("config", {}).get("warmup_steps", None),
        ckpt.get("metrics", {}).get("recon_recalls", {}).get(1, 0.0),
    )


# ---------------------------------------------------------------------------
# Warm-start initialization (weights only, fresh run)
# ---------------------------------------------------------------------------

def init_from_checkpoint(model: Pic2Word, path: Path, cfg: Config):
    """
    Load mapper weights from a checkpoint into the current model. This is
    NOT a resume — no optimizer state, no epoch/step counters, no schedulers
    are restored. The caller must use this only as a weight initializer and
    proceed with a fresh training run from epoch=0, global_step=0.

    Supported source -> destination cases:
      base -> vib  : remap net.0/3/6 → encoder.0/3 + fc_mu, leave fc_logvar fresh
      base -> base : strict load (sanity check)
      vib  -> vib  : strict load
      others       : strict load if shapes match, else error
    """
    if not path.exists():
        raise FileNotFoundError(f"init_from checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu")
    src_mapper_state = ckpt["mapper_state"]
    src_mapper_type  = ckpt.get("mapper_type", None)

    logger.info("=" * 55)
    logger.info(f"Warm-start init from: {path}")
    logger.info(f"  source mapper_type     : {src_mapper_type}")
    logger.info(f"  destination mapper_type: {cfg.mapper_type}")

    # ---- BaseMLP -> VIB key remap ----
    if cfg.mapper_type == "vib" and src_mapper_type == "base":
        vib_state = {
            "encoder.0.weight": src_mapper_state["net.0.weight"],
            "encoder.0.bias":   src_mapper_state["net.0.bias"],
            "encoder.3.weight": src_mapper_state["net.3.weight"],
            "encoder.3.bias":   src_mapper_state["net.3.bias"],
            "fc_mu.weight":     src_mapper_state["net.6.weight"],
            "fc_mu.bias":       src_mapper_state["net.6.bias"],
        }
        missing, unexpected = model.mapper.load_state_dict(vib_state, strict=False)
        logger.info("  remap applied:")
        logger.info("    net.0 -> encoder.0")
        logger.info("    net.3 -> encoder.3")
        logger.info("    net.6 -> fc_mu")
        logger.info(f"  missing keys (kept fresh from VIB init): {list(missing)}")
        logger.info(f"  unexpected keys                        : {list(unexpected)}")
        # missing should be only fc_logvar.weight + fc_logvar.bias
        expected_missing = {"fc_logvar.weight", "fc_logvar.bias"}
        if set(missing) != expected_missing:
            logger.warning(
                f"  Unexpected missing keys! got={set(missing)}, expected={expected_missing}"
            )

    # ---- Same mapper_type: strict load ----
    elif cfg.mapper_type == src_mapper_type:
        model.mapper.load_state_dict(src_mapper_state, strict=True)
        logger.info(f"  loaded mapper state strictly ({src_mapper_type} -> {cfg.mapper_type})")

    # ---- Unsupported combo ----
    else:
        raise ValueError(
            f"init_from: unsupported combo source={src_mapper_type} -> dest={cfg.mapper_type}. "
            f"Supported: base->base, base->vib, vib->vib."
        )

    # ---- Merger ----
    if ckpt.get("merger_state") is not None and model.merger is not None:
        model.merger.load_state_dict(ckpt["merger_state"])
        logger.info("  loaded merger state")

    # ---- Logit scale ----
    if "logit_scale" in ckpt:
        with torch.no_grad():
            model.clip.logit_scale.fill_(ckpt["logit_scale"])
        logger.info(
            f"  loaded logit_scale: {ckpt['logit_scale']:.4f} "
            f"(exp={model.clip.logit_scale.exp().item():.2f})"
        )

    logger.info("  optimizer state    : NOT loaded (fresh)")
    logger.info("  epoch / step       : reset to 0")
    logger.info("=" * 55)


# ---------------------------------------------------------------------------
# Optimizer / output dir builders
# ---------------------------------------------------------------------------

def _build_optimizer(model: Pic2Word, cfg: Config) -> AdamW:
    if hasattr(model.mapper, 'query_tokens'):
        query_params = [model.mapper.query_tokens]
        other_params = [p for n, p in model.mapper.named_parameters() if 'query_tokens' not in n]
        if model.clip.logit_scale.requires_grad:
            other_params += [model.clip.logit_scale]
        return AdamW([
            {"params": query_params, "lr": cfg.lr * 10},
            {"params": other_params, "lr": cfg.lr},
        ], weight_decay=cfg.weight_decay, betas=cfg.betas)

    if cfg.ls_lr_scale != 1.0 and model.clip.logit_scale.requires_grad:
        mapper_params = list(model.mapper.parameters())
        if model.merger is not None:
            mapper_params += list(model.merger.parameters())
        return AdamW([
            {"params": mapper_params,            "lr": cfg.lr},
            {"params": [model.clip.logit_scale], "lr": cfg.lr * cfg.ls_lr_scale},
        ], weight_decay=cfg.weight_decay, betas=cfg.betas)

    return AdamW(model.trainable_parameters(), lr=cfg.lr,
                 weight_decay=cfg.weight_decay, betas=cfg.betas)


def _build_out_dir(cfg: Config) -> Path:
    # If resuming, reuse the resumed checkpoint's folder verbatim.
    # The path passed to --resume is one of:
    #   <run_folder>/epoch_NN/checkpoint.pt        (end-of-epoch ckpt)   → parent.parent
    #   <run_folder>/epoch_NN_step_SSSSS.pt        (mid-epoch ckpt)      → parent
    # Either way, the run folder is the source of truth — never recompute
    # the suffix from current flags on resume, because flags like init_from
    # are intentionally not re-passed and would silently fork to a new dir.
    if cfg.resume is not None:
        resume_path = Path(cfg.resume)
        if resume_path.parent.name.startswith("epoch_") and resume_path.name == "checkpoint.pt":
            out_dir = resume_path.parent.parent
        else:
            out_dir = resume_path.parent
        logger.info(f"Resume: reusing folder {out_dir.name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # Fresh run — build suffix from current flags
    suffix = f"{cfg.mapper_type}_hd{cfg.hidden_dim}_tok{cfg.num_tokens}"
    if cfg.init_from is not None:
        # Tag warm-start runs so they don't collide with from-scratch runs
        # of the same hyperparameters.
        init_tag = "initbase" if "/base_" in cfg.init_from else "init"
        suffix += f"_{init_tag}"
    if cfg.repulsion_weight > 0.0 or cfg.decorr_weight > 0.0:
        suffix += f"_rep{cfg.repulsion_weight}_decorr{cfg.decorr_weight}"
    elif cfg.merge_tokens is not None:
        suffix += f"_merge{cfg.merge_tokens}_{cfg.merge_type}"
    if cfg.margin_weight > 0.0:
        suffix += f"_margin{cfg.margin}_mw{cfg.margin_weight}"
    if cfg.ls_lr_scale != 1.0:
        suffix += f"_lslr{cfg.ls_lr_scale}"
    if cfg.ls_max > 0.0:
        suffix += f"_lsmax{int(cfg.ls_max)}"
    if cfg.noise_sigma > 0.0:
        suffix += f"_noise{cfg.noise_sigma}"
    if cfg.pseudo_norm > 0.0:
        suffix += f"_pnorm{cfg.pseudo_norm}"
    if cfg.vib_beta > 0.0:
        suffix += f"_vib{cfg.vib_beta}"
    if cfg.vib_beta_initial > 0.0:
        suffix += f"_binit{cfg.vib_beta_initial}"
    if cfg.vib_logvar_bias != -2.0:
        suffix += f"_lvbias{cfg.vib_logvar_bias}"
    if cfg.vib_beta_delay > 0.0:
        suffix += f"_delay{cfg.vib_beta_delay}"
    if cfg.vib_delay > 0.0:
        suffix += f"_vibdelay{cfg.vib_delay}"
    if cfg.vib_beta > 0.0:
        suffix += f"_vwarm{cfg.vib_beta_warmup}"
    if cfg.vib_kl_mu_weight != 1.0:
        suffix += f"_klmuw{cfg.vib_kl_mu_weight}"
    if cfg.vib_kl_var_weight != 1.0:
        suffix += f"_klvw{cfg.vib_kl_var_weight}"
    if cfg.vib_logvar_clip_max != 2.0:
        suffix += f"_lvclip{cfg.vib_logvar_clip_max}"
    if cfg.l2_token_weight > 0.0:
        suffix += f"_l2tok{cfg.l2_token_weight}"

    # Seed tag — always present so every run folder is self-documenting.
    # Fixed seed → _seed{N}  ;  unseeded (full random init) → _seedR
    suffix += f"_seed{cfg.seed}" if cfg.seed is not None else "_seedR"

    out_dir = Path(cfg.output_dir) / suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seed(seed):
    """
    Seed all RNGs for a reproducible run.

    seed=None  → no seeding: full random (entropy-drawn) init. The run is NOT
                 reproducible. This is the intended 'unseeded' behaviour and
                 matches the headline run's regime. Logged loudly so it is
                 never silent.
    seed=int   → seed python / numpy / torch (CPU) / torch (all CUDA devices).
                 Note: this does NOT seed the WebDataset shard/buffer shuffle,
                 so data order is still nondeterministic even when seeded.
    """
    if seed is None:
        logger.warning("=" * 60)
        logger.warning("  NO SEED SET — full random init (entropy-drawn).")
        logger.warning("  This run is NOT reproducible. Pass --seed <int> to fix one.")
        logger.warning("=" * 60)
        return
    # import random as _random
    # import numpy as _np
    # _random.seed(seed)
    # _np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # logger.info(f"Seed set to {seed}")
    
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    logger.info(f"Seed set to {seed} (cudnn deterministic, data shuffle seeded)")

def main(cfg: Config):
    cfg.print_config()
    set_seed(cfg.seed)

    model = Pic2Word(cfg).to(cfg.device)
    model.print_info()

    _, preprocess = clip.load(cfg.clip_model, device="cpu")
    train_loader, val_loader, streaming_active, n_missing = build_loaders(
        cfg=cfg, clip_preprocess=preprocess)

    optimizer = _build_optimizer(model, cfg)

    total_steps = cfg.total_steps
    scheduler   = WarmupCosineScheduler(optimizer, cfg.warmup_steps, total_steps, min_lr=cfg.min_lr)
    scaler      = GradScaler(enabled=cfg.amp)

    start_epoch = 0
    start_step  = 0
    best_metric = 0.0
    best_epoch  = 0

    # init_from and resume are mutually exclusive
    if cfg.init_from is not None and cfg.resume is not None:
        raise ValueError("Use either --init_from or --resume, not both.")

    if cfg.init_from is not None:
        # Weight-only init from a compatible checkpoint, fresh run.
        init_from_checkpoint(model, Path(cfg.init_from), cfg)
        logger.info("Warm-start init complete. Starting fresh run "
                    "(epoch=0, step=0, fresh optimizer, fresh schedules).")
        # NOTE: optimizer was already built fresh above. Scheduler._step is 0.
        # Do NOT touch start_epoch / start_step / total_steps / scheduler.

        # Optional sanity eval on the loaded checkpoint before training begins.
        # This catches a wrong remap or shape mismatch in ~5 min of eval rather
        # than after 4+ hours of training.
        if cfg.run_analysis:
            logger.info("Running sanity Fashion-IQ eval on loaded init_from "
                        "checkpoint (deterministic mode)...")
            sanity_results = _run_fashion_eval(model, cfg, preprocess)
            model.train()
            if sanity_results:
                avg_r10 = sanity_results.get("avg", {}).get("10", 0.0)
                logger.info(f"Sanity eval: avg R@10 = {avg_r10*100:.2f}% "
                            "(should match source checkpoint within ~0.5%)")

    elif cfg.resume:
        start_epoch, start_step, restored_total, restored_warmup, best_metric = \
            load_checkpoint(model, optimizer, cfg.resume)
        if restored_total is not None:
            total_steps           = max(total_steps, restored_total)
            scheduler.total_steps = total_steps
        new_warmup = max(100, int(0.05 * total_steps))
        scheduler.warmup_steps = max(new_warmup, restored_warmup) if restored_warmup else new_warmup
        scheduler._step = start_epoch * cfg.steps_per_epoch + start_step

    out_dir       = _build_out_dir(cfg)

    if cfg.vib_delay > 0 or cfg.vib_beta_delay > 0:
        assert cfg.vib_delay == cfg.vib_beta_delay, (
            f"vib_delay ({cfg.vib_delay}) must equal vib_beta_delay ({cfg.vib_beta_delay}). "
            f"Sampling without KL causes norm inflation. "
            f"Comment out this assert to experiment with independent schedules."
        )

    # Warn if vib_beta_initial is silently ignored during a delay phase.
    if cfg.vib_delay > 0 and cfg.vib_beta_initial > 0:
        logger.warning(
            f"vib_beta_initial={cfg.vib_beta_initial} will have no effect during the "
            f"vib_delay={cfg.vib_delay} phase (KL is skipped while sampling is off). "
            f"It will activate once delay phase ends at "
            f"step {int(cfg.vib_delay * total_steps)}."
        )
    
    step_eval_dir = out_dir / "step_evals" if cfg.eval_every_steps > 0 else None
    if step_eval_dir:
        step_eval_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, cfg.epochs):
        logger.info(f"\n{'='*55}")
        logger.info(f"  Epoch {epoch + 1}/{cfg.epochs}")
        if epoch == start_epoch and start_step > 0:
            logger.info(f"  Resuming from step {start_step}")
        logger.info(f"{'='*55}")

        train_metrics = train_one_epoch(
            model            = model,
            loader           = train_loader,
            optimizer        = optimizer,
            scheduler        = scheduler,
            scaler           = scaler,
            epoch            = epoch,
            cfg              = cfg,
            out_dir          = out_dir,
            total_steps      = total_steps,
            start_step       = start_step if epoch == start_epoch else 0,
            save_every_steps = cfg.save_every_steps,
            preprocess       = preprocess,
            step_eval_dir    = step_eval_dir,
            streaming_active = streaming_active,
            n_missing        = n_missing,
        )
        start_step = 0

        # --- Reconstruction eval (fast, always runs) ---
        recon_recalls = eval_reconstruction(model=model, val_loader=val_loader, device=cfg.device)
        logger.info(
            f"Epoch {epoch:02d} | train_loss={train_metrics['loss']:.4f}"
            f" | recon R@1={recon_recalls[1]*100:.1f}%"
            f" | recon R@5={recon_recalls[5]*100:.1f}%"
        )

        # --- Create epoch folder and save checkpoint ---
        epoch_dir = out_dir / f"epoch_{epoch:02d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = epoch_dir / "checkpoint.pt"
        save_checkpoint(
            model, optimizer, epoch,
            {**train_metrics, "recon_recalls": recon_recalls},
            ckpt_path, cfg, total_steps, step=None,
        )

        # --- Fashion eval (task metric) ---
        fashion_results = {}
        if cfg.eval_epoch_end:
            logger.info(f"Running Fashion-IQ eval for epoch {epoch:02d}...")
            fashion_results = _run_fashion_eval(model, cfg, preprocess)
            model.train()
            if fashion_results:
                _save_json(fashion_results, epoch_dir / "fashion_eval.json")
                avg_r10 = fashion_results.get("avg", {}).get("10", 0.0)
                logger.info(f"Epoch {epoch:02d} | Fashion avg R@10={avg_r10*100:.1f}%")

        # --- Analysis (stickiness + CC12M) ---
        if cfg.run_analysis:
            logger.info(f"Running analysis for epoch {epoch:02d}...")
            analysis = _run_analysis(model, cfg, preprocess, val_loader)
            model.train()
            if analysis:
                _save_json(analysis, epoch_dir / "analysis.json")

        # --- Best model selection ---
        if fashion_results and "avg" in fashion_results:
            current_metric = fashion_results["avg"].get("10", 0.0)
        else:
            current_metric = recon_recalls[1]

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch  = epoch
            shutil.copyfile(ckpt_path, out_dir / "final.pt")
            logger.info(
                f"New best metric={best_metric*100:.1f}% at epoch {epoch:02d}"
                f" -> final.pt updated"
            )

    logger.info(
        f"\nTraining complete."
        f" Best metric={best_metric*100:.1f}% at epoch {best_epoch:02d}."
        f" final.pt -> epoch_{best_epoch:02d}/checkpoint.pt"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",           type=str,   default=None,
                   help="Path to YAML config file. CLI args override yaml values.")
    # model
    p.add_argument("--mapper_type",      type=str,   default=None)
    p.add_argument("--hidden_dim",       type=int,   default=None)
    p.add_argument("--num_tokens",       type=int,   default=None)
    p.add_argument("--num_heads",        type=int,   default=None)
    p.add_argument("--num_layers",       type=int,   default=None)
    p.add_argument("--dropout",          type=float, default=None)
    p.add_argument("--merge_tokens",     type=int,   default=None)
    p.add_argument("--merge_type",       type=str,   default=None)
    # data
    p.add_argument("--batch_size",       type=int,   default=None)
    p.add_argument("--num_workers",      type=int,   default=None)
    p.add_argument("--hf_token",         type=str,   default=None)
    # training
    p.add_argument("--epochs",           type=int,   default=None)
    p.add_argument("--lr",               type=float, default=None)
    p.add_argument("--min_lr",           type=float, default=None)
    p.add_argument("--weight_decay",     type=float, default=None)
    p.add_argument("--grad_clip",        type=float, default=None)
    p.add_argument("--amp",              action="store_true", default=None)
    p.add_argument("--use_captions",     action="store_true", default=None)
    p.add_argument("--repulsion_weight", type=float, default=None)
    p.add_argument("--decorr_weight",    type=float, default=None)
    # logit scale
    p.add_argument("--ls_lr_scale",      type=float, default=None)
    p.add_argument("--ls_max",           type=float, default=None)
    # margin
    p.add_argument("--margin",           type=float, default=None)
    p.add_argument("--margin_weight",    type=float, default=None)
    p.add_argument("--margin_delay",     type=float, default=None)
    # noise
    p.add_argument("--noise_sigma",      type=float, default=None)
    # pseudo_norm
    p.add_argument("--pseudo_norm",      type=float, default=None)
    # VIB
    p.add_argument("--vib_beta",         type=float, default=None)
    p.add_argument("--vib_beta_initial", type=float, default=None,
                   help="Floor for β at warmup start (default 0.0). Use small nonzero "
                        "value to avoid β=0 + sampling-on phase.")
    p.add_argument("--vib_beta_warmup",  type=float, default=None)
    p.add_argument("--vib_beta_delay",   type=float, default=None)
    p.add_argument("--vib_delay",        type=float, default=None)
    p.add_argument("--vib_logvar_bias",  type=float, default=None,
                   help="Initial bias for fc_logvar (default -2.0 → σ≈0.37). "
                        "Use -6.0 for warm-start runs (σ≈0.05).")
    p.add_argument("--vib_kl_mu_weight", type=float, default=None,
               help="Weight on KL_mu term in loss (default 1.0 = standard KL). "
                    "Lower values reduce mean-norm penalty.")
    p.add_argument("--vib_kl_var_weight", type=float, default=None,
                   help="Weight on KL_var term in loss (default 1.0 = standard KL). "
                        "Use 0.1 for warm-start to reduce variance-term dominance.")
    p.add_argument("--vib_logvar_clip_max", type=float, default=None,
                   help="Upper clamp on logvar in mapper (default 2.0 = effectively none). "
                        "Use -3.0 to cap σ ≤ 0.22 mechanically.")
    p.add_argument("--l2_token_weight", type=float, default=None,
                   help="Weight on 0.5*||z||^2 token penalty (deterministic VIB L2-ablation). "
                        "Reuses vib_beta_warmup/delay schedule. 0.0=disabled.")
    # checkpointing
    p.add_argument("--output_dir",       type=str,   default=None)
    p.add_argument("--resume",           type=str,   default=None)
    p.add_argument("--init_from",        type=str,   default=None,
                   help="Path to a checkpoint to use as weight initialization "
                        "(fresh run, no optimizer/schedule state). Mutually "
                        "exclusive with --resume.")
    p.add_argument("--save_every_steps", type=int,   default=None)
    # eval
    p.add_argument("--eval_every_steps", type=int,   default=None)
    p.add_argument("--eval_epoch_end",   action="store_true", default=None)
    p.add_argument("--run_analysis",     action="store_true", default=None)
    p.add_argument("--eval_data_dir",    type=str, default=None)
    # misc
    p.add_argument("--device",           type=str,   default=None)
    p.add_argument("--seed", type=int, default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load from yaml if provided, else start from defaults
    if args.config:
        cfg = load_from_yaml(args.config)
    else:
        cfg = Config()

    # Apply only explicitly provided CLI overrides
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k != "config"}
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    main(cfg)
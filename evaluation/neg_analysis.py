"""
evaluation/neg_analysis.py — Analyze negative similarity structure across checkpoints.

Tests hypothesis: "LS-induced softmax sharpening collapses negative distribution
structure, causing all negatives to look identical to the model."

For each checkpoint, loads CC12M val shard, computes query embeddings p and
target embeddings v, builds sim matrix, and measures:

  RAW SIMILARITY (what the mapper learned):
    neg_std          — std of neg similarities per query (higher = more graduated)
    hard_soft_gap    — max(neg) - min(neg) per query (higher = more spread)
    gt_minus_hardest — sim(p,gt) - max(neg) per query (higher = clearer positive)
    neg_mean         — mean of neg similarities

  POST-LS SOFTMAX (what the loss actually sees after LS amplification):
    prob_on_gt       — softmax prob on positive (near 1.0 = fully saturated)
    neg_entropy      — entropy of neg softmax (higher = more structure preserved)
    effective_negs   — exp(neg_entropy), how many negs "matter"
    top5_neg_mass    — prob mass in top-5 negatives (lower = more spread)

  RAW SOFTMAX (temperature=1, no LS — isolates mapper vs LS contribution):
    neg_entropy_raw  — entropy without LS
    effective_negs_raw — exp of raw entropy

  IDENTITY:
    sim_p_vraw       — sim(p, v_raw) reconstruction similarity

Usage:
    cd /notebooks/experiment/codes
    PYTHONPATH=/notebooks/experiment/codes python evaluation/neg_analysis.py
"""

import json
import logging
import math
import torch
import torch.nn.functional as F
import clip
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List
import webdataset as wds
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Checkpoint specification                                           #
# ------------------------------------------------------------------ #

@dataclass
class CheckpointSpec:
    name: str
    path: str
    group: str                          # group label for table display
    hidden_dim: int = 256
    num_tokens: int = 1
    mapper_type: str = "base"
    merge_tokens: Optional[int] = None
    merge_type: str = "learned"
    r10: str = ""                       # known R@10 for reference
    description: str = ""


# ------------------------------------------------------------------ #
#  Model loading                                                      #
# ------------------------------------------------------------------ #

def load_model_from_checkpoint(spec: CheckpointSpec, device: str = "cuda"):
    """
    Load Pic2Word model from checkpoint.

    Handles two checkpoint formats:
      - New format (4M runs): has 'config' dict with model hyperparams
      - Old format (1.5M/750k runs): no 'config' dict, uses spec defaults

    Returns: (model, logit_scale_value, clip_model_name)
    """
    from model.model import Pic2Word
    from config.config import Config

    ckpt = torch.load(spec.path, map_location="cpu", weights_only=False)

    # Extract model config — try checkpoint first, fall back to spec
    ckpt_cfg = ckpt.get("config", {})
    hidden_dim   = ckpt_cfg.get("hidden_dim", spec.hidden_dim)
    num_tokens   = ckpt_cfg.get("num_tokens", spec.num_tokens)
    mapper_type  = ckpt_cfg.get("mapper_type", spec.mapper_type)
    merge_tokens = ckpt_cfg.get("merge_tokens", spec.merge_tokens)
    merge_type   = ckpt_cfg.get("merge_type", spec.merge_type)
    clip_model   = ckpt_cfg.get("clip_model", "ViT-L/14")

    cfg = Config(
        clip_model   = clip_model,
        mapper_type  = mapper_type,
        hidden_dim   = hidden_dim,
        num_tokens   = num_tokens,
        merge_tokens = merge_tokens,
        merge_type   = merge_type,
        # Forward-pass edit z <- rho*z/||z|| of unit-norm runs must also apply at test; absent/0.0 = off.
        pseudo_norm  = ckpt_cfg.get("pseudo_norm", 0.0),
        device       = device,
    )

    model = Pic2Word(cfg).to(device).eval()

    # Load mapper weights — handle both key names
    mapper_state = ckpt.get("mapper_state", ckpt.get("mapper_state_dict", None))
    if mapper_state is not None:
        model.mapper.load_state_dict(mapper_state)
    else:
        logger.warning(f"No mapper state found in {spec.path}")

    # Load merger weights if present
    merger_state = ckpt.get("merger_state", None)
    if merger_state is not None and model.merger is not None:
        model.merger.load_state_dict(merger_state)

    # Load logit scale
    if "logit_scale" in ckpt:
        ls_val = ckpt["logit_scale"]
        with torch.no_grad():
            if isinstance(ls_val, torch.Tensor):
                model.clip.logit_scale.copy_(ls_val)
            else:
                model.clip.logit_scale.fill_(float(ls_val))

    ls = model.clip.logit_scale.exp().item()
    logger.info(
        f"Loaded {spec.name}: hd={hidden_dim}, tok={num_tokens}, "
        f"mapper={mapper_type}, merge={merge_tokens}, LS={ls:.2f}"
    )

    return model, ls, clip_model


# ------------------------------------------------------------------ #
#  Val loader                                                         #
# ------------------------------------------------------------------ #

def build_val_loader(clip_preprocess, val_shard_path: str, batch_size: int = 1024):
    """
    Build DataLoader for CC12M val shard.
    Returns batches of (images,) tensors.
    """
    def _is_valid(data):
        return "jpg" in data and data["jpg"] is not None

    def _transform(data):
        try:
            img = clip_preprocess(data["jpg"].convert("RGB"))
            return (img,)
        except Exception:
            return None

    dataset = (
        wds.WebDataset(val_shard_path, handler=wds.warn_and_continue)
        .decode("pil")
        .select(_is_valid)
        .map(_transform)
        .select(lambda x: x is not None)
        .batched(batch_size, partial=True)
    )

    loader = wds.WebLoader(dataset, batch_size=None, num_workers=1, pin_memory=True)
    return loader


# ------------------------------------------------------------------ #
#  Core metrics computation                                           #
# ------------------------------------------------------------------ #

@torch.no_grad()
def compute_metrics_for_batch(model, imgs, ls_value, device):
    """
    Compute all negative-structure metrics for one batch.

    Takes B images, computes query embeddings p and target embeddings v,
    builds BxB similarity matrix, then for each query analyzes the
    distribution of its B-1 negative similarities.

    Returns dict of metric_name -> list of per-query values.
    """
    from model.model import FusedWeightedMLPMapper

    B = imgs.shape[0]
    imgs = imgs.to(device)

    # ---- Forward pass to get p (query) and v (target) ---- #

    if model.use_multilayer:
        v_raw, intermediate = model.clip.encode_image_multilayer(imgs)
        if isinstance(model.mapper, FusedWeightedMLPMapper):
            pseudo = model.mapper(v_raw.detach(), intermediate)
        else:
            pseudo = model.mapper(intermediate)
    else:
        v_raw = model.clip.encoder_image_unnorm(imgs)
        pseudo = model.mapper(v_raw)

    if model.merger is not None:
        pseudo = model.merger(pseudo)

    # Same fixed-norm projection as Pic2Word.encode_query (unit-norm runs); no-op when 0.0.
    if model.pseudo_norm > 0:
        pseudo = F.normalize(pseudo, dim=-1) * model.pseudo_norm

    # v = normalized CLIP image embedding (retrieval target)
    v = F.normalize(v_raw, dim=-1)

    # p = query embedding (pseudo-token through text transformer)
    tok_emb, eot_pos = model._build_token_sequence(pseudo)
    p = model.clip.forward_text_transform(tok_emb, eot_pos)

    # ---- Similarity matrix ---- #

    sim_matrix = p @ v.T  # (B, B)

    # ---- Compute per-query metrics ---- #

    diag_mask = torch.eye(B, dtype=torch.bool, device=device)
    neg_mask = ~diag_mask

    metrics = {
        "neg_std": [],
        "hard_soft_gap": [],
        "gt_minus_hardest": [],
        "neg_mean": [],
        "sim_p_vraw": [],
        "prob_on_gt": [],
        "neg_entropy": [],
        "effective_negs": [],
        "top5_neg_mass": [],
        "neg_entropy_raw": [],
        "effective_negs_raw": [],
    }

    for i in range(B):
        neg_sims = sim_matrix[i][neg_mask[i]]       # (B-1,) negative similarities
        sim_gt   = sim_matrix[i, i].item()           # positive similarity

        hardest = neg_sims.max().item()
        softest = neg_sims.min().item()

        # --- Raw similarity metrics (mapper quality) --- #

        metrics["neg_std"].append(neg_sims.std().item())
        metrics["hard_soft_gap"].append(hardest - softest)
        metrics["gt_minus_hardest"].append(sim_gt - hardest)
        metrics["neg_mean"].append(neg_sims.mean().item())
        metrics["sim_p_vraw"].append(sim_gt)

        # --- Post-LS softmax metrics (what the loss sees) --- #

        # Full softmax including positive — to get P(gt)
        logits_full = ls_value * sim_matrix[i]       # (B,)
        probs_full  = F.softmax(logits_full, dim=-1)
        metrics["prob_on_gt"].append(probs_full[i].item())

        # Neg-only softmax — entropy over just negatives
        neg_logits = ls_value * neg_sims             # (B-1,)
        neg_probs  = F.softmax(neg_logits, dim=-1)   # (B-1,)
        neg_ent    = -(neg_probs * neg_probs.clamp_min(1e-10).log()).sum().item()

        metrics["neg_entropy"].append(neg_ent)
        metrics["effective_negs"].append(math.exp(neg_ent))

        top5_mass = neg_probs.topk(min(5, len(neg_probs))).values.sum().item()
        metrics["top5_neg_mass"].append(top5_mass)

        # --- Raw softmax (no LS, T=1) — isolates mapper vs LS --- #

        raw_neg_probs = F.softmax(neg_sims, dim=-1)
        raw_ent = -(raw_neg_probs * raw_neg_probs.clamp_min(1e-10).log()).sum().item()

        metrics["neg_entropy_raw"].append(raw_ent)
        metrics["effective_negs_raw"].append(math.exp(raw_ent))

    return metrics


# ------------------------------------------------------------------ #
#  Orchestration                                                      #
# ------------------------------------------------------------------ #

def analyze_checkpoint(spec: CheckpointSpec,
                       val_shard_path: str,
                       device: str = "cuda",
                       max_batches: int = 5,
                       batch_size: int = 1024) -> Dict:
    """
    Run full negative-structure analysis on one checkpoint.
    Loads model, iterates over val shard batches, aggregates metrics.
    """
    model, ls_value, clip_model_name = load_model_from_checkpoint(spec, device)
    _, clip_preprocess = clip.load(clip_model_name, device="cpu")
    val_loader = build_val_loader(clip_preprocess, val_shard_path, batch_size)

    all_metrics = None
    total_queries = 0

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        imgs = batch[0]
        batch_metrics = compute_metrics_for_batch(model, imgs, ls_value, device)

        if all_metrics is None:
            all_metrics = {k: [] for k in batch_metrics}

        for k, v_list in batch_metrics.items():
            all_metrics[k].extend(v_list)

        total_queries += imgs.shape[0]
        logger.info(f"  Batch {batch_idx + 1}: {imgs.shape[0]} queries (total: {total_queries})")

    # Aggregate across all queries
    result = {
        "name": spec.name,
        "group": spec.group,
        "description": spec.description,
        "r10": spec.r10,
        "logit_scale": ls_value,
        "n_queries": total_queries,
        "hidden_dim": spec.hidden_dim,
        "num_tokens": spec.num_tokens,
        "mapper_type": spec.mapper_type,
    }

    for k, vals in all_metrics.items():
        n = len(vals)
        mean = sum(vals) / n
        variance = sum((x - mean) ** 2 for x in vals) / n
        result[f"{k}_mean"] = mean
        result[f"{k}_std"] = variance ** 0.5

    # Free GPU memory before loading next checkpoint
    del model
    torch.cuda.empty_cache()

    return result


# ------------------------------------------------------------------ #
#  Output                                                             #
# ------------------------------------------------------------------ #

def print_results_table(results: List[Dict]):
    """Print formatted comparison table to stdout."""

    print("\n" + "=" * 160)
    print("  NEGATIVE STRUCTURE ANALYSIS — CC12M Val Shard")
    print("=" * 160)

    # Section 1: Raw similarity metrics
    print("\n--- RAW SIMILARITY (mapper quality, before LS) ---")
    header = (
        f"{'#':<3} {'Checkpoint':<28} {'Grp':<5} {'R@10':<7} {'hd':>4} "
        f"{'LS':>7} {'neg_std':>9} {'h-s_gap':>9} {'gt-hard':>9} "
        f"{'neg_mean':>10} {'sim_pv':>9}"
    )
    print(header)
    print("-" * 115)
    for idx, r in enumerate(results):
        print(
            f"{idx+1:<3} "
            f"{r['name']:<28} "
            f"{r['group']:<5} "
            f"{r['r10']:<7} "
            f"{r['hidden_dim']:>4} "
            f"{r['logit_scale']:>7.1f} "
            f"{r['neg_std_mean']:>9.5f} "
            f"{r['hard_soft_gap_mean']:>9.5f} "
            f"{r['gt_minus_hardest_mean']:>9.5f} "
            f"{r['neg_mean_mean']:>10.5f} "
            f"{r['sim_p_vraw_mean']:>9.5f}"
        )

    # Section 2: Post-LS softmax metrics
    print("\n--- POST-LS SOFTMAX (what the loss sees after LS amplification) ---")
    header = (
        f"{'#':<3} {'Checkpoint':<28} {'R@10':<7} {'LS':>7} "
        f"{'P(gt)':>10} {'neg_ent':>9} {'eff_negs':>10} {'top5_mass':>10}"
    )
    print(header)
    print("-" * 95)
    for idx, r in enumerate(results):
        print(
            f"{idx+1:<3} "
            f"{r['name']:<28} "
            f"{r['r10']:<7} "
            f"{r['logit_scale']:>7.1f} "
            f"{r['prob_on_gt_mean']:>10.6f} "
            f"{r['neg_entropy_mean']:>9.3f} "
            f"{r['effective_negs_mean']:>10.1f} "
            f"{r['top5_neg_mass_mean']:>10.5f}"
        )

    # Section 3: Raw softmax (no LS)
    print("\n--- RAW SOFTMAX (T=1, no LS — isolates mapper from LS effect) ---")
    header = f"{'#':<3} {'Checkpoint':<28} {'R@10':<7} {'raw_ent':>9} {'raw_eff_negs':>13}"
    print(header)
    print("-" * 65)
    for idx, r in enumerate(results):
        print(
            f"{idx+1:<3} "
            f"{r['name']:<28} "
            f"{r['r10']:<7} "
            f"{r['neg_entropy_raw_mean']:>9.3f} "
            f"{r['effective_negs_raw_mean']:>13.1f}"
        )

    print("\n" + "=" * 160)

    # Legend
    print("\nMetric Legend:")
    print("  neg_std      : Std of negative cosine sims (higher = more graduated structure)")
    print("  h-s_gap      : max(neg) - min(neg) per query (higher = wider spread)")
    print("  gt-hard      : sim(p,gt) - sim(p,hardest_neg) (higher = positive clearly separated)")
    print("  neg_mean     : Mean negative similarity level")
    print("  sim_pv       : sim(p, v_raw) reconstruction sim (higher = more identity-like)")
    print("  P(gt)        : Softmax prob on positive (1.0 = fully saturated, no gradient)")
    print("  neg_ent      : Entropy of neg softmax WITH LS (higher = more negs contribute)")
    print("  eff_negs     : exp(neg_entropy) = effective number of negatives (max=B-1)")
    print("  top5_mass    : Prob mass in top-5 negatives (higher = more concentrated)")
    print("  raw_ent      : Entropy WITHOUT LS (isolates mapper contribution)")
    print("  raw_eff_negs : Effective negs WITHOUT LS (mapper's intrinsic discrimination)")
    print()


def save_results(results: List[Dict], output_path: str):
    """Save full results to JSON for later analysis."""
    clean = []
    for r in results:
        clean.append({
            k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in r.items()
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(clean, f, indent=2)
    logger.info(f"Results saved to {output_path}")


# ------------------------------------------------------------------ #
#  Checkpoint definitions — 22 total                                  #
# ------------------------------------------------------------------ #

def build_checkpoint_list(base: str) -> List[CheckpointSpec]:
    """
    Build the full list of 22 checkpoints to analyze.

    Groups:
      A — Reference baselines (1.5M)
      B — Attention imbalance / multi-token (1.5M)
      C — Merger experiments (1.5M)
      D — Alternative architectures (1.5M)
      E — Auxiliary losses (1.5M)
      F — 4M LS dynamics
      G — 4M margin experiments (hd=256)
      H — 4M capacity scaling (hd=512)
      I — 750k early baselines
    """
    return [
        # ============================================================ #
        #  GROUP A — Reference baselines (what "working" looks like)    #
        # ============================================================ #
        CheckpointSpec(
            name="1.5M_frozenLS",
            path=f"{base}/checkpoints_1500k/base_hd256_tok1/final.pt",
            group="A",
            r10="24.3%",
            description="1.5M, hd256, frozen LS=100 — baseline",
        ),
        CheckpointSpec(
            name="1.5M_trainLS_BEST",
            path=f"{base}/checkpoints_1500k_trainable_ls/base_hd256_tok1/final.pt",
            group="A",
            r10="24.6%",
            description="1.5M, hd256, trainable LS~115 — BEST overall",
        ),

        # ============================================================ #
        #  GROUP B — Attention imbalance (1.5M)                         #
        # ============================================================ #
        CheckpointSpec(
            name="1.5M_tok2",
            path=f"{base}/checkpoints_1500k_trainable_ls/base_hd256_tok2/epoch_00.pt",
            group="B",
            num_tokens=2,
            r10="~18%",
            description="tok=2 no merger — mild attention imbalance",
        ),
        CheckpointSpec(
            name="1.5M_tok4",
            path=f"{base}/checkpoints_1500k_trainable_ls/base_hd256_tok4/epoch_00.pt",
            group="B",
            num_tokens=4,
            r10="8.6%",
            description="tok=4 no merger — severe imbalance, Ref@10=93%",
        ),

        # ============================================================ #
        #  GROUP C — Merger experiments (1.5M)                          #
        # ============================================================ #
        CheckpointSpec(
            name="1.5M_tok4_merge1_L",
            path=f"{base}/checkpoints_1500k_merger/base_hd256_tok4_merge1/final.pt",
            group="C",
            num_tokens=4, merge_tokens=1, merge_type="learned",
            r10="24.3%",
            description="tok=4->1 learned merger — recovered 8.6%->24.3%",
        ),
        CheckpointSpec(
            name="1.5M_tok4_merge1_W",
            path=f"{base}/checkpoints_1500k_merger/base_hd256_tok4_merge1_weighted/final.pt",
            group="C",
            num_tokens=4, merge_tokens=1, merge_type="weighted",
            r10="22.4%",
            description="tok=4->1 weighted merger — collapsed to uniform",
        ),
        CheckpointSpec(
            name="1.5M_tok2_merge1_L",
            path=f"{base}/checkpoints_1500k_merger/base_hd256_tok2_merge1_learned/final.pt",
            group="C",
            num_tokens=2, merge_tokens=1, merge_type="learned",
            r10="23.6%",
            description="tok=2->1 learned merger — plateaued",
        ),

        # ============================================================ #
        #  GROUP D — Alternative architectures (1.5M)                   #
        # ============================================================ #
        CheckpointSpec(
            name="1.5M_MultiLayerCLS",
            path=f"{base}/checkpoints_1500k_trainable_ls/multilayer_hd256_tok1/epoch_00.pt",
            group="D",
            mapper_type="multilayer",
            r10="~17%",
            description="MultiLayerCLS mapper — intermediate CLS tokens",
        ),
        CheckpointSpec(
            name="1.5M_FusedWeighted",
            path=f"{base}/checkpoints_1500k_trainable_ls/fused_weighted_hd256_tok1/epoch_00.pt",
            group="D",
            mapper_type="fused_weighted",
            r10="~15%",
            description="FusedWeightedMLP — bottleneck failure",
        ),

        # ============================================================ #
        #  GROUP E — Auxiliary loss experiments (1.5M)                   #
        # ============================================================ #
        CheckpointSpec(
            name="1.5M_repulsion",
            path=f"{base}/checkpoints_1500k_aux/base_hd256_tok1_rep0.1_decorr0.0/final.pt",
            group="E",
            r10="~22%",
            description="Repulsion loss (wt=0.1) — negative result",
        ),

        # ============================================================ #
        #  GROUP F — 4M LS dynamics experiments                         #
        # ============================================================ #
        CheckpointSpec(
            name="4M_frozenLS",
            path=f"{base}/4M_checkpoints/checkpoints_frozen_ls/base_hd256_tok1/final.pt",
            group="F",
            r10="~23%",
            description="4M, frozen LS=100",
        ),
        CheckpointSpec(
            name="4M_trainLS_EXPLODED",
            path=f"{base}/4M_checkpoints/checkpoints_trainable_ls/base_hd256_tok1/final.pt",
            group="F",
            r10="~22%",
            description="4M, LS exploded to 144 — CORE PATHOLOGY",
        ),

        # ============================================================ #
        #  GROUP G — 4M margin experiments (hd=256)                     #
        # ============================================================ #
        CheckpointSpec(
            name="4M_v1_broken",
            path=f"{base}/4M_checkpoints/base_hd256_tok1_margin0.1_mw0.1_lslr0.05_lsmax116/final.pt",
            group="G",
            r10="18.9%",
            description="margin delayed (mw=0.1, delay=0.2) — collapsed",
        ),
        CheckpointSpec(
            name="4M_v2_best256",
            path=f"{base}/4M_checkpoints/base_hd256_tok1_margin0.1_mw0.2_lslr0.2_lsmax116/final.pt",
            group="G",
            r10="24.1%",
            description="margin no-delay (mw=0.2) — best 4M hd256",
        ),
        CheckpointSpec(
            name="4M_v3_strong",
            path=f"{base}/4M_checkpoints/base_hd256_tok1_margin0.2_mw0.4_lslr0.4_lsmax116/final.pt",
            group="G",
            r10="23.6%",
            description="strong margin (mw=0.4) — stickiness down, R@10 worse",
        ),

        # ============================================================ #
        #  GROUP H — 4M capacity scaling (hd=512)                       #
        # ============================================================ #
        CheckpointSpec(
            name="4M_hd512_weak",
            path=f"{base}/4M_checkpoints/base_hd512_tok1_margin0.1_mw0.2_lslr0.2_lsmax116/final.pt",
            group="H",
            hidden_dim=512,
            r10="23.4%",
            description="hd512, mw=0.2 — stickiness 69%",
        ),
        CheckpointSpec(
            name="4M_hd512_strong",
            path=f"{base}/4M_checkpoints/base_hd512_tok1_margin0.2_mw0.4_lslr0.4_lsmax116/final.pt",
            group="H",
            hidden_dim=512,
            r10="22.9%",
            description="hd512, mw=0.4 — stickiness 75%, counterintuitive",
        ),
        CheckpointSpec(
            name="4M_hd512_nomargin",
            path=f"{base}/4M_checkpoints/base_hd512_tok1_lslr0.2_lsmax116/epoch_00_step_03000.pt",
            group="H",
            hidden_dim=512,
            r10="23.8%",
            description="hd512, NO margin — still climbing at step 3k",
        ),

        # ============================================================ #
        #  GROUP I — 750k early baselines                               #
        # ============================================================ #
        CheckpointSpec(
            name="750k_BaseMLP",
            path=f"{base}/checkpoints/base_hd256_tok1/final.pt",
            group="I",
            r10="~23%",
            description="750k, BaseMLP — earliest baseline",
        ),
        CheckpointSpec(
            name="750k_MultiLayer",
            path=f"{base}/checkpoints/multilayer_hd256_tok1/final.pt",
            group="I",
            mapper_type="multilayer",
            r10="~17%",
            description="750k, MultiLayerCLS",
        ),
        CheckpointSpec(
            name="750k_Weighted",
            path=f"{base}/checkpoints/weighted_hd256_tok1/final.pt",
            group="I",
            mapper_type="weighted",
            r10="~22%",
            description="750k, WeightedMLP",
        ),
    ]


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

if __name__ == "__main__":

    BASE = "/notebooks/experiment"
    VAL_SHARD = f"{BASE}/train_data/cc12m-train-0793.tar"
    OUTPUT_PATH = f"{BASE}/codes/evaluation/neg_structure_results.json"

    checkpoints = build_checkpoint_list(BASE)

    logger.info(f"Total checkpoints registered: {len(checkpoints)}")
    logger.info(f"Val shard: {VAL_SHARD}")
    logger.info(f"Output: {OUTPUT_PATH}")

    # Check val shard exists
    if not Path(VAL_SHARD).exists():
        logger.error(f"Val shard not found: {VAL_SHARD}")
        logger.error("Make sure train_data/cc12m-train-0793.tar exists.")
        exit(1)

    # Run analysis for each checkpoint
    results = []
    skipped = []
    failed = []

    for idx, spec in enumerate(checkpoints):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"[{idx+1}/{len(checkpoints)}] Analyzing: {spec.name} (Group {spec.group})")
        logger.info(f"  Path: {spec.path}")
        logger.info(f"  {spec.description}")
        logger.info(f"{'=' * 70}")

        if not Path(spec.path).exists():
            logger.warning(f"  SKIPPED — file not found: {spec.path}")
            skipped.append(spec.name)
            continue

        try:
            result = analyze_checkpoint(
                spec=spec,
                val_shard_path=VAL_SHARD,
                device="cuda",
                max_batches=5,
                batch_size=1024,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed.append(spec.name)
            continue

    # Summary
    logger.info(f"\n{'=' * 70}")
    logger.info(f"SUMMARY: {len(results)} analyzed, {len(skipped)} skipped, {len(failed)} failed")
    if skipped:
        logger.info(f"  Skipped: {', '.join(skipped)}")
    if failed:
        logger.info(f"  Failed:  {', '.join(failed)}")
    logger.info(f"{'=' * 70}")

    # Print comparison table
    if results:
        print_results_table(results)

    # Save to JSON
    save_results(results, OUTPUT_PATH)
"""
evaluation/retrieval.py
"""
import logging
from typing import List, Optional, Dict
import clip
import torch
import torch.nn.functional as F
from model.model import Pic2Word
logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_features(model      : Pic2Word,
                     loader,
                     device     : str,
                     text_list  : Optional[List[str]] = None) -> torch.Tensor:
    model.eval()
    all_feats = []
    text_idx  = 0

    for batch in loader:
        B    = batch[0].shape[0]
        imgs = batch[0].to(device)

        if text_list is None:
            feats = model.encode_candidates(imgs)
        else:
            batch_text  = text_list[text_idx : text_idx + B]
            text_tokens = clip.tokenize(batch_text, truncate=True).to(device)
            feats       = model.encode_query(imgs, text_tokens)
            text_idx   += B

        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0)


def recall_at_k(sims       : torch.Tensor,
                gt_indices : torch.Tensor,
                k_values   : List[int] = [1, 5, 10, 50]) -> Dict[int, float]:
    """
    sims       : (M, N) similarity matrix
    gt_indices : (M,)  one correct GT index per query
    """
    M       = sims.shape[0]
    recalls = {}

    for k in k_values:
        topk = torch.topk(sims, k, dim=-1).indices
        hits = sum(gt_indices[i].item() in topk[i].tolist() for i in range(M))
        recalls[k] = hits / M

    return recalls


def recall_subset_at_k(sims       : torch.Tensor,
                       gt_indices : torch.Tensor,
                       subset_ids : List[List[int]],
                       k_values   : List[int] = [1, 5, 10, 50]) -> Dict[int, float]:
    M       = sims.shape[0]
    recalls = {}

    for k in k_values:
        hits = 0
        for i, gt in enumerate(gt_indices):
            sub_idxs = torch.tensor(subset_ids[i], dtype=torch.int64)
            sub_sims = sims[i, sub_idxs]
            topk     = sub_idxs[torch.topk(sub_sims, k, dim=-1).indices]
            if gt.item() in topk.tolist():
                hits += 1
        recalls[k] = hits / M

    return recalls


def log_recalls(recalls : Dict[int, float], prefix : str = ""):
    r = " | ".join([f"R@{k}={v*100:.2f}%" for k, v in sorted(recalls.items())])
    logger.info(f"{prefix}{r}")
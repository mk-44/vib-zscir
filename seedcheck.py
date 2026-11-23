"""
seedcheck.py -- throwaway reproducibility check for the seeded WebDataset loader.
Builds the train loader TWICE with seed=3407 and confirms the first batches are
identical. Run on Paperspace from /notebooks/experiment/codes with PYTHONPATH set.

  export PYTHONPATH=/notebooks/experiment/codes
  python seedcheck.py
"""
import clip
import torch
from config.config import Config
from dataset.dataset import build_loaders


def make_cfg(seed):
    cfg = Config()          # same default-config path train.py uses when no --config
    cfg.seed = seed
    return cfg


def first_batches(seed, n=3):
    cfg = make_cfg(seed)
    _, preprocess = clip.load(cfg.clip_model, device="cpu")   # exactly as train.py
    train_loader, _, streaming, n_missing = build_loaders(cfg=cfg, clip_preprocess=preprocess)
    if streaming:
        print(f"  !!! WARNING: {n_missing} shard(s) STREAMING from HF -> run will NOT be reproducible")
    out = []
    for i, batch in enumerate(train_loader):
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        out.append(imgs.clone())
        if i + 1 >= n:
            break
    return out


print("Build A (seed=3407) ...")
A = first_batches(3407)
print("Build B (seed=3407) ...")
B = first_batches(3407)

allmatch = True
for i, (a, b) in enumerate(zip(A, B)):
    same = torch.equal(a, b)
    allmatch &= same
    print(f"  batch {i}: {tuple(a.shape)} vs {tuple(b.shape)} -> {'MATCH' if same else 'DIFFER'}")

print("\nRESULT:", "REPRODUCIBLE -- seeded loader works, safe to launch the 9h run"
      if allmatch else
      "NOT reproducible -- data order differs; do NOT trust 'reproducible' claim yet")
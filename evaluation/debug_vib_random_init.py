"""
evaluation/debug_vib_random_init.py

Test whether random initialization causes VIBMapper deterministic 
to diverge from BaseMLP.

Runs both mapper types with 5 different random seeds for 500 steps
on real training data shards. If results diverge for some seeds but
not others, init sensitivity is confirmed.

Usage:
    cd /notebooks/experiment/codes
    PYTHONPATH=/notebooks/experiment/codes python evaluation/debug_vib_random_init.py
"""

import torch
import torch.nn.functional as F
import clip
import math
import webdataset as wds
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from config.config import load_from_yaml
from model.model import Pic2Word, VIBMapper


def build_train_loader(preprocess, data_dir, batch_size=1024):
    """Use actual training shards, not val shard."""
    import os
    # Use first 10 shards for speed
    shard_pattern = os.path.join(data_dir, "cc12m-train-{0000..0009}.tar")
    ds = (wds.WebDataset(shard_pattern, shardshuffle=True)
      .shuffle(2000)
      .decode('pil')
      .to_tuple('jpg;png', handler=wds.handlers.warn_and_continue)
      .map_tuple(preprocess)
      .repeat())
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=2)


def run_one_seed(seed, mapper_type, preprocess, device, n_steps=500):
    """
    Train for n_steps with given seed and mapper_type.
    Returns dict of metrics at each checkpoint.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    cfg = load_from_yaml('config/config.yaml',
        mapper_type=mapper_type, hidden_dim=256, num_tokens=1,
        ls_lr_scale=0.2, ls_max=116.0, device=device)
    model = Pic2Word(cfg).to(device)
    
    # Build optimizer (same as _build_optimizer with ls_lr_scale=0.2)
    mapper_params = list(model.mapper.parameters())
    opt = AdamW([
        {'params': mapper_params, 'lr': 1e-4},
        {'params': [model.clip.logit_scale], 'lr': 1e-4 * 0.2},
    ], weight_decay=0.1)
    
    scaler = GradScaler(enabled=True)
    loader = build_train_loader(preprocess, '/notebooks/experiment/train_data', batch_size=1024)
    
    model.train()
    results = []
    step = 0
    
    for batch in loader:
        if step >= n_steps:
            break
        
        imgs = batch[0].to(device)
        opt.zero_grad(set_to_none=True)
        
        with autocast(enabled=True):
            if isinstance(model.mapper, VIBMapper):
                out = model(imgs, vib_beta=0.0, vib_delay=True)
            else:
                out = model(imgs)
            loss = out['loss']
        
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        scaler.step(opt)
        
        if cfg.ls_max > 0.0:
            with torch.no_grad():
                model.clip.logit_scale.clamp_(0.0, math.log(cfg.ls_max))
        
        scaler.update()
        
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                v_raw = model.clip.encoder_image_unnorm(imgs[:64])
                if isinstance(model.mapper, VIBMapper):
                    pseudo = model.mapper(v_raw, deterministic=True)
                else:
                    pseudo = model.mapper(v_raw)
                p_norm = pseudo.norm(dim=-1).mean().item()
            model.train()
            
            results.append({
                'step': step,
                'loss': loss.item(),
                'l_t2i': out['loss_t2i'].item(),
                'ls': out['logit_scale'].item(),
                'p_norm': p_norm,
            })
        
        step += 1
    
    del model, opt, scaler
    torch.cuda.empty_cache()
    return results


def main():
    device = 'cuda'
    seeds = [42, 123, 456, 789, 1337]
    n_steps = 500
    
    _, preprocess = clip.load('ViT-L/14', device='cpu')
    
    all_results = {}
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        
        # BaseMLP
        print(f"\n  --- BaseMLP (seed={seed}) ---")
        base_results = run_one_seed(seed, 'base', preprocess, device, n_steps)
        for r in base_results:
            print(f"    step {r['step']:4d} | loss={r['loss']:.4f} | l_t2i={r['l_t2i']:.4f} | ls={r['ls']:.2f} | p_norm={r['p_norm']:.4f}")
        
        # VIBMapper deterministic
        print(f"\n  --- VIBMapper deterministic (seed={seed}) ---")
        vib_results = run_one_seed(seed, 'vib', preprocess, device, n_steps)
        for r in vib_results:
            print(f"    step {r['step']:4d} | loss={r['loss']:.4f} | l_t2i={r['l_t2i']:.4f} | ls={r['ls']:.2f} | p_norm={r['p_norm']:.4f}")
        
        all_results[seed] = {'base': base_results, 'vib': vib_results}
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY — Final step comparison (step 400)")
    print(f"{'='*60}")
    print(f"  {'Seed':>6s}  {'BaseMLP loss':>12s}  {'VIB loss':>12s}  {'Diff':>8s}  {'Base norm':>10s}  {'VIB norm':>10s}  {'Norm diff':>10s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
    
    for seed in seeds:
        b = all_results[seed]['base'][-1]
        v = all_results[seed]['vib'][-1]
        loss_diff = v['loss'] - b['loss']
        norm_diff = v['p_norm'] - b['p_norm']
        print(f"  {seed:6d}  {b['loss']:12.4f}  {v['loss']:12.4f}  {loss_diff:+8.4f}  {b['p_norm']:10.4f}  {v['p_norm']:10.4f}  {norm_diff:+10.4f}")
    
    print(f"\nInterpretation:")
    print(f"  If all seeds show similar diffs → systematic issue (architecture/optimizer)")
    print(f"  If some seeds diverge, others don't → init sensitivity")
    print(f"  If no seeds show meaningful diff → issue is elsewhere (data order, longer training)")


if __name__ == '__main__':
    main()
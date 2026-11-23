"""
evaluation/debug_vib_vs_base.py

Compare VIBMapper (deterministic mode) vs BaseMLP training dynamics.
Isolates whether the performance gap comes from:
  1. Optimizer state (AdamW tracking fc_logvar with zero grads)
  2. Gradient clipping (clip_grad_norm_ over 9 vs 7 params)
  3. Weight decay on fc_logvar (AdamW decays params even without grads)
  4. Random init difference (fc_logvar creation consumes RNG)

Runs 4 configurations for 500 steps each on val shard:
  A. BaseMLP (7 params) — baseline
  B. VIBMapper deterministic, full optimizer (9 params) — current broken setup
  C. VIBMapper deterministic, fc_logvar frozen (7 params) — tests optimizer hypothesis
  D. VIBMapper deterministic, fc_logvar in optimizer but excluded from grad_clip (9 params) — tests grad_clip hypothesis

Usage:
    cd /notebooks/experiment/codes
    PYTHONPATH=/notebooks/experiment/codes python evaluation/debug_vib_vs_base.py
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


def build_small_loader(preprocess, shard_path, batch_size=256):
    ds = wds.WebDataset(shard_path).repeat().decode('pil').to_tuple(
        'jpg;png', handler=wds.handlers.warn_and_continue
    ).map_tuple(preprocess)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=1)


def train_loop(model, optimizer, loader, device, n_steps=500,
               grad_clip=1.0, ls_max=116.0, ls_lr_scale=0.2,
               label="", exclude_logvar_from_clip=False):
    """
    Train for n_steps, log loss and pseudo norm every 100 steps.
    Returns final pseudo norm.
    """
    model.train()
    scaler = GradScaler(enabled=True)
    step = 0
    
    # Identify which params to clip
    if exclude_logvar_from_clip:
        # Only clip encoder + fc_mu + logit_scale (exclude fc_logvar)
        clip_params = []
        for n, p in model.mapper.named_parameters():
            if 'fc_logvar' not in n and p.requires_grad:
                clip_params.append(p)
        if model.clip.logit_scale.requires_grad:
            clip_params.append(model.clip.logit_scale)
    else:
        clip_params = [p for p in model.trainable_parameters() if p.requires_grad]
    
    results = []
    
    for batch in loader:
        if step >= n_steps:
            break
        
        imgs = batch[0].to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=True):
            # VIBMapper in delay mode or BaseMLP
            if isinstance(model.mapper, VIBMapper):
                out = model(imgs, vib_beta=0.0, vib_delay=True)
            else:
                out = model(imgs)
            loss = out['loss']
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
        
        scaler.step(optimizer)
        
        # LS clamp
        if ls_max > 0.0 and model.clip.logit_scale.requires_grad:
            with torch.no_grad():
                model.clip.logit_scale.clamp_(0.0, math.log(ls_max))
        
        scaler.update()
        
        if step % 100 == 0:
            # Measure pseudo norm
            model.eval()
            with torch.no_grad():
                v_raw = model.clip.encoder_image_unnorm(imgs[:64])
                if isinstance(model.mapper, VIBMapper):
                    pseudo = model.mapper(v_raw, deterministic=True)
                else:
                    pseudo = model.mapper(v_raw)
                p_norm = pseudo.norm(dim=-1).mean().item()
            model.train()
            
            ls_val = out['logit_scale'].item()
            loss_val = loss.item()
            l_t2i = out['loss_t2i'].item()
            
            # Check grad norms
            grad_norms = {}
            for n, p in model.mapper.named_parameters():
                if p.grad is not None:
                    grad_norms[n] = p.grad.norm().item()
                else:
                    grad_norms[n] = 0.0
            
            results.append({
                'step': step, 'loss': loss_val, 'l_t2i': l_t2i,
                'ls': ls_val, 'p_norm': p_norm, 'grad_norms': grad_norms
            })
            
            print(f"  [{label}] step {step:4d} | loss={loss_val:.4f} | l_t2i={l_t2i:.4f} "
                  f"| ls={ls_val:.2f} | p_norm={p_norm:.4f}")
        
        step += 1
    
    return results


def main():
    device = 'cuda'
    VAL = '/notebooks/experiment/train_data/cc12m-train-0793.tar'
    N_STEPS = 500
    
    _, preprocess = clip.load('ViT-L/14', device='cpu')
    
    # =====================================================================
    # Config A: BaseMLP (7 params) — baseline
    # =====================================================================
    print("\n" + "="*60)
    print("CONFIG A: BaseMLP (baseline)")
    print("="*60)
    
    torch.manual_seed(42)
    cfg_a = load_from_yaml('config/config.yaml',
        mapper_type='base', hidden_dim=256, num_tokens=1,
        ls_lr_scale=0.2, ls_max=116.0, device=device)
    model_a = Pic2Word(cfg_a).to(device)
    
    # Save init weights for later comparison
    init_weights_base = {n: p.data.clone() for n, p in model_a.mapper.named_parameters()}
    
    opt_a = AdamW([
        {'params': list(model_a.mapper.parameters()), 'lr': 1e-4},
        {'params': [model_a.clip.logit_scale], 'lr': 1e-4 * 0.2},
    ], weight_decay=0.1)
    
    loader_a = build_small_loader(preprocess, VAL)
    results_a = train_loop(model_a, opt_a, loader_a, device, N_STEPS, label="BaseMLP")
    
    # =====================================================================
    # Config B: VIBMapper deterministic, full optimizer (9 params) — broken
    # =====================================================================
    print("\n" + "="*60)
    print("CONFIG B: VIBMapper deterministic, FULL optimizer (9 params)")
    print("="*60)
    
    torch.manual_seed(42)
    cfg_b = load_from_yaml('config/config.yaml',
        mapper_type='vib', hidden_dim=256, num_tokens=1,
        ls_lr_scale=0.2, ls_max=116.0, device=device)
    model_b = Pic2Word(cfg_b).to(device)
    
    # Copy BaseMLP init weights to VIBMapper
    model_b.mapper.encoder[0].weight.data = init_weights_base['net.0.weight'].clone()
    model_b.mapper.encoder[0].bias.data   = init_weights_base['net.0.bias'].clone()
    model_b.mapper.encoder[3].weight.data = init_weights_base['net.3.weight'].clone()
    model_b.mapper.encoder[3].bias.data   = init_weights_base['net.3.bias'].clone()
    model_b.mapper.fc_mu.weight.data      = init_weights_base['net.6.weight'].clone()
    model_b.mapper.fc_mu.bias.data        = init_weights_base['net.6.bias'].clone()
    model_b.clip.logit_scale.data         = model_a.clip.logit_scale.data.clone()
    
    # Reload model_a's logit_scale since it was trained
    # Actually need fresh logit_scale — reload from CLIP
    _, _ = clip.load('ViT-L/14', device='cpu')  # just to reset
    # Reset both to same init
    init_ls = torch.tensor(4.6052)  # ln(100)
    model_b.clip.logit_scale.data.fill_(init_ls.item())
    
    # Rebuild model_a from scratch with same seed for fair comparison
    torch.manual_seed(42)
    model_a2 = Pic2Word(cfg_a).to(device)
    model_a2.clip.logit_scale.data.fill_(init_ls.item())
    
    # Copy same init to VIB
    model_b.mapper.encoder[0].weight.data = model_a2.mapper.net[0].weight.data.clone()
    model_b.mapper.encoder[0].bias.data   = model_a2.mapper.net[0].bias.data.clone()
    model_b.mapper.encoder[3].weight.data = model_a2.mapper.net[3].weight.data.clone()
    model_b.mapper.encoder[3].bias.data   = model_a2.mapper.net[3].bias.data.clone()
    model_b.mapper.fc_mu.weight.data      = model_a2.mapper.net[6].weight.data.clone()
    model_b.mapper.fc_mu.bias.data        = model_a2.mapper.net[6].bias.data.clone()
    model_b.clip.logit_scale.data         = model_a2.clip.logit_scale.data.clone()
    
    opt_b = AdamW([
        {'params': list(model_b.mapper.parameters()), 'lr': 1e-4},  # includes fc_logvar!
        {'params': [model_b.clip.logit_scale], 'lr': 1e-4 * 0.2},
    ], weight_decay=0.1)
    
    loader_b = build_small_loader(preprocess, VAL)
    results_b = train_loop(model_b, opt_b, loader_b, device, N_STEPS, label="VIB-full-opt")
    
    # =====================================================================
    # Config C: VIBMapper deterministic, fc_logvar FROZEN (7 params)
    # =====================================================================
    print("\n" + "="*60)
    print("CONFIG C: VIBMapper deterministic, fc_logvar FROZEN (7 params)")
    print("="*60)
    
    torch.manual_seed(42)
    model_c = Pic2Word(cfg_b).to(device)
    
    # Copy same init weights
    model_c.mapper.encoder[0].weight.data = model_a2.mapper.net[0].weight.data.clone()
    model_c.mapper.encoder[0].bias.data   = model_a2.mapper.net[0].bias.data.clone()
    model_c.mapper.encoder[3].weight.data = model_a2.mapper.net[3].weight.data.clone()
    model_c.mapper.encoder[3].bias.data   = model_a2.mapper.net[3].bias.data.clone()
    model_c.mapper.fc_mu.weight.data      = model_a2.mapper.net[6].weight.data.clone()
    model_c.mapper.fc_mu.bias.data        = model_a2.mapper.net[6].bias.data.clone()
    model_c.clip.logit_scale.data         = model_a2.clip.logit_scale.data.clone()
    
    # FREEZE fc_logvar
    model_c.mapper.fc_logvar.weight.requires_grad = False
    model_c.mapper.fc_logvar.bias.requires_grad = False
    
    # Only trainable params in optimizer
    trainable_c = [p for p in model_c.mapper.parameters() if p.requires_grad]
    trainable_c.append(model_c.clip.logit_scale)
    
    opt_c = AdamW([
        {'params': [p for p in model_c.mapper.parameters() if p.requires_grad], 'lr': 1e-4},
        {'params': [model_c.clip.logit_scale], 'lr': 1e-4 * 0.2},
    ], weight_decay=0.1)
    
    loader_c = build_small_loader(preprocess, VAL)
    results_c = train_loop(model_c, opt_c, loader_c, device, N_STEPS, label="VIB-frozen-logvar")
    
    # =====================================================================
    # Config D: VIBMapper deterministic, fc_logvar in optimizer but excluded from grad_clip
    # =====================================================================
    print("\n" + "="*60)
    print("CONFIG D: VIBMapper deterministic, fc_logvar in opt but EXCLUDED from grad_clip")
    print("="*60)
    
    torch.manual_seed(42)
    model_d = Pic2Word(cfg_b).to(device)
    
    # Copy same init weights
    model_d.mapper.encoder[0].weight.data = model_a2.mapper.net[0].weight.data.clone()
    model_d.mapper.encoder[0].bias.data   = model_a2.mapper.net[0].bias.data.clone()
    model_d.mapper.encoder[3].weight.data = model_a2.mapper.net[3].weight.data.clone()
    model_d.mapper.encoder[3].bias.data   = model_a2.mapper.net[3].bias.data.clone()
    model_d.mapper.fc_mu.weight.data      = model_a2.mapper.net[6].weight.data.clone()
    model_d.mapper.fc_mu.bias.data        = model_a2.mapper.net[6].bias.data.clone()
    model_d.clip.logit_scale.data         = model_a2.clip.logit_scale.data.clone()
    
    opt_d = AdamW([
        {'params': list(model_d.mapper.parameters()), 'lr': 1e-4},  # includes fc_logvar
        {'params': [model_d.clip.logit_scale], 'lr': 1e-4 * 0.2},
    ], weight_decay=0.1)
    
    loader_d = build_small_loader(preprocess, VAL)
    results_d = train_loop(model_d, opt_d, loader_d, device, N_STEPS, label="VIB-no-clip-logvar",
                           exclude_logvar_from_clip=True)
    
    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "="*60)
    print("SUMMARY after 500 steps")
    print("="*60)
    
    for name, results in [
        ("A. BaseMLP (baseline)",           results_a),
        ("B. VIB full opt (9 params)",      results_b),
        ("C. VIB frozen logvar (7 params)", results_c),
        ("D. VIB no-clip logvar",           results_d),
    ]:
        final = results[-1]
        print(f"  {name:40s} | loss={final['loss']:.4f} | l_t2i={final['l_t2i']:.4f} "
              f"| ls={final['ls']:.2f} | p_norm={final['p_norm']:.4f}")
    
    print("\nInterpretation:")
    print("  If C matches A → optimizer/weight_decay on fc_logvar is the cause")
    print("  If D matches A → gradient clipping including fc_logvar is the cause")
    print("  If both C and D match A → both contribute")
    print("  If neither matches A → something else entirely")


if __name__ == '__main__':
    main()
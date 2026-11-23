"""Neg-structure analysis: all checkpoints × {Fashion-IQ, CIRCO}. Auto-gen v_raw if missing."""
import json, logging, math, time, sys, random
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, '/notebooks/experiment/codes')
sys.path.insert(0, '/notebooks/experiment/codes/evaluation')

from metrics.paths import METRICS_ROOT, derive_run_id_and_ckpt_tag
from config.config import load_from_yaml
from model.model import Pic2Word

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)

SEARCH_ROOTS = [
    "/notebooks/experiment/4M_checkpoints",
    "/notebooks/experiment/checkpoints_1500k",
    "/notebooks/experiment/checkpoints_1500k_trainable_ls",
    "/notebooks/experiment/checkpoints_1500k_merger",
    "/notebooks/experiment/checkpoints_1500k_aux",
]
DATASETS = {
    'fashion': '/notebooks/experiment/eval_data/fashion_iq',
    'circo':   '/notebooks/experiment/eval_data/circo',
}
OUT_ROOT    = METRICS_ROOT / "neg_structure"
BATCH_SIZE  = 1024
MAX_BATCHES = 5
DEVICE      = 'cuda'


def _generate_vraw(eval_dir, name):
    """No v_raw cache found — encode images in eval_dir with CLIP, save cache, return."""
    import clip as _clip
    from PIL import Image
    log.info(f"  Generating v_raw for {name} (no cache found)...")
    imgs = []
    for ext in ('*.png','*.jpg','*.jpeg','*.PNG','*.JPG','*.JPEG'):
        imgs.extend(Path(eval_dir).rglob(ext))
    if not imgs:
        raise RuntimeError(f"No images under {eval_dir} to generate v_raw")
    log.info(f"  Found {len(imgs)} images")
    n_want = MAX_BATCHES * BATCH_SIZE * 2
    if len(imgs) > n_want:
        random.seed(42); imgs = random.sample(imgs, n_want)
    clip_model, preprocess = _clip.load("ViT-L/14", device=DEVICE)
    clip_model.eval()
    embs, buf = [], []
    with torch.no_grad():
        for i, ip in enumerate(imgs):
            try:
                buf.append(preprocess(Image.open(ip).convert("RGB")))
            except Exception:
                continue
            if len(buf) >= 128:
                e = clip_model.visual(torch.stack(buf).to(DEVICE).half() if next(clip_model.parameters()).dtype == torch.float16 else torch.stack(buf).to(DEVICE))
                embs.append(e.cpu().float()); buf = []
                done = sum(x.shape[0] for x in embs)
                if done % 1024 == 0: log.info(f"    encoded {done}/{len(imgs)}")
        if buf:
            e = clip_model.visual(torch.stack(buf).to(DEVICE).half() if next(clip_model.parameters()).dtype == torch.float16 else torch.stack(buf).to(DEVICE))
            embs.append(e.cpu().float())
    all_emb = torch.cat(embs, dim=0)
    cache_path = Path(eval_dir) / f"vraw_cache_generated_ViT-L-14.pt"
    torch.save({'embeddings': all_emb}, cache_path)
    log.info(f"  Saved {tuple(all_emb.shape)} → {cache_path}")
    del clip_model; torch.cuda.empty_cache()
    return all_emb


def load_dataset(eval_dir, name):
    parts = []
    for f in sorted(Path(eval_dir).rglob("*.pt")):
        try:
            d = torch.load(f, map_location='cpu', weights_only=False)
            emb = d.get('embeddings') if isinstance(d, dict) else d
            if isinstance(emb, torch.Tensor) and emb.dim() == 2 and emb.shape[1] == 768 and emb.shape[0] >= 64:
                parts.append(emb.float())
                log.info(f"  cache {f.relative_to(eval_dir)}: {tuple(emb.shape)}")
        except Exception:
            pass
    if not parts:
        parts = [_generate_vraw(eval_dir, name)]
    v = torch.cat(parts, dim=0)
    g = torch.Generator().manual_seed(42)
    v = v[torch.randperm(v.shape[0], generator=g)][:MAX_BATCHES * BATCH_SIZE]
    return [v[i:i+BATCH_SIZE] for i in range(0, v.shape[0], BATCH_SIZE) if v[i:i+BATCH_SIZE].shape[0] >= 64]


def discover_checkpoints():
    seen, uniq = set(), []
    for root in SEARCH_ROOTS:
        rp = Path(root)
        if not rp.exists():
            log.warning(f"Root missing: {root}"); continue
        for p in sorted(rp.rglob("*.pt")):
            if p.resolve() not in seen:
                seen.add(p.resolve()); uniq.append(p)
    return uniq


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cc = ck.get("config", {}); mk = ck.get("mapper_kwargs", {})
    mtype = ck.get('mapper_type', 'base')
    if mtype in ('multilayer', 'weighted', 'fused_weighted'):
        return None, None, {'mapper_type': mtype, 'skipped_reason': 'needs_intermediate_features'}
    cfg = load_from_yaml(
        '/notebooks/experiment/codes/config/config.yaml',
        clip_model=cc.get('clip_model','ViT-L/14'), mapper_type=mtype,
        hidden_dim=mk.get('hidden_dim', cc.get('hidden_dim',256)),
        num_tokens=mk.get('num_tokens', cc.get('num_tokens',1)),
        num_heads=cc.get('num_heads',8), num_layers=cc.get('num_layers',1),
        dropout=mk.get('dropout', cc.get('dropout',0.1)),
        merge_tokens=cc.get('merge_tokens',None), merge_type=cc.get('merge_type','learned'),
        max_token_len=77, clip_prompt=cc.get('clip_prompt','a photo of'),
        vib_logvar_bias=cc.get('vib_logvar_bias',-2.0),
        vib_logvar_clip_max=cc.get('vib_logvar_clip_max',2.0),
        # Forward-pass edit z <- rho*z/||z|| of unit-norm runs must also apply at test; absent/0.0 = off.
        pseudo_norm=cc.get('pseudo_norm',0.0), device=DEVICE,
    )
    m = Pic2Word(cfg).to(DEVICE)
    m.mapper.load_state_dict(ck['mapper_state'])
    if ck.get('merger_state') and m.merger is not None:
        m.merger.load_state_dict(ck['merger_state'])
    if ck.get('logit_scale') is not None:
        with torch.no_grad(): m.clip.logit_scale.fill_(ck['logit_scale'])
    m = m.float().eval()
    ls = m.clip.logit_scale.exp().item()
    meta = {
        'logit_scale': ls, 'mapper_type': mtype,
        'hidden_dim': mk.get('hidden_dim', cc.get('hidden_dim')),
        'num_tokens': mk.get('num_tokens', cc.get('num_tokens')),
        'epoch': ck.get('epoch'), 'step': ck.get('step'),
        'ls_lr_scale': cc.get('ls_lr_scale'), 'ls_max': cc.get('ls_max'),
        'margin': cc.get('margin'), 'margin_weight': cc.get('margin_weight'),
        'margin_delay': cc.get('margin_delay'),
        'noise_sigma': cc.get('noise_sigma'), 'pseudo_norm': cc.get('pseudo_norm'),
        'vib_beta': cc.get('vib_beta'),
    }
    return m, ls, meta


@torch.no_grad()
def compute_metrics(model, batch, ls):
    B = batch.shape[0]
    v_raw = batch.to(DEVICE).float()
    pseudo = model.mapper(v_raw)
    if model.merger is not None: pseudo = model.merger(pseudo)
    # Same fixed-norm projection as Pic2Word.encode_query (unit-norm runs); no-op when 0.0.
    if model.pseudo_norm > 0: pseudo = F.normalize(pseudo, dim=-1) * model.pseudo_norm
    v = F.normalize(v_raw, dim=-1)
    tok_emb, eot_pos = model._build_token_sequence(pseudo)
    p = model.clip.forward_text_transform(tok_emb, eot_pos)
    sim = p @ v.T
    diag = torch.eye(B, dtype=torch.bool, device=DEVICE)
    out = {k: [] for k in ['neg_std','hard_soft_gap','gt_minus_hardest','neg_mean','sim_p_vraw',
                            'prob_on_gt','neg_entropy','effective_negs','top5_neg_mass',
                            'neg_entropy_raw','effective_negs_raw']}
    for i in range(B):
        neg = sim[i][~diag[i]]; pos = sim[i, i].item()
        hardest, softest = neg.max().item(), neg.min().item()
        out['neg_std'].append(neg.std().item())
        out['hard_soft_gap'].append(hardest - softest)
        out['gt_minus_hardest'].append(pos - hardest)
        out['neg_mean'].append(neg.mean().item())
        out['sim_p_vraw'].append(pos)
        fp = F.softmax(ls * sim[i], dim=-1)
        out['prob_on_gt'].append(fp[i].item())
        np_ = F.softmax(ls * neg, dim=-1)
        e = -(np_ * np_.clamp_min(1e-10).log()).sum().item()
        out['neg_entropy'].append(e); out['effective_negs'].append(math.exp(e))
        out['top5_neg_mass'].append(np_.topk(min(5, len(np_))).values.sum().item())
        rp = F.softmax(neg, dim=-1)
        re = -(rp * rp.clamp_min(1e-10).log()).sum().item()
        out['neg_entropy_raw'].append(re); out['effective_negs_raw'].append(math.exp(re))
    return out


def aggregate_per_split(name):
    d = OUT_ROOT / name
    if not d.exists(): return
    agg = []
    for jp in sorted(d.glob("*.json")):
        if jp.name.startswith("all_"): continue
        try:
            with open(jp) as f: agg.append(json.load(f))
        except Exception: pass
    with open(d / f"all_{name}.json", "w") as f: json.dump(agg, f, indent=2)
    log.info(f"  aggregated {len(agg)} → {d / f'all_{name}.json'}")


def main():
    for n in DATASETS: (OUT_ROOT / n).mkdir(parents=True, exist_ok=True)
    pts = discover_checkpoints()
    log.info(f"Found {len(pts)} checkpoints")

    datasets = {}
    for name, edir in DATASETS.items():
        log.info(f"\nLoading {name} v_raw from {edir}")
        datasets[name] = load_dataset(edir, name)
        log.info(f"  → {len(datasets[name])} batches × {datasets[name][0].shape[0]}")

    todo = []
    for pt in pts:
        try:
            run_id, tag = derive_run_id_and_ckpt_tag(pt)
        except Exception:
            continue
        missing = [n for n in DATASETS if not (OUT_ROOT / n / f"{run_id}__{tag}.json").exists()]
        if missing: todo.append((pt, run_id, tag, missing))
    log.info(f"\nTo process: {len(todo)}  (already done: {len(pts) - len(todo)})")

    if not todo:
        for n in DATASETS: aggregate_per_split(n)
        return

    t0 = time.time()
    for i, (pt, run_id, tag, missing) in enumerate(todo):
        elapsed = time.time() - t0
        eta = (elapsed / max(1, i)) * (len(todo) - i) if i > 0 else 0
        log.info(f"\n[{i+1}/{len(todo)}] {pt.name}  (elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m)")
        try:
            model, ls, meta = load_model(pt)
        except Exception as e:
            log.error(f"  load failed: {e}"); continue
        if model is None:
            for n in missing:
                jp = OUT_ROOT / n / f"{run_id}__{tag}.json"
                with open(jp, 'w') as f: json.dump({'checkpoint': str(pt), 'skipped': True, **meta}, f, indent=2)
            log.info(f"  skipped ({meta.get('skipped_reason')})")
            continue
        for ds_name in missing:
            try:
                all_m, n_q = None, 0
                for b in datasets[ds_name]:
                    bm = compute_metrics(model, b, ls)
                    if all_m is None: all_m = {k: [] for k in bm}
                    for k, v in bm.items(): all_m[k].extend(v)
                    n_q += b.shape[0]
                res = {'checkpoint': str(pt), 'dataset': ds_name, 'n_queries': n_q, **meta}
                for k, vs in all_m.items():
                    mn = sum(vs)/len(vs); var = sum((x-mn)**2 for x in vs)/len(vs)
                    res[f'{k}_mean'] = mn; res[f'{k}_std'] = var**0.5
                jp = OUT_ROOT / ds_name / f"{run_id}__{tag}.json"
                with open(jp, 'w') as f: json.dump(res, f, indent=2)
                log.info(f"  {ds_name}: LS={ls:.1f}  P(gt)={res['prob_on_gt_mean']:.4f}  "
                         f"eff={res['effective_negs_mean']:.1f}  gt-hard={res['gt_minus_hardest_mean']:.4f}")
            except Exception as e:
                log.error(f"  {ds_name} failed: {e}")
        del model; torch.cuda.empty_cache()

    log.info("\nAggregating...")
    for n in DATASETS: aggregate_per_split(n)


if __name__ == "__main__":
    main()

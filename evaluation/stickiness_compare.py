"""
evaluation/stickiness_compare.py

ONE script for the 5-way CIRCO-val reference-stickiness comparison:
    baseline (ours) | vib (ours) | pic2word | searle | lincir

Goal: test the paper's claim that reference over-reliance is a SHARED property of
pseudo-token ZS-CIR, not a quirk of our re-implementation. Output is a single
comparison table (R10_ref, sim_delta, median rank of the reference) plus optional
CIRCO val mAP for end-to-end validation.

=============================================================================
WHY THIS IS REVIEWER-PROOF
=============================================================================
Stickiness depends ONLY on (composed query p, gallery). Both are unit-normalized,
so sim = p @ gallery.T is cosine. `compute_stickiness()` is byte-identical to the
diagnostic block of eval_circo.py and is the SAME function for every method. The
only per-method difference is how the composed query p is produced, and each
producer follows that method's official inference recipe. So the comparison is
apples-to-apples by construction.

CIRCO TEMPLATE (verified shared convention): Pic2Word, SEARLE, and LinCIR all use
    "a photo of $ that {caption}"
on CIRCO. So one template is simultaneously each method's own recipe AND a
controlled constant. (--template overrides; verify against each repo if unsure.)

EMBEDDING SPACES / GALLERIES
    baseline, vib, pic2word, searle : OpenAI CLIP ViT-L/14  -> gallery_cache_ViT-L-14.pt
                                      (SEARLE/Pic2Word ARE OpenAI ViT-L/14, so their
                                       R10_ref is directly comparable to ours)
    lincir                          : HF CLIP-L/14 image_embeds -> gallery_cache_hf-ViT-L-14.pt
                                      (own space; comparable in spirit, measured natively.
                                       --reuse_openai_gallery approximates, caveat in paper.)

=============================================================================
VALIDATION CHAIN (do this, in order)
=============================================================================
1. SELF-CHECK (proves the harness):  --selfcheck   (runs baseline, must hit ~85.9%)
       baseline R10_ref MUST reproduce 85.9% / sim_delta +0.066
       vib      R10_ref MUST reproduce 67.7% / sim_delta +0.040
2. PER-EXTERNAL-METHOD mAP CHECK (proves each adapter):  --with_map
       each external method's CIRCO val mAP@10 must match its published value
       before you trust its stickiness. If mAP is off, the adapter is wrong --
       use --from_embeddings for that method instead (see below).

ESCAPE HATCH (maximum faithfulness): --from_embeddings LABEL=queries.pt
    If you would rather not trust this script's reimplementation of an external
    method, run THAT method's own official CIRCO-val inference, dump the 220
    composed-query embeddings as a (220, D) tensor in ds.annotations order, and
    feed it here. Stickiness is then computed by the identical core on queries
    the method's own code produced. This is the most defensible route for the
    external rows.

=============================================================================
RUN (on Paperspace, from codes/, PYTHONPATH=.)
=============================================================================
  # stage A: ours + the two OpenAI-CLIP externals (share your gallery, light deps)
  PYTHONPATH=. python evaluation/stickiness_compare.py \
      --methods baseline vib pic2word searle --with_map \
      --baseline_ckpt /notebooks/experiment/4M_checkpoints/<base>/epoch_02_step_03000.pt \
      --vib_ckpt      /notebooks/experiment/4M_checkpoints/<vib>/epoch_01_step_01500.pt \
      --pic2word_repo /path/to/composed_image_retrieval --pic2word_ckpt /path/to/pic2word_model.pt

  # stage B: add lincir (HF stack + its own gallery; heaviest)
  PYTHONPATH=.:/path/to/lincir python evaluation/stickiness_compare.py \
      --methods lincir --lincir_repo /path/to/lincir --with_map

  # escape hatch for any method:
  PYTHONPATH=. python evaluation/stickiness_compare.py \
      --methods from_emb --from_embeddings searle=/path/to/searle_circoval_queries.pt
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

EXPECTED = {  # self-check targets from the paper
    'baseline': {'r10_ref': 0.859, 'sim_delta': 0.066},
    'vib':      {'r10_ref': 0.677, 'sim_delta': 0.040},
}


# ===========================================================================
# METHOD-AGNOSTIC CORE  (identical arithmetic to eval_circo.py; unit-tested)
# ===========================================================================
@torch.no_grad()
def compute_stickiness(query_embs, gallery_embs, img_ids, img_id_to_idx,
                       annotations, tau=None):
    device = gallery_embs.device
    diags, predictions = [], {}
    for q, ann in enumerate(annotations):
        p = query_embs[q].to(device)
        scores = (p @ gallery_embs.T).squeeze(0)

        ref_idx = img_id_to_idx[str(ann['reference_img_id'])]
        sim_ref = scores[ref_idx].item()
        rank_ref = (scores > scores[ref_idx]).sum().item() + 1

        gt_idx = [img_id_to_idx[str(g)] for g in ann['gt_img_ids'] if str(g) in img_id_to_idx]
        sim_gt = scores[gt_idx].cpu().tolist() if gt_idx else []
        gt_ranks = [(scores > scores[gi]).sum().item() + 1 for gi in gt_idx]

        scores_masked = scores.clone()
        scores_masked[ref_idx] = float('-inf')
        top50 = [int(img_ids[i]) for i in scores_masked.topk(50).indices.cpu().tolist()]
        assert len(set(top50)) == 50 and int(ann['reference_img_id']) not in top50
        predictions[str(ann.get('query_id', q))] = top50

        if tau is not None:
            probs = torch.softmax(tau * scores_masked, dim=0)
            p_gtset = probs[gt_idx].sum().item() if gt_idx else 0.0
        else:
            p_gtset = None

        diags.append({'sim_ref': sim_ref, 'rank_ref': rank_ref,
                      'sim_gt_mean': float(np.mean(sim_gt)) if sim_gt else None,
                      'best_gt_rank': min(gt_ranks) if gt_ranks else None,
                      'p_gtset': p_gtset})

    # R10_ref / rank stats span ALL queries (need only the reference rank).
    ranks    = [d['rank_ref'] for d in diags]
    sim_refs = [d['sim_ref'] for d in diags]
    # sim_delta / P(gt) / best_gt_rank span only queries that HAVE GT (val).
    # sim_delta is computed over the SAME subset for both terms so it stays
    # internally consistent: mean(sim_ref | has_gt) - mean(sim_gt | has_gt).
    gt_rows  = [d for d in diags if d['sim_gt_mean'] is not None]
    bgr      = [d['best_gt_rank'] for d in diags if d['best_gt_rank'] is not None]
    s = {
        'n_queries':       len(diags),                 # e.g. 1020 for val+test
        'n_gt':            len(gt_rows),                # e.g. 220 (val only)
        'r10_ref':         sum(r <= 10 for r in ranks) / len(ranks),
        'median_rank_ref': int(np.median(ranks)),
        'mean_rank_ref':   float(np.mean(ranks)),
        'sim_ref_mean':    float(np.mean(sim_refs)),
    }
    if gt_rows:
        sr = float(np.mean([d['sim_ref'] for d in gt_rows]))
        sg = float(np.mean([d['sim_gt_mean'] for d in gt_rows]))
        s['sim_gt_mean'] = sg
        s['sim_delta'] = sr - sg                        # consistent: both over has_gt
        s['best_gt_rank_median'] = int(np.median(bgr)) if bgr else None
    else:
        s['sim_gt_mean'] = None
        s['sim_delta'] = None
        s['best_gt_rank_median'] = None
    if any(d['p_gtset'] is not None for d in diags):
        s['p_gtset_mean'] = float(np.mean([d['p_gtset'] for d in gt_rows]))
    return s, predictions


# ===========================================================================
# Loaders (mirror eval_circo.py)
# ===========================================================================
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_circo_split(circo_data_path, preprocess, split='val'):
    """split in {'val','test','val+test'} -> (ds_for_paths, anns).
    Test annotations carry reference_img_id + relative_caption but NO GT (sealed
    on the server), so gt_img_ids=[] / target_img_id=None there. We never read
    test GT. Per-query and NOT deduped: the caption is part of the composed query,
    so the same reference under two captions is two distinct stickiness measurements.
    The gallery (img_paths / img_ids_indexes_map) is the shared COCO pool, identical
    across splits, so the first loaded split resolves reference paths for all."""
    repo_src = Path(circo_data_path) / 'circo_repo' / 'src'
    ds_mod = _load_module('_circo_ds', repo_src / 'dataset.py')
    parts = split.split('+')
    prefix = len(parts) > 1                       # unique query_ids only when mixing
    ds_for_paths, anns = None, []
    for part in parts:
        ds = ds_mod.CIRCODataset(data_path=str(circo_data_path), split=part,
                                 mode='relative', preprocess=preprocess)
        if ds_for_paths is None:
            ds_for_paths = ds
        for a in ds.annotations:
            anns.append({
                'query_id':        f"{part}_{a['id']}" if prefix else a['id'],
                'raw_id':           a['id'],
                'reference_img_id': a['reference_img_id'],
                'target_img_id':    a.get('target_img_id'),
                'gt_img_ids':       a.get('gt_img_ids', []) or [],
                'relative_caption': a['relative_caption'],
                'split':            part,
            })
    n_cap = sum(1 for a in anns if a['relative_caption'])
    n_gt  = sum(1 for a in anns if a['gt_img_ids'])
    print(f"  CIRCO {split}: {len(anns)} queries  ({n_cap} with caption, {n_gt} with GT)")
    if n_cap < len(anns):
        raise SystemExit(f"{len(anns)-n_cap} queries lack relative_caption; "
                         f"cannot build a composed query for the {split} split")
    return ds_for_paths, anns


def load_official_compute_metrics(circo_data_path):
    repo_src = Path(circo_data_path) / 'circo_repo' / 'src'
    ds_module = _load_module('_circo_ds2', repo_src / 'dataset.py')
    saved = sys.modules.get('dataset')
    sys.modules['dataset'] = ds_module
    try:
        ev = _load_module('_circo_eval', repo_src / 'evaluation.py')
    finally:
        sys.modules.pop('dataset', None)
        if saved is not None:
            sys.modules['dataset'] = saved
    return ev.compute_metrics


def load_gallery(path, device):
    cache = torch.load(path, map_location='cpu')
    embs = F.normalize(cache['embeddings'].float(), dim=-1).to(device)
    return embs, cache['img_ids'], cache['img_id_to_idx']


# ===========================================================================
# PER-METHOD COMPOSED-QUERY PRODUCERS
# Each returns (Q[220,D] on cpu, anns, tau_or_None, space_tag)
# ===========================================================================
@torch.no_grad()
def queries_ours(ckpt_path, circo_data_path, device, split):
    """baseline / vib via our model.encode_query (reproduces known R10_ref)."""
    import clip
    from config.config import load_from_yaml
    from model.model import Pic2Word
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cc = ckpt.get('config', {})
    cfg = load_from_yaml('config/config.yaml',
                         clip_model=cc.get('clip_model', 'ViT-L/14'),
                         mapper_type=ckpt['mapper_type'],
                         hidden_dim=cc.get('hidden_dim', 256),
                         num_tokens=cc.get('num_tokens', 1),
                         clip_prompt=cc.get('clip_prompt', 'a photo of'),
                         device=device)
    model = Pic2Word(cfg).to(device).eval()
    model.mapper.load_state_dict(ckpt['mapper_state'])
    if ckpt.get('merger_state') is not None and model.merger is not None:
        model.merger.load_state_dict(ckpt['merger_state'])
    if 'logit_scale' in ckpt:
        model.clip.logit_scale.fill_(ckpt['logit_scale'])
    _, preprocess = clip.load(cfg.clip_model, device='cpu')
    ds, anns = load_circo_split(circo_data_path, preprocess, split)
    embs = []
    for a in anns:
        rp = ds.img_paths[ds.img_ids_indexes_map[str(a['reference_img_id'])]]
        img = preprocess(PIL.Image.open(rp).convert('RGB')).unsqueeze(0).to(device)
        tok = clip.tokenize([a['relative_caption']], truncate=True).to(device)
        embs.append(model.encode_query(img, tok).squeeze(0).cpu())
    return torch.stack(embs), anns, model.clip.logit_scale.exp().item(), 'openai-ViT-L/14'


@torch.no_grad()
def queries_pic2word(repo, ckpt_path, circo_data_path, device, template, split):
    """Official Pic2Word (google-research/composed_image_retrieval), mirroring
    evaluate_cirr in src/eval_utils.py EXACTLY, with everything loaded from the
    Pic2Word repo by absolute path so nothing from our codebase contaminates it:
      - their patched CLIP via repo/model/clip.py load() (has encode_text_img_retrieval)
      - their IM2TEXT from repo/model/model.py (constructor defaults)
      - their bundled open_clip tokenize (repo/src/third_party/open_clip)
      - both state dicts loaded with 'module.' stripping
      - template 'a photo of * , {caption}', id_split = tokenize(['*'])[0][1]
      - image NOT normalized before img2text
      - encode_text_img_retrieval(text, token, split_ind=id_split, repeat=False)
    The only shared object is the gallery (same frozen OpenAI ViT-L/14 space);
    the valmAP check confirms that space alignment. --template is ignored: Pic2Word's
    CIRR template is fixed in their data.py."""
    repo = Path(repo)
    saved_path = list(sys.path)                  # snapshot so we can restore
    saved_mods = dict(sys.modules)               # snapshot to restore shadowed modules
    try:
        # Put Pic2Word's tree FIRST and evict any of our cached packages that share
        # names (model, clip, third_party) so Pic2Word's internal imports
        # (e.g. clip.py's "from model.model import build_model") resolve to ITS files,
        # not ours. repo root first (for `model`, `clip`), then src (for third_party).
        import importlib, types
        sys.path.insert(0, str(repo))            # any repo-root modules
        sys.path.insert(0, str(repo / 'src'))    # third_party.open_clip
        # Evict cached model/clip/third_party so none of ours lingers.
        for name in list(sys.modules):
            if name in ('model', 'clip', 'third_party') \
               or name.startswith(('model.', 'clip.', 'third_party.')):
                del sys.modules[name]
        # Force 'model' to resolve to Pic2Word's repo/model. Our codes/model is a
        # REGULAR package (has __init__.py), which Python prefers over repo/model's
        # namespace dir regardless of sys.path order -- so path tricks don't work.
        # Registering the package explicitly with __path__ pinned to repo/model makes
        # clip.py's "from model.model import build_model" find Pic2Word's file.
        model_pkg = types.ModuleType('model')
        model_pkg.__path__ = [str(repo / 'model')]
        sys.modules['model'] = model_pkg
        model_mod = importlib.import_module('model.model')    # Pic2Word's CLIP/IM2TEXT defs
        clip_mod = importlib.import_module('model.clip')      # Pic2Word's patched CLIP loader
        from third_party.open_clip.clip import tokenize

        model, _, preprocess = clip_mod.load('ViT-L/14', jit=False)
        model = model.to(device).float().eval()
        img2text = model_mod.IM2TEXT(embed_dim=model.embed_dim,
                                     output_dim=model.token_embedding.weight.shape[1])
        ck = torch.load(ckpt_path, map_location='cpu')
        sd, sdi = ck['state_dict'], ck['state_dict_img2text']
        if next(iter(sd)).startswith('module.'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        if next(iter(sdi)).startswith('module.'):
            sdi = {k[len('module.'):]: v for k, v in sdi.items()}
        model.load_state_dict(sd)
        img2text.load_state_dict(sdi)
        img2text = img2text.to(device).float().eval()

        ds, anns = load_circo_split(circo_data_path, preprocess, split)
        id_split = tokenize(["*"])[0][1]
        embs = []
        for a in anns:
            rp = ds.img_paths[ds.img_ids_indexes_map[str(a['reference_img_id'])]]
            img = preprocess(PIL.Image.open(rp).convert('RGB')).unsqueeze(0).to(device)
            token = img2text(model.encode_image(img))        # NOT normalized (matches their code)
            text = tokenize(['a photo of * , {}'.format(a['relative_caption'])]).to(device)
            feat = model.encode_text_img_retrieval(text, token, split_ind=id_split, repeat=False)
            embs.append(F.normalize(feat.float(), dim=-1).squeeze(0).cpu())
        tau = model.logit_scale.exp().item()
        return torch.stack(embs), anns, tau, 'openai-ViT-L/14'
    finally:
        # Fully revert: restore sys.path, drop Pic2Word's model/clip/third_party
        # modules, and put OUR originals back so nothing is shadowed afterwards.
        sys.path[:] = saved_path
        for name in list(sys.modules):
            if (name in ('model', 'clip', 'third_party')
                    or name.startswith(('model.', 'clip.', 'third_party.', 'open_clip'))):
                sys.modules.pop(name, None)
        for name, mod in saved_mods.items():
            if (name in ('model', 'clip', 'third_party')
                    or name.startswith(('model.', 'clip.', 'third_party.', 'open_clip'))):
                sys.modules[name] = mod


@torch.no_grad()
def queries_searle(circo_data_path, device, template, split):
    """SEARLE-XL via torch.hub. OpenAI CLIP ViT-L/14 space (your gallery)."""
    import clip
    clip_model, preprocess = clip.load('ViT-L/14', device=device)
    clip_model = clip_model.float().eval()
    phi, encode_with_pseudo_tokens = torch.hub.load(
        'miccunifi/SEARLE', 'searle', source='github', backbone='ViT-L/14')
    phi = phi.to(device).eval()
    ds, anns = load_circo_split(circo_data_path, preprocess, split)
    embs = []
    for a in anns:
        rp = ds.img_paths[ds.img_ids_indexes_map[str(a['reference_img_id'])]]
        img = preprocess(PIL.Image.open(rp).convert('RGB')).unsqueeze(0).to(device)
        pseudo = phi(clip_model.encode_image(img).float())
        prompt = template.format(caption=a['relative_caption'])
        tok = clip.tokenize([prompt], truncate=True).to(device)
        feat = encode_with_pseudo_tokens(clip_model, tok, pseudo)
        embs.append(F.normalize(feat.float(), dim=-1).squeeze(0).cpu())
    return torch.stack(embs), anns, clip_model.logit_scale.exp().item(), 'openai-ViT-L/14'


@torch.no_grad()
def queries_lincir(repo, circo_data_path, device, template, split):
    """Official LinCIR (navervision/lincir), mirroring generate_test_submission.py,
    with everything loaded from the LinCIR repo so nothing of ours contaminates it:
      - build_text_encoder(args.clip_model_name='large') -> HF openai/clip-vit-large-patch14
        (same weights as our OpenAI ViT-L/14 gallery, so --reuse_openai_gallery is valid)
      - tokenizer is SDXL tokenizer_2 with '[$]' added BY build_text_encoder (id 49408);
        we use it as-returned (do NOT re-add the special token)
      - Phi(input_dim=projection_dim, hidden_dim=projection_dim*4, output_dim=hidden_size, 0)
        loaded from hf_hub_download('navervision/zeroshot-cir-models','lincir_large.pt')['Phi']
      - template 'a photo of $ that {caption}' -> '$'->'[$]' -> tokenize -> 49408 mapped to 259
      - encode_with_pseudo_tokens_HF injects the pseudo-token at id 259.
    Space: HF CLIP-L/14 image_embeds. Use --reuse_openai_gallery (shared weights)."""
    repo = Path(repo)
    saved_path = list(sys.path)
    saved_mods = dict(sys.modules)
    try:
        sys.path.insert(0, str(repo))
        for name in list(sys.modules):
            if name in ('models', 'encode_with_pseudo_tokens', 'clip') \
               or name.startswith(('models.', 'clip.')):
                del sys.modules[name]
        import importlib
        models_mod = importlib.import_module('models')
        ewpt_mod = importlib.import_module('encode_with_pseudo_tokens')
        from huggingface_hub import hf_hub_download

        class _A:
            clip_model_name = 'large'
            mixed_precision = 'fp32'
            cache_dir = None
        clip_vision, clip_pp, clip_text, tokenizer = models_mod.build_text_encoder(_A())
        # NOTE: build_text_encoder already added '[$]' (id 49408). Do not re-add.
        phi = models_mod.Phi(input_dim=clip_text.config.projection_dim,
                             hidden_dim=clip_text.config.projection_dim * 4,
                             output_dim=clip_text.config.hidden_size, dropout=0).eval()
        ckpt = hf_hub_download('navervision/zeroshot-cir-models', 'lincir_large.pt')
        phi.load_state_dict(torch.load(ckpt, map_location='cpu')['Phi'])
        clip_vision = clip_vision.to(device).eval()
        clip_text = clip_text.to(device).eval()
        phi = phi.to(device).eval()

        ds, anns = load_circo_split(circo_data_path, (lambda x: x), split)
        embs = []
        for a in anns:
            rp = ds.img_paths[ds.img_ids_indexes_map[str(a['reference_img_id'])]]
            pil = PIL.Image.open(rp).convert('RGB')
            pix = clip_pp(pil, return_tensors='pt')['pixel_values'].to(device)
            pseudo = phi(clip_vision(pixel_values=pix).image_embeds)
            prompt = 'a photo of $ that {}'.format(a['relative_caption']).replace('$', '[$]')
            ids = tokenizer(text=prompt, return_tensors='pt',
                            padding='max_length', truncation=True)['input_ids'].to(device)
            ids = torch.where(ids == 49408, torch.ones_like(ids) * 259, ids)
            feat = ewpt_mod.encode_with_pseudo_tokens_HF(clip_text, ids, pseudo)
            embs.append(F.normalize(feat.float(), dim=-1).squeeze(0).cpu())
        return torch.stack(embs), anns, None, 'hf-ViT-L/14'
    finally:
        sys.path[:] = saved_path
        for name in list(sys.modules):
            if name in ('models', 'encode_with_pseudo_tokens', 'clip') \
               or name.startswith(('models.', 'clip.')):
                sys.modules.pop(name, None)
        for name, mod in saved_mods.items():
            if name in ('models', 'encode_with_pseudo_tokens', 'clip') \
               or name.startswith(('models.', 'clip.')):
                sys.modules[name] = mod


# ===========================================================================
# DRIVER
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+',
                    default=['baseline', 'vib', 'pic2word', 'searle', 'lincir'],
                    help="subset of: baseline vib pic2word searle lincir from_emb")
    ap.add_argument('--selfcheck', action='store_true',
                    help='run baseline+vib and assert they reproduce the paper numbers')
    ap.add_argument('--with_map', action='store_true',
                    help='also compute official CIRCO val mAP (per-method adapter check)')
    ap.add_argument('--template', default='a photo of $ that {caption}')
    ap.add_argument('--split', default='val', choices=['val', 'test', 'val+test'],
                    help="stickiness reference set. 'val+test' (1020) gives R10_ref on "
                         "all refs; sim_delta/P(gt) stay val-only (test GT is sealed).")
    ap.add_argument('--circo_data_path', type=Path,
                    default=Path('/notebooks/experiment/eval_data/circo'))
    ap.add_argument('--baseline_ckpt', type=Path)
    ap.add_argument('--vib_ckpt', type=Path)
    ap.add_argument('--pic2word_repo', type=Path)
    ap.add_argument('--pic2word_ckpt', type=Path)
    ap.add_argument('--lincir_repo', type=Path)
    ap.add_argument('--from_embeddings', nargs='*', default=[],
                    help='LABEL=/path/to/queries.pt (220xD in ds.annotations order)')
    ap.add_argument('--openai_gallery', type=Path, default=None)
    ap.add_argument('--hf_gallery', type=Path, default=None)
    ap.add_argument('--reuse_openai_gallery', action='store_true')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', type=Path,
                    default=Path('/notebooks/experiment/codes/metrics/stickiness_compare.json'))
    args = ap.parse_args()
    dev = args.device
    cdp = args.circo_data_path
    if args.selfcheck and 'baseline' not in args.methods:
        args.methods = ['baseline', 'vib'] + args.methods
    if args.openai_gallery is None:
        args.openai_gallery = cdp / 'gallery_cache_ViT-L-14.pt'
    if args.hf_gallery is None:
        args.hf_gallery = cdp / 'gallery_cache_hf-ViT-L-14.pt'

    # galleries are loaded lazily (only what we need)
    galleries = {}
    def get_gallery(space):
        key = 'hf' if space.startswith('hf') and not args.reuse_openai_gallery else 'openai'
        if key not in galleries:
            path = args.hf_gallery if key == 'hf' else args.openai_gallery
            galleries[key] = load_gallery(path, dev)
            print(f'  [gallery] {key}: {galleries[key][0].shape[0]:,} imgs ({path.name})')
        return galleries[key]

    # --- produce queries per requested method ---
    def produce(method):
        if method == 'baseline':
            return queries_ours(args.baseline_ckpt, cdp, dev, args.split)
        if method == 'vib':
            return queries_ours(args.vib_ckpt, cdp, dev, args.split)
        if method == 'pic2word':
            return queries_pic2word(args.pic2word_repo, args.pic2word_ckpt, cdp, dev, args.template, args.split)
        if method == 'searle':
            return queries_searle(cdp, dev, args.template, args.split)
        if method == 'lincir':
            return queries_lincir(args.lincir_repo, cdp, dev, args.template, args.split)
        raise SystemExit(f'unknown method {method}')

    from_emb = dict(kv.split('=', 1) for kv in args.from_embeddings)
    rows = {}
    compute_metrics = load_official_compute_metrics(cdp) if args.with_map else None

    run_list = [m for m in args.methods if m != 'from_emb'] + list(from_emb.keys())
    for method in run_list:
        print(f'\n=== {method} ===')
        if method in from_emb:                          # escape hatch
            Q = torch.load(from_emb[method], map_location='cpu')
            _, anns = load_circo_split(cdp, (lambda x: x), args.split)
            tau, space = None, 'provided'
        else:
            Q, anns, tau, space = produce(method)
        gallery, img_ids, id2idx = get_gallery(space)
        Q = F.normalize(Q.float(), dim=-1)
        s, preds = compute_stickiness(Q, gallery, img_ids, id2idx, anns, tau=tau)
        s['space'] = space
        # Official mAP needs local GT -> always computed over the VAL subset only,
        # re-keyed from the (possibly prefixed) query_id back to the raw val id,
        # so 'val' and 'val+test' runs both report the same 220-query val mAP.
        if args.with_map:
            val_preds = {str(a['raw_id']): preds[str(a['query_id'])]
                         for a in anns if a['split'] == 'val'}
            if val_preds:
                map_atk, _, _ = compute_metrics(Path(cdp), val_preds, [5, 10, 25, 50])
                s['map@10'] = round(float(map_atk[10]) * 100, 2)
                s['map_n'] = len(val_preds)
        rows[method] = s

        # self-check assertion. R10_ref target is val-only (85.9%); on val+test the
        # reference set is 1020 so R10_ref differs -- there we sanity-check sim_delta
        # (still val-220) and just report R10_ref over the larger set.
        if args.selfcheck and method in EXPECTED:
            exp = EXPECTED[method]
            if args.split == 'val':
                ok = (abs(s['r10_ref'] - exp['r10_ref']) < 0.02
                      and s['sim_delta'] is not None
                      and abs(s['sim_delta'] - exp['sim_delta']) < 0.01)
                print(f"  SELF-CHECK {method}: R10_ref={s['r10_ref']*100:.1f}% "
                      f"(expect {exp['r10_ref']*100:.1f}%)  -> {'PASS' if ok else 'FAIL'}")
                if not ok:
                    print('  !! harness does not reproduce the paper number; fix before trusting externals')
            else:
                ok = s['sim_delta'] is not None and abs(s['sim_delta'] - exp['sim_delta']) < 0.01
                print(f"  SELF-CHECK {method}: sim_delta={s['sim_delta']:+.4f} "
                      f"(expect {exp['sim_delta']:+.4f}, val-220)  -> {'PASS' if ok else 'FAIL'}; "
                      f"R10_ref={s['r10_ref']*100:.1f}% over {s['n_queries']} refs")

    # --- comparison table ---
    print('\n' + '=' * 84)
    print(f'CIRCO reference stickiness  [split={args.split}]   '
          '(lower R10_ref / sim_delta = less sticky)')
    print('R10_ref/medRank over all refs; sim_delta/P(gt)/mAP@10 over the val-220 (GT) subset.')
    print('=' * 84)
    hdr = (f"{'method':10s} {'n_ref':>6s} {'R10_ref':>8s} {'medRank':>8s} "
           f"{'n_gt':>5s} {'sim_delta':>10s} {'P(gt)':>7s}")
    if args.with_map:
        hdr += f" {'valmAP':>7s}"
    print(hdr); print('-' * len(hdr))
    for m in run_list:
        s = rows[m]
        sd = s.get('sim_delta')
        pg = s.get('p_gtset_mean')
        line = (f"{m:10s} {s['n_queries']:6d} {s['r10_ref']*100:7.1f}% "
                f"{s['median_rank_ref']:8d} {s.get('n_gt', 0):5d} "
                f"{('%+.4f' % sd) if sd is not None else '--':>10s} "
                f"{('%.4f' % pg) if pg is not None else '--':>7s}")
        if args.with_map:
            line += f" {('%.2f' % s['map@10']) if 'map@10' in s else '--':>7s}"
        print(line)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({'template': args.template, 'rows': rows}, open(args.out, 'w'), indent=2)
    print(f'\nsaved -> {args.out}')


if __name__ == '__main__':
    main()
"""
count_baseline_params.py

Counts TRAINABLE mapper parameters for the three external baselines
(Pic2Word, SEARLE-XL, LinCIR) by loading each via the SAME official paths
used in evaluation/stickiness_compare.py, then summing the parameters of the
mapper object only (img2text / phi). This is the ground-truth count -- the
same method that gave our own mapper 460,032 -- so it directly confirms or
corrects the 1.64M / 14.16M / 14.16M figures in Table 2.

The CLIP encoders are frozen and NOT counted; only the trainable mapper is,
matching how "#Params (trainable mapper size)" is defined in the paper.

RUN (on Paperspace, from codes/, PYTHONPATH=.):
  # all three
  PYTHONPATH=. python count_baseline_params.py \
      --methods pic2word searle lincir \
      --pic2word_repo /path/to/composed_image_retrieval \
      --pic2word_ckpt /path/to/pic2word_model.pt \
      --lincir_repo   /path/to/lincir

  # just SEARLE (the one we most need to confirm; torch.hub, no local repo)
  PYTHONPATH=. python count_baseline_params.py --methods searle

  # just LinCIR
  PYTHONPATH=. python count_baseline_params.py --methods lincir --lincir_repo /path/to/lincir
"""

import argparse
import importlib
import sys
import types
from pathlib import Path

import torch


def count_params(module):
    """Return (trainable, total) parameter counts for an nn.Module."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


# ---------------------------------------------------------------------------
# Pic2Word: the trainable mapper is IM2TEXT (img2text), loaded from the official
# google-research/composed_image_retrieval repo, exactly as in stickiness_compare.
# ---------------------------------------------------------------------------
def count_pic2word(repo, ckpt_path, device):
    repo = Path(repo)
    saved_path = list(sys.path)
    saved_mods = dict(sys.modules)
    try:
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / 'src'))
        for name in list(sys.modules):
            if name in ('model', 'clip', 'third_party') \
               or name.startswith(('model.', 'clip.', 'third_party.')):
                del sys.modules[name]
        model_pkg = types.ModuleType('model')
        model_pkg.__path__ = [str(repo / 'model')]
        sys.modules['model'] = model_pkg
        model_mod = importlib.import_module('model.model')
        clip_mod = importlib.import_module('model.clip')

        model, _, _ = clip_mod.load('ViT-L/14', jit=False)
        model = model.to(device).float().eval()
        img2text = model_mod.IM2TEXT(embed_dim=model.embed_dim,
                                     output_dim=model.token_embedding.weight.shape[1])
        ck = torch.load(ckpt_path, map_location='cpu')
        sdi = ck['state_dict_img2text']
        if next(iter(sdi)).startswith('module.'):
            sdi = {k[len('module.'):]: v for k, v in sdi.items()}
        img2text.load_state_dict(sdi)
        return count_params(img2text)
    finally:
        sys.path[:] = saved_path
        for name in list(sys.modules):
            if (name in ('model', 'clip', 'third_party')
                    or name.startswith(('model.', 'clip.', 'third_party.', 'open_clip'))):
                sys.modules.pop(name, None)
        for name, mod in saved_mods.items():
            if (name in ('model', 'clip', 'third_party')
                    or name.startswith(('model.', 'clip.', 'third_party.', 'open_clip'))):
                sys.modules[name] = mod


# ---------------------------------------------------------------------------
# SEARLE-XL: the trainable mapper is phi, loaded via torch.hub (miccunifi/SEARLE),
# exactly as in stickiness_compare. No local repo needed.
# ---------------------------------------------------------------------------
def count_searle(device):
    phi, _ = torch.hub.load('miccunifi/SEARLE', 'searle',
                            source='github', backbone='ViT-L/14')
    phi = phi.to(device).eval()
    return count_params(phi)


# ---------------------------------------------------------------------------
# LinCIR: the trainable mapper is Phi, loaded from the official navervision/lincir
# repo + HF checkpoint, exactly as in stickiness_compare.
# ---------------------------------------------------------------------------
def count_lincir(repo, device):
    repo = Path(repo)
    saved_path = list(sys.path)
    saved_mods = dict(sys.modules)
    try:
        sys.path.insert(0, str(repo))
        for name in list(sys.modules):
            if name in ('models', 'encode_with_pseudo_tokens', 'clip') \
               or name.startswith(('models.', 'clip.')):
                del sys.modules[name]
        models_mod = importlib.import_module('models')
        from huggingface_hub import hf_hub_download

        class _A:
            clip_model_name = 'large'
            mixed_precision = 'fp32'
            cache_dir = None
        _, _, clip_text, _ = models_mod.build_text_encoder(_A())
        phi = models_mod.Phi(input_dim=clip_text.config.projection_dim,
                             hidden_dim=clip_text.config.projection_dim * 4,
                             output_dim=clip_text.config.hidden_size, dropout=0).eval()
        ckpt = hf_hub_download('navervision/zeroshot-cir-models', 'lincir_large.pt')
        phi.load_state_dict(torch.load(ckpt, map_location='cpu')['Phi'])
        return count_params(phi)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+',
                    default=['pic2word', 'searle', 'lincir'])
    ap.add_argument('--pic2word_repo', type=Path)
    ap.add_argument('--pic2word_ckpt', type=Path)
    ap.add_argument('--lincir_repo', type=Path)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    # paper's claimed values, for at-a-glance comparison
    PAPER = {'pic2word': 1.64, 'searle': 14.16, 'lincir': 14.16}

    results = {}
    for m in args.methods:
        print(f'\n=== {m} ===')
        try:
            if m == 'pic2word':
                if not (args.pic2word_repo and args.pic2word_ckpt):
                    print('  skipped: need --pic2word_repo and --pic2word_ckpt')
                    continue
                tr, tot = count_pic2word(args.pic2word_repo, args.pic2word_ckpt, args.device)
            elif m == 'searle':
                tr, tot = count_searle(args.device)
            elif m == 'lincir':
                if not args.lincir_repo:
                    print('  skipped: need --lincir_repo')
                    continue
                tr, tot = count_lincir(args.lincir_repo, args.device)
            else:
                print(f'  unknown method {m}')
                continue
            results[m] = tr
            print(f'  trainable mapper params : {tr:,}  ({tr/1e6:.2f}M)')
            print(f'  total mapper params     : {tot:,}  ({tot/1e6:.2f}M)')
            if m in PAPER:
                claimed = PAPER[m]
                got = round(tr / 1e6, 2)
                match = 'MATCH' if abs(got - claimed) < 0.01 else 'MISMATCH'
                print(f'  paper says {claimed:.2f}M  ->  measured {got:.2f}M  [{match}]')
        except Exception as e:
            print(f'  ERROR loading {m}: {e}')

    print('\n' + '=' * 56)
    print(f"{'method':10s} {'measured (M)':>14s} {'paper (M)':>12s}")
    print('-' * 56)
    for m in args.methods:
        if m in results:
            got = results[m] / 1e6
            claimed = PAPER.get(m, float('nan'))
            print(f"{m:10s} {got:14.2f} {claimed:12.2f}")
    print('=' * 56)
    print('\nNote: counts the TRAINABLE mapper only (img2text / phi); '
          'frozen CLIP encoders excluded, matching the paper definition.')


if __name__ == '__main__':
    main()
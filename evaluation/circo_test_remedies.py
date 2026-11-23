"""
Print config + generate CIRCO test submission for the 4M remedy checkpoints,
so Table 1's C.test column can be filled (margin / unit-norm / additive noise),
with the bounded baseline included to re-confirm the existing 11.4.

Reuses load_model / generate_test_submission from eval_circo_test.py, so the
submission JSON is byte-identical in format to the baseline/VIB submissions.

Place this at:
    /notebooks/experiment/codes/evaluation/circo_test_remedies.py
Run:
    cd /notebooks/experiment/codes
    PYTHONPATH=/notebooks/experiment/codes python evaluation/circo_test_remedies.py

NOTE: 1.5M runs (repulsion, decorrelation, mergers) are intentionally excluded —
they are not in Table 1 and would waste rate-limited server submissions.
"""
import gc
import math
from pathlib import Path

import torch

from evaluation.eval_circo_test import load_model, generate_test_submission  # noqa: F401

# --------------------------------------------------------------------------- #
# 4M checkpoints for Table 1's bottom rows (+ baseline to re-confirm 11.4).
# --------------------------------------------------------------------------- #
CKPTS = [
    # baseline (bounded) — already 11.4 test; include to confirm pipeline parity
    Path('/notebooks/experiment/4M_checkpoints/'
         'base_hd256_tok1_lslr0.2_lsmax116/epoch_02_step_03000.pt'),
    # + margin
    Path('/notebooks/experiment/4M_checkpoints/'
         'base_hd256_tok1_margin0.2_mw0.4_lslr0.4_lsmax116/epoch_01_step_01500.pt'),
    # + unit-norm (pseudo_norm = 1.0)
    Path('/notebooks/experiment/4M_checkpoints/'
         'base_hd256_tok1_lslr0.2_lsmax116_pnorm1.0/epoch_01_step_01500.pt'),
    # + additive noise (sigma 0.1)  --  CONFIRM this is the best-val ckpt (step 500 is early)
    Path('/notebooks/experiment/4M_checkpoints/'
         'base_hd256_tok1_lslr0.2_lsmax116_noise0.1/epoch_01_step_00500.pt'),
]

CIRCO_DATA = Path('/notebooks/experiment/eval_data/circo')
DEVICE = 'cuda'

# Discriminating hyperparameters worth eyeballing to confirm which remedy a ckpt is.
KEY_FIELDS = [
    'mapper_type', 'num_tokens', 'hidden_dim', 'merge_tokens', 'merge_type',
    'ls_lr_scale', 'ls_max', 'vib_beta',
    'margin', 'margin_weight', 'margin_delay',
    'noise_sigma', 'pseudo_norm', 'pnorm',
    'repulsion_weight', 'decorr_weight',
    'lr', 'batch_size', 'epochs',
]


def print_config(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt.get('config', {})
    if not isinstance(cfg, dict):                     # dataclass / namespace fallback
        try:
            cfg = vars(cfg)
        except TypeError:
            cfg = {}

    print('=' * 70)
    print(f'CONFIG  {ckpt_path}')
    print('-' * 70)
    print(f"  epoch={ckpt.get('epoch')}  step={ckpt.get('step')}  "
          f"total_steps={ckpt.get('total_steps')}")
    ls = ckpt.get('logit_scale')
    if ls is not None:
        # stored as the log value; show tau = exp for readability
        try:
            print(f"  logit_scale(log)={float(ls):.4f}   tau=exp={math.exp(float(ls)):.2f}")
        except (TypeError, ValueError):
            print(f"  logit_scale={ls}")
    print(f"  mapper_type={ckpt.get('mapper_type')}  "
          f"mapper_kwargs={ckpt.get('mapper_kwargs')}")

    shown = {k: cfg[k] for k in KEY_FIELDS if k in cfg}
    for k, v in shown.items():
        print(f"    {k} = {v}")
    rest = {k: v for k, v in cfg.items() if k not in shown}
    if rest:
        print('  -- other config --')
        for k, v in rest.items():
            print(f"    {k} = {v}")
    print('=' * 70)

    del ckpt
    gc.collect()


def main():
    # Derive the gallery-cache path exactly like eval_circo_test.main().
    sample = torch.load(CKPTS[0], map_location='cpu')
    clip_model_name = sample.get('config', {}).get('clip_model', 'ViT-L/14')
    del sample
    gallery_cache = CIRCO_DATA / f"gallery_cache_{clip_model_name.replace('/', '-')}.pt"
    assert gallery_cache.exists(), f'Gallery cache not found: {gallery_cache}'

    made = []
    for i, ckpt in enumerate(CKPTS):
        print(f'\n\n[{i + 1}/{len(CKPTS)}] {ckpt}')
        if not ckpt.exists():
            print(f'  MISSING — skipping: {ckpt}')
            continue

        print_config(ckpt)                 # 1) config first

        try:                               # 2) then the submission (existing pipeline)
            out = generate_test_submission(ckpt, DEVICE, CIRCO_DATA, gallery_cache)
            made.append(out)
        except Exception as e:
            print(f'  FAILED: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print('\n' + '=' * 70)
    print('SUBMISSIONS WRITTEN:')
    for p in made:
        print(' ', p)
    print('\nUpload each JSON to https://circo.micc.unifi.it/evaluation')
    print('(manual upload + email turnaround ~1-2 days)')


if __name__ == '__main__':
    main()
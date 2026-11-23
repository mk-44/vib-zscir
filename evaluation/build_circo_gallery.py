"""
Build CIRCO gallery cache: pre-encode all ~123k COCO 2017 unlabeled images with frozen CLIP ViT-L/14. 
One-time script; cache reused across all checkpoints, all VIB runs, all future experiments.

Output schema (saved as torch .pt):
    {
        'embeddings'   : torch.Tensor (N, 768), fp32, normalized, on CPU
        'img_ids'      : list[str], length N, parallel to embeddings rows
        'img_id_to_idx': dict[str, int], O(1) lookup at eval time
        'clip_model'   : str, e.g. 'ViT-L/14'
    }
Usage:
    cd /notebooks/experiment/codes
    PYTHONPATH = /notebooks/experiment/codes python evaluation/build_circo_gallery.py
"""

import argparse
import sys
import time
from pathlib import Path

import clip
import PIL.Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import importlib.util


def _load_circo_dataset_class(circo_data_path):
    circo_dataset_py = Path(circo_data_path) / 'circo_repo' / 'src' / 'dataset.py'
    if not circo_dataset_py.exists():
        raise FileNotFoundError(f'CIRCO dataset.py not found at {circo_dataset_py}')
    
    spec = importlib.util.spec_from_file_location('circo_official_dataset_module', circo_dataset_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CIRCODataset


# ---------------------------------------------------------------------------
# Wrapper that delegates to official CIRCODataset(classic) but adds RGB conversion.
# CIRCO's loader does PIL.Image.open(path) without .convert('RGB'). For 220 ref
# images you usually get away with it; for 123k images you almost certainly hit
# at least one grayscale or RGBA image and CLIP preprocess fails.
# ---------------------------------------------------------------------------

class CircoGalleryDataset(Dataset):
    def __init__(self, circo_data_path, preprocess):
        CIRCODataset = _load_circo_dataset_class(circo_data_path)
        self.inner = CIRCODataset(
            data_path=circo_data_path,
            split='val',
            mode='classic',
            preprocess=None,
        )
        self.preprocess = preprocess

    def __len__(self):
        return len(self.inner.img_ids)

    def __getitem__(self, idx):
        img_path = self.inner.img_paths[idx]
        img_id   = str(self.inner.img_ids[idx])
        img      = PIL.Image.open(img_path).convert('RGB')
        img      = self.preprocess(img)
        return {'img': img, 'img_id': img_id}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type = Path, default = Path('/notebooks/experiment/eval_data/circo'))
    p.add_argument('--cache_path', type = Path, default = None, help='Default: <data_path>/gallery_cache_<clip_model_tag>.pt')
    p.add_argument('--clip_model', type = str, default = 'ViT-L/14')
    p.add_argument('--batch_size', type = int, default = 64)
    p.add_argument('--num_workers', type = int, default = 4)
    p.add_argument('--device', type = str, default = 'cuda')
    p.add_argument('--force', action = 'store_true', help = 'Rebuild even if cache exists')
    return p.parse_args()


def main():
    args = parse_args()

    if args.cache_path is None:
        clip_tag = args.clip_model.replace('/', '-')
        args.cache_path = args.data_path / f'gallery_cache_{clip_tag}.pt'

    if args.cache_path.exists() and not args.force:
        print(f'Cache already exists at {args.cache_path}')
        print(f'  (use --force to rebuild)')
        return

    print('=' * 70)
    print('Building CIRCO gallery cache')
    print('=' * 70)
    print(f'  data_path : {args.data_path}')
    print(f'  cache_path: {args.cache_path}')
    print(f'  clip_model: {args.clip_model}')
    print(f'  batch_size: {args.batch_size}')
    print(f'  device    : {args.device}')
    print()

    print(f'Loading CLIP {args.clip_model}...')
    model, preprocess = clip.load(args.clip_model, device = args.device)
    model.eval()

    print('Initializing CircoGalleryDataset...')
    ds = CircoGalleryDataset(args.data_path, preprocess)
    print(f'  Total images to encode: {len(ds):,}')

    loader = DataLoader(
        ds,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.num_workers,
        pin_memory = True
    )

    all_embs = []
    all_img_ids = []
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc = 'Encoding'):
            imgs = batch['img'].to(args.device, non_blocking=True)
            img_ids = batch['img_id']                          # list[str], len=B
            emb = model.encode_image(imgs)                 # (B, 768) fp16 on GPU
            emb = F.normalize(emb.float(), dim = -1)         # cast fp32 + normalize
            all_embs.append(emb.cpu())
            all_img_ids.extend(img_ids)

    embeddings = torch.cat(all_embs, dim = 0)
    elapsed = time.time() - t0
    print(f'\nEncoded {embeddings.shape[0]:,} images in {elapsed:.1f}s ({embeddings.shape[0]/elapsed:.1f} img/sec)')

    
    assert len(all_img_ids) == embeddings.shape[0], f'img_ids ({len(all_img_ids)}) != embeddings ({embeddings.shape[0]})'
    assert len(set(all_img_ids)) == len(all_img_ids), f'Duplicate img_ids! ({len(all_img_ids)} ids, {len(set(all_img_ids))} unique)'
    assert embeddings.shape[1] == 768, f'Unexpected embedding dim: {embeddings.shape[1]}'
    assert embeddings.dtype == torch.float32, f'Embeddings should be fp32, got {embeddings.dtype}'
    norms = embeddings.norm(dim=-1)
    assert (norms - 1.0).abs().max() < 1e-3, f'Embeddings not normalized: norm range {norms.min():.4f} to {norms.max():.4f}'

    img_id_to_idx = {iid: i for i, iid in enumerate(all_img_ids)}
    cache = {
        'embeddings'   : embeddings,
        'img_ids'      : all_img_ids,
        'img_id_to_idx': img_id_to_idx,
        'clip_model'   : args.clip_model,
    }
    args.cache_path.parent.mkdir(parents = True, exist_ok = True)
    torch.save(cache, args.cache_path)

    file_size_mb = args.cache_path.stat().st_size / 1e6
    print()
    print('=' * 70)
    print('Cache built successfully')
    print('=' * 70)
    print(f'  Path           : {args.cache_path}')
    print(f'  File size      : {file_size_mb:.1f} MB')
    print(f'  Num images     : {embeddings.shape[0]:,}')
    print(f'  Embedding shape: {tuple(embeddings.shape)} {embeddings.dtype}')
    print(f'  First 5 img_ids: {all_img_ids[:5]}')
    print(f'  Last 5 img_ids : {all_img_ids[-5:]}')
    print('=' * 70)


if __name__ == '__main__':
    main()
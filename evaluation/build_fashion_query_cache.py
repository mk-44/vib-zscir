"""
Build Fashion-IQ query-side cache: pre-encode all val-split images with
frozen CLIP ViT-L/14 (UNNORMALIZED, since this is what the mapper takes
as input). One-time script; cache reused across all checkpoints during
bulk Fashion-IQ eval.

Three caches built (one per category):
    /notebooks/experiment/eval_data/fashion_iq/query_cache_dress_ViT-L-14.pt
    /notebooks/experiment/eval_data/fashion_iq/query_cache_shirt_ViT-L-14.pt
    /notebooks/experiment/eval_data/fashion_iq/query_cache_toptee_ViT-L-14.pt

Output schema (saved as torch .pt dict):
    {
        'embeddings' : torch.Tensor (N, 768), fp32, UNNORMALIZED, on CPU
        'image_names': list[str], length N — Fashion-IQ image names (no extension)
        'name_to_idx': dict[str, int], O(1) lookup at eval time
        'clip_model' : 'ViT-L/14',
        'category'   : 'dress' / 'shirt' / 'toptee',
        'split'      : 'val',
        'built_at'   : ISO timestamp,
        'n_images'   : N,
        'normalized' : False,
        'source_file': 'image_splits/split.<cat>.val.json',
        'note'       : '...',
    }

Usage:
    cd /notebooks/experiment/codes
    PYTHONPATH=/notebooks/experiment/codes python evaluation/build_fashion_query_cache.py
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import clip
import PIL.Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


CATEGORIES = ['dress', 'shirt', 'toptee']


class _FashionIQRawImageDataset(Dataset):
    """
    Iterates val-split images in the order they appear in
    image_splits/split.<category>.val.json. RGB conversion enforced.
    """
    def __init__(self, image_names, images_dir, preprocess):
        self.image_names = image_names
        self.images_dir  = images_dir
        self.preprocess  = preprocess

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        path = self.images_dir / f'{name}.png'
        try:
            img = PIL.Image.open(path).convert('RGB')
        except Exception:
            # Fallback to blank — matches eval_fashion.py behavior
            img = PIL.Image.new('RGB', (224, 224))
        img = self.preprocess(img)
        return {'img': img, 'name': name}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',   type=Path,
                   default=Path('/notebooks/experiment/eval_data/fashion_iq'))
    p.add_argument('--clip_model',  type=str, default='ViT-L/14')
    p.add_argument('--batch_size',  type=int, default=128)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device',      type=str, default='cuda')
    p.add_argument('--split',       type=str, default='val')
    p.add_argument('--force',       action='store_true',
                   help='Rebuild even if cache exists')
    return p.parse_args()


def build_one_category(category, args, model, preprocess):
    """Build cache for one category. Skip if exists (unless --force)."""
    clip_tag    = args.clip_model.replace('/', '-')
    cache_path  = args.data_path / f'query_cache_{category}_{clip_tag}.pt'

    if cache_path.exists() and not args.force:
        print(f'  [{category}] cache already exists at {cache_path.name} (skip)')
        return

    # Load image list from the split file
    split_file  = args.data_path / 'image_splits' / f'split.{category}.{args.split}.json'
    with open(split_file) as f:
        image_names = json.load(f)
    print(f'  [{category}] {len(image_names)} images to encode')

    images_dir = args.data_path / 'images' / 'images'
    ds         = _FashionIQRawImageDataset(image_names, images_dir, preprocess)
    loader     = DataLoader(
        ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
    )

    all_embs  = []
    all_names = []

    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'  [{category}] encoding', leave=False):
            imgs  = batch['img'].to(args.device, non_blocking=True)
            names = batch['name']                                # list[str]
            # UNNORMALIZED — input to mapper
            emb   = model.encode_image(imgs)                     # CLIP visual encoder output
            # Note: CLIP's encode_image() actually returns the same as visual_encoder(img),
            # which is what model.clip.encoder_image_unnorm() wraps.
            # No F.normalize here — that's the whole point.
            all_embs.append(emb.float().cpu())
            all_names.extend(names)

    embeddings = torch.cat(all_embs, dim=0)
    elapsed = time.time() - t0
    print(f'  [{category}] encoded {embeddings.shape[0]} images in {elapsed:.1f}s '
          f'({embeddings.shape[0]/elapsed:.1f} img/sec)')

    # ----- Sanity assertions -----
    assert len(all_names) == embeddings.shape[0], \
        f'names ({len(all_names)}) != embeddings ({embeddings.shape[0]})'
    assert len(set(all_names)) == len(all_names), \
        f'Duplicate image names'
    assert embeddings.shape[1] == 768, \
        f'Unexpected embedding dim: {embeddings.shape[1]}'
    assert embeddings.dtype == torch.float32, \
        f'Embeddings should be fp32, got {embeddings.dtype}'

    # ----- Save -----
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    cache = {
        'embeddings' : embeddings,
        'image_names': all_names,
        'name_to_idx': name_to_idx,
        'clip_model' : args.clip_model,
        'category'   : category,
        'split'      : args.split,
        'built_at'   : datetime.now().isoformat(timespec='seconds'),
        'n_images'   : embeddings.shape[0],
        'normalized' : False,
        'source_file': str(split_file.relative_to(args.data_path)),
        'note'       : 'Unnormalized CLIP visual features. Input to mapper. '
                       'Cosine sim with these is NOT meaningful; pass through mapper first.',
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)

    file_size_mb = cache_path.stat().st_size / 1e6
    print(f'  [{category}] saved cache: {file_size_mb:.1f} MB → {cache_path.name}')


def main():
    args = parse_args()

    print('=' * 70)
    print('Building Fashion-IQ query-side cache (UNNORMALIZED CLIP visual features)')
    print('=' * 70)
    print(f'  data_path : {args.data_path}')
    print(f'  clip_model: {args.clip_model}')
    print(f'  split     : {args.split}')
    print(f'  device    : {args.device}')
    print(f'  batch_size: {args.batch_size}')
    print()

    print(f'Loading CLIP {args.clip_model}...')
    model, preprocess = clip.load(args.clip_model, device=args.device)
    model.eval()

    print()
    for category in CATEGORIES:
        build_one_category(category, args, model, preprocess)
        print()

    print('=' * 70)
    print('All caches built.')
    print('=' * 70)
    print('\nFiles created:')
    clip_tag = args.clip_model.replace('/', '-')
    for cat in CATEGORIES:
        path = args.data_path / f'query_cache_{cat}_{clip_tag}.pt'
        if path.exists():
            print(f'  {path}  ({path.stat().st_size/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
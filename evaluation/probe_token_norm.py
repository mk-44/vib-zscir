"""
evaluation/probe_token_norm.py

Measure the inference pseudo-token norm ||z|| (z = mu for VIB, z = f(v) for the
deterministic mappers) over the CIRCO validation reference images, identically
for every checkpoint, so the token norms reported in the paper come from one
defined set and one code path.

Why this is the defensible probe
--------------------------------
* Model construction reuses eval_circo_v2.load_model, so mapper_type,
  VIB-vs-base, logit_scale and eval() mode all match the mAP evaluation. A VIB
  mapper in eval mode returns mu (no sampling), exactly the inference token.
* The reference set is the CIRCO val split loaded through the same official
  CIRCODataset that the mAP is computed on (via _load_circo_official), so the
  norms are measured on identically the images the retrieval numbers use.
* The token is built exactly as model.encode_query builds it before the text
  transform: z = mapper(encoder_image_unnorm(ref)), then merger / pseudo_norm
  if the model has them. Nothing is re-derived by hand.
* The pseudo-token depends only on the reference image, not the caption, so we
  measure over the UNIQUE reference images (each token once) and report n.

Usage (from codes/):
    cd /notebooks/experiment/codes
    PYTHONPATH=/notebooks/experiment/codes python evaluation/probe_token_norm.py \
        baseline=/notebooks/experiment/4M_checkpoints/<baseline_run>/epoch_02/epoch_02_step_03000.pt \
        noise=/notebooks/experiment/4M_checkpoints/<noise_run>/<ckpt>.pt \
        l2_only=/notebooks/experiment/4M_checkpoints/base_hd256_tok1_lslr0.2_lsmax116_l2tok0.1024_seedR/epoch_01_step_01000.pt \
        vib=/notebooks/experiment/4M_checkpoints/<vib_run>/epoch_01_step_01500.pt

Each argument is LABEL=PATH. Prints mean / std / min / max per checkpoint over
the same reference set, and writes token_norm_probe.json.
"""

import argparse
import json
from pathlib import Path

import clip
import PIL.Image
import torch
import torch.nn.functional as F

from evaluation.eval_circo_v2 import load_model, _load_circo_official


# ---------------------------------------------------------------------------
# Reference set: one path per UNIQUE reference image across the val queries.
# (Token norm depends only on the image, so duplicate references are redundant.)
# Uses the same CIRCODataset attribute access as eval_circo_v2.eval_circo.
# ---------------------------------------------------------------------------

def _unique_reference_paths(ds):
    seen, paths = set(), []
    for q in range(len(ds)):
        rid = str(ds.annotations[q]["reference_img_id"])
        if rid in seen:
            continue
        seen.add(rid)
        paths.append(ds.img_paths[ds.img_ids_indexes_map[rid]])
    return paths


# ---------------------------------------------------------------------------
# Token norms — built exactly as encode_query builds the token pre-text-transform.
# ---------------------------------------------------------------------------

@torch.no_grad()
def token_norms(model, ref_tensors, device, batch=64):
    out = []
    for i in range(0, ref_tensors.shape[0], batch):
        chunk = ref_tensors[i:i + batch].to(device)
        v_raw = model.clip.encoder_image_unnorm(chunk)   # frozen CLIP visual, unnormalized
        z = model.mapper(v_raw)                           # VIB eval -> mu ; base -> z
        if model.merger is not None:
            z = model.merger(z)
        if model.pseudo_norm > 0:
            z = F.normalize(z, dim=-1) * model.pseudo_norm
        out.append(z.norm(dim=-1).reshape(-1).float().cpu())
    return torch.cat(out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="LABEL=/path/to/checkpoint.pt (one or more)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--circo_data_path", type=Path,
                   default=Path("/notebooks/experiment/eval_data/circo"))
    p.add_argument("--out", type=Path, default=Path("token_norm_probe.json"))
    return p.parse_args()


def main():
    args = parse_args()

    runs = []
    for tok in args.runs:
        if "=" not in tok:
            raise SystemExit(f"expected LABEL=PATH, got {tok!r}")
        label, path = tok.split("=", 1)
        if not Path(path).exists():
            raise SystemExit(f"[{label}] checkpoint not found: {path}")
        runs.append((label, path))

    # clip backbone name from the first checkpoint's stored config (no model build needed)
    ck0 = torch.load(runs[0][1], map_location="cpu")
    clip_model_name = ck0.get("config", {}).get("clip_model", "ViT-L/14")
    del ck0

    # Build the CIRCO val reference set ONCE, shared across all checkpoints.
    CIRCODataset, _ = _load_circo_official(args.circo_data_path)
    _, preprocess = clip.load(clip_model_name, device="cpu")
    ds = CIRCODataset(data_path=str(args.circo_data_path), split="val",
                      mode="relative", preprocess=preprocess)
    ref_paths = _unique_reference_paths(ds)
    print(f"CIRCO val: {len(ds)} queries, {len(ref_paths)} unique reference images "
          f"| backbone {clip_model_name}")
    ref_tensors = torch.stack(
        [preprocess(PIL.Image.open(p).convert("RGB")) for p in ref_paths]
    )

    results = {}
    print(f"\n{'label':12s}  {'mean':>8s}  {'std':>7s}  {'min':>7s}  {'max':>7s}   n")
    print("-" * 56)
    for label, path in runs:
        model, cm = load_model(Path(path), args.device)
        if cm != clip_model_name:
            raise SystemExit(f"[{label}] backbone {cm} != {clip_model_name}; "
                             f"cannot share one preprocess/reference set")
        n = token_norms(model, ref_tensors, args.device)
        results[label] = {
            "checkpoint": path,
            "n_refs": int(n.numel()),
            "mean": round(float(n.mean()), 4),
            "std":  round(float(n.std()),  4),
            "min":  round(float(n.min()),  4),
            "max":  round(float(n.max()),  4),
        }
        r = results[label]
        print(f"{label:12s}  {r['mean']:8.3f}  {r['std']:7.3f}  {r['min']:7.3f}  "
              f"{r['max']:7.3f}   {r['n_refs']}")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    args.out.write_text(json.dumps(
        {"backbone": clip_model_name, "n_unique_refs": len(ref_paths),
         "n_val_queries": len(ds), "results": results}, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
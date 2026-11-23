#!/usr/bin/env python
"""evaluation/probe_sigma.py

For each VIB checkpoint, report posterior statistics over the CIRCO val
reference images, using the EXACT mu/logvar path of VIBMapper:
    h      = mapper.encoder(v_raw)
    mu     = mapper.fc_mu(h)
    logvar = mapper.fc_logvar(h).clamp(-10, logvar_clip_max)
    sigma  = exp(0.5 * logvar)

Reports per checkpoint:
    sigma_mean : mean of sigma over all coords and images   (the "sigma_bar")
    sigma_min  : min sigma over coords (mean over images of the per-image min)
    mu_norm    : mean L2 norm of mu over images              (the "||mu||")

Reuse eval_circo_v2.load_model and probe_token_norm's reference set.

Usage (from codes/):
    PYTHONPATH=/notebooks/experiment/codes python evaluation/probe_sigma.py LABEL=/path/ckpt.pt [LABEL2=...]
"""
import argparse, json
from pathlib import Path
import clip, PIL.Image, torch

from evaluation.eval_circo_v2 import load_model, _load_circo_official
from evaluation.probe_token_norm import _unique_reference_paths


@torch.no_grad()
def sigma_stats(model, ref_tensors, device, batch=64):
    mapper = model.mapper
    clip_max = getattr(mapper, "logvar_clip_max", 2.0)
    sig_means, sig_mins, mu_norms = [], [], []
    for i in range(0, ref_tensors.shape[0], batch):
        chunk = ref_tensors[i:i + batch].to(device)
        v_raw = model.clip.encoder_image_unnorm(chunk)
        h = mapper.encoder(v_raw)
        B = v_raw.shape[0]
        mu = mapper.fc_mu(h).view(B, mapper.num_tokens, mapper.token_dim)
        logvar = mapper.fc_logvar(h).view(B, mapper.num_tokens, mapper.token_dim)
        logvar = logvar.clamp(-10.0, clip_max)
        sigma = (0.5 * logvar).exp()                       # (B, tok, dim)
        sig_means.append(sigma.float().mean(dim=(1, 2)).cpu())      # per-image mean sigma
        sig_mins.append(sigma.float().amin(dim=(1, 2)).cpu())       # per-image min sigma
        mu_norms.append(mu.float().norm(dim=-1).reshape(B, -1).mean(dim=1).cpu())
    return (torch.cat(sig_means), torch.cat(sig_mins), torch.cat(mu_norms))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="LABEL=/path/to/ckpt.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--circo_data_path", type=Path,
                   default=Path("/notebooks/experiment/eval_data/circo"))
    p.add_argument("--out", type=Path, default=Path("sigma_probe.json"))
    return p.parse_args()


def main():
    args = parse_args()
    runs = []
    for tok in args.runs:
        label, path = tok.split("=", 1)
        if not Path(path).exists():
            print(f"[{label}] not found: {path}")
            continue
        runs.append((label, path))

    ck0 = torch.load(runs[0][1], map_location="cpu")
    clip_model_name = ck0.get("config", {}).get("clip_model", "ViT-L/14")
    del ck0
    CIRCODataset, _ = _load_circo_official(args.circo_data_path)
    _, preprocess = clip.load(clip_model_name, device="cpu")
    ds = CIRCODataset(data_path=str(args.circo_data_path), split="val",
                      mode="relative", preprocess=preprocess)
    ref_paths = _unique_reference_paths(ds)
    print(f"CIRCO val: {len(ref_paths)} unique refs | backbone {clip_model_name}")
    ref_tensors = torch.stack(
        [preprocess(PIL.Image.open(p).convert("RGB")) for p in ref_paths])

    results = {}
    print(f"\n{'label':28s} {'sigma_bar':>9s} {'sigma_min':>9s} {'||mu||':>8s}")
    print("-" * 58)
    for label, path in runs:
        model, cm = load_model(Path(path), args.device)
        if not hasattr(model.mapper, "fc_logvar"):
            print(f"{label:28s}  (not a VIB mapper)")
            del model; continue
        sm, smin, mn = sigma_stats(model, ref_tensors, args.device)
        r = {"sigma_bar": round(float(sm.mean()), 4),
             "sigma_min": round(float(smin.mean()), 4),
             "mu_norm":   round(float(mn.mean()), 4)}
        results[label] = {"checkpoint": path, **r}
        print(f"{label:28s} {r['sigma_bar']:9.4f} {r['sigma_min']:9.4f} {r['mu_norm']:8.3f}")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
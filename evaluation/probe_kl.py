#!/usr/bin/env python
"""evaluation/probe_kl.py

Compute KL_mu and KL_sigma (in NATS) from a VIB checkpoint over the CIRCO
validation reference images, using the EXACT formula and code path of model.py:

    KL_mu    = 0.5 * sum_d mu_d^2                          (then mean over images)
    KL_sigma = 0.5 * sum_d (exp(logvar_d) - logvar_d - 1)  (then mean over images)
    KL_total = KL_mu + KL_sigma

mu and logvar are reproduced exactly as VIBMapper.forward builds them:
    h      = mapper.encoder(v_raw)
    mu     = mapper.fc_mu(h)
    logvar = mapper.fc_logvar(h).clamp(-10, logvar_clip_max)

Usage (from codes/):
    PYTHONPATH=/notebooks/experiment/codes python evaluation/probe_kl.py \
        vib01=/notebooks/experiment/4M_checkpoints/vib_hd256_tok1_lslr0.2_lsmax116_vib0.0001/epoch_01_step_01500.pt
"""
import argparse, json, math
from pathlib import Path
import clip, PIL.Image, torch

from evaluation.eval_circo_v2 import load_model, _load_circo_official
from evaluation.probe_token_norm import _unique_reference_paths


@torch.no_grad()
def kl_terms(model, ref_tensors, device, batch=64):
    mapper = model.mapper
    kl_mu_list, kl_sig_list = [], []
    clip_max = getattr(mapper, "logvar_clip_max", 2.0)
    for i in range(0, ref_tensors.shape[0], batch):
        chunk = ref_tensors[i:i + batch].to(device)
        v_raw = model.clip.encoder_image_unnorm(chunk)
        h = mapper.encoder(v_raw)
        B = v_raw.shape[0]
        mu = mapper.fc_mu(h).view(B, mapper.num_tokens, mapper.token_dim)
        logvar = mapper.fc_logvar(h).view(B, mapper.num_tokens, mapper.token_dim)
        logvar = logvar.clamp(-10.0, clip_max)
        kl_mu  = 0.5 * mu.pow(2).sum(dim=(1, 2))
        kl_sig = 0.5 * (logvar.exp() - logvar - 1).sum(dim=(1, 2))
        kl_mu_list.append(kl_mu.float().cpu())
        kl_sig_list.append(kl_sig.float().cpu())
    return torch.cat(kl_mu_list), torch.cat(kl_sig_list)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="LABEL=/path/to/checkpoint.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--circo_data_path", type=Path,
                   default=Path("/notebooks/experiment/eval_data/circo"))
    p.add_argument("--out", type=Path, default=Path("kl_probe.json"))
    return p.parse_args()


def main():
    args = parse_args()
    runs = []
    for tok in args.runs:
        label, path = tok.split("=", 1)
        if not Path(path).exists():
            raise SystemExit(f"[{label}] checkpoint not found: {path}")
        runs.append((label, path))

    ck0 = torch.load(runs[0][1], map_location="cpu")
    clip_model_name = ck0.get("config", {}).get("clip_model", "ViT-L/14")
    del ck0

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

    ln2 = math.log(2)
    results = {}
    print(f"\n{'label':10s}  {'KLmu_nat':>9s} {'KLsig_nat':>9s} {'KLtot_nat':>9s} "
          f"| {'KLmu_bit':>8s} {'KLsig_bit':>9s} {'KLtot_bit':>9s}")
    print("-" * 78)
    for label, path in runs:
        model, cm = load_model(Path(path), args.device)
        if not hasattr(model.mapper, "fc_logvar"):
            print(f"[{label}] not a VIB mapper (no fc_logvar) -- skipping")
            del model
            continue
        klmu, klsig = kl_terms(model, ref_tensors, args.device)
        mu_n, sig_n = float(klmu.mean()), float(klsig.mean())
        tot_n = mu_n + sig_n
        results[label] = {
            "checkpoint": path,
            "kl_mu_nats": round(mu_n, 4), "kl_sigma_nats": round(sig_n, 4),
            "kl_total_nats": round(tot_n, 4),
            "kl_mu_bits": round(mu_n / ln2, 4),
            "kl_sigma_bits": round(sig_n / ln2, 4),
            "kl_total_bits": round(tot_n / ln2, 4),
        }
        print(f"{label:10s}  {mu_n:9.3f} {sig_n:9.3f} {tot_n:9.3f} "
              f"| {mu_n/ln2:8.3f} {sig_n/ln2:9.3f} {tot_n/ln2:9.3f}")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    args.out.write_text(json.dumps(
        {"backbone": clip_model_name, "n_unique_refs": len(ref_paths),
         "results": results}, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
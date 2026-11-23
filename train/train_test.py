"""
train/train_test.py

Comprehensive test suite for train/train.py.
All outputs go to train/temp_test_outputs/ — deleted at exit unless --no-delete.

Run:
    python train/train_test.py
    python train/train_test.py --no-delete   # keep temp for inspection
"""

import sys
import math
import atexit
import shutil
import argparse
import traceback
import json
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler

from config.config import Config, get_config, load_from_yaml
from model.model import Pic2Word
from train.train import (
    WarmupCosineScheduler,
    eval_reconstruction,
    _compute_lambda_m,
    _save_json,
    _append_eval_log,
    save_checkpoint,
    load_checkpoint,
    _build_optimizer,
    _build_out_dir,
    train_one_epoch,
    main,
)

# ---------------------------------------------------------------------------
# Temp dir — always cleaned up at exit (unless --no-delete)
# ---------------------------------------------------------------------------

DELETE_TEMP = True
TEMP_DIR    = Path(__file__).resolve().parent / "temp_test_outputs"
TEMP_DIR.mkdir(exist_ok=True)

def _cleanup():
    if DELETE_TEMP and TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        print(f"\nTemp dir deleted: {TEMP_DIR}")
    else:
        print(f"\nTemp dir kept: {TEMP_DIR}")

atexit.register(_cleanup)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIP_MODEL   = "ViT-B/32"
BATCH_SIZE   = 4
N_STEPS      = 5                       # steps per epoch in tests
APPROX_TRAIN = BATCH_SIZE * N_STEPS    # 20 → 5 steps/epoch
DEVICE       = "cpu"

MOCK_FASHION = {
    "dress"  : {"1": 0.05, "5": 0.10, "10": 0.20, "50": 0.50},
    "shirt"  : {"1": 0.06, "5": 0.11, "10": 0.21, "50": 0.51},
    "toptee" : {"1": 0.04, "5": 0.09, "10": 0.19, "50": 0.49},
    "avg"    : {"1": 0.05, "5": 0.10, "10": 0.20, "50": 0.50},
}

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

PASS   = 0
FAIL   = 0
ERRORS = []

def run(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  PASS  {name}")
        PASS += 1
    except Exception as e:
        print(f"  FAIL  {name}")
        print(f"        {e}")
        FAIL += 1
        ERRORS.append((name, traceback.format_exc()))

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_cfg(subdir="default", **overrides):
    cfg = Config(
        clip_model           = CLIP_MODEL,
        mapper_type          = "base",
        hidden_dim           = 64,
        num_tokens           = 1,
        dropout              = 0.0,
        batch_size           = BATCH_SIZE,
        approx_train_samples = APPROX_TRAIN,
        approx_val_samples   = BATCH_SIZE * 2,
        epochs               = 2,
        lr                   = 1e-3,
        min_lr               = 1e-6,
        weight_decay         = 0.1,
        grad_clip            = 1.0,
        device               = DEVICE,
        output_dir           = str(TEMP_DIR / subdir),
        save_every_steps     = 3,
        eval_every_steps     = 0,
        eval_epoch_end       = False,
        run_analysis         = False,
        amp                  = False,
        ls_max               = 0.0,
        ls_lr_scale          = 1.0,
        margin               = 0.0,
        margin_weight        = 0.0,
        margin_delay         = 0.2,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class SyntheticLoader:
    """Yields (imgs, captions) batches of random data."""
    def __init__(self, n_batches=N_STEPS, batch_size=BATCH_SIZE, seed=0):
        self.n_batches  = n_batches
        self.batch_size = batch_size
        self.seed       = seed

    def __iter__(self):
        torch.manual_seed(self.seed)
        for _ in range(self.n_batches):
            imgs     = torch.randn(self.batch_size, 3, 224, 224)
            captions = ["a photo of a cat"] * self.batch_size
            yield (imgs, captions)

    def __len__(self):
        return self.n_batches


def make_model(cfg=None):
    if cfg is None:
        cfg = make_cfg()
    return Pic2Word(cfg).to(DEVICE)


def make_optimizer(model, cfg=None):
    if cfg is None:
        cfg = make_cfg()
    return _build_optimizer(model, cfg)


def make_scheduler(optimizer, cfg=None, total_steps=None):
    if cfg is None:
        cfg = make_cfg()
    ts = total_steps or cfg.total_steps
    return WarmupCosineScheduler(optimizer, cfg.warmup_steps, ts, cfg.min_lr)


def _make_and_save_ckpt(subdir, epoch, step=None, ls_val=4.6,
                        recon_r1=0.95, recon_r5=0.99):
    """Helper: build model, save checkpoint, return (path, cfg, model, opt)."""
    cfg  = make_cfg(subdir)
    m    = make_model(cfg)
    opt  = make_optimizer(m, cfg)
    with torch.no_grad():
        m.clip.logit_scale.fill_(ls_val)
    path    = TEMP_DIR / subdir / f"epoch_{epoch:02d}.pt"
    metrics = {"recon_recalls": {1: recon_r1, 5: recon_r5}}
    save_checkpoint(m, opt, epoch, metrics, path, cfg, total_steps=100, step=step)
    return path, cfg, m, opt


def _run_main(subdir, **cfg_overrides):
    """Run main() with synthetic loaders, return cfg."""
    cfg        = make_cfg(subdir, **cfg_overrides)
    loader     = SyntheticLoader(cfg.steps_per_epoch, BATCH_SIZE)
    val_loader = SyntheticLoader(2, BATCH_SIZE)
    with patch("train.train.build_loaders", return_value=(loader, val_loader)):
        main(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Load CLIP once (cached for all subsequent Pic2Word inits)
# ---------------------------------------------------------------------------

print("\nLoading CLIP (ViT-B/32)...")
_BASE_CFG   = make_cfg()
_BASE_MODEL = make_model(_BASE_CFG)
_BASE_MODEL.eval()
print("Done.\n")


# ===========================================================================
# 1. WarmupCosineScheduler
# ===========================================================================

def test_scheduler_warmup_increases():
    """LR increases linearly during warmup."""
    m   = make_model()
    opt = AdamW(m.trainable_parameters(), lr=1e-3)
    sch = WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100)
    lrs = []
    for _ in range(10):
        sch.step()
        lrs.append(sch.get_lr())
    for i in range(1, len(lrs)):
        assert lrs[i] >= lrs[i-1] - 1e-12, \
            f"LR did not increase at step {i}: {lrs[i-1]:.8f} -> {lrs[i]:.8f}"

def test_scheduler_cosine_decay_after_warmup():
    """LR decreases after warmup ends."""
    m   = make_model()
    opt = AdamW(m.trainable_parameters(), lr=1e-3)
    sch = WarmupCosineScheduler(opt, warmup_steps=5, total_steps=100)
    for _ in range(5):
        sch.step()
    peak = sch.get_lr()
    for _ in range(40):
        sch.step()
    assert sch.get_lr() < peak, \
        f"LR did not decay: peak={peak:.6f}, now={sch.get_lr():.6f}"

def test_scheduler_min_lr_floor():
    """LR never drops below min_lr * base_lr."""
    m      = make_model()
    min_lr = 1e-6
    base_lr = 1e-3
    opt    = AdamW(m.trainable_parameters(), lr=base_lr)
    sch    = WarmupCosineScheduler(opt, warmup_steps=5, total_steps=50, min_lr=min_lr)
    for _ in range(50):
        sch.step()
    assert sch.get_lr() >= base_lr * min_lr - 1e-12, \
        f"LR {sch.get_lr():.2e} below floor {base_lr * min_lr:.2e}"


def test_scheduler_peak_at_warmup_end():
    """At warmup_steps, scale == 1.0 → peak lr."""
    m   = make_model()
    opt = AdamW(m.trainable_parameters(), lr=1e-3)
    sch = WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100)
    for _ in range(10):
        sch.step()
    assert abs(sch.get_lr() - 1e-3) < 1e-8, \
        f"Peak LR should be 1e-3, got {sch.get_lr():.2e}"

def test_scheduler_step_counter():
    """_step tracks number of calls."""
    m   = make_model()
    opt = AdamW(m.trainable_parameters(), lr=1e-3)
    sch = WarmupCosineScheduler(opt, warmup_steps=5, total_steps=50)
    assert sch._step == 0
    for _ in range(7):
        sch.step()
    assert sch._step == 7

def test_scheduler_multiple_param_groups_scale_equally():
    """All param groups scale by same factor."""
    m     = make_model()
    lr1, lr2 = 1e-3, 5e-5
    opt   = AdamW([
        {"params": list(m.mapper.parameters()), "lr": lr1},
        {"params": [m.clip.logit_scale],        "lr": lr2},
    ], weight_decay=0.1)
    sch   = WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100)
    for _ in range(3):
        sch.step()
    scale = 3 / 10  # warmup scale at step 3
    assert abs(opt.param_groups[0]["lr"] - lr1 * scale) < 1e-10
    assert abs(opt.param_groups[1]["lr"] - lr2 * scale) < 1e-10

def test_scheduler_zero_lr_at_step_zero():
    """Before any step, lr is not yet scaled."""
    m   = make_model()
    opt = AdamW(m.trainable_parameters(), lr=1e-3)
    WarmupCosineScheduler(opt, warmup_steps=10, total_steps=100)
    # base_lrs are captured at init — param group lr unchanged
    assert abs(opt.param_groups[0]["lr"] - 1e-3) < 1e-10


# ===========================================================================
# 2. _compute_lambda_m
# ===========================================================================

def test_lambda_m_disabled_when_zero_weight():
    assert _compute_lambda_m(500, 1000, 0.0, 0.2) == 0.0

def test_lambda_m_zero_during_delay():
    """global_step < delay_steps → 0.0."""
    result = _compute_lambda_m(100, 1000, margin_weight=0.1, margin_delay=0.2)
    assert result == 0.0, f"Expected 0.0 during delay, got {result}"

def test_lambda_m_zero_at_exact_delay_boundary():
    """global_step == delay_steps - 1 → still 0."""
    delay = int(0.2 * 1000)
    result = _compute_lambda_m(delay - 1, 1000, 0.1, 0.2)
    assert result == 0.0

def test_lambda_m_positive_after_delay():
    result = _compute_lambda_m(300, 1000, 0.1, 0.2)
    assert result > 0.0, f"Expected > 0 after delay, got {result}"

def test_lambda_m_reaches_max_at_end():
    result = _compute_lambda_m(1000, 1000, 0.1, 0.2)
    assert abs(result - 0.1) < 1e-6, f"Expected 0.1 at end, got {result}"

def test_lambda_m_clamps_at_margin_weight():
    """Does not exceed margin_weight even past total_steps."""
    result = _compute_lambda_m(5000, 1000, 0.1, 0.2)
    assert result <= 0.1 + 1e-10

def test_lambda_m_linear_ramp():
    """Values at later steps should be larger."""
    r1 = _compute_lambda_m(500, 1000, 0.1, 0.2)
    r2 = _compute_lambda_m(700, 1000, 0.1, 0.2)
    r3 = _compute_lambda_m(900, 1000, 0.1, 0.2)
    assert r1 < r2 < r3, f"Not monotone: {r1:.4f}, {r2:.4f}, {r3:.4f}"


# ===========================================================================
# 3. eval_reconstruction
# ===========================================================================

def test_eval_reconstruction_returns_dict():
    loader = SyntheticLoader(3, BATCH_SIZE)
    result = eval_reconstruction(_BASE_MODEL, loader, DEVICE, k_vals=[1, 5])
    assert isinstance(result, dict)
    assert 1 in result and 5 in result

def test_eval_reconstruction_values_in_range():
    loader = SyntheticLoader(3, BATCH_SIZE)
    result = eval_reconstruction(_BASE_MODEL, loader, DEVICE, k_vals=[1, 5])
    for k, v in result.items():
        assert 0.0 <= v <= 1.0, f"R@{k}={v} out of [0, 1]"

def test_eval_reconstruction_sets_eval_mode():
    m = make_model(); m.train()
    loader = SyntheticLoader(2, BATCH_SIZE)
    eval_reconstruction(m, loader, DEVICE)
    assert not m.training, "eval_reconstruction must call model.eval()"

def test_eval_reconstruction_identity_r1_high():
    """When P == V (identity reconstruction), R@1 == 1.0."""
    B, D = 20, 512
    V    = F.normalize(torch.randn(B, D), dim=-1)
    sims = V @ V.T
    top1 = sims.topk(1, dim=-1).indices
    r1   = sum(top1[i].item() == i for i in range(B)) / B
    assert r1 == 1.0, f"R@1 should be 1.0 with P==V, got {r1}"

def test_eval_reconstruction_k_vals_respected():
    loader = SyntheticLoader(5, BATCH_SIZE)   # 20 samples — topk(10) safe
    result = eval_reconstruction(_BASE_MODEL, loader, DEVICE, k_vals=[1, 5, 10])
    assert set(result.keys()) == {1, 5, 10}


# ===========================================================================
# 4. _save_json
# ===========================================================================

def test_save_json_creates_file():
    path = TEMP_DIR / "json_test" / "out.json"
    _save_json({"a": 1}, path)
    assert path.exists()

def test_save_json_creates_parent_dirs():
    path = TEMP_DIR / "json_nested" / "a" / "b" / "c.json"
    _save_json({"x": 2}, path)
    assert path.exists()

def test_save_json_content_correct():
    data = {"epoch": 0, "r10": 0.234, "nested": {"k": [1, 2, 3]}}
    path = TEMP_DIR / "json_content" / "data.json"
    _save_json(data, path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == data

def test_save_json_overwrites_existing():
    path = TEMP_DIR / "json_overwrite" / "data.json"
    _save_json({"v": 1}, path)
    _save_json({"v": 2}, path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["v"] == 2


# ===========================================================================
# 5. _append_eval_log
# ===========================================================================

def test_append_eval_log_creates_file():
    path = TEMP_DIR / "log_test" / "eval_log.txt"
    _append_eval_log(MOCK_FASHION, path, epoch=0, step=100, global_step=100)
    assert path.exists()

def test_append_eval_log_appends_multiple_lines():
    path = TEMP_DIR / "log_multi" / "eval_log.txt"
    _append_eval_log(MOCK_FASHION, path, 0, 100, 100)
    _append_eval_log(MOCK_FASHION, path, 0, 200, 200)
    with open(path) as f:
        lines = [l for l in f.readlines() if l.strip()]
    assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"

def test_append_eval_log_contains_epoch_step():
    path = TEMP_DIR / "log_content" / "eval_log.txt"
    _append_eval_log(MOCK_FASHION, path, epoch=3, step=1500, global_step=9500)
    content = path.read_text()
    assert "Epoch 03" in content
    assert "Step 01500" in content

def test_append_eval_log_contains_r10_value():
    path = TEMP_DIR / "log_r10" / "eval_log.txt"
    _append_eval_log(MOCK_FASHION, path, epoch=0, step=100, global_step=100)
    content = path.read_text()
    assert "R@10" in content
    assert "20.0%" in content   # avg R@10 = 0.20


# ===========================================================================
# 6. save_checkpoint
# ===========================================================================

def test_save_checkpoint_file_created():
    cfg  = make_cfg("ckpt_created")
    m, opt = make_model(cfg), make_optimizer(make_model(cfg), cfg)
    path = TEMP_DIR / "ckpt_created" / "test.pt"
    save_checkpoint(m, opt, 0, {}, path, cfg, 100)
    assert path.exists()

def test_save_checkpoint_creates_parent_dir():
    cfg  = make_cfg("ckpt_parent")
    m, opt = make_model(cfg), make_optimizer(make_model(cfg), cfg)
    path = TEMP_DIR / "ckpt_parent" / "epoch_00" / "checkpoint.pt"
    save_checkpoint(m, opt, 0, {}, path, cfg, 100)
    assert path.parent.exists()

def test_save_checkpoint_all_keys_present():
    cfg  = make_cfg("ckpt_keys")
    m, opt = make_model(cfg), make_optimizer(make_model(cfg), cfg)
    path = TEMP_DIR / "ckpt_keys" / "ckpt.pt"
    save_checkpoint(m, opt, 0, {}, path, cfg, 100)
    ckpt = torch.load(path, map_location="cpu")
    for key in ["epoch", "step", "mapper_state", "optimizer", "metrics",
                "mapper_type", "mapper_kwargs", "config",
                "total_steps", "logit_scale", "merger_state"]:
        assert key in ckpt, f"Missing key: {key}"

def test_save_checkpoint_epoch_end_step_is_none():
    cfg  = make_cfg("ckpt_step_none")
    m, opt = make_model(cfg), make_optimizer(make_model(cfg), cfg)
    path = TEMP_DIR / "ckpt_step_none" / "ckpt.pt"
    save_checkpoint(m, opt, 2, {}, path, cfg, 100, step=None)
    ckpt = torch.load(path, map_location="cpu")
    assert ckpt["step"] is None
    assert ckpt["epoch"] == 2

def test_save_checkpoint_mid_epoch_step_stored():
    cfg  = make_cfg("ckpt_step_int")
    m, opt = make_model(cfg), make_optimizer(make_model(cfg), cfg)
    path = TEMP_DIR / "ckpt_step_int" / "ckpt.pt"
    save_checkpoint(m, opt, 1, {}, path, cfg, 100, step=500)
    ckpt = torch.load(path, map_location="cpu")
    assert ckpt["step"] == 500

def test_save_checkpoint_logit_scale_stored_correctly():
    cfg = make_cfg("ckpt_ls")
    m   = make_model(cfg)
    with torch.no_grad():
        m.clip.logit_scale.fill_(4.321)
    opt  = make_optimizer(m, cfg)
    path = TEMP_DIR / "ckpt_ls" / "ckpt.pt"
    save_checkpoint(m, opt, 0, {}, path, cfg, 100)
    ckpt = torch.load(path, map_location="cpu")
    assert abs(ckpt["logit_scale"] - 4.321) < 1e-5

def test_save_checkpoint_merger_state_none_when_no_merger():
    cfg  = make_cfg("ckpt_no_merger")
    m, opt = make_model(cfg), make_optimizer(make_model(cfg), cfg)
    path = TEMP_DIR / "ckpt_no_merger" / "ckpt.pt"
    save_checkpoint(m, opt, 0, {}, path, cfg, 100)
    ckpt = torch.load(path, map_location="cpu")
    assert ckpt["merger_state"] is None

def test_save_checkpoint_config_has_clip_model():
    cfg  = make_cfg("ckpt_config")
    m, opt = make_model(cfg), make_optimizer(make_model(cfg), cfg)
    path = TEMP_DIR / "ckpt_config" / "ckpt.pt"
    save_checkpoint(m, opt, 0, {}, path, cfg, 100)
    ckpt = torch.load(path, map_location="cpu")
    assert ckpt["config"]["clip_model"] == CLIP_MODEL


# ===========================================================================
# 7. load_checkpoint
# ===========================================================================

def test_load_checkpoint_epoch_end_resumes_next_epoch():
    path, cfg, m, opt = _make_and_save_ckpt("load_epoch_end", epoch=2, step=None)
    m2, opt2 = make_model(cfg), make_optimizer(make_model(cfg), cfg)
    start_epoch, start_step, *_ = load_checkpoint(m2, opt2, path)
    assert start_epoch == 3, f"Expected 3, got {start_epoch}"
    assert start_step  == 0, f"Expected 0, got {start_step}"

def test_load_checkpoint_mid_epoch_resumes_next_step():
    path, cfg, m, opt = _make_and_save_ckpt("load_mid", epoch=1, step=500)
    m2, opt2 = make_model(cfg), make_optimizer(make_model(cfg), cfg)
    start_epoch, start_step, *_ = load_checkpoint(m2, opt2, path)
    assert start_epoch == 1,   f"Expected 1, got {start_epoch}"
    assert start_step  == 501, f"Expected 501, got {start_step}"

def test_load_checkpoint_mapper_weights_restored():
    path, cfg, m_orig, _ = _make_and_save_ckpt("load_wts", epoch=0)
    m_new  = make_model(cfg)
    opt_new = make_optimizer(m_new, cfg)
    load_checkpoint(m_new, opt_new, path)
    for (n, p1), (_, p2) in zip(m_orig.mapper.named_parameters(),
                                  m_new.mapper.named_parameters()):
        assert torch.allclose(p1, p2), f"Mapper param '{n}' not restored"

def test_load_checkpoint_logit_scale_restored():
    path, cfg, _, _ = _make_and_save_ckpt("load_ls", epoch=0, ls_val=4.111)
    m   = make_model(cfg)
    opt = make_optimizer(m, cfg)
    load_checkpoint(m, opt, path)
    assert abs(m.clip.logit_scale.item() - 4.111) < 1e-5

def test_load_checkpoint_returns_best_metric():
    path, cfg, _, _ = _make_and_save_ckpt("load_metric", epoch=0, recon_r1=0.95)
    m, opt = make_model(cfg), make_optimizer(make_model(cfg), cfg)
    _, _, _, _, best_metric = load_checkpoint(m, opt, path)
    assert isinstance(best_metric, float)
    assert abs(best_metric - 0.95) < 1e-6

def test_load_checkpoint_total_steps_returned():
    path, cfg, _, _ = _make_and_save_ckpt("load_total", epoch=0)
    m, opt = make_model(cfg), make_optimizer(make_model(cfg), cfg)
    _, _, total, _, _ = load_checkpoint(m, opt, path)
    assert total == 100   # passed to save_checkpoint


# ===========================================================================
# 8. _build_optimizer
# ===========================================================================

def test_build_optimizer_single_group_default():
    cfg = make_cfg("opt_single")
    m   = make_model(cfg)
    opt = _build_optimizer(m, cfg)
    assert len(opt.param_groups) == 1

def test_build_optimizer_two_groups_when_ls_lr_scale_ne_1():
    cfg = make_cfg("opt_ls", ls_lr_scale=0.05)
    m   = make_model(cfg)
    opt = _build_optimizer(m, cfg)
    assert len(opt.param_groups) == 2

def test_build_optimizer_ls_lr_is_scaled():
    cfg = make_cfg("opt_ls_lr", ls_lr_scale=0.05)
    m   = make_model(cfg)
    opt = _build_optimizer(m, cfg)
    mapper_lr = opt.param_groups[0]["lr"]
    ls_lr     = opt.param_groups[1]["lr"]
    assert abs(mapper_lr - cfg.lr) < 1e-12
    assert abs(ls_lr - cfg.lr * 0.05) < 1e-12

def test_build_optimizer_weight_decay_correct():
    cfg = make_cfg("opt_wd")
    m   = make_model(cfg)
    opt = _build_optimizer(m, cfg)
    for g in opt.param_groups:
        assert abs(g["weight_decay"] - cfg.weight_decay) < 1e-12

def test_build_optimizer_returns_adamw():
    cfg = make_cfg("opt_type")
    m   = make_model(cfg)
    opt = _build_optimizer(m, cfg)
    assert isinstance(opt, AdamW)

def test_build_optimizer_logit_scale_in_trainable_group():
    """logit_scale should be in one of the optimizer param groups."""
    cfg = make_cfg("opt_ls_in_group", ls_lr_scale=0.05)
    m   = make_model(cfg)
    opt = _build_optimizer(m, cfg)
    all_params = [p for g in opt.param_groups for p in g["params"]]
    param_ids  = {id(p) for p in all_params}
    assert id(m.clip.logit_scale) in param_ids, "logit_scale not in any optimizer group"


# ===========================================================================
# 9. _build_out_dir
# ===========================================================================

def test_build_out_dir_base_suffix():
    cfg = make_cfg("outdir_base")
    d   = _build_out_dir(cfg)
    assert d.name == f"base_hd{cfg.hidden_dim}_tok{cfg.num_tokens}"

def test_build_out_dir_margin_in_suffix():
    cfg = make_cfg("outdir_margin", margin=0.1, margin_weight=0.1)
    d   = _build_out_dir(cfg)
    assert "margin" in d.name and "mw" in d.name

def test_build_out_dir_lslr_in_suffix():
    cfg = make_cfg("outdir_lslr", ls_lr_scale=0.05)
    d   = _build_out_dir(cfg)
    assert "lslr" in d.name

def test_build_out_dir_lsmax_in_suffix():
    cfg = make_cfg("outdir_lsmax", ls_max=116.0)
    d   = _build_out_dir(cfg)
    assert "lsmax116" in d.name

def test_build_out_dir_is_created():
    cfg = make_cfg("outdir_exists")
    d   = _build_out_dir(cfg)
    assert d.exists() and d.is_dir()

def test_build_out_dir_no_margin_in_suffix_when_disabled():
    cfg = make_cfg("outdir_no_margin", margin=0.0, margin_weight=0.0)
    d   = _build_out_dir(cfg)
    assert "margin" not in d.name


# ===========================================================================
# 10. train_one_epoch
# ===========================================================================

def _make_train_components(subdir, **cfg_kw):
    cfg    = make_cfg(subdir, **cfg_kw)
    m      = make_model(cfg)
    opt    = _build_optimizer(m, cfg)
    sch    = make_scheduler(opt, cfg, total_steps=N_STEPS * cfg.epochs)
    scaler = GradScaler(enabled=False)
    loader = SyntheticLoader(N_STEPS, BATCH_SIZE)
    out    = TEMP_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    return m, opt, sch, scaler, loader, cfg, out

def test_train_one_epoch_returns_loss_dict():
    m, opt, sch, scaler, loader, cfg, out = _make_train_components("toe_loss")
    result = train_one_epoch(m, loader, opt, sch, scaler, 0, cfg, out, N_STEPS*2)
    assert isinstance(result, dict) and "loss" in result

def test_train_one_epoch_loss_is_positive_finite():
    m, opt, sch, scaler, loader, cfg, out = _make_train_components("toe_finite")
    result = train_one_epoch(m, loader, opt, sch, scaler, 0, cfg, out, N_STEPS*2)
    loss   = result["loss"]
    assert math.isfinite(loss), f"Loss not finite: {loss}"
    assert loss > 0, f"Loss should be positive, got {loss}"

def test_train_one_epoch_scheduler_advances():
    m, opt, sch, scaler, loader, cfg, out = _make_train_components("toe_sch")
    assert sch._step == 0
    train_one_epoch(m, loader, opt, sch, scaler, 0, cfg, out, N_STEPS*2)
    assert sch._step == N_STEPS

def test_train_one_epoch_mid_epoch_checkpoints_at_correct_steps():
    m, opt, sch, scaler, loader, cfg, out = _make_train_components(
        "toe_mid", save_every_steps=2)
    train_one_epoch(m, loader, opt, sch, scaler, 0, cfg, out, N_STEPS*2,
                    save_every_steps=2)
    assert (out / "epoch_00_step_00002.pt").exists(), "No checkpoint at step 2"
    assert (out / "epoch_00_step_00004.pt").exists(), "No checkpoint at step 4"

def test_train_one_epoch_no_checkpoint_at_step_zero():
    m, opt, sch, scaler, loader, cfg, out = _make_train_components(
        "toe_no_zero", save_every_steps=1)
    train_one_epoch(m, loader, opt, sch, scaler, 0, cfg, out, N_STEPS*2,
                    save_every_steps=1)
    assert not (out / "epoch_00_step_00000.pt").exists(), \
        "Should not save at step 0 (condition: step > 0)"

def test_train_one_epoch_resume_skips_steps():
    """start_step=3: steps 0,1,2 skipped; scheduler still advances all N_STEPS."""
    m, opt, sch, scaler, loader, cfg, out = _make_train_components("toe_resume")
    train_one_epoch(m, loader, opt, sch, scaler, 0, cfg, out, N_STEPS*2,
                    start_step=3)
    # Scheduler must advance for all N_STEPS (including skipped ones)
    assert sch._step == N_STEPS

def test_train_one_epoch_model_in_train_mode_after():
    m, opt, sch, scaler, loader, cfg, out = _make_train_components("toe_mode")
    train_one_epoch(m, loader, opt, sch, scaler, 0, cfg, out, N_STEPS*2)
    assert m.training, "Model must be in train mode after train_one_epoch"

def test_train_one_epoch_logit_scale_clamped():
    """ls_max clamps logit_scale after each optimizer step."""
    m, opt, sch, scaler, _, cfg, out = _make_train_components(
        "toe_clamp", ls_max=100.0)
    with torch.no_grad():
        m.clip.logit_scale.fill_(math.log(200.0))   # 200 >> 100
    loader = SyntheticLoader(3, BATCH_SIZE)
    train_one_epoch(m, loader, opt, sch, scaler, 0, cfg, out, N_STEPS*2,
                    save_every_steps=0)
    assert m.clip.logit_scale.exp().item() <= 100.0 + 0.1, \
        f"logit_scale not clamped: {m.clip.logit_scale.exp().item():.2f}"

def test_train_one_epoch_step_eval_json_created():
    """eval_every_steps=2 → step 2 and 4 get json files (mocked fashion eval)."""
    m, opt, sch, scaler, loader, cfg, out = _make_train_components(
        "toe_stepeval", eval_every_steps=2)
    step_dir = out / "step_evals"
    step_dir.mkdir(parents=True, exist_ok=True)

    with patch("train.train._run_fashion_eval", return_value=MOCK_FASHION):
        train_one_epoch(m, loader, opt, sch, scaler, 0, cfg, out, N_STEPS*2,
                        save_every_steps=0,
                        preprocess=object(),   # non-None to pass the guard
                        step_eval_dir=step_dir)

    assert (step_dir / "step_00002_epoch_00.json").exists()
    assert (step_dir / "step_00004_epoch_00.json").exists()

def test_train_one_epoch_step_eval_json_content():
    """Step eval json contains epoch, step, fashion_eval keys."""
    m, opt, sch, scaler, loader, cfg, out = _make_train_components(
        "toe_stepeval_content", eval_every_steps=2)
    step_dir = out / "step_evals"
    step_dir.mkdir(parents=True, exist_ok=True)

    with patch("train.train._run_fashion_eval", return_value=MOCK_FASHION):
        train_one_epoch(m, loader, opt, sch, scaler, 0, cfg, out, N_STEPS*2,
                        save_every_steps=0, preprocess=object(),
                        step_eval_dir=step_dir)

    with open(step_dir / "step_00002_epoch_00.json") as f:
        data = json.load(f)
    assert "epoch" in data and "step" in data and "fashion_eval" in data
    assert data["epoch"] == 0 and data["step"] == 2

def test_train_one_epoch_step_eval_log_appended():
    """Step eval writes to eval_log.txt."""
    m, opt, sch, scaler, loader, cfg, out = _make_train_components(
        "toe_stepeval_log", eval_every_steps=2)
    step_dir = out / "step_evals"
    step_dir.mkdir(parents=True, exist_ok=True)

    with patch("train.train._run_fashion_eval", return_value=MOCK_FASHION):
        train_one_epoch(m, loader, opt, sch, scaler, 0, cfg, out, N_STEPS*2,
                        save_every_steps=0, preprocess=object(),
                        step_eval_dir=step_dir)

    log = (step_dir / "eval_log.txt").read_text()
    assert log.count("Epoch") == 2   # step 2 and step 4


# ===========================================================================
# 11. main — integration
# ===========================================================================

def test_main_epoch_dirs_created():
    cfg = _run_main("main_epoch_dirs", epochs=2)
    out = _build_out_dir(cfg)
    assert (out / "epoch_00").exists()
    assert (out / "epoch_01").exists()

def test_main_checkpoint_pt_in_each_epoch_dir():
    cfg = _run_main("main_ckpt_pt", epochs=2)
    out = _build_out_dir(cfg)
    assert (out / "epoch_00" / "checkpoint.pt").exists()
    assert (out / "epoch_01" / "checkpoint.pt").exists()

def test_main_final_pt_created():
    cfg = _run_main("main_final", epochs=2)
    assert (_build_out_dir(cfg) / "final.pt").exists()

def test_main_mid_epoch_checkpoints_are_flat():
    """Mid-epoch .pt files sit directly in out_dir, not in epoch_XX/."""
    cfg = _run_main("main_mid_flat", save_every_steps=3, epochs=1)
    out = _build_out_dir(cfg)
    assert (out / "epoch_00_step_00003.pt").exists()
    assert not (out / "epoch_00" / "epoch_00_step_00003.pt").exists()

def test_main_fashion_eval_skipped_gracefully_no_crash():
    """eval_epoch_end=True but no eval data → no crash."""
    cfg = _run_main("main_fashion_skip", eval_epoch_end=True, epochs=1)
    assert (_build_out_dir(cfg) / "epoch_00").exists()

def test_main_analysis_skipped_gracefully_no_crash():
    """run_analysis=True but no eval data → no crash."""
    cfg = _run_main("main_analysis_skip", run_analysis=True, epochs=1)
    assert (_build_out_dir(cfg) / "epoch_00").exists()

def test_main_fashion_eval_json_saved():
    """With mocked fashion eval and eval_epoch_end=True → json saved."""
    cfg        = make_cfg("main_fashion_json", eval_epoch_end=True, epochs=1)
    loader     = SyntheticLoader(cfg.steps_per_epoch, BATCH_SIZE)
    val_loader = SyntheticLoader(2, BATCH_SIZE)

    with patch("train.train.build_loaders", return_value=(loader, val_loader)), \
         patch("train.train._run_fashion_eval", return_value=MOCK_FASHION):
        main(cfg)

    out = _build_out_dir(cfg)
    json_path = out / "epoch_00" / "fashion_eval.json"
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)
    assert "dress" in data and "avg" in data

def test_main_step_evals_dir_created_when_eval_every_steps():
    cfg        = make_cfg("main_stepdir", eval_every_steps=2, epochs=1)
    loader     = SyntheticLoader(cfg.steps_per_epoch, BATCH_SIZE)
    val_loader = SyntheticLoader(2, BATCH_SIZE)

    with patch("train.train.build_loaders", return_value=(loader, val_loader)):
        main(cfg)

    assert (_build_out_dir(cfg) / "step_evals").exists()

def test_main_best_model_is_highest_fashion_r10():
    """final.pt should come from epoch with higher fashion R@10."""
    fashion_e0 = {**MOCK_FASHION, "avg": {"1": 0.05, "5": 0.10, "10": 0.20}}
    fashion_e1 = {**MOCK_FASHION, "avg": {"1": 0.08, "5": 0.15, "10": 0.35}}

    call_count = [0]
    def _mock(model, cfg, preprocess):
        r = fashion_e0 if call_count[0] == 0 else fashion_e1
        call_count[0] += 1
        return r

    cfg        = make_cfg("main_best", eval_epoch_end=True, epochs=2)
    loader     = SyntheticLoader(cfg.steps_per_epoch, BATCH_SIZE)
    val_loader = SyntheticLoader(2, BATCH_SIZE)

    with patch("train.train.build_loaders", return_value=(loader, val_loader)), \
         patch("train.train._run_fashion_eval", side_effect=_mock):
        main(cfg)

    out   = _build_out_dir(cfg)
    final = torch.load(out / "final.pt", map_location="cpu")
    assert final["epoch"] == 1, \
        f"final.pt should be epoch 1 (better R@10), got epoch {final['epoch']}"

def test_main_resume_from_mid_epoch_no_crash():
    """Resume from a mid-epoch checkpoint completes without error."""
    cfg        = make_cfg("main_resume_mid", epochs=2, save_every_steps=3)
    loader     = SyntheticLoader(cfg.steps_per_epoch, BATCH_SIZE)
    val_loader = SyntheticLoader(2, BATCH_SIZE)
    out        = _build_out_dir(cfg)

    # First full run to generate a mid-epoch checkpoint
    with patch("train.train.build_loaders", return_value=(loader, val_loader)):
        main(cfg)

    mid_ckpt = out / "epoch_00_step_00003.pt"
    assert mid_ckpt.exists()

    cfg2 = make_cfg("main_resume_mid", epochs=2, save_every_steps=3,
                    resume=str(mid_ckpt))
    with patch("train.train.build_loaders", return_value=(loader, val_loader)):
        main(cfg2)   # must not raise

def test_main_checkpoint_has_correct_epoch():
    """epoch_00/checkpoint.pt must have epoch=0."""
    cfg = _run_main("main_epoch_val", epochs=2)
    out = _build_out_dir(cfg)
    ckpt = torch.load(out / "epoch_00" / "checkpoint.pt", map_location="cpu")
    assert ckpt["epoch"] == 0
    ckpt1 = torch.load(out / "epoch_01" / "checkpoint.pt", map_location="cpu")
    assert ckpt1["epoch"] == 1


# ===========================================================================
# 12. Config / YAML
# ===========================================================================

def test_yaml_loads_values():
    path = TEMP_DIR / "yaml_test.yaml"
    path.write_text("batch_size: 256\nepochs: 3\ndevice: cpu\n")
    cfg = load_from_yaml(str(path))
    assert cfg.batch_size == 256
    assert cfg.epochs     == 3

def test_yaml_cli_override_wins():
    path = TEMP_DIR / "yaml_override.yaml"
    path.write_text("batch_size: 256\n")
    cfg = load_from_yaml(str(path), batch_size=512)
    assert cfg.batch_size == 512

def test_yaml_missing_raises():
    try:
        load_from_yaml("nonexistent_config_file.yaml")
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError:
        pass

def test_yaml_betas_list_becomes_tuple():
    path = TEMP_DIR / "yaml_betas.yaml"
    path.write_text("betas: [0.9, 0.95]\n")
    cfg = load_from_yaml(str(path))
    assert isinstance(cfg.betas, tuple)
    assert cfg.betas == (0.9, 0.95)

def test_get_config_unknown_key_raises():
    try:
        get_config(totally_unknown_field_xyz=99)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# ===========================================================================
# Runner
# ===========================================================================

TESTS = [
    # WarmupCosineScheduler
    ("scheduler | warmup increases linearly",              test_scheduler_warmup_increases),
    ("scheduler | cosine decay after warmup",              test_scheduler_cosine_decay_after_warmup),
    ("scheduler | min_lr floor respected",                 test_scheduler_min_lr_floor),
    ("scheduler | peak lr at warmup end",                  test_scheduler_peak_at_warmup_end),
    ("scheduler | step counter tracks calls",              test_scheduler_step_counter),
    ("scheduler | multiple groups scale equally",          test_scheduler_multiple_param_groups_scale_equally),
    ("scheduler | lr unchanged before first step",         test_scheduler_zero_lr_at_step_zero),
    # _compute_lambda_m
    ("lambda_m  | zero when margin_weight=0",              test_lambda_m_disabled_when_zero_weight),
    ("lambda_m  | zero during delay phase",                test_lambda_m_zero_during_delay),
    ("lambda_m  | zero at delay boundary - 1",             test_lambda_m_zero_at_exact_delay_boundary),
    ("lambda_m  | positive after delay",                   test_lambda_m_positive_after_delay),
    ("lambda_m  | reaches margin_weight at end",           test_lambda_m_reaches_max_at_end),
    ("lambda_m  | clamps at margin_weight",                test_lambda_m_clamps_at_margin_weight),
    ("lambda_m  | monotone ramp",                          test_lambda_m_linear_ramp),
    # eval_reconstruction
    ("eval_recon| returns correct keys",                   test_eval_reconstruction_returns_dict),
    ("eval_recon| values in [0, 1]",                       test_eval_reconstruction_values_in_range),
    ("eval_recon| sets model to eval mode",                test_eval_reconstruction_sets_eval_mode),
    ("eval_recon| R@1=1.0 when P==V",                      test_eval_reconstruction_identity_r1_high),
    ("eval_recon| k_vals respected",                       test_eval_reconstruction_k_vals_respected),
    # _save_json
    ("json      | creates file",                           test_save_json_creates_file),
    ("json      | creates parent dirs",                    test_save_json_creates_parent_dirs),
    ("json      | content correct",                        test_save_json_content_correct),
    ("json      | overwrites existing",                    test_save_json_overwrites_existing),
    # _append_eval_log
    ("eval_log  | creates file",                           test_append_eval_log_creates_file),
    ("eval_log  | appends multiple lines",                 test_append_eval_log_appends_multiple_lines),
    ("eval_log  | contains epoch and step",                test_append_eval_log_contains_epoch_step),
    ("eval_log  | contains R@10 value",                    test_append_eval_log_contains_r10_value),
    # save_checkpoint
    ("save_ckpt | file created",                           test_save_checkpoint_file_created),
    ("save_ckpt | creates parent dir",                     test_save_checkpoint_creates_parent_dir),
    ("save_ckpt | all keys present",                       test_save_checkpoint_all_keys_present),
    ("save_ckpt | step=None for epoch end",                test_save_checkpoint_epoch_end_step_is_none),
    ("save_ckpt | step=int for mid-epoch",                 test_save_checkpoint_mid_epoch_step_stored),
    ("save_ckpt | logit_scale stored correctly",           test_save_checkpoint_logit_scale_stored_correctly),
    ("save_ckpt | merger_state=None when no merger",       test_save_checkpoint_merger_state_none_when_no_merger),
    ("save_ckpt | config has clip_model",                  test_save_checkpoint_config_has_clip_model),
    # load_checkpoint
    ("load_ckpt | epoch end → resumes next epoch",         test_load_checkpoint_epoch_end_resumes_next_epoch),
    ("load_ckpt | mid-epoch → resumes next step",          test_load_checkpoint_mid_epoch_resumes_next_step),
    ("load_ckpt | mapper weights restored",                test_load_checkpoint_mapper_weights_restored),
    ("load_ckpt | logit_scale restored",                   test_load_checkpoint_logit_scale_restored),
    ("load_ckpt | returns best_metric",                    test_load_checkpoint_returns_best_metric),
    ("load_ckpt | returns total_steps",                    test_load_checkpoint_total_steps_returned),
    # _build_optimizer
    ("optimizer | single group by default",                test_build_optimizer_single_group_default),
    ("optimizer | two groups when ls_lr_scale != 1",       test_build_optimizer_two_groups_when_ls_lr_scale_ne_1),
    ("optimizer | ls lr is scaled correctly",              test_build_optimizer_ls_lr_is_scaled),
    ("optimizer | weight_decay correct",                   test_build_optimizer_weight_decay_correct),
    ("optimizer | returns AdamW",                          test_build_optimizer_returns_adamw),
    ("optimizer | logit_scale in a param group",           test_build_optimizer_logit_scale_in_trainable_group),
    # _build_out_dir
    ("out_dir   | correct base suffix",                    test_build_out_dir_base_suffix),
    ("out_dir   | margin in suffix",                       test_build_out_dir_margin_in_suffix),
    ("out_dir   | lslr in suffix",                         test_build_out_dir_lslr_in_suffix),
    ("out_dir   | lsmax in suffix",                        test_build_out_dir_lsmax_in_suffix),
    ("out_dir   | directory is created",                   test_build_out_dir_is_created),
    ("out_dir   | no margin when disabled",                test_build_out_dir_no_margin_in_suffix_when_disabled),
    # train_one_epoch
    ("toe       | returns loss dict",                      test_train_one_epoch_returns_loss_dict),
    ("toe       | loss is positive and finite",            test_train_one_epoch_loss_is_positive_finite),
    ("toe       | scheduler advances N_STEPS",             test_train_one_epoch_scheduler_advances),
    ("toe       | mid-epoch checkpoints at correct steps", test_train_one_epoch_mid_epoch_checkpoints_at_correct_steps),
    ("toe       | no checkpoint at step 0",                test_train_one_epoch_no_checkpoint_at_step_zero),
    ("toe       | resume skips steps correctly",           test_train_one_epoch_resume_skips_steps),
    ("toe       | model in train mode after",              test_train_one_epoch_model_in_train_mode_after),
    ("toe       | logit_scale clamped by ls_max",          test_train_one_epoch_logit_scale_clamped),
    ("toe       | step eval json created",                 test_train_one_epoch_step_eval_json_created),
    ("toe       | step eval json content correct",         test_train_one_epoch_step_eval_json_content),
    ("toe       | step eval log appended",                 test_train_one_epoch_step_eval_log_appended),
    # main integration
    ("main      | epoch dirs created",                     test_main_epoch_dirs_created),
    ("main      | checkpoint.pt in each epoch dir",        test_main_checkpoint_pt_in_each_epoch_dir),
    ("main      | final.pt created",                       test_main_final_pt_created),
    ("main      | mid-epoch checkpoints are flat files",   test_main_mid_epoch_checkpoints_are_flat),
    ("main      | fashion eval skip no crash",             test_main_fashion_eval_skipped_gracefully_no_crash),
    ("main      | analysis skip no crash",                 test_main_analysis_skipped_gracefully_no_crash),
    ("main      | fashion eval json saved",                test_main_fashion_eval_json_saved),
    ("main      | step_evals dir created",                 test_main_step_evals_dir_created_when_eval_every_steps),
    ("main      | best model is highest fashion R@10",     test_main_best_model_is_highest_fashion_r10),
    ("main      | resume from mid-epoch no crash",         test_main_resume_from_mid_epoch_no_crash),
    ("main      | checkpoint has correct epoch number",    test_main_checkpoint_has_correct_epoch),
    # Config / YAML
    ("yaml      | loads values correctly",                 test_yaml_loads_values),
    ("yaml      | CLI override wins over yaml",            test_yaml_cli_override_wins),
    ("yaml      | missing file raises FileNotFoundError",  test_yaml_missing_raises),
    ("yaml      | betas list → tuple",                     test_yaml_betas_list_becomes_tuple),
    ("yaml      | unknown key raises ValueError",          test_get_config_unknown_key_raises),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-delete", action="store_true",
                        help="Keep temp outputs after tests")
    args = parser.parse_args()
    if args.no_delete:
        DELETE_TEMP = False

    print(f"{'='*70}")
    print(f"  train_test.py  —  {len(TESTS)} tests")
    print(f"  Temp dir: {TEMP_DIR}")
    print(f"{'='*70}")

    for name, fn in TESTS:
        run(name, fn)

    print(f"\n{'='*70}")
    print(f"  {PASS} passed   {FAIL} failed   {PASS+FAIL} total")
    print(f"{'='*70}\n")

    if ERRORS:
        print("-- Failed test details --")
        for name, tb in ERRORS:
            print(f"\n[ {name} ]")
            print(tb)

    sys.exit(0 if FAIL == 0 else 1)
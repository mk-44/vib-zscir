"""
config/config_h.py  (ViT-H / open_clip config; same as config.py + backbone fields)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Literal
import yaml


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Data constants
    # ------------------------------------------------------------------
    hf_base_url          : str = "https://huggingface.co/datasets/pixparse/cc12m-wds/resolve/main"
    hf_eval_repo_id      : str = "BUAADreamer/cir_dataset"
    train_data_dir       : str = "/notebooks/experiment/train_data"
    eval_data_dir        : str = "/notebooks/experiment/eval_data"
    train_shards         : int = 793          # shards 0000 → 0792
    val_shard            : int = 793          # shard  0793
    approx_train_samples : int = 4_000_000
    approx_val_samples   : int = 5_000

    # ------------------------------------------------------------------
    # Model constants (previously hardcoded in model.py)
    # ------------------------------------------------------------------
    max_token_len : int = 77              # CLIP max sequence length
    clip_prompt   : str = "a photo of"   # pseudo-token prompt prefix

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    clip_model   : str   = "ViT-H-14"
    clip_backend : str   = "open_clip"                     # "openai" (clip.load) or "open_clip"
    clip_pretrained : Optional[str] = "laion2b_s32b_b79k"  # open_clip weights tag; None for the openai backend
    mapper_type : Literal["base", "multilayer", "weighted", "fused_weighted", "vib"] = "base"
    hidden_dim   : int   = 256
    num_tokens   : int   = 1
    num_heads    : int   = 8
    num_layers   : int   = 1
    dropout      : float = 0.1
    merge_tokens : Optional[int] = None
    merge_type  : Literal["learned", "weighted"] = "learned"
    pseudo_norm: float = 0.0 # 0.0 = disabled (no norm constraint)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    batch_size   : int           = 1024
    num_workers  : int           = 4
    shuffle_buf  : int           = 2_000
    hf_token     : Optional[str] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    epochs       : int   = 5
    lr           : float = 1e-4
    min_lr       : float = 1e-6
    weight_decay : float = 0.1
    betas        : Tuple[float, float] = (0.9, 0.98)
    grad_clip    : float = 1.0
    amp          : bool  = True
    use_captions : bool  = False
    repulsion_weight : float = 0.0
    decorr_weight    : float = 0.0

    # ------------------------------------------------------------------
    # Logit scale control
    # ------------------------------------------------------------------
    ls_lr_scale : float = 1.0    # lr multiplier for logit_scale (1.0 = same as mapper)
    ls_max      : float = 0.0    # clamp logit_scale <= log(ls_max); 0.0 = no clamp

    # ------------------------------------------------------------------
    # Soft margin auxiliary loss
    # ------------------------------------------------------------------
    margin        : float = 0.0  # margin value; 0.0 = disabled
    margin_weight : float = 0.0  # peak weight;  0.0 = disabled
    margin_delay  : float = 0.2  # fraction of total_steps before ramp starts

    # ------------------------------------------------------------------
    # Add noise to pseudo_token
    # ------------------------------------------------------------------
    noise_sigma: float = 0.0

    # ------------------------------------------------------------------
    # Variational Information Bottleneck (VIB)
    # ------------------------------------------------------------------
    vib_beta         : float = 0.0  # KL weight ceiling (β_target); 0.0 = disabled
    vib_beta_initial : float = 0.0  # KL weight floor at warmup start (β_initial)
    vib_beta_warmup  : float = 0.3  # fraction of total_steps for linear β warmup (β_initial → β_target)
    vib_beta_delay   : float = 0.0  # fraction of training with β=0 before warmup starts
    vib_delay        : float = 0.0  # fraction of training with deterministic mapper (no sampling)
    vib_logvar_bias  : float = -2.0 # initial bias for fc_logvar (controls initial σ at step 0)
    vib_kl_mu_weight: float = 1.0
    vib_kl_var_weight: float = 1.0
    vib_logvar_clip_max: float = 2.0

    # ------------------------------------------------------------------
    # Warm-start / weights-only initialization
    # ------------------------------------------------------------------
    # Distinct from `resume`. init_from loads compatible mapper weights only,
    # starts a fresh run (epoch=0, step=0, fresh optimizer, fresh schedules).
    # Mutually exclusive with `resume`.
    init_from : Optional[str] = None

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    output_dir       : str          = "checkpoints"
    save_every_steps : int          = 500
    resume           : Optional[str] = None

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device : str = "cuda"
    seed : Optional[int] = None


    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------
    eval_every_steps : int  = 1000
    eval_epoch_end   : bool = True
    run_analysis     : bool = True

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def steps_per_epoch(self) -> int:
        return self.approx_train_samples // self.batch_size

    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * self.epochs

    @property
    def warmup_steps(self) -> int:
        return max(100, int(0.05 * self.total_steps))

    @property
    def mapper_kwargs(self):
        base_kwargs = {
            'hidden_dim': self.hidden_dim,
            'dropout':    self.dropout,
            'num_tokens': self.num_tokens,
        }
        if self.mapper_type == 'multilayer':
            return {**base_kwargs,
                    'num_heads':  self.num_heads,
                    'num_layers': self.num_layers}
        if self.mapper_type == 'vib':
            return {**base_kwargs,
                    'logvar_bias': self.vib_logvar_bias, 
                    'logvar_clip_max': self.vib_logvar_clip_max}
        return base_kwargs

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_config(self):
        print(f"{'='*50}")
        print(f"  Config")
        print(f"{'='*50}")
        print(f"  clip_model          : {self.clip_model}")
        print(f"  clip_backend        : {self.clip_backend}")
        print(f"  clip_pretrained     : {self.clip_pretrained}")
        print(f"  mapper_type         : {self.mapper_type}")
        print(f"  hidden_dim          : {self.hidden_dim}")
        print(f"  num_tokens          : {self.num_tokens}")
        print(f"  num_heads           : {self.num_heads}")
        print(f"  num_layers          : {self.num_layers}")
        print(f"  dropout             : {self.dropout}")
        print(f"  merge_tokens        : {self.merge_tokens}")
        print(f"  merge_type          : {self.merge_type}")
        print(f"  ---")
        print(f"  train_shards        : {self.train_shards}  (0000 → {self.train_shards - 1:04d})")
        print(f"  val_shard           : {self.val_shard:04d}")
        print(f"  approx_train_samples: {self.approx_train_samples:,}")
        print(f"  train_data_dir      : {self.train_data_dir}")
        print(f"  eval_data_dir       : {self.eval_data_dir}")
        print(f"  batch_size          : {self.batch_size}")
        print(f"  num_workers         : {self.num_workers}")
        print(f"  shuffle_buf         : {self.shuffle_buf}")
        print(f"  ---")
        print(f"  epochs              : {self.epochs}")
        print(f"  lr                  : {self.lr}")
        print(f"  min_lr              : {self.min_lr}")
        print(f"  weight_decay        : {self.weight_decay}")
        print(f"  warmup_steps        : {self.warmup_steps}")
        print(f"  grad_clip           : {self.grad_clip}")
        print(f"  amp                 : {self.amp}")
        print(f"  use_captions        : {self.use_captions}")
        print(f"  steps_per_epoch     : {self.steps_per_epoch}")
        print(f"  total_steps         : {self.total_steps}")
        print(f"  ---")
        print(f"  ls_lr_scale         : {self.ls_lr_scale}")
        print(f"  ls_max              : {self.ls_max}")
        print(f"  margin              : {self.margin}")
        print(f"  margin_weight       : {self.margin_weight}")
        print(f"  margin_delay        : {self.margin_delay}")
        print(f"  noise_sigma         : {self.noise_sigma}")
        print(f"  pseudo_norm         : {self.pseudo_norm}")
        print(f"  vib_beta            : {self.vib_beta}")
        print(f"  vib_beta_initial    : {self.vib_beta_initial}")
        print(f"  vib_beta_warmup     : {self.vib_beta_warmup}")
        print(f"  vib_beta_delay      : {self.vib_beta_delay}")
        print(f"  vib_delay           : {self.vib_delay}")
        print(f"  vib_logvar_bias     : {self.vib_logvar_bias}")
        print(f"  vib_kl_mu_weight    : {self.vib_kl_mu_weight}")
        print(f"  vib_kl_var_weight   : {self.vib_kl_var_weight}")
        print(f"  vib_logvar_clip_max : {self.vib_logvar_clip_max}")
        print(f"  ---")
        print(f"  init_from           : {self.init_from}")
        print(f"  ---")
        print(f"  output_dir          : {self.output_dir}")
        print(f"  save_every_steps    : {self.save_every_steps}")
        print(f"  resume              : {self.resume}")
        print(f"  device              : {self.device}")
        print(f"  seed                : {self.seed}")
        print(f"  ---")
        print(f"  eval_every_steps    : {self.eval_every_steps}")
        print(f"  eval_epoch_end      : {self.eval_epoch_end}")
        print(f"  run_analysis        : {self.run_analysis}")
        print(f"{'='*50}")


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def get_config(**overrides) -> Config:
    """
    Build a Config from defaults + keyword overrides.

    Example:
        cfg = get_config(batch_size=1024, epochs=5, device="cuda")
    """
    cfg = Config()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown config key: '{k}'")
        setattr(cfg, k, v)
    return cfg


def load_from_yaml(yaml_path: str, **overrides) -> Config:
    """
    Build a Config from a YAML file + optional keyword overrides.
    Overrides take priority over YAML values.

    Example:
        cfg = load_from_yaml("config/config.yaml", batch_size=1024)
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config YAML not found: {yaml_path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    # YAML betas comes in as a list → convert to tuple
    if "betas" in data and isinstance(data["betas"], list):
        data["betas"] = tuple(data["betas"])

    # CLI overrides win over YAML
    data.update(overrides)

    return get_config(**data)
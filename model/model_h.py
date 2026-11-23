"""
model/model_h.py

ViT-H / open_clip variant of model.py.

The ONLY differences vs model.py are inside CLIPWrapper (it can now load an
OpenAI `clip` backbone OR an `open_clip` backbone) and two lines of
Pic2Word.__init__ (pass the backend through, and tokenize the prompt with the
wrapper's tokenizer). Everything else - all mappers, mergers, the entire
Pic2Word forward / loss / KL / inference path - is byte-identical to model.py.

With backend="openai" (the default) this file is behaviourally identical to
model.py: attn_mask is None, so the text transformer is called exactly as
before, the OpenAI tokenizer is used, and `open_clip` is never imported.
"""
import torch
from torch import nn
import torch.nn.functional as F
import clip
import math
from typing import List, Literal, Optional, Tuple

from config.config import Config


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads."
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        self.scale      = math.sqrt(self.head_dim)
        self.W_q = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim)
        self.W_k = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim)
        self.W_v = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim)
        self.W_o = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self._init_wts()

    def _init_wts(self):
        for w in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(w.weight)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, T, _ = q.shape
        S       = k.shape[1]
        H, d    = self.num_heads, self.head_dim
        Q = self.W_q(q).reshape(B, T, H, d).transpose(1, 2).contiguous()
        K = self.W_k(k).reshape(B, S, H, d).transpose(1, 2).contiguous()
        V = self.W_v(v).reshape(B, S, H, d).transpose(1, 2).contiguous()
        attn_out = self.attn_dropout(
            F.softmax(Q @ K.transpose(-2, -1) / self.scale, dim=-1)
        ) @ V
        return self.W_o(attn_out.transpose(1, 2).contiguous().reshape(B, T, H * d))


# ---------------------------------------------------------------------------
# CLIP wrapper
# ---------------------------------------------------------------------------

class CLIPWrapper(nn.Module):
    _ALTERNATING_LAYERS = list(range(1, 24, 2))  # ViT-L specific; only used by the
                                                 # multilayer mappers (not VIB).

    def __init__(self, model_name: str = "ViT-L/14", device: str = "cuda",
                 backend: str = "openai", pretrained: Optional[str] = None):
        super().__init__()
        self.device  = device
        self.backend = backend

        if backend == "openai":
            clip_model, self.preprocess = clip.load(model_name, device=self.device, jit=False)
            clip_model = clip_model.float()
            self._tokenize_fn = clip.tokenize
            self.attn_mask    = None                      # causal mask is baked into the
                                                          # OpenAI text transformer
        elif backend == "open_clip":
            import open_clip
            clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device)
            clip_model = clip_model.float()
            self._tokenize_fn = open_clip.get_tokenizer(model_name)
            # open_clip does NOT bake the mask in - it must be passed at call time.
            self.attn_mask    = clip_model.attn_mask.to(self.device)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'openai' or 'open_clip'.")

        self.visual_dim       = clip_model.visual.output_dim
        self.token_dim        = clip_model.token_embedding.embedding_dim
        try:
            self.internal_dim = clip_model.visual.transformer.width
        except AttributeError:
            self.internal_dim = self.visual_dim           # only multilayer mappers read this
        self.visual_encoder   = clip_model.visual
        self.text_transformer = clip_model.transformer
        if backend == "open_clip":
            # ViT-H text activations are kept for backprop into the mapper and
            # overflow 48GB at batch 1024. Checkpointing recomputes them in the
            # backward pass instead of storing them: identical math, a bit slower,
            # and a no-op under eval / torch.no_grad.
            self.text_transformer.grad_checkpointing = True
        self.token_embedding  = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final         = clip_model.ln_final
        self.text_projection  = clip_model.text_projection
        self.logit_scale      = clip_model.logit_scale

        self._freeze_clip()
        self._intermediate_cls: List[torch.Tensor] = []
        self._hooks: List = []

    def _freeze_clip(self):
        for p in self.parameters():
            p.requires_grad = False
        self.logit_scale.requires_grad = True  # trainable by default

    def _register_hooks(self):
        resblocks = self.visual_encoder.transformer.resblocks
        for li in self._ALTERNATING_LAYERS:
            hook = resblocks[li].register_forward_hook(self._make_hook())
            self._hooks.append(hook)

    def _make_hook(self):
        def hook_fn(module, inp, out):
            self._intermediate_cls.append(out[0, :, :]) # CLS token.
        return hook_fn

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Text transformer call (backend-agnostic, batch-first [B, seq, dim] I/O).
    # ------------------------------------------------------------------
    def _text_forward(self, x_nld: torch.Tensor) -> torch.Tensor:
        # Input/output are batch-first [batch, seq, dim] for BOTH backends.
        #   open_clip : its text transformer is batch-first and takes the causal
        #               mask at call time, so feed [batch, seq, dim] straight through.
        #   openai    : its text transformer expects [seq, batch, dim] with the
        #               mask baked in, so permute in, run, permute back.
        # NOT @torch.no_grad: on the training text path, grad must flow to the mapper.
        if self.attn_mask is not None:                       # open_clip (batch-first)
            L = x_nld.shape[1]
            return self.text_transformer(x_nld, attn_mask=self.attn_mask[:L, :L])
        x_lnd = x_nld.permute(1, 0, 2)                        # openai: NLD -> LND
        x_lnd = self.text_transformer(x_lnd)
        return x_lnd.permute(1, 0, 2)                         # LND -> NLD

    def tokenize(self, texts):
        return self._tokenize_fn(texts)

    @torch.no_grad()
    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.visual_encoder(img.to(self.device)), dim=-1)

    @torch.no_grad()
    def encoder_image_unnorm(self, img: torch.Tensor) -> torch.Tensor:
        return self.visual_encoder(img.to(self.device))

    @torch.no_grad()
    def encode_image_multilayer(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._intermediate_cls = []
        self._register_hooks()
        vis_raw = self.visual_encoder(img.to(self.device))
        res     = torch.stack(self._intermediate_cls, dim=1)
        self._remove_hooks()
        self._intermediate_cls = []
        return vis_raw, res

    @torch.no_grad()
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        x   = self.token_embedding(text_tokens) + self.positional_embedding
        x   = self._text_forward(x)
        x   = self.ln_final(x)
        eot = text_tokens.argmax(dim=-1)
        return F.normalize(x[torch.arange(x.shape[0]), eot] @ self.text_projection, dim=-1)

    def forward_text_transform(self, token_embeddings: torch.Tensor, eot_pos: torch.Tensor) -> torch.Tensor:
        x = self._text_forward(token_embeddings)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0], device=token_embeddings.device), eot_pos]
        return F.normalize(x @ self.text_projection, dim=-1)

    def get_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.exp()


# ---------------------------------------------------------------------------
# Mappers
# ---------------------------------------------------------------------------

class BaseMLP(nn.Module):
    def __init__(self, visual_dim=768, token_dim=768, hidden_dim=512, dropout=0.1, num_tokens=1):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim  = token_dim
        self.net = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim * num_tokens),
        )

    def forward(self, v_raw: torch.Tensor) -> torch.Tensor:
        B, _ = v_raw.shape
        return self.net(v_raw).view(B, self.num_tokens, self.token_dim)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)


class VIBMapper(nn.Module):
    """
    Variational Information Bottleneck mapper.

    Replaces deterministic BaseMLP with stochastic encoder.
    Shared encoder (vis_dim -> hidden -> hidden) then splits into:
      fc_mu     (hidden -> token_dim * num_tokens)  - mean of q(z|v)
      fc_logvar (hidden -> token_dim * num_tokens)  - log variance of q(z|v)

    Training:  z = mu + sigma*eps,  eps ~ N(0,I)   -> returns (z, mu, logvar)
    Inference: z = mu  (deterministic)             -> returns mu (plain tensor)

    KL divergence (computed in Pic2Word.forward):
      KL(q(z|v) || N(0,I)) = 0.5 * sum_d (mu_d^2 + sigma_d^2 - log sigma_d^2 - 1)

    The mu^2 term acts as a norm penalty (prevents norm inflation).
    The sigma^2 terms control per-dimension precision (learned noise).
    Together they bound total mutual information I(v_raw; z).

    Param count vs BaseMLP (ViT-L, dims 768; ViT-H scales these up to 1024):
      Shared encoder: Linear(768,256) + Linear(256,256) = 262K  (same)
      fc_mu:          Linear(256,768)                    = 197K  (replaces BaseMLP's 3rd layer)
      fc_logvar:      Linear(256,768)                    = 197K  (NEW)
      Total:          656K  (vs BaseMLP's 460K, +196K)
    """

    def __init__(self, visual_dim=768, token_dim=768, hidden_dim=256, dropout=0.1,
                 num_tokens=1, logvar_bias: float = -2.0, logvar_clip_max: float = 2.0):
        super().__init__()
        self.num_tokens  = num_tokens
        self.token_dim   = token_dim
        self.logvar_bias = logvar_bias
        self.logvar_clip_max = logvar_clip_max

        out_dim = token_dim * num_tokens

        # Shared encoder - same architecture as BaseMLP's first two layers
        self.encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Split heads for mean and log-variance
        self.fc_mu     = nn.Linear(hidden_dim, out_dim)
        self.fc_logvar = nn.Linear(hidden_dim, out_dim)

        # Initialize fc_logvar:
        #   weight = 0    -> initial sigma is uniform across input features
        #   bias   = `logvar_bias` -> initial sigma = exp(0.5 * logvar_bias)
        # Default -2.0 -> sigma ~ 0.37 (original behaviour)
        # Warm-start  -6.0 -> sigma ~ 0.05 (gentle, recommended for warm-start runs)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.constant_(self.fc_logvar.bias, logvar_bias)

    def forward(self, v_raw: torch.Tensor, deterministic : bool = False):
        """
        Training:  returns (z_sampled, mu, logvar) - tuple of 3
        Eval:      returns mu - plain tensor (same interface as BaseMLP)
        """
        B = v_raw.shape[0]
        h = self.encoder(v_raw)
        mu = self.fc_mu(h).view(B, self.num_tokens, self.token_dim)

        if self.training and not deterministic:
            logvar = self.fc_logvar(h).view(B, self.num_tokens, self.token_dim)
            # Clamp logvar to prevent numerical instability
            logvar = logvar.clamp(-10.0, self.logvar_clip_max)
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            z = mu + std * eps        # reparameterization trick
            return z, mu, logvar
        else:
            return mu                 # deterministic at inference

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WeightedMLPMapper(nn.Module):
    NUM_LAYERS = 12

    def __init__(self, internal_dim=1024, token_dim=768, hidden_dim=256, dropout=0.1, num_tokens=1):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim  = token_dim
        self.layer_wts  = nn.Parameter(torch.zeros(self.NUM_LAYERS))
        self.net = nn.Sequential(
            nn.Linear(internal_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim,   hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim,   token_dim * num_tokens),
        )

    def forward(self, intermediate_cls: torch.Tensor) -> torch.Tensor:
        B      = intermediate_cls.shape[0]
        wts    = F.softmax(self.layer_wts, dim=0)
        pooled = (wts.unsqueeze(0).unsqueeze(-1) * intermediate_cls).sum(dim=1)
        return self.net(pooled).view(B, self.num_tokens, self.token_dim)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FusedWeightedMLPMapper(nn.Module):
    NUM_LAYERS = 12

    def __init__(self, internal_dim=1024, visual_dim=768, token_dim=768, hidden_dim=256, dropout=0.1, num_tokens=1):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim  = token_dim
        self.layer_wts  = nn.Parameter(torch.zeros(self.NUM_LAYERS))
        self.net = nn.Sequential(
            nn.Linear(internal_dim + visual_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim,                hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim,                token_dim * num_tokens),
        )

    def forward(self, v_raw: torch.Tensor, intermediate_cls: torch.Tensor) -> torch.Tensor:
        B      = v_raw.shape[0]
        wts    = F.softmax(self.layer_wts, dim=0)
        pooled = (wts.unsqueeze(0).unsqueeze(-1) * intermediate_cls).sum(dim=1)
        return self.net(torch.cat([pooled, v_raw], dim=-1)).reshape(B, self.num_tokens, self.token_dim)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiLayerCLSMapper(nn.Module):
    def __init__(self, internal_dim=1024, token_dim=768, hidden_dim=256,
                 num_tokens=1, num_heads=8, num_layers=1, dropout=0.1):
        super().__init__()
        self.num_tokens   = num_tokens
        self.token_dim    = token_dim
        self.input_proj   = nn.Sequential(nn.Linear(internal_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.query_tokens = nn.Parameter(torch.randn(num_tokens, hidden_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        self.cross_attn_layers = nn.ModuleList([
            nn.ModuleDict({
                "attn"   : MultiHeadCrossAttention(hidden_dim, num_heads, dropout),
                "ln_q"   : nn.LayerNorm(hidden_dim),
                "ln_kv"  : nn.LayerNorm(hidden_dim),
                "ffn"    : nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout),
                ),
                "ffn_ln" : nn.LayerNorm(hidden_dim),
            }) for _ in range(num_layers)
        ])
        self.output_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, token_dim))

    def forward(self, intermediate_cls: torch.Tensor) -> torch.Tensor:
        B  = intermediate_cls.shape[0]
        kv = self.input_proj(intermediate_cls)
        q  = self.query_tokens.unsqueeze(0).expand(B, -1, -1).clone()
        for layer in self.cross_attn_layers:
            q = q + layer["attn"](layer["ln_q"](q), layer["ln_kv"](kv), layer["ln_kv"](kv))
            q = q + layer["ffn"](layer["ffn_ln"](q))
        return self.output_proj(q)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Token mergers
# ---------------------------------------------------------------------------

class TokenMerger(nn.Module):
    def __init__(self, num_tokens: int, merge_tokens: int, token_dim: int):
        super().__init__()
        self.merge_tokens = merge_tokens
        self.token_dim    = token_dim
        self.ln   = nn.LayerNorm(num_tokens * token_dim)
        self.proj = nn.Linear(num_tokens * token_dim, merge_tokens * token_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        return self.proj(F.gelu(self.ln(tokens.reshape(B, -1)))).reshape(B, self.merge_tokens, self.token_dim)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WeightedMerger(nn.Module):
    def __init__(self, num_tokens: int, merge_tokens: int, token_dim: int):
        super().__init__()
        self.merge_tokens = merge_tokens
        self.token_dim    = token_dim
        self.wts = nn.Parameter(torch.zeros(merge_tokens, num_tokens))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.einsum('mt,btd->bmd', F.softmax(self.wts, dim=-1), tokens)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Pic2Word
# ---------------------------------------------------------------------------

class Pic2Word(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.device           = cfg.device
        self.max_token_len    = cfg.max_token_len
        self.repulsion_weight = cfg.repulsion_weight
        self.decorr_weight    = cfg.decorr_weight
        self.noise_sigma      = cfg.noise_sigma
        self.pseudo_norm      = cfg.pseudo_norm

        self.clip = CLIPWrapper(model_name=cfg.clip_model, device=cfg.device,
                                backend=getattr(cfg, "clip_backend", "openai"),
                                pretrained=getattr(cfg, "clip_pretrained", None))

        # --- Mapper ---
        if cfg.mapper_type == "base":
            self.mapper         = BaseMLP(visual_dim=self.clip.visual_dim, token_dim=self.clip.token_dim, **cfg.mapper_kwargs)
            self.use_multilayer = False
        elif cfg.mapper_type == "vib":
            self.mapper         = VIBMapper(visual_dim=self.clip.visual_dim, token_dim=self.clip.token_dim, **cfg.mapper_kwargs)
            self.use_multilayer = False
        elif cfg.mapper_type == "multilayer":
            self.mapper         = MultiLayerCLSMapper(internal_dim=self.clip.internal_dim, token_dim=self.clip.token_dim, **cfg.mapper_kwargs)
            self.use_multilayer = True
        elif cfg.mapper_type == "weighted":
            self.mapper         = WeightedMLPMapper(internal_dim=self.clip.internal_dim, token_dim=self.clip.token_dim, **cfg.mapper_kwargs)
            self.use_multilayer = True
        elif cfg.mapper_type == "fused_weighted":
            self.mapper         = FusedWeightedMLPMapper(
                internal_dim=self.clip.internal_dim, visual_dim=self.clip.visual_dim,
                token_dim=self.clip.token_dim, **cfg.mapper_kwargs)
            self.use_multilayer = True
        else:
            raise ValueError(f"Unknown mapper_type '{cfg.mapper_type}'.")

        # --- Merger ---
        self.merger = None
        if cfg.merge_tokens is not None and cfg.merge_tokens < self.mapper.num_tokens:
            MergerCls   = WeightedMerger if cfg.merge_type == "weighted" else TokenMerger
            self.merger = MergerCls(
                num_tokens   = self.mapper.num_tokens,
                merge_tokens = cfg.merge_tokens,
                token_dim    = self.clip.token_dim,
            )

        # --- Prompt buffer ---
        prompt_tokens = self.clip.tokenize([cfg.clip_prompt])[0]
        self.register_buffer("prompt_tokens", prompt_tokens)
        self._prompt_eot_idx = int(prompt_tokens.argmax())

    # ------------------------------------------------------------------
    # Token sequence builder
    # ------------------------------------------------------------------

    def _build_token_sequence(self,
                               pseudo_tokens: torch.Tensor,
                               text_tokens:   Optional[torch.Tensor] = None
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D   = pseudo_tokens.shape
        MAX_LEN   = self.max_token_len
        prompt_ids = self.prompt_tokens.unsqueeze(0).expand(B, -1).to(self.device)
        full_emb   = self.clip.token_embedding(prompt_ids)
        eot_idx    = self._prompt_eot_idx
        pref       = full_emb[:, :eot_idx, :]

        if text_tokens is None:
            eos_emb = full_emb[:, eot_idx : eot_idx + 1, :]
            seq     = torch.cat([pref, pseudo_tokens, eos_emb], dim=1)
            new_eot = torch.full((B,), eot_idx + T, dtype=torch.int64, device=self.device)
            pad_len = MAX_LEN - seq.shape[1]
            if pad_len > 0:
                seq = torch.cat([seq, torch.zeros(B, pad_len, D, dtype=seq.dtype, device=self.device)], dim=1)
        else:
            txt_ids = text_tokens.to(self.device)
            txt_eot = txt_ids.argmax(dim=-1)
            txt_emb = self.clip.token_embedding(txt_ids)
            seqs, eots = [], []
            for b in range(B):
                t_len   = int(txt_eot[b].item())
                eos_emb = txt_emb[b, t_len : t_len + 1, :]
                seq_b   = torch.cat(
                    [pref[b], pseudo_tokens[b], txt_emb[b, 1:t_len, :], eos_emb], dim=0
                )[:MAX_LEN]
                eot_b   = seq_b.shape[0] - 1
                pad_len = MAX_LEN - seq_b.shape[0]
                if pad_len > 0:
                    seq_b = torch.cat(
                        [seq_b, torch.zeros(pad_len, D, dtype=seq_b.dtype, device=self.device)], dim=0
                    )
                seqs.append(seq_b)
                eots.append(eot_b)
            seq     = torch.stack(seqs, dim=0)
            new_eot = torch.tensor(eots, dtype=torch.int64, device=seq.device)

        return seq + self.clip.positional_embedding[:seq.shape[1]].unsqueeze(0), new_eot

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self,
                imgs          : torch.Tensor,
                captions                    = None,
                margin        : float       = 0.0,
                margin_weight : float       = 0.0,
                vib_beta      : float       = 0.0,
                vib_delay     : bool        = False,
                vib_kl_mu_weight  : float   = 1.0,
                vib_kl_var_weight : float   = 1.0,
                v_raw                       = None) -> dict:
        # v_raw: optional precomputed UNNORMALIZED frozen-CLIP image embedding
        # [B, visual_dim]. When supplied (embedding-cache path), the image tower
        # is skipped entirely and this v_raw feeds the mapper directly, exactly
        # as the live encode would. The mapper, normalization, text path, loss
        # and KL below are identical either way, so the recipe is unchanged --
        # only the source of v_raw differs. Cache path requires the non-multilayer
        # mappers (base / vib); the cache stores only the final v_raw.

        # --- Visual encoding + mapper ---
        mu, logvar = None, None

        if v_raw is not None:
            # Embedding-cache path: use the supplied frozen v_raw, no image tower.
            v_raw = v_raw.to(self.device)
            if isinstance(self.mapper, VIBMapper):
                if vib_delay:
                    pseudo_tokens = self.mapper(v_raw, deterministic = True)
                    mu, logvar = None, None
                else:
                    pseudo_tokens, mu, logvar = self.mapper(v_raw)
            else:
                pseudo_tokens = self.mapper(v_raw)
        elif self.use_multilayer:
            imgs = imgs.to(self.device)
            v_raw, intermediate = self.clip.encode_image_multilayer(imgs)
            pseudo_tokens = (self.mapper(v_raw.detach(), intermediate)
                             if isinstance(self.mapper, FusedWeightedMLPMapper)
                             else self.mapper(intermediate))
        else:
            imgs = imgs.to(self.device)
            v_raw = self.clip.encoder_image_unnorm(imgs)
            # VIBMapper returns (z, mu, logvar) during training
            if isinstance(self.mapper, VIBMapper):
                if vib_delay:
                    pseudo_tokens = self.mapper(v_raw, deterministic = True)
                    mu, logvar = None, None
                else:
                    pseudo_tokens, mu, logvar = self.mapper(v_raw)
            else:
                pseudo_tokens = self.mapper(v_raw)

        if self.merger is not None:
            pseudo_tokens = self.merger(pseudo_tokens)

        if self.pseudo_norm > 0:
            pseudo_tokens = F.normalize(pseudo_tokens, dim=-1) * self.pseudo_norm

        if self.training and self.noise_sigma > 0:
            pseudo_tokens = pseudo_tokens + self.noise_sigma * torch.randn_like(pseudo_tokens)

        # --- Text encoding + contrastive loss ---
        v = F.normalize(v_raw.detach(), dim=-1)
        tok_emb, eot_pos = self._build_token_sequence(pseudo_tokens, captions)
        p           = self.clip.forward_text_transform(tok_emb, eot_pos)
        logit_scale = self.clip.get_logit_scale()

        loss, l_t2i, l_i2t = self._contrastive_loss(
            p, v, logit_scale, margin=margin, margin_weight=margin_weight)
        loss = loss + self._aux_loss(p, v, pseudo_tokens, self.repulsion_weight, self.decorr_weight)

        # --- VIB KL divergence ---
        # KL(q(z|v) || N(0,I)) = 0.5 * sum_d (mu_d^2 + exp(logvar_d) - logvar_d - 1)
        # mu^2 term = norm penalty (prevents norm inflation)
        # sigma^2 terms = per-dimension precision control


        kl_loss = torch.tensor(0.0, device=self.device)
        kl_mu   = torch.tensor(0.0, device=self.device)
        kl_var  = torch.tensor(0.0, device=self.device)
        kl_mu_weighted  = torch.tensor(0.0, device=self.device)
        kl_var_weighted = torch.tensor(0.0, device=self.device)

        if mu is not None and logvar is not None:
            kl_mu_per_sample  = 0.5 * mu.pow(2).sum(dim=(1, 2))                          # (B,)
            kl_var_per_sample = 0.5 * (logvar.exp() - logvar - 1).sum(dim=(1, 2))        # (B,)
            kl_mu   = kl_mu_per_sample.mean()
            kl_var  = kl_var_per_sample.mean()
            kl_mu_weighted  = vib_kl_mu_weight  * kl_mu                        # weighted (what kl_loss includes)
            kl_var_weighted = vib_kl_var_weight * kl_var

            kl_loss = vib_kl_mu_weight * kl_mu + vib_kl_var_weight * kl_var
            if vib_beta > 0:
                loss = loss + vib_beta * kl_loss

        out = {
            "loss"        : loss,
            "loss_t2i"    : l_t2i,
            "loss_i2t"    : l_i2t,
            "logit_scale" : logit_scale,
            "kl_loss"     : kl_loss,
            "kl_mu"       : kl_mu,
            "kl_var"      : kl_var,
            "kl_mu_weighted"  : kl_mu_weighted,     # weighted contribution to kl_loss
            "kl_var_weighted" : kl_var_weighted    # weighted contribution to kl_loss
        }

        # --- VIB stats (only when sampling is active) ---
        # mu/logvar are None when:
        #   - mapper is not VIBMapper (BaseMLP etc.)
        #   - VIBMapper is in deterministic mode (vib_delay=True or eval)
        # In both cases skip - no stats to log.
        if mu is not None and logvar is not None:
            with torch.no_grad():
                out["mu_norm"]      = mu.float().norm(dim=-1).mean().detach()
                out["logvar_mean"]  = logvar.float().mean().detach()
                out["std_mean"]     = (0.5 * logvar.float()).exp().mean().detach()

        return out

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    @staticmethod
    def _contrastive_loss(p, v, logit_scale,
                          margin: float = 0.0,
                          margin_weight: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = logit_scale * (p @ v.T)
        labels = torch.arange(p.shape[0], dtype=torch.int64, device=p.device)
        l_t2i  = F.cross_entropy(logits,   labels)
        l_i2t  = F.cross_entropy(logits.T, labels)
        loss   = (l_t2i + l_i2t) / 2.0

        if margin_weight > 0.0:
            sims     = p @ v.T
            pos      = sims.diagonal()
            hard_t2i = sims.clone().fill_diagonal_(-float('inf')).max(dim=1).values
            hard_i2t = sims.T.clone().fill_diagonal_(-float('inf')).max(dim=1).values
            loss_margin = (F.softplus(margin - pos + hard_t2i).mean() +
                           F.softplus(margin - pos + hard_i2t).mean()) / 2.0
            loss = loss + margin_weight * loss_margin

        return loss, l_t2i, l_i2t

    @staticmethod
    def _aux_loss(p, v, pseudo_tokens, repulsion_wt, decor_wt) -> torch.Tensor:
        aux = 0.0
        if repulsion_wt > 0.0:
            aux = aux + repulsion_wt * F.cosine_similarity(p.float(), v.float(), dim=-1).mean()
        if decor_wt > 0.0 and pseudo_tokens.shape[1] > 1:
            B, T, D   = pseudo_tokens.shape
            toks_norm = F.normalize(pseudo_tokens, dim=-1)
            gram      = torch.bmm(toks_norm, toks_norm.transpose(1, 2))
            mask      = ~torch.eye(T, dtype=torch.bool, device=gram.device)
            aux       = aux + decor_wt * gram[:, mask].pow(2).mean()
        return aux

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_query(self, imgs: torch.Tensor,
                     text_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        imgs = imgs.to(self.device)
        if self.use_multilayer:
            v_raw, intermediate = self.clip.encode_image_multilayer(imgs)
            pseudo_tokens = (self.mapper(v_raw, intermediate)
                             if isinstance(self.mapper, FusedWeightedMLPMapper)
                             else self.mapper(intermediate))
        else:
            v_raw         = self.clip.encoder_image_unnorm(imgs)
            # VIBMapper in eval mode returns mu directly (plain tensor)
            pseudo_tokens = self.mapper(v_raw)
        if self.merger is not None:
            pseudo_tokens = self.merger(pseudo_tokens)
        if self.pseudo_norm > 0:
            pseudo_tokens = F.normalize(pseudo_tokens, dim=-1) * self.pseudo_norm
        tok_emb, eot_pos = self._build_token_sequence(pseudo_tokens, text_tokens)
        return self.clip.forward_text_transform(tok_emb, eot_pos)

    @torch.no_grad()
    def encode_candidates(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_image(imgs.to(self.device))

    # ------------------------------------------------------------------
    # Optimizer helpers
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> list:
        params = list(self.mapper.parameters())
        if self.clip.logit_scale.requires_grad:
            params += [self.clip.logit_scale]
        if self.merger is not None:
            params += list(self.merger.parameters())
        return params

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def print_info(self):
        clip_p   = sum(p.numel() for p in self.clip.parameters())
        mapper_p = self.mapper.num_parameters()
        print(f"{'='*55}")
        print(f"Pic2Word")
        print(f"{'='*55}")
        print(f"CLIP          : {clip_p/1e6:.1f}M params  [FROZEN]")
        print(f"Mapper        : {mapper_p/1e6:.3f}M params  [TRAINABLE]")
        print(f"Mapper type   : {type(self.mapper).__name__}")
        print(f"Num tokens    : {self.mapper.num_tokens}")
        print(f"Visual dim    : {self.clip.visual_dim}")
        print(f"Token dim     : {self.clip.token_dim}")
        print(f"Internal dim  : {self.clip.internal_dim}")
        print(f"max_token_len : {self.max_token_len}")
        print(f"logit_scale   : {'TRAINABLE' if self.clip.logit_scale.requires_grad else 'FROZEN'}")
        print(f"repulsion_wt  : {self.repulsion_weight}")
        print(f"decorr_wt     : {self.decorr_weight}")
        if self.merger is not None:
            print(f"Merger        : {self.merger.num_parameters()/1e6:.3f}M params  [TRAINABLE]")
            print(f"Merger type   : {type(self.merger).__name__}")
            print(f"Merge tokens  : {self.merger.merge_tokens}")
        print(f"noise_sigma   : {self.noise_sigma}")
        print(f"pseudo_norm   : {self.pseudo_norm}")
        if isinstance(self.mapper, VIBMapper):
            init_std = math.exp(0.5 * self.mapper.logvar_bias)
            max_std  = math.exp(0.5 * self.mapper.logvar_clip_max)
            print(f"VIB logvar_bias : {self.mapper.logvar_bias}  (initial sigma ~ {init_std:.4f})")
            print(f"VIB logvar_clip_max : {self.mapper.logvar_clip_max}  (max sigma ~ {max_std:.4f})")
        print(f"{'='*55}")
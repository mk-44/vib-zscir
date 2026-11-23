"""
model/model_test.py
"""

import sys
import math
import copy
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import clip

from config.config import Config
from model.model import (
    Pic2Word,
    CLIPWrapper,
    BaseMLP,
    WeightedMLPMapper,
    FusedWeightedMLPMapper,
    MultiLayerCLSMapper,
    TokenMerger,
    WeightedMerger,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEVICE         = "cpu"
CLIP_MODEL     = "ViT-B/32"
BATCH          = 4
IMG_SIZE       = 224
VIS_DIM        = 512
TOKEN_DIM      = 512
PROMPT_EOT_IDX = 4   # "a photo of" → [BOS, a, photo, of, EOT] → EOT at 4

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

PASS  = 0
FAIL  = 0
TESTS = []

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

# ---------------------------------------------------------------------------
# Shared fixtures (built once)
# ---------------------------------------------------------------------------

def make_cfg(num_tokens=1, margin=0.0, margin_weight=0.0):
    return Config(
        clip_model    = CLIP_MODEL,
        mapper_type   = "base",
        hidden_dim    = 256,
        num_tokens    = num_tokens,
        dropout       = 0.0,
        device        = DEVICE,
        max_token_len = 77,
        clip_prompt   = "a photo of",
        margin        = margin,
        margin_weight = margin_weight,
    )

def make_model(num_tokens=1):
    m = Pic2Word(make_cfg(num_tokens)).to(DEVICE)
    m.eval()
    return m

def get_pseudo(model, imgs):
    with torch.no_grad():
        v_raw = model.clip.encoder_image_unnorm(imgs)
        return model.mapper(v_raw)

torch.manual_seed(0)
IMGS  = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
IMGS2 = torch.randn(2,     3, IMG_SIZE, IMG_SIZE)

print("\nLoading CLIP (ViT-B/32)...")
M1 = make_model(num_tokens=1)
M2 = make_model(num_tokens=2)
print("Done.\n")

# ===========================================================================
# 1. Config integration
# ===========================================================================

def test_max_token_len_stored():
    assert M1.max_token_len == 77

def test_prompt_eot_idx_correct():
    assert M1._prompt_eot_idx == PROMPT_EOT_IDX, \
        f"Expected {PROMPT_EOT_IDX}, got {M1._prompt_eot_idx}"

def test_clip_prompt_tokenized():
    expected = clip.tokenize(["a photo of"])[0]
    assert torch.equal(M1.prompt_tokens.cpu(), expected)

def test_device_stored():
    assert M1.device == DEVICE

# ===========================================================================
# 2. CLIPWrapper
# ===========================================================================

def test_visual_dim():
    assert M1.clip.visual_dim == VIS_DIM

def test_token_dim():
    assert M1.clip.token_dim == TOKEN_DIM

def test_encode_image_normalised():
    with torch.no_grad():
        feats = M1.clip.encode_image(IMGS)
    norms = feats.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5), \
        f"Not unit-normalised: {norms}"

def test_encoder_image_unnorm_shape():
    with torch.no_grad():
        feats = M1.clip.encoder_image_unnorm(IMGS)
    assert feats.shape == (BATCH, VIS_DIM)

def test_logit_scale_trainable():
    assert M1.clip.logit_scale.requires_grad

def test_visual_encoder_frozen():
    for name, p in M1.clip.visual_encoder.named_parameters():
        assert not p.requires_grad, f"Visual encoder '{name}' should be frozen"

def test_text_transformer_frozen():
    for name, p in M1.clip.text_transformer.named_parameters():
        assert not p.requires_grad, f"Text transformer '{name}' should be frozen"

def test_logit_scale_positive():
    assert M1.clip.get_logit_scale().item() > 0

# ===========================================================================
# 3. Mapper shapes
# ===========================================================================

def test_base_mlp_shape_tok1():
    pseudo = get_pseudo(M1, IMGS)
    assert pseudo.shape == (BATCH, 1, TOKEN_DIM)

def test_base_mlp_shape_tok2():
    pseudo = get_pseudo(M2, IMGS)
    assert pseudo.shape == (BATCH, 2, TOKEN_DIM)

def test_weighted_mlp_mapper_shape():
    mapper = WeightedMLPMapper(internal_dim=512, token_dim=512, hidden_dim=256, num_tokens=1)
    x      = torch.randn(BATCH, 12, 512)
    out    = mapper(x)
    assert out.shape == (BATCH, 1, 512)

def test_weighted_mlp_layer_weights_sum_to_one():
    mapper = WeightedMLPMapper(internal_dim=512, token_dim=512, hidden_dim=256, num_tokens=1)
    wts    = F.softmax(mapper.layer_wts, dim=0)
    assert torch.allclose(wts.sum(), torch.tensor(1.0), atol=1e-6)

def test_fused_weighted_mapper_shape():
    mapper = FusedWeightedMLPMapper(
        internal_dim=512, visual_dim=512, token_dim=512, hidden_dim=256, num_tokens=1)
    out = mapper(torch.randn(BATCH, 512), torch.randn(BATCH, 12, 512))
    assert out.shape == (BATCH, 1, 512)

# ===========================================================================
# 4. Mergers
# ===========================================================================

def test_token_merger_shape():
    merger = TokenMerger(num_tokens=4, merge_tokens=1, token_dim=512)
    out    = merger(torch.randn(BATCH, 4, 512))
    assert out.shape == (BATCH, 1, 512)

def test_weighted_merger_shape():
    merger = WeightedMerger(num_tokens=4, merge_tokens=1, token_dim=512)
    out    = merger(torch.randn(BATCH, 4, 512))
    assert out.shape == (BATCH, 1, 512)

def test_weighted_merger_weights_sum_to_one():
    merger   = WeightedMerger(num_tokens=4, merge_tokens=2, token_dim=512)
    row_sums = F.softmax(merger.wts, dim=-1).sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(2), atol=1e-6)

def test_weighted_merger_param_count():
    merger = WeightedMerger(num_tokens=4, merge_tokens=1, token_dim=512)
    assert merger.num_parameters() == 4  # 1 * 4 scalars only

# ===========================================================================
# 5. Token sequence — no-text branch
# ===========================================================================

def test_no_text_eot_exact_tok1():
    pseudo   = get_pseudo(M1, IMGS)
    _, eot   = M1._build_token_sequence(pseudo, None)
    expected = PROMPT_EOT_IDX + 1  # 5
    assert (eot == expected).all(), f"Expected {expected}, got {eot}"

def test_no_text_eot_exact_tok2():
    pseudo   = get_pseudo(M2, IMGS)
    _, eot   = M2._build_token_sequence(pseudo, None)
    expected = PROMPT_EOT_IDX + 2  # 6
    assert (eot == expected).all(), f"Expected {expected}, got {eot}"

def test_no_text_output_shape():
    pseudo          = get_pseudo(M1, IMGS)
    tok_emb, eot    = M1._build_token_sequence(pseudo, None)
    assert tok_emb.shape == (BATCH, 77, TOKEN_DIM)
    assert eot.shape     == (BATCH,)

def test_no_text_padding_after_eot_is_zero():
    pseudo           = get_pseudo(M1, IMGS)
    tok_emb, eot_pos = M1._build_token_sequence(pseudo, None)
    pos_emb          = M1.clip.positional_embedding[:77].unsqueeze(0)
    seq              = tok_emb - pos_emb
    for b in range(BATCH):
        pad = seq[b, eot_pos[b].item() + 1:, :]
        assert torch.allclose(pad, torch.zeros_like(pad), atol=1e-5), \
            f"Sample {b}: non-zero padding after eot_pos={eot_pos[b].item()}"

def test_no_text_eot_embedding_not_zero():
    pseudo           = get_pseudo(M1, IMGS)
    tok_emb, eot_pos = M1._build_token_sequence(pseudo, None)
    pos_emb          = M1.clip.positional_embedding[:77].unsqueeze(0)
    seq              = tok_emb - pos_emb
    for b in range(BATCH):
        eot_embed = seq[b, eot_pos[b].item(), :]
        assert eot_embed.norm().item() > 1e-3, \
            f"Sample {b}: token at eot_pos is near-zero (padding?)"

# ===========================================================================
# 6. Token sequence — text branch  (THE CRITICAL EOT BUG TESTS)
# ===========================================================================

def test_text_eot_not_always_76():
    """
    THE BUG: if eot_b computed after padding → always 76.
    Short modifiers must give eot_pos well below 76.
    """
    modifiers   = ["red", "blue", "big", "small"]
    text_tokens = clip.tokenize(modifiers, truncate=True)
    pseudo      = get_pseudo(M1, IMGS)
    _, eot_pos  = M1._build_token_sequence(pseudo, text_tokens)
    assert not (eot_pos == 76).all(), \
        "eot_pos is always 76 for all samples — EOT padding bug is present!"

def test_text_eot_exact_single_word():
    """
    Modifier 'red' → [BOS, red, EOT, PAD...]
    pref(4) + pseudo(1) + red(1) + EOT(1) = 7 tokens → eot_b = 6.
    Bug gives 76.
    """
    text_tokens = clip.tokenize(["red", "red"], truncate=True)
    pseudo      = get_pseudo(M1, IMGS2)
    _, eot_pos  = M1._build_token_sequence(pseudo, text_tokens)
    assert (eot_pos == 6).all(), \
        f"Expected eot_pos=6 for 'red', got {eot_pos.tolist()}. " \
        f"If 76, EOT padding bug is present."

def test_text_eot_scales_with_length():
    """Longer modifier → higher eot_pos."""
    short_tok = clip.tokenize(["red"],                                          truncate=True)
    long_tok  = clip.tokenize(["completely different style long description"],  truncate=True)
    pseudo    = get_pseudo(M1, IMGS2[:1])
    _, eot_s  = M1._build_token_sequence(pseudo[:1], short_tok)
    _, eot_l  = M1._build_token_sequence(pseudo[:1], long_tok)
    assert eot_l[0].item() > eot_s[0].item(), \
        f"Longer modifier must give higher eot_pos. short={eot_s[0]}, long={eot_l[0]}"

def test_text_eot_within_bounds():
    text_tokens = clip.tokenize(["red", "long blue coat with buttons and details"], truncate=True)
    pseudo      = get_pseudo(M1, IMGS2)
    _, eot_pos  = M1._build_token_sequence(pseudo, text_tokens)
    assert (eot_pos > 0).all()
    assert (eot_pos < M1.max_token_len).all()

def test_text_output_shape():
    text_tokens      = clip.tokenize(["red", "blue", "big", "small"], truncate=True)
    pseudo           = get_pseudo(M1, IMGS)
    tok_emb, eot_pos = M1._build_token_sequence(pseudo, text_tokens)
    assert tok_emb.shape == (BATCH, 77, TOKEN_DIM)
    assert eot_pos.shape == (BATCH,)

def test_text_padding_after_eot_is_zero():
    text_tokens      = clip.tokenize(["red", "blue", "big", "small"], truncate=True)
    pseudo           = get_pseudo(M1, IMGS)
    tok_emb, eot_pos = M1._build_token_sequence(pseudo, text_tokens)
    pos_emb          = M1.clip.positional_embedding[:77].unsqueeze(0)
    seq              = tok_emb - pos_emb
    for b in range(BATCH):
        pad = seq[b, eot_pos[b].item() + 1:, :]
        assert torch.allclose(pad, torch.zeros_like(pad), atol=1e-5), \
            f"Sample {b}: non-zero padding after eot_pos={eot_pos[b].item()}"

def test_text_eot_embedding_matches_clip_eot_token():
    """
    tok_emb[b, eot_b] - pos_emb[eot_b] must equal clip.token_embedding(49407).
    49407 is CLIP's EOT token id.
    """
    text_tokens      = clip.tokenize(["red"], truncate=True)
    pseudo           = get_pseudo(M1, IMGS2[:1])
    tok_emb, eot_pos = M1._build_token_sequence(pseudo[:1], text_tokens)

    eot_id           = 49407
    eot_embed_expected = M1.clip.token_embedding(torch.tensor([eot_id]))[0]

    b       = 0
    pos     = eot_pos[b].item()
    pos_emb = M1.clip.positional_embedding[pos]
    raw_at_eot = tok_emb[b, pos, :] - pos_emb

    assert torch.allclose(raw_at_eot, eot_embed_expected, atol=1e-5), \
        "Token at eot_pos does not match CLIP EOT embedding — wrong position."

def test_text_different_lengths_give_different_eot():
    text_tokens = clip.tokenize(["red", "long blue coat with many buttons"], truncate=True)
    pseudo      = get_pseudo(M1, IMGS2)
    _, eot_pos  = M1._build_token_sequence(pseudo, text_tokens)
    assert eot_pos[0].item() != eot_pos[1].item(), \
        "Short and long modifiers must produce different eot_pos values"

# ===========================================================================
# 7. EOT attention — functional
# ===========================================================================

def test_different_modifiers_give_different_embeddings():
    """
    If eot_pos is wrong (padding), modifiers produce identical outputs.
    This catches any eot bug regardless of root cause.
    """
    img  = IMGS2[:1]
    tok1 = clip.tokenize(["red"],                                         truncate=True)
    tok2 = clip.tokenize(["completely different style long description"],  truncate=True)
    q1   = M1.encode_query(img, tok1)
    q2   = M1.encode_query(img, tok2)
    cos  = (q1 * q2).sum().item()
    assert cos < 0.9999, \
        f"Different modifiers produce near-identical output (cos={cos:.6f}). EOT may be wrong."

def test_same_modifier_deterministic():
    img = IMGS2[:1]
    tok = clip.tokenize(["red"], truncate=True)
    q1  = M1.encode_query(img, tok)
    q2  = M1.encode_query(img, tok)
    assert torch.allclose(q1, q2, atol=1e-6), "encode_query is not deterministic"

def test_modifier_changes_vs_no_modifier():
    img      = IMGS2[:1]
    tok      = clip.tokenize(["red"], truncate=True)
    q_text   = M1.encode_query(img, tok)
    q_notext = M1.encode_query(img, None)
    cos      = (q_text * q_notext).sum().item()
    assert cos < 0.9999, "Query with text must differ from query without text"

def test_transformer_output_at_eot_has_signal():
    img              = IMGS2[:1]
    tok              = clip.tokenize(["red"], truncate=True)
    pseudo           = get_pseudo(M1, img)
    tok_emb, eot_pos = M1._build_token_sequence(pseudo, tok)
    with torch.no_grad():
        x = M1.clip.text_transformer(tok_emb.permute(1, 0, 2)).permute(1, 0, 2)
        x = M1.clip.ln_final(x)
    pos          = eot_pos[0].item()
    norm_at_eot  = x[0, pos, :].norm().item()
    assert norm_at_eot > 1.0, \
        f"Text transformer output at eot_pos has near-zero norm={norm_at_eot:.4f}"

# ===========================================================================
# 8. Forward pass
# ===========================================================================

def test_forward_output_keys():
    m = make_model(); m.train()
    out = m(IMGS)
    assert set(out.keys()) == {"loss", "loss_t2i", "loss_i2t", "logit_scale"}

def test_forward_loss_scalar():
    m = make_model(); m.train()
    assert m(IMGS)["loss"].shape == torch.Size([])

def test_forward_loss_positive():
    m = make_model(); m.train()
    assert m(IMGS)["loss"].item() > 0

def test_forward_with_captions():
    m   = make_model(); m.train()
    cap = clip.tokenize(["red", "blue", "big", "small"], truncate=True)
    assert m(IMGS, captions=cap)["loss"].item() > 0

def test_margin_adds_to_loss():
    m_plain  = make_model(); m_plain.eval()
    m_margin = make_model(); m_margin.eval()
    m_margin.load_state_dict(m_plain.state_dict())
    loss_plain  = m_plain (IMGS, margin=0.0, margin_weight=0.0)["loss"].item()
    loss_margin = m_margin(IMGS, margin=0.1, margin_weight=0.1)["loss"].item()
    assert loss_margin >= loss_plain - 1e-5, \
        f"Margin loss {loss_margin:.4f} should be >= InfoNCE {loss_plain:.4f}"

def test_pure_infonce_when_margin_zero():
    """margin_weight=0 → total loss == InfoNCE exactly."""
    m = make_model(); m.eval()
    out = m(IMGS, margin=0.5, margin_weight=0.0)
    with torch.no_grad():
        v_raw  = m.clip.encoder_image_unnorm(IMGS)
        pseudo = m.mapper(v_raw)
        v      = F.normalize(v_raw, dim=-1)
        tok_emb, eot_pos = m._build_token_sequence(pseudo)
        p      = m.clip.forward_text_transform(tok_emb, eot_pos)
        ls     = m.clip.get_logit_scale()
        logits = ls * (p @ v.T)
        labels = torch.arange(BATCH)
        expected = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0
    assert torch.allclose(out["loss"], expected, atol=1e-5), \
        f"With margin_weight=0, loss must equal InfoNCE. Got {out['loss']:.6f} vs {expected:.6f}"

# ===========================================================================
# 9. Gradients
# ===========================================================================

def test_mapper_params_get_gradients():
    m = make_model(); m.train()
    m(IMGS)["loss"].backward()
    for name, p in m.mapper.named_parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0, \
            f"Mapper param '{name}' has no gradient"

def test_logit_scale_gets_gradient():
    m = make_model(); m.train()
    m(IMGS)["loss"].backward()
    assert m.clip.logit_scale.grad is not None
    assert m.clip.logit_scale.grad.abs().item() > 0

def test_clip_visual_no_gradient():
    m = make_model(); m.train()
    m(IMGS)["loss"].backward()
    for name, p in m.clip.visual_encoder.named_parameters():
        assert p.grad is None, f"Visual encoder '{name}' must have no grad (frozen)"

def test_clip_text_no_gradient():
    m = make_model(); m.train()
    m(IMGS)["loss"].backward()
    for name, p in m.clip.text_transformer.named_parameters():
        assert p.grad is None, f"Text transformer '{name}' must have no grad (frozen)"

def test_loss_decreases_after_step():
    torch.manual_seed(0)
    m   = make_model(); m.train()
    opt = torch.optim.SGD(m.trainable_parameters(), lr=1e-1)
    l0  = m(IMGS)["loss"].item()
    for _ in range(5):
        m(IMGS)["loss"].backward(); opt.step(); opt.zero_grad()
    l1  = m(IMGS)["loss"].item()
    assert l1 < l0, f"Loss did not decrease after 5 steps: before={l0:.4f}, after={l1:.4f}"

# ===========================================================================
# 10. Contrastive loss static method
# ===========================================================================

def test_infonce_perfect_predictions_near_zero():
    B  = 8; D = 512
    v  = F.normalize(torch.randn(B, D), dim=-1)
    loss, _, _ = Pic2Word._contrastive_loss(v, v, torch.tensor(100.0))
    assert loss.item() < 0.1, f"InfoNCE with p==v should be ~0, got {loss.item():.4f}"

def test_infonce_random_near_log_B():
    B  = 8; D = 512
    p  = F.normalize(torch.randn(B, D), dim=-1)
    v  = F.normalize(torch.randn(B, D), dim=-1)
    loss, _, _ = Pic2Word._contrastive_loss(p, v, torch.tensor(1.0))
    assert abs(loss.item() - math.log(B)) < 1.0, \
        f"Random loss={loss.item():.2f}, expected~{math.log(B):.2f}"

def test_margin_term_non_negative():
    B  = 8; D = 512
    p  = F.normalize(torch.randn(B, D), dim=-1)
    v  = F.normalize(torch.randn(B, D), dim=-1)
    ls = torch.tensor(50.0)
    l_plain, _, _  = Pic2Word._contrastive_loss(p, v, ls, 0.0, 0.0)
    l_margin, _, _ = Pic2Word._contrastive_loss(p, v, ls, 0.1, 1.0)
    assert l_margin.item() >= l_plain.item() - 1e-5

def test_hard_negatives_off_diagonal():
    B  = 8; D = 512
    v  = F.normalize(torch.randn(B, D), dim=-1)
    sims     = v @ v.T
    pos      = sims.diagonal()
    hard_t2i = sims.clone().fill_diagonal_(-float('inf')).max(dim=1).values
    assert (pos > hard_t2i).all(), "With p==v, diagonal must exceed all off-diagonal sims"

# ===========================================================================
# 11. Inference
# ===========================================================================

def test_encode_query_no_text_normalised():
    feats = M1.encode_query(IMGS)
    norms = feats.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5)

def test_encode_query_with_text_normalised():
    tok   = clip.tokenize(["red", "blue", "big", "small"], truncate=True)
    feats = M1.encode_query(IMGS, tok)
    norms = feats.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5)

def test_encode_candidates_normalised():
    feats = M1.encode_candidates(IMGS)
    norms = feats.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5)

def test_encode_query_shape():
    assert M1.encode_query(IMGS).shape == (BATCH, TOKEN_DIM)

def test_encode_candidates_shape():
    assert M1.encode_candidates(IMGS).shape == (BATCH, VIS_DIM)

# ===========================================================================
# 12. Trainable parameters
# ===========================================================================

def test_logit_scale_in_trainable():
    ids = {id(p) for p in M1.trainable_parameters()}
    assert id(M1.clip.logit_scale) in ids

def test_clip_visual_not_in_trainable():
    ids = {id(p) for p in M1.trainable_parameters()}
    for name, p in M1.clip.visual_encoder.named_parameters():
        assert id(p) not in ids, f"Frozen visual param '{name}' must not be trainable"

def test_clip_text_not_in_trainable():
    ids = {id(p) for p in M1.trainable_parameters()}
    for name, p in M1.clip.text_transformer.named_parameters():
        assert id(p) not in ids, f"Frozen text param '{name}' must not be trainable"

def test_trainable_count_reasonable():
    n = sum(p.numel() for p in M1.trainable_parameters())
    assert 0 < n < 10_000_000, f"Unexpected trainable param count: {n}"

# ===========================================================================
# 13. Merger integration
# ===========================================================================

def test_token_merger_model_output_shape():
    cfg = Config(clip_model=CLIP_MODEL, mapper_type="base", hidden_dim=256,
                 num_tokens=4, merge_tokens=1, merge_type="learned", device=DEVICE)
    m   = Pic2Word(cfg).to(DEVICE); m.eval()
    assert m.merger is not None
    assert m.encode_query(IMGS2).shape == (2, TOKEN_DIM)

def test_merger_none_when_not_configured():
    assert M1.merger is None

def test_merger_none_when_equal_tokens():
    cfg = Config(clip_model=CLIP_MODEL, mapper_type="base", num_tokens=2,
                 merge_tokens=2, device=DEVICE)
    m   = Pic2Word(cfg).to(DEVICE)
    assert m.merger is None, "merge_tokens == num_tokens must not create a merger"

# ===========================================================================
# 14. State dict roundtrip
# ===========================================================================

def test_save_load_identical_output():
    cfg = make_cfg()
    m_a = Pic2Word(cfg).to(DEVICE); m_a.eval()
    m_b = Pic2Word(cfg).to(DEVICE); m_b.eval()
    m_b.mapper.load_state_dict(m_a.mapper.state_dict())
    with torch.no_grad():
        m_b.clip.logit_scale.fill_(m_a.clip.logit_scale.item())
    assert torch.allclose(m_a.encode_query(IMGS2), m_b.encode_query(IMGS2), atol=1e-6)

def test_different_init_different_output():
    cfg = make_cfg()
    m_a = Pic2Word(cfg).to(DEVICE); m_a.eval()
    m_b = Pic2Word(cfg).to(DEVICE); m_b.eval()
    assert not torch.allclose(m_a.encode_query(IMGS2), m_b.encode_query(IMGS2), atol=1e-4)

# ===========================================================================
# Runner
# ===========================================================================

TESTS = [
    # config
    ("config | max_token_len stored",          test_max_token_len_stored),
    ("config | prompt_eot_idx correct",        test_prompt_eot_idx_correct),
    ("config | clip_prompt tokenized",         test_clip_prompt_tokenized),
    ("config | device stored",                 test_device_stored),
    # clip wrapper
    ("clip   | visual_dim",                    test_visual_dim),
    ("clip   | token_dim",                     test_token_dim),
    ("clip   | encode_image normalised",       test_encode_image_normalised),
    ("clip   | encoder_image_unnorm shape",    test_encoder_image_unnorm_shape),
    ("clip   | logit_scale trainable",         test_logit_scale_trainable),
    ("clip   | visual encoder frozen",         test_visual_encoder_frozen),
    ("clip   | text transformer frozen",       test_text_transformer_frozen),
    ("clip   | logit_scale positive",          test_logit_scale_positive),
    # mappers
    ("mapper | base tok1 shape",               test_base_mlp_shape_tok1),
    ("mapper | base tok2 shape",               test_base_mlp_shape_tok2),
    ("mapper | weighted shape",                test_weighted_mlp_mapper_shape),
    ("mapper | weighted layer wts sum 1",      test_weighted_mlp_layer_weights_sum_to_one),
    ("mapper | fused weighted shape",          test_fused_weighted_mapper_shape),
    # mergers
    ("merger | token merger shape",            test_token_merger_shape),
    ("merger | weighted merger shape",         test_weighted_merger_shape),
    ("merger | weighted wts sum 1",            test_weighted_merger_weights_sum_to_one),
    ("merger | weighted param count",          test_weighted_merger_param_count),
    # token seq no text
    ("tokseq | no text eot exact tok1",        test_no_text_eot_exact_tok1),
    ("tokseq | no text eot exact tok2",        test_no_text_eot_exact_tok2),
    ("tokseq | no text output shape",          test_no_text_output_shape),
    ("tokseq | no text padding after eot=0",   test_no_text_padding_after_eot_is_zero),
    ("tokseq | no text eot embed not zero",    test_no_text_eot_embedding_not_zero),
    # token seq with text  ← CRITICAL EOT BUG TESTS
    ("EOT    | not always 76",                 test_text_eot_not_always_76),
    ("EOT    | exact value single word",       test_text_eot_exact_single_word),
    ("EOT    | scales with length",            test_text_eot_scales_with_length),
    ("EOT    | within bounds",                 test_text_eot_within_bounds),
    ("EOT    | text output shape",             test_text_output_shape),
    ("EOT    | padding after eot=0",           test_text_padding_after_eot_is_zero),
    ("EOT    | embedding matches CLIP EOT",    test_text_eot_embedding_matches_clip_eot_token),
    ("EOT    | diff lengths diff eot",         test_text_different_lengths_give_different_eot),
    # eot attention functional
    ("EOT    | diff modifiers diff embeds",    test_different_modifiers_give_different_embeddings),
    ("EOT    | same modifier deterministic",   test_same_modifier_deterministic),
    ("EOT    | modifier vs no modifier",       test_modifier_changes_vs_no_modifier),
    ("EOT    | transformer output has signal", test_transformer_output_at_eot_has_signal),
    # forward
    ("fwd    | output keys",                   test_forward_output_keys),
    ("fwd    | loss scalar",                   test_forward_loss_scalar),
    ("fwd    | loss positive",                 test_forward_loss_positive),
    ("fwd    | with captions",                 test_forward_with_captions),
    ("fwd    | margin adds to loss",           test_margin_adds_to_loss),
    ("fwd    | pure infonce when mw=0",        test_pure_infonce_when_margin_zero),
    # gradients
    ("grad   | mapper gets gradients",         test_mapper_params_get_gradients),
    ("grad   | logit scale gets gradient",     test_logit_scale_gets_gradient),
    ("grad   | clip visual no gradient",       test_clip_visual_no_gradient),
    ("grad   | clip text no gradient",         test_clip_text_no_gradient),
    ("grad   | loss decreases after step",     test_loss_decreases_after_step),
    # contrastive loss
    ("loss   | perfect preds near 0",          test_infonce_perfect_predictions_near_zero),
    ("loss   | random near log(B)",            test_infonce_random_near_log_B),
    ("loss   | margin term non-negative",      test_margin_term_non_negative),
    ("loss   | hard negatives off diagonal",   test_hard_negatives_off_diagonal),
    # inference
    ("infer  | encode_query no text norm",     test_encode_query_no_text_normalised),
    ("infer  | encode_query text norm",        test_encode_query_with_text_normalised),
    ("infer  | encode_candidates norm",        test_encode_candidates_normalised),
    ("infer  | encode_query shape",            test_encode_query_shape),
    ("infer  | encode_candidates shape",       test_encode_candidates_shape),
    # trainable params
    ("params | logit scale in trainable",      test_logit_scale_in_trainable),
    ("params | clip visual not in trainable",  test_clip_visual_not_in_trainable),
    ("params | clip text not in trainable",    test_clip_text_not_in_trainable),
    ("params | count reasonable",              test_trainable_count_reasonable),
    # merger integration
    ("merger | model output shape",            test_token_merger_model_output_shape),
    ("merger | none when not configured",      test_merger_none_when_not_configured),
    ("merger | none when equal tokens",        test_merger_none_when_equal_tokens),
    # state dict
    ("ckpt   | identical weights same output", test_save_load_identical_output),
    ("ckpt   | different init diff output",    test_different_init_different_output),
]


if __name__ == "__main__":
    print(f"{'='*65}")
    print(f"  model_test.py  —  {len(TESTS)} tests")
    print(f"{'='*65}")

    for name, fn in TESTS:
        run(name, fn)

    print(f"\n{'='*65}")
    print(f"  {PASS} passed   {FAIL} failed   {PASS + FAIL} total")
    print(f"{'='*65}\n")

    sys.exit(0 if FAIL == 0 else 1)
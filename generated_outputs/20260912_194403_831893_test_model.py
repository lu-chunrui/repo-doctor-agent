import math
import pytest
import torch
import torch.nn as nn

from model import (
    RMSNorm,
    create_norm,
    precompute_rope,
    apply_rope,
    CausalSelfAttention,
    MultiHeadAttention,
    SwiGLUFeedForward,
    GELUFeedForward,
    create_feedforward,
    TransformerBlock,
    TransformerLanguageModel,
)


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(0)


class TestRMSNorm:
    def test_forward_normal(self):
        dim = 8
        norm = RMSNorm(dim)
        x = torch.randn(2, 3, dim)
        out = norm(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_forward_single_element(self):
        norm = RMSNorm(1)
        x = torch.tensor([[[3.0]]])
        out = norm(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_forward_zero_input(self):
        norm = RMSNorm(4)
        x = torch.zeros(1, 2, 4)
        out = norm(x)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_eps_effect(self):
        norm = RMSNorm(4, eps=1e-5)
        assert norm.eps == 1e-5


class TestCreateNorm:
    def test_rmsnorm(self):
        norm = create_norm(16, "rmsnorm")
        assert isinstance(norm, RMSNorm)

    def test_layer_norm(self):
        norm = create_norm(16, "layer_norm")
        assert isinstance(norm, nn.LayerNorm)

    def test_unknown_norm_type(self):
        with pytest.raises(ValueError):
            create_norm(16, "unknown")


class TestPrecomputeRope:
    def test_shapes(self):
        head_size = 8
        max_length = 16
        cos, sin = precompute_rope(head_size, max_length)
        assert cos.shape == (max_length, head_size // 2)
        assert sin.shape == (max_length, head_size // 2)

    def test_values_at_position_zero(self):
        cos, sin = precompute_rope(4, 4)
        assert torch.allclose(cos[0], torch.ones(2))
        assert torch.allclose(sin[0], torch.zeros(2))

    def test_odd_head_size_raises(self):
        with pytest.raises(ValueError):
            precompute_rope(5, 8)

    def test_custom_base(self):
        cos, sin = precompute_rope(4, 4, base=100.0)
        assert cos.shape == (4, 2)
        assert sin.shape == (4, 2)


class TestApplyRope:
    def test_forward_normal(self):
        head_size = 4
        max_length = 8
        cos, sin = precompute_rope(head_size, max_length)
        x = torch.randn(2, 3, head_size)
        out = apply_rope(x, cos, sin)
        assert out.shape == x.shape

    def test_position_offset(self):
        head_size = 4
        max_length = 8
        cos, sin = precompute_rope(head_size, max_length)
        x = torch.randn(1, 2, head_size)
        out = apply_rope(x, cos, sin, position_offset=2)
        assert out.shape == x.shape

    def test_position_end_exceeds_raises(self):
        head_size = 4
        max_length = 4
        cos, sin = precompute_rope(head_size, max_length)
        x = torch.randn(1, 5, head_size)
        with pytest.raises(ValueError):
            apply_rope(x, cos, sin)


class TestCausalSelfAttention:
    def test_forward_normal(self):
        attn = CausalSelfAttention(embedding_dim=16, head_size=8, max_seq_len=10)
        x = torch.randn(2, 5, 16)
        out, cache = attn(x)
        assert out.shape == (2, 5, 8)
        assert cache is None

    def test_forward_with_cache(self):
        attn = CausalSelfAttention(embedding_dim=16, head_size=8, max_seq_len=10)
        x = torch.randn(2, 5, 16)
        out, cache = attn(x, use_cache=True)
        assert out.shape == (2, 5, 8)
        assert cache is not None
        k, v = cache
        assert k.shape == (2, 5, 8)
        assert v.shape == (2, 5, 8)

    def test_forward_with_past_kv(self):
        attn = CausalSelfAttention(embedding_dim=16, head_size=8, max_seq_len=10)
        x = torch.randn(2, 5, 16)
        _, cache = attn(x, use_cache=True)
        x_new = torch.randn(2, 1, 16)
        out, new_cache = attn(x_new, past_kv=cache, use_cache=True)
        assert out.shape == (2, 1, 8)
        k, v = new_cache
        assert k.shape == (2, 6, 8)
        assert v.shape == (2, 6, 8)

    def test_causal_mask(self):
        attn = CausalSelfAttention(embedding_dim=8, head_size=4, max_seq_len=4)
        x = torch.randn(1, 4, 8)
        out, _ = attn(x)
        assert out.shape == (1, 4, 4)


class TestMultiHeadAttention:
    def test_forward_normal(self):
        mha = MultiHeadAttention(embedding_dim=16, num_heads=4, max_seq_len=10)
        x = torch.randn(2, 5, 16)
        out, cache = mha(x)
        assert out.shape == (2, 5, 16)
        assert cache is None

    def test_forward_with_cache(self):
        mha = MultiHeadAttention(embedding_dim=16, num_heads=4, max_seq_len=10)
        x = torch.randn(2, 5, 16)
        out, cache = mha(x, use_cache=True)
        assert out.shape == (2, 5, 16)
        assert cache is not None
        assert len(cache) == 4

    def test_forward_with_past_kv(self):
        mha = MultiHeadAttention(embedding_dim=16, num_heads=4, max_seq_len=10)
        x = torch.randn(2, 5, 16)
        _, cache = mha(x, use_cache=True)
        x_new = torch.randn(2, 1, 16)
        out, new_cache = mha(x_new, past_kv=cache, use_cache=True)
        assert out.shape == (2, 1, 16)
        assert len(new_cache) == 4

    def test_indivisible_heads_raises(self):
        with pytest.raises(ValueError):
            MultiHeadAttention(embedding_dim=15, num_heads=4, max_seq_len=10)


class TestFeedForward:
    def test_swiglu_forward(self):
        ff = SwiGLUFeedForward(embedding_dim=16, feedforward_dim=32)
        x = torch.randn(2, 5, 16)
        out = ff(x)
        assert out.shape == x.shape

    def test_gelu_forward(self):
        ff = GELUFeedForward(embedding_dim=16, feedforward_dim=32)
        x = torch.randn(2, 5, 16)
        out = ff(x)
        assert out.shape == x.shape

    def test_create_feedforward_swiglu(self):
        ff = create_feedforward(16, 32, "swiglu")
        assert isinstance(ff, SwiGLUFeedForward)

    def test_create_feedforward_gelu(self):
        ff = create_feedforward(16, 32, "gelu")
        assert isinstance(ff, GELUFeedForward)

    def test_create_feedforward_unknown_raises(self):
        with pytest.raises(ValueError):
            create_feedforward(16, 32, "unknown")


class TestTransformerBlock:
    def test_forward_normal(self):
        block = TransformerBlock(
            embedding_dim=16,
            num_heads=4,
            feedforward_dim=32,
            max_seq_len=10,
            norm_type="rmsnorm",
            feedforward_type="swiglu",
        )
        x = torch.randn(2, 5, 16)
        out, cache = block(x)
        assert out.shape == x.shape
        assert cache is None

    def test_forward_with_cache(self):
        block = TransformerBlock(
            embedding_dim=16,
            num_heads=4,
            feedforward_dim=32,
            max_seq_len=10,
            norm_type="rmsnorm",
            feedforward_type="swiglu",
        )
        x = torch.randn(2, 5, 16)
        out, cache = block(x, use_cache=True)
        assert out.shape == x.shape
        assert cache is not None

    def test_forward_with_past_kv(self):
        block = TransformerBlock(
            embedding_dim=16,
            num_heads=4,
            feedforward_dim=32,
            max_seq_len=10,
            norm_type="rmsnorm",
            feedforward_type="swiglu",
        )
        x = torch.randn(2, 5, 16)
        _, cache = block(x, use_cache=True)
        x_new = torch.randn(2, 1, 16)
        out, new_cache = block(x_new, past_kv=cache, use_cache=True)
        assert out.shape == x_new.shape
        assert new_cache is not None


class TestTransformerLanguageModel:
    def _make_model(self, **kwargs):
        defaults = dict(
            vocab_size=20,
            embedding_dim=16,
            num_heads=4,
            num_layers=2,
            feedforward_dim=32,
            max_seq_len=10,
            norm_type="rmsnorm",
            feedforward_type="swiglu",
        )
        defaults.update(kwargs)
        return TransformerLanguageModel(**defaults)

    def test_forward_normal(self):
        model = self._make_model()
        x = torch.randint(0, 20, (2, 5))
        logits, loss = model(x)
        assert logits.shape == (2, 5, 20)
        assert loss is None

    def test_forward_with_target(self):
        model = self._make_model()
        x = torch.randint(0, 20, (2, 5))
        target = torch.randint(0, 20, (2, 5))
        logits, loss = model(x, target=target)
        assert logits.shape == (2, 5, 20)
        assert loss is not None
        assert loss.dim() == 0

    def test_forward_with_cache(self):
        model = self._make_model()
        x = torch.randint(0, 20, (2, 5))
        logits, loss, cache = model(x, use_cache=True)
        assert logits.shape == (2, 5, 20)
        assert loss is None
        assert cache is not None
        assert len(cache) == 2

    def test_forward_with_past_kv(self):
        model = self._make_model()
        x = torch.randint(0, 20, (2, 5))
        _, _, cache = model(x, use_cache=True)
        x_new = torch.randint(0, 20, (2, 1))
        logits, loss, new_cache = model(x_new, past_kv=cache, use_cache=True)
        assert logits.shape == (2, 1, 20)
        assert loss is None
        assert new_cache is not None

    def test_forward_exceeds_max_seq_len_raises(self):
        model = self._make_model(max_seq_len=5)
        x = torch.randint(0, 20, (2, 6))
        with pytest.raises(ValueError):
            model(x)

    def test_sample_next_token_normal(self):
        model = self._make_model()
        logits = torch.randn(2, 20)
        token = model.sample_next_token(logits, temperature=1.0, top_k=5)
        assert token.shape == (2, 1)
        assert token.min() >= 0
        assert token.max() < 20

    def test_sample_next_token_top_k_one(self):
        model = self._make_model()
        logits = torch.randn(2, 20)
        token = model.sample_next_token(logits, temperature=1.0, top_k=1)
        assert token.shape == (2, 1)

    def test_sample_next_token_invalid_temperature_raises(self):
        model = self._make_model()
        logits = torch.randn(2, 20)
        with pytest.raises(ValueError):
            model.sample_next_token(logits, temperature=0.0)

    def test_generate_without_cache(self):
        model = self._make_model(max_seq_len=10)
        x = torch.randint(0, 20, (1, 3))
        out = model.generate_without_cache(x, max_new_tokens=2, temperature=1.0, top_k=5)
        assert out.shape == (1, 5)

    def test_generate_without_cache_exceeds_raises(self):
        model = self._make_model(max_seq_len=5)
        x = torch.randint(0, 20, (1, 4))
        with pytest.raises(ValueError):
            model.generate_without_cache(x, max_new_tokens=2)

    def test_generate_with_cache(self):
        model = self._make_model(max_seq_len=10)
        x = torch.randint(0, 20, (1, 3))
        out = model.generate_with_cache(x, max_new_tokens=2, temperature=1.0, top_k=5)
        assert out.shape == (1, 5)

    def test_generate_with_cache_exceeds_raises(self):
        model = self._make_model(max_seq_len=5)
        x = torch.randint(0, 20, (1, 4))
        with pytest.raises(ValueError):
            model.generate_with_cache(x, max_new_tokens=2)
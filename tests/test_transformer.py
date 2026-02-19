import json
from functools import lru_cache

from einops import rearrange
import numpy
import torch
import torch.nn.functional as F

from .common import NumpySnapshot, FIXTURES_PATH

N_LAYERS = 3
VOCAB_SIZE = 10_000
BATCH_SIZE = 4
N_QUERIES = 12
N_KEYS = 16
N_HEADS = 4
D_HEAD = 16
D_MODEL = N_HEADS * D_HEAD   # 64
D_FF = 128
THETA = 10_000.0

def make_q(batch_size=BATCH_SIZE, n_queries=N_QUERIES, d_model=D_MODEL):
    torch.manual_seed(1)
    return torch.randn(batch_size, n_queries, d_model)

def make_k(batch_size=BATCH_SIZE, n_keys=N_KEYS, d_model=D_MODEL):
    torch.manual_seed(2)
    return torch.randn(batch_size, n_keys, d_model)

def make_v(batch_size=BATCH_SIZE, n_keys=N_KEYS, d_model=D_MODEL):
    torch.manual_seed(3)
    return torch.randn(batch_size, n_keys, d_model)

def make_in_embeddings(batch_size=BATCH_SIZE, n_queries=N_QUERIES, d_model=D_MODEL):
    torch.manual_seed(4)
    return torch.randn(batch_size, n_queries, d_model)

def make_mask(batch_size=BATCH_SIZE, n_queries=N_QUERIES, n_keys=N_KEYS):
    torch.manual_seed(5)
    return torch.randn(batch_size, n_queries, n_keys) > 0.5

def make_in_indices(batch_size=BATCH_SIZE, n_queries=N_QUERIES):
    torch.manual_seed(6)
    return torch.randint(0, 10_000, (batch_size, n_queries))

def make_pos_ids(n_queries=N_QUERIES):
    return torch.arange(0, n_queries)

@lru_cache
def load_ts_state_dict():
    state_dict = torch.load(FIXTURES_PATH / 'ts_tests' / 'model.pt', map_location='cpu')
    config = json.load(open(FIXTURES_PATH / 'ts_tests' / 'model_config.json'))
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    return state_dict, config

def test_linear(run_linear):
    snap = NumpySnapshot('test_linear')
    state_dict, _ = load_ts_state_dict()
    in_embeddings = make_in_embeddings()
    w1_weight = state_dict['layers.0.ffn.w1.weight']
    output = run_linear(d_in=D_MODEL, d_out=D_FF, weights=w1_weight, in_features=in_embeddings)
    snap.assert_match(output)
    print('PASSED: test_linear')

def test_embedding(run_embedding):
    snap = NumpySnapshot('test_embedding')
    state_dict, _ = load_ts_state_dict()
    in_indices = make_in_indices()
    embedding_weight = state_dict['token_embeddings.weight']
    output = run_embedding(vocab_size=VOCAB_SIZE, d_model=D_MODEL, weights=embedding_weight, token_ids=in_indices)
    snap.assert_match(output)
    print('PASSED: test_embedding')

def test_rmsnorm(run_rmsnorm):
    snap = NumpySnapshot('test_rmsnorm')
    state_dict, _ = load_ts_state_dict()
    in_embeddings = make_in_embeddings()
    reference_weights = state_dict['layers.1.ln1.weight']
    d_model = reference_weights.shape[0]
    actual_output = run_rmsnorm(d_model=d_model, eps=1e-5, weights=reference_weights, in_features=in_embeddings)
    snap.assert_match(actual_output, atol=1e-6)
    print('PASSED: test_rmsnorm')

def test_silu_matches_pytorch(run_silu):
    x = torch.tensor([
        [0.2352, 0.9259, 0.5189, 0.4725, 0.9730],
        [0.7581, 0.9692, 0.2129, 0.9345, 0.0149],
    ])
    expected_output = F.silu(x)
    actual_output = run_silu(x)
    numpy.testing.assert_allclose(actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6)
    print('PASSED: test_silu_matches_pytorch')

def test_swiglu(run_swiglu):
    snap = NumpySnapshot('test_swiglu')
    state_dict, _ = load_ts_state_dict()
    in_embeddings = make_in_embeddings()
    w1_weight, w2_weight, w3_weight = [state_dict[f'layers.0.ffn.{k}.weight'] for k in ['w1', 'w2', 'w3']]
    actual_output = run_swiglu(d_model=D_MODEL, d_ff=D_FF, w1_weight=w1_weight, w2_weight=w2_weight, w3_weight=w3_weight, in_features=in_embeddings)
    snap.assert_match(actual_output, atol=1e-5)
    print('PASSED: test_swiglu')

def test_rope(run_rope):
    snap = NumpySnapshot('test_rope')
    in_embeddings = make_in_embeddings()
    pos_ids = make_pos_ids()
    output = run_rope(D_MODEL, theta=THETA, max_seq_len=len(pos_ids), in_query_or_key=in_embeddings, token_positions=pos_ids)
    snap.assert_match(output, atol=1e-6)
    print('PASSED: test_rope')

def test_scaled_dot_product_attention(run_scaled_dot_product_attention):
    snap = NumpySnapshot('test_scaled_dot_product_attention')
    q, k, v, mask = make_q(), make_k(), make_v(), make_mask()
    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    snap.assert_match(actual_output, atol=1e-6)
    print('PASSED: test_scaled_dot_product_attention')

def test_4d_scaled_dot_product_attention(run_scaled_dot_product_attention):
    snap = NumpySnapshot('test_4d_scaled_dot_product_attention')
    q, k, v, mask = make_q(), make_k(), make_v(), make_mask()
    q, k, v = (rearrange(x, '(batch head) seq d -> batch head seq d', head=2) for x in (q, k, v))
    mask = rearrange(mask, '(batch head) query key -> batch head query key', head=2)
    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    snap.assert_match(actual_output, atol=1e-6)
    print('PASSED: test_4d_scaled_dot_product_attention')

def test_multihead_self_attention(run_multihead_self_attention):
    snap = NumpySnapshot('test_multihead_self_attention')
    state_dict, _ = load_ts_state_dict()
    in_embeddings = make_in_embeddings()
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [state_dict[f'layers.0.attn.{k}_proj.weight'] for k in ['q', 'k', 'v', 'output']]
    actual_output = run_multihead_self_attention(d_model=D_MODEL, num_heads=N_HEADS, max_seq_len=N_KEYS, q_proj_weight=q_proj_weight,
                                                 k_proj_weight=k_proj_weight, v_proj_weight=v_proj_weight, o_proj_weight=o_proj_weight,
                                                 in_features=in_embeddings)
    snap.assert_match(actual_output, atol=1e-6)
    print('PASSED: test_multihead_self_attention')

def test_multihead_self_attention_with_rope(run_multihead_self_attention_with_rope):
    snap = NumpySnapshot('test_multihead_self_attention_with_rope')
    state_dict, _ = load_ts_state_dict()
    in_embeddings = make_in_embeddings()
    pos_ids = rearrange(make_pos_ids(), 'seq -> 1 seq')
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [state_dict[f'layers.0.attn.{k}_proj.weight'] for k in ['q', 'k', 'v', 'output']]
    actual_output = run_multihead_self_attention_with_rope(
        d_model=D_MODEL, num_heads=N_HEADS, max_seq_len=N_KEYS, theta=THETA, q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight, o_proj_weight=o_proj_weight, in_features=in_embeddings, token_positions=pos_ids)
    snap.assert_match(actual_output, atol=1e-6)
    print('PASSED: test_multihead_self_attention_with_rope')

def test_transformer_block(run_transformer_block):
    snap = NumpySnapshot('test_transformer_block')
    state_dict, _ = load_ts_state_dict()
    in_embeddings = make_in_embeddings()
    block_weights = {k.replace('layers.0.', ''): v for k, v in state_dict.items() if 'layers.0.' in k}
    actual_output = run_transformer_block(d_model=D_MODEL, num_heads=N_HEADS, d_ff=D_FF, max_seq_len=N_KEYS, theta=THETA, weights=block_weights,
                                          in_features=in_embeddings)
    snap.assert_match(actual_output, atol=1e-6)
    print('PASSED: test_transformer_block')

def test_transformer_lm(run_transformer_lm):
    snap = NumpySnapshot('test_transformer_lm')
    state_dict, _ = load_ts_state_dict()
    in_indices = make_in_indices()
    actual_output = run_transformer_lm(vocab_size=VOCAB_SIZE, context_length=N_KEYS, d_model=D_MODEL, num_layers=N_LAYERS,
                                       num_heads=N_HEADS, d_ff=D_FF, rope_theta=THETA, weights=state_dict, in_indices=in_indices)
    snap.assert_match(actual_output, atol=1e-4, rtol=1e-2)
    print('PASSED: test_transformer_lm')

def test_transformer_lm_truncated_input(run_transformer_lm):
    snap = NumpySnapshot('test_transformer_lm_truncated_input')
    state_dict, _ = load_ts_state_dict()
    in_indices = make_in_indices()
    in_indices_truncated = in_indices[..., : in_indices.shape[-1] // 2]
    truncated_actual_output = run_transformer_lm(vocab_size=VOCAB_SIZE, context_length=N_KEYS, d_model=D_MODEL, d_ff=D_FF, num_layers=N_LAYERS,
                                                 num_heads=N_HEADS, rope_theta=THETA, weights=state_dict, in_indices=in_indices_truncated)
    snap.assert_match(truncated_actual_output, atol=1e-4)
    print('PASSED: test_transformer_lm_truncated_input')

def test_softmax_matches_pytorch(run_softmax):
    x = torch.tensor([
        [0.4655, 0.8303, 0.9608, 0.9656, 0.6840],
        [0.2583, 0.2198, 0.9334, 0.2995, 0.1722],
        [0.1573, 0.6860, 0.1327, 0.7284, 0.6811],
    ])
    expected = F.softmax(x, dim=-1)
    numpy.testing.assert_allclose(run_softmax(x, dim=-1).detach().numpy(), expected.detach().numpy(), atol=1e-6)
    # Test that softmax handles numerical overflow issues
    numpy.testing.assert_allclose(run_softmax(x + 100, dim=-1).detach().numpy(), expected.detach().numpy(), atol=1e-6)
    print('PASSED: test_softmax_matches_pytorch')

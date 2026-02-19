import hashlib
import torch
from .common import NumpySnapshot

BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE = 2, 10, 100
MODEL_ID = 'Qwen/Qwen2.5-Math-1.5B'

def _logits():
    torch.manual_seed(42)
    return torch.randn(BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE)

def _input_ids():
    torch.manual_seed(42)
    return torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))

def _labels():
    ids = _input_ids()
    return torch.cat([ids[:, 1:], torch.zeros(ids.shape[0], 1, dtype=ids.dtype)], dim=1)

def _raw_rewards_or_advantages():
    torch.manual_seed(42)
    return torch.rand(BATCH_SIZE, 1)

def _policy_log_probs():
    torch.manual_seed(42)
    return torch.randn(BATCH_SIZE, SEQ_LENGTH)

def _old_log_probs():
    plp = _policy_log_probs()
    torch.manual_seed(42)
    return plp + torch.randn_like(plp)

def _advantages():
    rra = _raw_rewards_or_advantages()
    return rra - torch.mean(rra, dim=0)

def _mask():
    torch.manual_seed(42)
    return torch.rand(BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE) > 0.5

def _response_mask():
    torch.manual_seed(42)
    return torch.rand(BATCH_SIZE, SEQ_LENGTH) > 0.5

def _reward_fn():
    def fn(response, ground_truth):
        h = int(hashlib.sha256(response.encode()).hexdigest(), 16)
        r = (h % 10) / 10.0
        return {'reward': r, 'format_reward': r, 'answer_reward': r}
    return fn

def test_tokenize_prompt_and_output(run_tokenize_prompt_and_output):
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception:
        print(f'SKIPPED: test_tokenize_prompt_and_output (could not load tokenizer {MODEL_ID})')
        return
    prompt_strs = ['Hello, world!', 'This is a test.', 'This is another test.']
    output_strs = ['Hello, world!', 'This is a test.', 'This is another test.']
    output = run_tokenize_prompt_and_output(prompt_strs=prompt_strs, output_strs=output_strs, tokenizer=tokenizer)
    NumpySnapshot('test_tokenize_prompt_and_output').assert_match(output)
    print('PASSED: test_tokenize_prompt_and_output')

def test_compute_entropy(run_compute_entropy):
    output = run_compute_entropy(_logits())
    NumpySnapshot('test_compute_entropy').assert_match(output)
    print('PASSED: test_compute_entropy')

def test_get_response_log_probs(run_get_response_log_probs):
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    except Exception:
        print(f'SKIPPED: test_get_response_log_probs (could not load model {MODEL_ID})')
        return
    output = run_get_response_log_probs(model=model, input_ids=_input_ids(), labels=_labels(), return_token_entropy=True)
    NumpySnapshot('test_get_response_log_probs').assert_match(output)
    print('PASSED: test_get_response_log_probs')

def test_masked_normalize_dim_0(run_masked_normalize):
    output = run_masked_normalize(tensor=_logits(), mask=_mask(), dim=0, normalize_constant=42.0)
    NumpySnapshot('test_masked_normalize_dim0').assert_match(output)
    print('PASSED: test_masked_normalize_dim_0')

def test_masked_normalize_dim_1(run_masked_normalize):
    output = run_masked_normalize(tensor=_logits(), mask=_mask(), dim=1, normalize_constant=42.0)
    NumpySnapshot('test_masked_normalize_dim1').assert_match(output)
    print('PASSED: test_masked_normalize_dim_1')

def test_masked_normalize_dim_last(run_masked_normalize):
    output = run_masked_normalize(tensor=_logits(), mask=_mask(), dim=-1, normalize_constant=42.0)
    NumpySnapshot('test_masked_normalize_dimlast').assert_match(output)
    print('PASSED: test_masked_normalize_dim_last')

def test_masked_normalize_dim_none(run_masked_normalize):
    output = run_masked_normalize(tensor=_logits(), mask=_mask(), dim=None, normalize_constant=42.0)
    NumpySnapshot('test_masked_normalize_dimNone').assert_match(output)
    print('PASSED: test_masked_normalize_dim_none')

def test_sft_microbatch_train_step(run_sft_microbatch_train_step):
    plp = _policy_log_probs()
    plp.requires_grad = True
    loss, _ = run_sft_microbatch_train_step(policy_log_probs=plp, response_mask=_response_mask(),
                                            gradient_accumulation_steps=2, normalization_constant=1.0)
    NumpySnapshot('test_sft_microbatch_train_step').assert_match({'loss': loss, 'policy_log_probs_grad': plp.grad})
    print('PASSED: test_sft_microbatch_train_step')

def test_sft_microbatch_train_step_normalize(run_sft_microbatch_train_step):
    plp = _policy_log_probs()
    plp.requires_grad = True
    loss, _ = run_sft_microbatch_train_step(policy_log_probs=plp, response_mask=_response_mask(),
                                            gradient_accumulation_steps=2, normalization_constant=42.0)
    NumpySnapshot('test_sft_microbatch_train_step_normalize').assert_match({'loss': loss, 'policy_log_probs_grad': plp.grad})
    print('PASSED: test_sft_microbatch_train_step_normalize')

def test_sft_microbatch_train_step_10_steps(run_sft_microbatch_train_step):
    plp = _policy_log_probs()
    plp.requires_grad = True
    loss_list, grad_list = [], []
    for _ in range(10):
        loss, _ = run_sft_microbatch_train_step(policy_log_probs=plp, response_mask=_response_mask(),
                                                gradient_accumulation_steps=2, normalization_constant=1.0)
        loss_list.append(loss)
        grad_list.append(plp.grad)
    NumpySnapshot('test_sft_microbatch_train_step_10_steps').assert_match(
        {'loss': torch.stack(loss_list), 'policy_log_probs_grad': torch.stack(grad_list)})
    print('PASSED: test_sft_microbatch_train_step_10_steps')

def test_compute_group_normalized_rewards_normalize_by_std(run_compute_group_normalized_rewards):
    rollout_responses = [f'hmm I think ths answer is {i}' for i in range(8)]
    repeated_ground_truths = ['42'] * 8
    normalized_rewards, raw_rewards, metadata = run_compute_group_normalized_rewards(
        reward_fn=_reward_fn(), rollout_responses=rollout_responses, repeated_ground_truths=repeated_ground_truths,
        group_size=4, advantage_eps=1e-6, normalize_by_std=True)
    NumpySnapshot('test_compute_group_normalized_rewards_normalize_by_std').assert_match(
        {'normalized_rewards': normalized_rewards, 'raw_rewards': raw_rewards})
    print('PASSED: test_compute_group_normalized_rewards_normalize_by_std')

def test_compute_group_normalized_rewards_no_normalize_by_std(run_compute_group_normalized_rewards):
    rollout_responses = [f'hmm I think ths answer is {i}' for i in range(8)]
    repeated_ground_truths = ['42'] * 8
    normalized_rewards, raw_rewards, metadata = run_compute_group_normalized_rewards(
        reward_fn=_reward_fn(), rollout_responses=rollout_responses, repeated_ground_truths=repeated_ground_truths,
        group_size=4, advantage_eps=1e-6, normalize_by_std=False)
    NumpySnapshot('test_compute_group_normalized_rewards_no_normalize_by_std').assert_match(
        {'normalized_rewards': normalized_rewards, 'raw_rewards': raw_rewards})
    print('PASSED: test_compute_group_normalized_rewards_no_normalize_by_std')

def test_compute_naive_policy_gradient_loss(run_compute_naive_policy_gradient_loss):
    output = run_compute_naive_policy_gradient_loss(raw_rewards_or_advantages=_raw_rewards_or_advantages(), policy_log_probs=_policy_log_probs())
    NumpySnapshot('test_compute_naive_policy_gradient_loss').assert_match(output)
    print('PASSED: test_compute_naive_policy_gradient_loss')

def test_compute_grpo_clip_loss_large_cliprange(run_compute_grpo_clip_loss):
    output, _ = run_compute_grpo_clip_loss(advantages=_advantages(), policy_log_probs=_policy_log_probs(),
                                           old_log_probs=_old_log_probs(), cliprange=10.0)
    NumpySnapshot('test_compute_grpo_clip_loss_large_cliprange').assert_match(output)
    print('PASSED: test_compute_grpo_clip_loss_large_cliprange')

def test_compute_grpo_clip_loss_small_cliprange(run_compute_grpo_clip_loss):
    output, _ = run_compute_grpo_clip_loss(advantages=_advantages(), policy_log_probs=_policy_log_probs(),
                                           old_log_probs=_old_log_probs(), cliprange=0.1)
    NumpySnapshot('test_compute_grpo_clip_loss_small_cliprange').assert_match(output)
    print('PASSED: test_compute_grpo_clip_loss_small_cliprange')

def test_compute_policy_gradient_loss_no_baseline(run_compute_policy_gradient_loss):
    output, _ = run_compute_policy_gradient_loss(policy_log_probs=_policy_log_probs(), loss_type='no_baseline',
                                                 raw_rewards=_raw_rewards_or_advantages(), advantages=_advantages(),
                                                 old_log_probs=_old_log_probs(), cliprange=0.5)
    NumpySnapshot('test_compute_policy_gradient_loss_no_baseline').assert_match(output)
    print('PASSED: test_compute_policy_gradient_loss_no_baseline')

def test_compute_policy_gradient_loss_reinforce_with_baseline(run_compute_policy_gradient_loss):
    output, _ = run_compute_policy_gradient_loss(policy_log_probs=_policy_log_probs(), loss_type='reinforce_with_baseline',
                                                 raw_rewards=_raw_rewards_or_advantages(), advantages=_advantages(),
                                                 old_log_probs=_old_log_probs(), cliprange=0.5)
    NumpySnapshot('test_compute_policy_gradient_loss_reinforce_with_baseline').assert_match(output)
    print('PASSED: test_compute_policy_gradient_loss_reinforce_with_baseline')

def test_compute_policy_gradient_loss_grpo_clip(run_compute_policy_gradient_loss):
    output, _ = run_compute_policy_gradient_loss(policy_log_probs=_policy_log_probs(), loss_type='grpo_clip',
                                                 raw_rewards=_raw_rewards_or_advantages(), advantages=_advantages(),
                                                 old_log_probs=_old_log_probs(), cliprange=0.5)
    NumpySnapshot('test_compute_policy_gradient_loss_grpo_clip').assert_match(output)
    print('PASSED: test_compute_policy_gradient_loss_grpo_clip')

def test_masked_mean_dim_0(run_masked_mean):
    output = run_masked_mean(tensor=_logits(), mask=_mask(), dim=0)
    NumpySnapshot('test_masked_mean_dim0').assert_match(output)
    print('PASSED: test_masked_mean_dim_0')

def test_masked_mean_dim_1(run_masked_mean):
    output = run_masked_mean(tensor=_logits(), mask=_mask(), dim=1)
    NumpySnapshot('test_masked_mean_dim1').assert_match(output)
    print('PASSED: test_masked_mean_dim_1')

def test_masked_mean_dim_last(run_masked_mean):
    output = run_masked_mean(tensor=_logits(), mask=_mask(), dim=-1)
    NumpySnapshot('test_masked_mean_dimlast').assert_match(output)
    print('PASSED: test_masked_mean_dim_last')

def test_masked_mean_dim_none(run_masked_mean):
    output = run_masked_mean(tensor=_logits(), mask=_mask())
    NumpySnapshot('test_masked_mean_dimNone').assert_match(output)
    print('PASSED: test_masked_mean_dim_none')

def test_grpo_microbatch_train_step_grpo_clip(run_grpo_microbatch_train_step):
    plp = _policy_log_probs()
    plp.requires_grad = True
    loss, _ = run_grpo_microbatch_train_step(
        policy_log_probs=plp, response_mask=_response_mask(), gradient_accumulation_steps=2, loss_type='grpo_clip',
        raw_rewards=_raw_rewards_or_advantages(), advantages=_advantages(), old_log_probs=_old_log_probs(), cliprange=0.1)
    NumpySnapshot('test_grpo_microbatch_train_step_grpo_clip').assert_match({'loss': loss, 'policy_log_probs_grad': plp.grad})
    print('PASSED: test_grpo_microbatch_train_step_grpo_clip')

def test_grpo_microbatch_train_step_grpo_clip_10_steps(run_grpo_microbatch_train_step):
    plp = _policy_log_probs()
    plp.requires_grad = True
    loss_list, grad_list = [], []
    for _ in range(10):
        loss, _ = run_grpo_microbatch_train_step(
            policy_log_probs=plp, response_mask=_response_mask(), gradient_accumulation_steps=2, loss_type='grpo_clip',
            raw_rewards=_raw_rewards_or_advantages(), advantages=_advantages(), old_log_probs=_old_log_probs(), cliprange=0.1)
        loss_list.append(loss)
        grad_list.append(plp.grad)
    NumpySnapshot('test_grpo_microbatch_train_step_grpo_clip_10_steps').assert_match(
        {'loss': torch.stack(loss_list), 'policy_log_probs_grad': torch.stack(grad_list)})
    print('PASSED: test_grpo_microbatch_train_step_grpo_clip_10_steps')

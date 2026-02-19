import json
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import FIXTURES_PATH

def test_parse_mmlu_response(run_parse_mmlu_response):
    mmlu_example = {
        'subject': 'virology',
        'question': 'How many human polyomaviruses are known at present?',
        'options': ['100', '1', '10', 'unknown'],
        'answer': 'A',
    }
    model_output = 'The correct answer is B. There is only one human polyomavirus known at present, which is the BK virus.'
    assert run_parse_mmlu_response(mmlu_example=mmlu_example, model_output=model_output) == 'B'
    print('PASSED: test_parse_mmlu_response')

def test_parse_mmlu_response_unknown(run_parse_mmlu_response):
    mmlu_example = {
        'subject': 'virology',
        'question': 'How many human polyomaviruses are known at present?',
        'options': ['100', '1', '10', 'unknown'],
        'answer': 'A',
    }
    model_output = 'The correct answer is 10000 polyomaviruses.'
    assert not run_parse_mmlu_response(mmlu_example=mmlu_example, model_output=model_output)
    print('PASSED: test_parse_mmlu_response_unknown')

def test_parse_gsm8k_response(run_parse_gsm8k_response):
    model_output = 'Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May.'
    assert run_parse_gsm8k_response(model_output=model_output) == '72'
    print('PASSED: test_parse_gsm8k_response')

def test_parse_gsm8k_response_unknown(run_parse_gsm8k_response):
    model_output = 'Natalia sold twenty-four clips in May. Thus, Natalia sold seventy-two clips altogether in April and May.'
    assert not run_parse_gsm8k_response(model_output=model_output)
    print('PASSED: test_parse_gsm8k_response_unknown')

def test_packed_sft_dataset(get_packed_sft_dataset):
    sft_sample_path = FIXTURES_PATH / 'sft_sample.jsonl'
    tokenizer = AutoTokenizer.from_pretrained(FIXTURES_PATH / 'Meta-Llama-3-8B')
    seq_length = 32
    packed_sft_dataset = get_packed_sft_dataset(tokenizer=tokenizer, dataset_path=sft_sample_path, seq_length=seq_length, shuffle=False)
    with open(FIXTURES_PATH / 'tokenized_sft_sample.json') as f:
        expected_examples = json.load(f)
    assert len(packed_sft_dataset) == len(expected_examples)
    for example, expected_example in zip(packed_sft_dataset, expected_examples):
        assert example['input_ids'].tolist() == expected_example['input_ids']
        assert example['labels'].tolist() == expected_example['labels']
    shuffled = get_packed_sft_dataset(tokenizer=tokenizer, dataset_path=sft_sample_path, seq_length=seq_length, shuffle=True)
    all_unshuffled = [{k: v.tolist() for k, v in ex.items()} for ex in packed_sft_dataset]
    all_shuffled = [{k: v.tolist() for k, v in ex.items()} for ex in shuffled]
    assert all_unshuffled != all_shuffled
    print('PASSED: test_packed_sft_dataset')

def test_iterate_batches(get_packed_sft_dataset, run_iterate_batches):
    sft_sample_path = FIXTURES_PATH / 'sft_sample.jsonl'
    tokenizer = AutoTokenizer.from_pretrained(FIXTURES_PATH / 'Meta-Llama-3-8B')
    seq_length = 32
    batch_size = 8
    packed_sft_dataset = get_packed_sft_dataset(tokenizer=tokenizer, dataset_path=sft_sample_path, seq_length=seq_length, shuffle=True)
    train_dataloader = run_iterate_batches(dataset=packed_sft_dataset, batch_size=batch_size, shuffle=True)
    assert len(train_dataloader) == math.ceil(75 / batch_size)
    for batch_idx, batch in enumerate(train_dataloader):
        if batch_idx != len(train_dataloader) - 1:
            assert batch['input_ids'].shape == (batch_size, seq_length)
            assert batch['labels'].shape == (batch_size, seq_length)
        assert batch['input_ids'].dtype in (torch.long, torch.int64)
        assert batch['labels'].dtype in (torch.long, torch.int64)
    print('PASSED: test_iterate_batches')

def test_per_instance_dpo_loss(run_compute_per_instance_dpo_loss):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained(FIXTURES_PATH / 'tiny-gpt2')
    model_ref = AutoModelForCausalLM.from_pretrained(FIXTURES_PATH / 'tiny-gpt2-ref')
    prompt = 'The quick brown fox jumps over'
    good_response = 'the lazy dog.'
    bad_response = 'their crazy frog.'
    loss = run_compute_per_instance_dpo_loss(lm=model, lm_ref=model_ref, tokenizer=tokenizer, beta=0.5,
                                             prompt=prompt, response_chosen=good_response, response_rejected=bad_response)
    assert torch.isclose(loss, torch.tensor(0.5785), atol=1e-4)
    print('PASSED: test_per_instance_dpo_loss')

import math
import tempfile
from collections import Counter
from pathlib import Path

import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from .common import NumpySnapshot

def test_cross_entropy(run_cross_entropy):
    inputs = torch.tensor([
        [
            [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
            [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
            [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
            [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
        ],
        [
            [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
            [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
            [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
            [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
        ]])
    targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])
    expected = F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1))
    numpy.testing.assert_allclose(
        run_cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1)).detach().numpy(), expected.detach().numpy(), atol=1e-4,
    )
    # Test that cross-entropy handles numerical overflow issues
    large_inputs = 1000.0 * inputs
    large_expected = F.cross_entropy(large_inputs.view(-1, large_inputs.size(-1)), targets.view(-1))
    numpy.testing.assert_allclose(
        run_cross_entropy(large_inputs.view(-1, large_inputs.size(-1)), targets.view(-1)).detach().numpy(),
        large_expected.detach().numpy(), atol=1e-4,
    )
    print('PASSED: test_cross_entropy')

def test_gradient_clipping(run_gradient_clipping):
    tensors = [torch.randn((5, 5)) for _ in range(6)]
    max_norm = 1e-2
    t1 = tuple(torch.nn.Parameter(torch.clone(t)) for t in tensors)
    t1[-1].requires_grad_(False)
    loss = torch.cat(t1).sum()
    loss.backward()
    clip_grad_norm_(t1, max_norm)
    t1_grads = [torch.clone(t.grad) for t in t1 if t.grad is not None]
    t1_c = tuple(torch.nn.Parameter(torch.clone(t)) for t in tensors)
    t1_c[-1].requires_grad_(False)
    loss_c = torch.cat(t1_c).sum()
    loss_c.backward()
    run_gradient_clipping(t1_c, max_norm)
    t1_c_grads = [torch.clone(t.grad) for t in t1_c if t.grad is not None]
    assert len(t1_grads) == len(t1_c_grads)
    for t1_grad, t1_c_grad in zip(t1_grads, t1_c_grads):
        numpy.testing.assert_allclose(t1_grad.detach().numpy(), t1_c_grad.detach().numpy(), atol=1e-6)
    print('PASSED: test_gradient_clipping')

def _optimize(opt_class):
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_class(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    for _ in range(1000):
        opt.zero_grad()
        x = torch.rand(model.in_features)
        y_hat = model(x)
        y = torch.tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()
    return model.weight.detach()

def test_adamw(get_adamw_cls):
    snap = NumpySnapshot('test_adamw')
    pytorch_weights = _optimize(torch.optim.AdamW)
    actual_weights = _optimize(get_adamw_cls())
    if torch.allclose(actual_weights, pytorch_weights, atol=1e-4):
        print('PASSED: test_adamw')
        return
    snap.assert_match(actual_weights, atol=1e-4)
    print('PASSED: test_adamw')

def test_get_lr_cosine_schedule(run_get_lr_cosine_schedule):
    max_learning_rate = 1
    min_learning_rate = 1 * 0.1
    warmup_iters = 7
    cosine_cycle_iters = 21
    expected_lrs = [0, 0.14285714285714285, 0.2857142857142857, 0.42857142857142855, 0.5714285714285714, 0.7142857142857143, 0.8571428571428571,
                    1.0, 0.9887175604818206, 0.9554359905560885, 0.9018241671106134, 0.8305704108364301, 0.7452476826029011, 0.6501344202803414,
                    0.55, 0.44986557971965857, 0.3547523173970989, 0.26942958916356996, 0.19817583288938662, 0.14456400944391146,
                    0.11128243951817937, 0.1, 0.1, 0.1, 0.1]
    actual_lrs = [
        run_get_lr_cosine_schedule(it=it, max_learning_rate=max_learning_rate, min_learning_rate=min_learning_rate, warmup_iters=warmup_iters,
                                   cosine_cycle_iters=cosine_cycle_iters) for it in range(25)]
    numpy.testing.assert_allclose(numpy.array(actual_lrs), numpy.array(expected_lrs))
    print('PASSED: test_get_lr_cosine_schedule')

def test_get_batch(run_get_batch):
    dataset = np.arange(0, 100)
    context_length = 7
    batch_size = 32
    device = 'cpu'
    starting_indices = Counter()
    num_iters = 1000
    for _ in range(num_iters):
        x, y = run_get_batch(dataset=dataset, batch_size=batch_size, context_length=context_length, device=device)
        assert x.shape == (batch_size, context_length)
        assert y.shape == (batch_size, context_length)
        np.testing.assert_allclose((x + 1).detach().numpy(), y.detach().numpy())
        starting_indices.update(x[:, 0].tolist())
    num_possible_starting_indices = len(dataset) - context_length
    assert max(starting_indices) == num_possible_starting_indices - 1
    assert min(starting_indices) == 0
    expected_count = (num_iters * batch_size) / num_possible_starting_indices
    std = math.sqrt((num_iters * batch_size) * (1 / num_possible_starting_indices) * (1 - (1 / num_possible_starting_indices)))
    lower = expected_count - 5 * std
    upper = expected_count + 5 * std
    for idx, count in starting_indices.items():
        if count < lower:
            raise ValueError(f'Starting index {idx} occurs {count} times, but expected at least {lower}')
        if count > upper:
            raise ValueError(f'Starting index {idx} occurs {count} times, but expected at most {upper}')
    # Verify the device flag is handled (invalid device should raise)
    try:
        run_get_batch(dataset=dataset, batch_size=batch_size, context_length=context_length, device='cuda:99')
        raise AssertionError('Expected RuntimeError or AssertionError for invalid device')
    except (RuntimeError, AssertionError):
        pass
    print('PASSED: test_get_batch')

class _TestNet(nn.Module):
    def __init__(self, d_input=100, d_output=10):
        super().__init__()
        self.fc1 = nn.Linear(d_input, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, d_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def _are_optimizers_equal(state1_dict, state2_dict, atol=1e-8, rtol=1e-5):
    if set(state1_dict.keys()) != set(state2_dict.keys()):
        return False
    if state1_dict['param_groups'] != state2_dict['param_groups']:
        return False
    s1, s2 = state1_dict['state'], state2_dict['state']
    if set(s1.keys()) != set(s2.keys()):
        return False
    for key in s1:
        if set(s1[key].keys()) != set(s2[key].keys()):
            return False
        for sub_key in s1[key]:
            a, b = s1[key][sub_key], s2[key][sub_key]
            if torch.is_tensor(a) and torch.is_tensor(b):
                if not torch.allclose(a, b, atol=atol, rtol=rtol):
                    return False
            elif a != b:
                return False
    return True

def test_checkpointing(get_adamw_cls, run_save_checkpoint, run_load_checkpoint):
    torch.manual_seed(42)
    d_input, d_output, num_iters = 100, 10, 10
    model = _TestNet(d_input=d_input, d_output=d_output)
    optimizer = get_adamw_cls()(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    it = 0
    for _ in range(num_iters):
        optimizer.zero_grad()
        x = torch.rand(d_input)
        y = torch.rand(d_output)
        y_hat = model(x)
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        optimizer.step()
        it += 1
    tmp_dir = tempfile.mkdtemp()
    serialization_path = Path(tmp_dir) / 'checkpoint.pt'
    run_save_checkpoint(model, optimizer, iteration=it, out=serialization_path)
    new_model = _TestNet(d_input=d_input, d_output=d_output)
    new_optimizer = get_adamw_cls()(new_model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    loaded_iterations = run_load_checkpoint(src=serialization_path, model=new_model, optimizer=new_optimizer)
    assert it == loaded_iterations
    original_model_state = model.state_dict()
    new_model_state = new_model.state_dict()
    assert set(original_model_state.keys()) == set(new_model_state.keys())
    for key in original_model_state:
        numpy.testing.assert_allclose(original_model_state[key].detach().numpy(), new_model_state[key].detach().numpy())
    assert _are_optimizers_equal(optimizer.state_dict(), new_optimizer.state_dict())
    print('PASSED: test_checkpointing')

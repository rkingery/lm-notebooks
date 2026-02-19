import os
from copy import deepcopy

import numpy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from .common import FIXTURES_PATH

def _validate_ddp_net_equivalence(net):
    net_module_states = list(net.module.state_dict().values())
    for t in net_module_states:
        tensor_list = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, t)
        for tensor in tensor_list:
            assert torch.allclose(tensor, t)

class _FC2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 50, bias=True)
        self.fc.bias.requires_grad = False
    def forward(self, x):
        return self.fc(x)

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.fc2 = _FC2()
        self.fc3 = nn.Linear(50, 5, bias=False)
        self.relu = nn.ReLU()
        self.no_grad_fixed_param = nn.Parameter(torch.tensor([2.0, 2.0]), requires_grad=False)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class ToyModelWithTiedWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 10, bias=False)
        self.fc4 = nn.Linear(10, 50, bias=False)
        self.fc5 = nn.Linear(50, 5, bias=False)
        self.fc4.weight = self.fc2.weight
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.fc5(x)

def _setup_process_group(rank, world_size, backend):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12390'
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 0:
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)
        else:
            raise ValueError('Unable to find CUDA devices.')
        device = f'cuda:{local_rank}'
    else:
        device = 'cpu'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device

def _cleanup_process_group():
    dist.barrier()
    dist.destroy_process_group()

def _run_ddp_training_loop(rank, world_size, ddp_model, non_parallel_model, on_train_batch_start=None, on_after_backward=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_x = torch.load(FIXTURES_PATH / 'ddp_test_data.pt')
    all_y = torch.load(FIXTURES_PATH / 'ddp_test_labels.pt')
    assert all_x.size(0) % world_size == 0
    local_bs = int(all_y.size(0) / world_size)
    loss_fn = nn.MSELoss()
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)
    non_parallel_optimizer = optim.SGD(non_parallel_model.parameters(), lr=0.1)
    for i in range(5):
        if on_train_batch_start:
            on_train_batch_start(ddp_model=ddp_model, optimizer=ddp_optimizer)
        ddp_optimizer.zero_grad()
        non_parallel_optimizer.zero_grad()
        non_parallel_outputs = non_parallel_model(all_x.to(device))
        non_parallel_loss = loss_fn(non_parallel_outputs, all_y.to(device))
        non_parallel_loss.backward()
        non_parallel_optimizer.step()
        if rank == 0:
            for np_param, ddp_param in zip(non_parallel_model.parameters(), ddp_model.parameters()):
                if np_param.requires_grad and ddp_param.requires_grad:
                    assert not torch.allclose(np_param, ddp_param)
                else:
                    assert torch.allclose(np_param, ddp_param)
        offset = rank * local_bs
        ddp_data = all_x[offset : offset + local_bs, :].to(device)
        ddp_labels = all_y[offset : offset + local_bs, :].to(device)
        ddp_outputs = ddp_model(ddp_data)
        ddp_loss = loss_fn(ddp_outputs, ddp_labels)
        ddp_loss.backward()
        if on_after_backward:
            on_after_backward(ddp_model=ddp_model, optimizer=ddp_optimizer)
        ddp_optimizer.step()
        if rank == 0:
            for np_param, ddp_param in zip(non_parallel_model.parameters(), ddp_model.parameters()):
                assert torch.allclose(np_param, ddp_param)
        torch.manual_seed(42 + i)
        shuffle_idxs = torch.randperm(all_x.size(0))
        all_x = all_x[shuffle_idxs]
        all_y = all_y[shuffle_idxs]
    if rank == 0:
        for np_param, ddp_param in zip(non_parallel_model.parameters(), ddp_model.parameters()):
            assert torch.allclose(np_param, ddp_param)

def test_ddp_individual_parameters(get_ddp_individual_parameters, ddp_individual_parameters_on_after_backward):
    world_size = 2
    for model_class in [ToyModel, ToyModelWithTiedWeights]:
        mp.spawn(
            _worker_ddp_individual_parameters,
            args=(world_size, model_class, get_ddp_individual_parameters, ddp_individual_parameters_on_after_backward),
            nprocs=world_size, join=True, start_method='fork',
        )
    print('PASSED: test_ddp_individual_parameters')

def _worker_ddp_individual_parameters(rank, world_size, model_class, get_ddp_individual_parameters, ddp_individual_parameters_on_after_backward):
    device = _setup_process_group(rank=rank, world_size=world_size, backend='gloo')
    dist.barrier()
    torch.manual_seed(rank)
    non_parallel_model = model_class().to(device)
    ddp_base = deepcopy(non_parallel_model)
    ddp_model = get_ddp_individual_parameters(ddp_base)
    for (np_name, np_param), (ddp_name, ddp_param) in zip(
        non_parallel_model.named_parameters(), ddp_model.named_parameters()
    ):
        is_no_grad_fixed = 'no_grad_fixed_param' in ddp_name or 'no_grad_fixed_param' in np_name
        if rank == 0 or is_no_grad_fixed:
            assert torch.allclose(np_param, ddp_param)
        else:
            assert not torch.allclose(np_param, ddp_param)
    _validate_ddp_net_equivalence(ddp_model)
    _run_ddp_training_loop(
        rank, world_size, ddp_model, non_parallel_model,
        on_after_backward=ddp_individual_parameters_on_after_backward,
    )
    _cleanup_process_group()

def test_ddp_bucketed(get_ddp_bucketed, ddp_bucketed_on_after_backward, ddp_bucketed_on_train_batch_start):
    world_size = 2
    for model_class in [ToyModel, ToyModelWithTiedWeights]:
        for bucket_size_mb in [0.0016, 0.0001, 0.01]:
            mp.spawn(
                _worker_ddp_bucketed,
                args=(world_size, bucket_size_mb, model_class, get_ddp_bucketed, ddp_bucketed_on_after_backward, ddp_bucketed_on_train_batch_start),
                nprocs=world_size, join=True, start_method='fork',
            )
    print('PASSED: test_ddp_bucketed')

def _worker_ddp_bucketed(rank, world_size, bucket_size_mb, model_class, get_ddp_bucketed, ddp_bucketed_on_after_backward,
                          ddp_bucketed_on_train_batch_start):
    device = _setup_process_group(rank=rank, world_size=world_size, backend='gloo')
    dist.barrier()
    torch.manual_seed(rank)
    non_parallel_model = model_class().to(device)
    ddp_base = deepcopy(non_parallel_model)
    ddp_model = get_ddp_bucketed(ddp_base, bucket_size_mb=bucket_size_mb)
    for (np_name, np_param), (ddp_name, ddp_param) in zip(
        non_parallel_model.named_parameters(), ddp_model.named_parameters()
    ):
        is_no_grad_fixed = 'no_grad_fixed_param' in ddp_name or 'no_grad_fixed_param' in np_name
        if rank == 0 or is_no_grad_fixed:
            assert torch.allclose(np_param, ddp_param)
        else:
            assert not torch.allclose(np_param, ddp_param)
    _validate_ddp_net_equivalence(ddp_model)
    _run_ddp_training_loop(
        rank, world_size, ddp_model, non_parallel_model,
        on_train_batch_start=ddp_bucketed_on_train_batch_start,
        on_after_backward=ddp_bucketed_on_after_backward,
    )
    _cleanup_process_group()

def test_sharded_optimizer(get_sharded_optimizer):
    world_size = 2
    for model_class in [ToyModel, ToyModelWithTiedWeights]:
        mp.spawn(
            _worker_sharded_optimizer,
            args=(world_size, model_class, get_sharded_optimizer),
            nprocs=world_size, join=True, start_method='fork',
        )
    print('PASSED: test_sharded_optimizer')

def _worker_sharded_optimizer(rank, world_size, model_class, get_sharded_optimizer):
    device = _setup_process_group(rank=rank, world_size=world_size, backend='gloo')
    torch.manual_seed(42)
    optimizer_cls = torch.optim.AdamW
    non_sharded_model = model_class().to(device)
    non_sharded_optimizer = optimizer_cls(non_sharded_model.parameters(), lr=0.1, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-8)
    sharded_model = deepcopy(non_sharded_model)
    sharded_optimizer = get_sharded_optimizer(sharded_model.parameters(), optimizer_cls, lr=0.1, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-8)
    for _ in range(10):
        non_sharded_optimizer.zero_grad()
        sharded_optimizer.zero_grad()
        input_ = torch.rand((32, 10)).to(device)
        labels = torch.rand((32, 5)).to(device)
        non_sharded_input, sharded_input = deepcopy(input_), deepcopy(input_)
        non_sharded_labels, sharded_labels = deepcopy(labels), deepcopy(labels)
        non_sharded_loss = ((non_sharded_labels - non_sharded_model(non_sharded_input)) ** 2).sum()
        sharded_loss = ((sharded_labels - sharded_model(sharded_input)) ** 2).sum()
        non_sharded_loss.backward()
        sharded_loss.backward()
        non_sharded_optimizer.step()
        sharded_optimizer.step()
    for non_sharded_p, sharded_p in zip(non_sharded_model.parameters(), sharded_model.parameters()):
        numpy.testing.assert_allclose(non_sharded_p.detach().cpu().numpy(), sharded_p.detach().cpu().numpy())
    _cleanup_process_group()

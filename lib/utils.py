import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

import sys
from torch import optim as optim
from typing import List, Union
import json
import csv
import random
import torch.nn.functional as F
import numpy as np


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / max(1, self.count)

    @property
    def max(self):
        return 0 if len(self.deque)==0 else max(self.deque)

    @property
    def value(self):
        return 0 if len(self.deque)==0 else self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                    sys.stdout.flush()
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / max(1, len(iterable))))
        sys.stdout.flush()

    def log_every_no_print(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                # if torch.cuda.is_available():
                #     print(log_msg.format(
                #         i, len(iterable), eta=eta_string,
                #         meters=str(self),
                #         time=str(iter_time), data=str(data_time),
                #         memory=torch.cuda.max_memory_allocated() / MB))
                #     sys.stdout.flush()
                # else:
                #     print(log_msg.format(
                #         i, len(iterable), eta=eta_string,
                #         meters=str(self),
                #         time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('{} Total time: {} ({:.4f} s / it)'.format(
        #     header, total_time_str, total_time / max(1, len(iterable))))
        sys.stdout.flush()


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        args.rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
        args.world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE'))
        args.gpu = args.rank % torch.cuda.device_count()
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def build_optimizer(config, model, momentum=0.9):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    high_lr_keywords = {}
    low_keywords = {}  # low_keywords disabled
    # if hasattr(model, 'low_weight_decay_keywords'):
    #     low_keywords = model.low_weight_decay_keywords()
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    if hasattr(model, 'high_lr_keywords'):
        high_lr_keywords = model.high_lr_keywords()

    parameters = set_weight_decay(model, skip_list=skip, skip_keywords=skip_keywords, low_keywords=low_keywords,
                                  high_lr_keywords=high_lr_keywords, high_lr_num=config.lr * 10)

    opt_lower = config.opt.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=float(momentum), nesterov=True,
                              lr=config.lr, weight_decay=config.weight_decay)
    elif opt_lower == 'adamw':
        optimizer = AdamWCustomized(parameters, eps=config.opt_eps,
                                lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=(), low_keywords=(), high_lr_keywords=(), high_lr_num=None):
    has_decay = []
    has_decay_name = []
    no_decay = []
    no_decay_name = []
    low_decay = []
    low_decay_name = []
    high_lr = []
    high_lr_name = []

    for name, param in model.named_parameters():
        # if 'adapter' in name:
        #     print(
        #         'hi'
        #     )
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):

            no_decay.append(param)
            no_decay_name.append(name)
        # elif check_keywords_in_name(name, low_keywords):
        #     low_decay.append(param)
        #     low_decay_name.append(name)
        elif check_keywords_in_name(name, high_lr_keywords):
            high_lr.append(param)
            high_lr_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)

    if high_lr_keywords:
        print('high lr params: ', high_lr_name)
        return [{'params': has_decay},
                {'params': low_decay, 'weight_decay': 0.0001},
                {'params': no_decay, 'weight_decay': 0.},
                {'params': high_lr, 'lr': high_lr_num}]
    else:
        return [{'params': has_decay},
                {'params': low_decay, 'weight_decay': 0.0001},
                {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


class AdamWCustomized(optim.AdamW):
    def fake_step(self, closure=None):  # for testing...
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                beta1, beta2 = group['betas']
                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])


def read_json(filename: str) -> Union[list, dict]:
    """read json files"""
    with open(filename, "rb") as fin:
        data = json.load(fin, encoding="utf-8")
    return data


DATASETS = {
    'cifar': 1,
    'caltech101': 2,
    'dtd': 3,
    'oxford_flowers102': 4,
    'svhn': 5,
    'sun397': 6,
    'oxford_iiit_pet': 7,
    'natural': -1,
    'patch_camelyon': 8,
    'eurosat': 9,
    'resisc45': 10,
    'diabetic_retinopathy': 11,
    'specialized': -1,
    'clevr_count': 12,
    'clevr_dist': 13,
    'dmlab': 14,
    'kitti': 15,
    'dsprites_loc': 16,
    'dsprites_ori': 17,
    'smallnorb_azi': 18,
    'smallnorb_ele': 19,
    'structured': -1,
    'cub': 1,
    'nabirds': 2,
    'oxfordflower': 3,
    'stanforddog': 4,
    'stanfordcar': 5
}


def save_to_csv(path, dataset, acc):
    if not path.endswith('.csv'):
        path += '.csv'

    try:
        f = open(path, 'r')
    except:
        f = open(path, 'w', newline='')
        writer = csv.DictWriter(f, fieldnames=list(DATASETS.keys()))
        writer.writerow({})
        f.close()
        f = open(path, 'r')

    reader = csv.DictReader(f, fieldnames=list(DATASETS.keys()))

    # Always save to the last line
    my_dict = reader.__next__()
    f.close()

    my_dict[dataset] = acc
    f = open(path, 'w+')
    writer = csv.DictWriter(f, fieldnames=list(DATASETS.keys()))
    writer.writerow(my_dict)
    f.close()


def softmax(x, t=0.0005):
    x = [i / t for i in x]
    return np.exp(x) / np.sum(np.exp(x))
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import MultiStepLR

from .optimizer_nadam import Nadam

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        if len(self.tensors) == 2:
            y = self.tensors[1][index]
            return (x, y)

        if len(self.tensors) == 3:
            y = self.tensors[1][index]
            w = self.tensors[2][index]
            return (x, y, w)

        return (x, )

    def __len__(self):
        return self.tensors[0].size(0)

def get_optimizer(model, optimizer: str, learning_rate: float, momentum, weight_decay, additional_vars=None):
    if additional_vars is None:
        parameters = model.parameters()
    else:
        parameters = [p for p in model.parameters()] + additional_vars

    if optimizer == 'nadam':
        ret = Nadam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'adam':
        ret = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        ret = optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'adagrad':
        ret = optim.Adagrad(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'rms':
        ret = optim.RMSprop(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Not supported optimizer {optimizer}")
    return ret

def get_loss(loss_name: str, reduction='sum'):
    if 'ce' in loss_name:
        ret = nn.CrossEntropyLoss(reduction=reduction)
    elif 'mse' in loss_name:
        ret = nn.MSELoss(reduction=reduction)
    else:
        raise ValueError(f"Not supported loss {loss_name}")
    return ret

def get_scheduler(optimizer, n_epochs: int, loss_name=None):
    scheduler = MultiStepLR

    if n_epochs <= 20:
        scheduler = scheduler(optimizer, milestones=[10, 15], gamma=0.1)
    elif n_epochs <= 30:
        scheduler = scheduler(optimizer, milestones=[15, 25], gamma=0.1)
    elif n_epochs <= 40:
        scheduler = scheduler(optimizer, milestones=[20, 30], gamma=0.1)
    elif n_epochs <= 50:
        scheduler = scheduler(optimizer, milestones=[25, 40], gamma=0.1)
    elif n_epochs <= 60:
        scheduler = scheduler(optimizer, milestones=[30, 50], gamma=0.1)
    elif n_epochs <= 70:
        scheduler = scheduler(optimizer, milestones=[40, 60], gamma=0.1)
    elif n_epochs <= 80:
        scheduler = scheduler(optimizer, milestones=[30, 50, 70], gamma=0.1)
    elif n_epochs <= 120:
        scheduler = scheduler(optimizer, milestones=[40, 80, 100], gamma=0.1)
    elif n_epochs <= 160:
        scheduler = scheduler(optimizer, milestones=[40, 80, 120, 140], gamma=0.1)
    else:
        scheduler = scheduler(optimizer, milestones=[60, 100, 140, 180], gamma=0.1)
    return scheduler

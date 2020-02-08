"""
Improving the Adversarial Robustness and Interpretability of Deep Neural Networks by Regularizing their Input Gradients
"""

import torch
from torch.autograd import grad


def gradient_regularization(model, loss_fn, optimizer, x, y, lambd):
    optimizer.zero_grad()

    x.requires_grad_(True)
    outputs = model(x)
    lx = loss_fn(outputs, y)
    #x_grad = grad(lx, x, retain_graph=True, create_graph=True)[0]
    lx.backward(retain_graph=True)
    x_grad = x.grad.data#.detach()
    x.requires_grad_(False)

    regularization = x_grad**2
    regularization = torch.sum(regularization)
    #print(regularization)
    #x.grad.zero_()

    #outputs = model(x)
    #loss_natural = loss_fn(outputs, y)
    loss_natural = lx

    loss = loss_natural + lambd * regularization

    return outputs, loss

"""
https://arxiv.org/pdf/1907.02610.pdf
"""

import torch
import numpy as np
import torch.optim as optim

def locally_linearity_regularization(model,
                loss_fn, x, y, norm, optimizer,
                step_size, epsilon=0.031, perturb_steps=10,
                lambd=4.0, mu=3.0, version=None):
    model.eval()
    batch_size = len(x)

    def model_grad(x):
        x.requires_grad_(True)
        lx = loss_fn(model(x), y)
        lx.backward()
        ret = x.grad
        x.grad.zero_()
        x.requires_grad_(False)
        return ret

    def grad_dot(x, delta, model_grad):
        ret = torch.matmul(model_grad.flatten(start_dim=1), delta.flatten(start_dim=1).T)
        return torch.mean(ret)

    # calc gamma(eps, x)
    def g(x, delta: torch.Tensor, model_grad):
        ret = loss_fn(model(x+delta), y) - grad_dot(x, delta, model_grad)
        #ret = loss_fn(model(x+delta), y) - loss_fn(model(x), y) - grad_dot(x, delta)
        return torch.abs(ret)

    mg = model_grad(x)

    if norm in [2, np.inf]:
        delta = 0.001 * torch.randn(x.shape).cuda().detach()
        delta = torch.autograd.Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=step_size)

        for _ in range(perturb_steps):
            # optimize
            optimizer_delta.zero_grad()
            loss = (-1) * g(x, delta, mg)

            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=norm, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x)
            delta.data.clamp_(0, 1).sub_(x)
            delta.data.renorm_(p=norm, dim=0, maxnorm=epsilon)

        delta.requires_grad_(False)
        #x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f"[LLR] Not supported norm: {norm}")

    model.train()

    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    outputs = model(x)
    loss_natural = loss_fn(outputs, y)
    if version == "sum":
        loss = loss_natural + lambd * g(x, delta, mg) + mu * grad_dot(x, delta, mg) * len(x)
    else:
        loss = loss_natural + lambd * g(x, delta, mg) + mu * grad_dot(x, delta, mg)

    return outputs, loss

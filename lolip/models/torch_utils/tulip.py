"""
SCALEABLE INPUT GRADIENT REGULARIZATION FOR ADVERSARIAL ROBUSTNESS
https://github.com/cfinlay/tulip/blob/master/cifar10/train.py
"""

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import grad


def tulip_loss(model_fn, loss_fn, x, y, step_size=1e-0, lambd=1.):
    x.requires_grad_(True)

    outputs = model_fn(x)
    lx = loss_fn(outputs, y)
    loss = lx.sum()

    loss.backward(retain_graph=True)
    dx = x.grad.data#.detach()
    #dx = grad(loss, x, retain_graph=True)[0]

    sh = dx.shape
    x.requires_grad_(False)

    # v is the finite difference direction.
    # For example, if norm=='L2', v is the gradient of the loss wrt inputs
    v = dx.view(sh[0], -1)
    #Nb, Nd = v.shape

    nv = v.norm(2, dim=-1, keepdim=True)
    nz = nv.view(-1) > 0
    v[nz] = v[nz].div(nv[nz])

    v = v.view(sh)
    xf = x + step_size * v

    mf = model_fn(xf)
    lf = loss_fn(mf, y)
    #if args.fd_order=='O2':
    #    xb = x - step_size * v
    #    mb = model_fn(xb)
    #    lb = loss_fn(mb, y)
    #    H = 2 * step_size
    #else:
    H = step_size
    lb = lx
    dl = (lf-lb)/H # This is the finite difference approximation
                    # of the directional derivative of the loss

    dl2 = dl.pow(2)
    tik_penalty = dl2.sum() / 2

    loss = loss + lambd * tik_penalty

    return outputs, loss
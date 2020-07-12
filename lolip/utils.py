import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils


def local_lip(model, x, xp, top_norm, btm_norm, reduction='mean'):
    model.eval()
    down = torch.flatten(x - xp, start_dim=1)
    if top_norm == "kl":
        criterion_kl = nn.KLDivLoss(reduction='none')
        top = criterion_kl(F.log_softmax(model(xp), dim=1),
                           F.softmax(model(x), dim=1))
        ret = torch.sum(top, dim=1) / torch.norm(down + 1e-6, dim=1, p=btm_norm)
    else:
        top = torch.flatten(model(x), start_dim=1) - torch.flatten(model(xp), start_dim=1)
        ret = torch.norm(top, dim=1, p=top_norm) / torch.norm(down + 1e-6, dim=1, p=btm_norm)

    if reduction == 'mean':
        return torch.mean(ret)
    elif reduction == 'sum':
        return torch.sum(ret)
    else:
        raise ValueError(f"Not supported reduction: {reduction}")

def preprocess_x(x):
    return torch.from_numpy(x.transpose(0, 3, 1, 2)).float()

def estimate_local_lip_v2(model, X, top_norm, btm_norm,
        batch_size=16, perturb_steps=10, step_size=0.003, epsilon=0.01,
        device="cuda"):
    model.eval()
    if isinstance(X, torch.utils.data.Dataset):
        loader = torch.utils.data.DataLoader(X,
            batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        dataset = data_utils.TensorDataset(preprocess_x(X))
        loader = torch.utils.data.DataLoader(dataset,
            batch_size=batch_size, shuffle=False, num_workers=2)

    total_loss = 0.
    ret = []
    for x in loader:
        x = x[0]
        x = x.to(device)
        # generate adversarial example
        if btm_norm in [1, 2, np.inf]:
            x_adv = x + 0.001 * torch.randn(x.shape).to(device)

            # Setup optimizers
            optimizer = optim.SGD([x_adv], lr=step_size)

            for _ in range(perturb_steps):
                x_adv.requires_grad_(True)
                optimizer.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * local_lip(model, x, x_adv, top_norm, btm_norm)
                loss.backward()
                # renorming gradient
                eta = step_size * x_adv.grad.data.sign().detach()
                x_adv = x_adv.data.detach() + eta.detach()
                eta = torch.clamp(x_adv.data - x.data, -epsilon, epsilon)
                x_adv = x.data.detach() + eta.detach()
                x_adv = torch.clamp(x_adv, 0, 1.0)
        else:
            raise ValueError(f"Unsupported norm {btm_norm}")

        total_loss += local_lip(model, x, x_adv, top_norm, btm_norm, reduction='sum').item()

        ret.append(x_adv.detach().cpu().numpy().transpose(0, 2, 3, 1))
    ret_v = total_loss / len(X)
    return ret_v, np.concatenate(ret, axis=0)

def estimate_local_mse(model, X,
        batch_size=16, perturb_steps=10, step_size=0.003, epsilon=0.01,
        device="cuda"):
    model.eval()
    dataset = data_utils.TensorDataset(preprocess_x(X))
    loader = torch.utils.data.DataLoader(dataset,
        batch_size=batch_size, shuffle=False, num_workers=2)
    
    total_loss = 0.
    ret = []
    for [x] in loader:
        x = x.to(device)
        # generate adversarial example
        x_adv = x + 0.001 * torch.randn(x.shape).to(device)

        # Setup optimizers
        optimizer = optim.SGD([x_adv], lr=step_size)

        for _ in range(perturb_steps):
            x_adv.requires_grad_(True)
            optimizer.zero_grad()
            with torch.enable_grad():
                loss = nn.MSELoss()(model(x_adv), model(x))
            loss.backward()
            # renorming gradient
            eta = step_size * x_adv.grad.data.sign().detach()
            x_adv = x_adv.data.detach() + eta.detach()
            eta = torch.clamp(x_adv.data - x.data, -epsilon, epsilon)
            x_adv = x.data.detach() + eta.detach()
            x_adv = torch.clamp(x_adv, 0, 1.0)

        loss_adv = nn.MSELoss(reduction="sum")(model(x_adv), model(x))
        loss_adv = torch.sqrt(loss_adv.detach()).sum()
        total_loss += loss_adv.item()

        ret.append(x_adv.detach().cpu().numpy().transpose(0, 2, 3, 1))
    ret_v = total_loss / len(X)
    return ret_v, np.concatenate(ret, axis=0)

#def estimate_local_lip(model, X, top_norm, btm_norm,
#        batch_size=16, perturb_steps=10, step_size=0.003, epsilon=0.01, device="cuda"):
#    model.eval()
#    dataset = data_utils.TensorDataset(preprocess_x(X))
#    loader = torch.utils.data.DataLoader(dataset,
#        batch_size=batch_size, shuffle=False, num_workers=2)
#    
#    total_loss = 0.
#    ret = []
#    for [x] in loader:
#        x = x.to(device)
#        # generate adversarial example
#        if btm_norm in [2, np.inf]:
#            delta = 0.001 * torch.randn(x.shape).to(device).detach()
#            delta = torch.autograd.Variable(delta.data, requires_grad=True)
#
#            # Setup optimizers
#            optimizer = optim.SGD([delta], lr=step_size)
#
#            for _ in range(perturb_steps):
#                x_adv = x + delta
#
#                # optimize
#                optimizer.zero_grad()
#                with torch.enable_grad():
#                    loss = (-1) * local_lip(model, x, x_adv, top_norm, btm_norm)
#                loss.backward()
#                # renorming gradient
#                grad_norms = delta.grad.view(len(x), -1).norm(p=btm_norm, dim=1)
#                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
#                # avoid nan or inf if gradient is 0
#                if (grad_norms == 0).any():
#                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
#                optimizer.step()
#
#                # projection
#                delta.data.add_(x)
#                delta.data.clamp_(0, 1).sub_(x)
#                delta.data.renorm_(p=btm_norm, dim=0, maxnorm=epsilon)
#            x_adv = x + delta
#        else:
#            raise ValueError(f"Unsupported norm {btm_norm}")
#        ret.append(x_adv.detach().cpu().numpy().transpose(0, 2, 3, 1))
#    return np.concatenate(ret, axis=0)

import copy
from functools import partial
import gc

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.functional import softmax
from tqdm import tqdm

from ..base import AttackModel
from .projected_gradient_descent import projected_gradient_descent

class MultiTarget(AttackModel):

  def __init__(self, model_fn, eps, eps_iter, nb_iter, norm, n_classes,
               loss_fn=None, clip_min=None, clip_max=None, y=None,
               batch_size=128, rand_init=True, rand_minmax=None):
    self.n_classes = n_classes
    self.model_fn = model_fn
    self.eps = eps
    self.batch_size = batch_size
    self.attack_fn = partial(projected_gradient_descent, model_fn=model_fn,
      eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=norm, loss_fn=loss_fn,
      clip_min=clip_min, clip_max=clip_max, targeted=True, rand_init=rand_init,
      rand_minmax=rand_minmax)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

  def _preprocess_x(self, X):
    return torch.from_numpy(X.transpose(0, 3, 1, 2)).float()

  def perturb_ds(self, ds, eps=None):
    loader = torch.utils.data.DataLoader(ds,
        batch_size=self.batch_size, shuffle=False, num_workers=8)

    ret = []
    for [x, y] in tqdm(loader, desc="Attacking (MT)"):
      x = x.to(self.device)

      pred = self.model_fn(x)
      pred = softmax(pred, dim=1).detach().cpu().numpy()
      pred = np.array([pred[i, yi] for i, yi in enumerate(y)])

      r = []
      scores = []
      for j in range(self.n_classes):
        yp = j * torch.ones(len(x)).long().to(self.device)
        adv_x = self.attack_fn(x=x, y=yp).detach()
        adv_pred = softmax(self.model_fn(adv_x), dim=1)[:, j]
        scores.append(adv_pred.detach().cpu().numpy() - pred)
        r.append(adv_x.cpu().numpy())
      scores = np.array(scores)
      #idx = scores.argmax(axis=0)
      #r = np.array(r)
      #for i in range(len(x)):
      #  ret.append(r[idx[i], i])


      idx_top2 = np.argsort(scores, axis=0)[-2:, :]
      y = y.to("cpu").numpy()
      r = np.array(r)
      for i in range(len(x)):
        if idx_top2[1, i] != y[i]:
          ret.append(r[idx_top2[1, i], i])
        else:
          ret.append(r[idx_top2[0, i], i])

      del scores
      del r
      gc.collect()

    return np.array(ret).transpose(0, 2, 3, 1)

  def perturb(self, X, y=None, eps=None):
    """
    y: correct: label
    """
    #self.model_fn.eval()
    dataset = torch.utils.data.TensorDataset(
      self._preprocess_x(X), torch.from_numpy(y).long())
    #loader = torch.utils.data.DataLoader(dataset,
    #  batch_size=self.batch_size, shuffle=False, num_workers=1)

    return self.perturb_ds(dataset, eps=eps)

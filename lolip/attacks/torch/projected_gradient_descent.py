"""The Projected Gradient Descent attack."""
from functools import partial

import numpy as np
import torch
import torch.utils.data as data_utils
from tqdm import tqdm

from .fast_gradient_method import fast_gradient_method
from ..base import AttackModel


def clip_eta(eta, norm, eps):
  """
  PyTorch implementation of the clip_eta in utils_tf.
  :param eta: Tensor
  :param norm: np.inf, 1, or 2
  :param eps: float
  """
  if norm not in [np.inf, 1, 2]:
    raise ValueError('norm must be np.inf, 1, or 2.')

  avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
  reduc_ind = list(range(1, len(eta.size())))
  if norm == np.inf:
    eta = torch.clamp(eta, -eps, eps)
  else:
    if norm == 1:
      raise NotImplementedError("L1 clip is not implemented.")
      norm = torch.max(
          avoid_zero_div,
          torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True)
      )
    elif norm == 2:
      norm = torch.sqrt(torch.max(
          avoid_zero_div,
          torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
      ))
    factor = torch.min(
        torch.tensor(1., dtype=eta.dtype, device=eta.device),
        eps / norm
        )
    eta = eta * factor
  return eta

class ProjectedGradientDescent(AttackModel):

  def __init__(self, model_fn, eps, eps_iter, nb_iter, norm, loss_fn=None,
               clip_min=None, clip_max=None, y=None, targeted=False,
               batch_size=16, rand_init=True, rand_minmax=None, device=None,
               preprocess_img=True):
    self.model_fn = model_fn
    self.eps = eps
    self.batch_size = batch_size
    self.preprocess_img = preprocess_img
    self.attack_fn = partial(projected_gradient_descent, model_fn=model_fn,
      eps=eps, eps_iter=eps_iter, nb_iter=nb_iter, norm=norm, loss_fn=loss_fn,
      clip_min=clip_min, clip_max=clip_max, targeted=targeted, rand_init=rand_init,
      rand_minmax=rand_minmax)
    if device is None:
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
      self.device=device

  def _preprocess_x(self, X):
    if self.preprocess_img == True:
      return torch.from_numpy(X.transpose(0, 3, 1, 2)).float()
    else:
      return torch.from_numpy(X).float()

  def perturb(self, X, y, eps=None):
    dataset = data_utils.TensorDataset(self._preprocess_x(X), torch.from_numpy(y).long())
    #loader = torch.utils.data.DataLoader(dataset,
    #    batch_size=self.batch_size, shuffle=False, num_workers=2)

    return self.perturb_ds(dataset, eps=eps)

  def perturb_ds(self, ds, eps=None):
    loader = torch.utils.data.DataLoader(ds,
        batch_size=self.batch_size, shuffle=False, num_workers=2)

    ret = []
    for [x, y] in tqdm(loader, desc="Attacking (PGD)"):
      x, y = x.to(self.device), y.to(self.device)
      ret.append(self.attack_fn(x=x, y=y).detach().cpu().numpy())
    if self.preprocess_img == True:
      return np.concatenate(ret, axis=0).transpose(0, 2, 3, 1)
    else:
      return np.concatenate(ret, axis=0)


def projected_gradient_descent(model_fn, x, eps, eps_iter, nb_iter, norm, loss_fn=None,
                               clip_min=None, clip_max=None, y=None, targeted=False,
                               rand_init=True, rand_minmax=None, sanity_checks=True):
  """
  This class implements either the Basic Iterative Method
  (Kurakin et al. 2016) when rand_init is set to False. or the
  Madry et al. (2017) method if rand_init is set to True.
  Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
  Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
  :param model_fn: a callable that takes an input tensor and returns the model logits.
  :param x: input tensor.
  :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
  :param eps_iter: step size for each attack iteration
  :param nb_iter: Number of attack iterations.
  :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
  :param clip_min: (optional) float. Minimum float value for adversarial example components.
  :param clip_max: (optional) float. Maximum float value for adversarial example components.
  :param y: (optional) Tensor with true labels. If targeted is true, then provide the
            target label. Otherwise, only provide this parameter if you'd like to use true
            labels when crafting adversarial samples. Otherwise, model predictions are used
            as labels to avoid the "label leaking" effect (explained in this paper:
            https://arxiv.org/abs/1611.01236). Default is None.
  :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being more like y.
  :param rand_init: (optional) bool. Whether to start the attack from a randomly perturbed x.
  :param rand_minmax: (optional) bool. Support of the continuous uniform distribution from
            which the random perturbation on x was drawn. Effective only when rand_init is
            True. Default equals to eps.
  :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
            memory or for unit tests that intentionally pass strange input)
  :return: a tensor for the adversarial example
  """
  if norm == 1:
    raise NotImplementedError("It's not clear that FGM is a good inner loop"
                              " step for PGD when norm=1, because norm=1 FGM "
                              " changes only one pixel at a time. We need "
                              " to rigorously test a strong norm=1 PGD "
                              "before enabling this feature.")
  if norm not in [np.inf, 2]:
    raise ValueError("Norm order must be either np.inf or 2.")
  if eps < 0:
    raise ValueError(
        "eps must be greater than or equal to 0, got {} instead".format(eps))
  if eps == 0:
    return x
  if eps_iter < 0:
    raise ValueError(
        "eps_iter must be greater than or equal to 0, got {} instead".format(eps_iter))
  if eps_iter == 0:
    return x

  assert eps_iter <= eps, (eps_iter, eps)
  if clip_min is not None and clip_max is not None:
    if clip_min > clip_max:
      raise ValueError(
          "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
              clip_min, clip_max))

  asserts = []

  # If a data range was specified, check that the input was in that range
  if clip_min is not None:
    assert_ge = torch.all(torch.ge(x, torch.tensor(clip_min, device=x.device, dtype=x.dtype)))
    asserts.append(assert_ge)

  if clip_max is not None:
    assert_le = torch.all(torch.le(x, torch.tensor(clip_max, device=x.device, dtype=x.dtype)))
    asserts.append(assert_le)

  # Initialize loop variables
  if rand_init:
    if rand_minmax is None:
      rand_minmax = eps
    eta = torch.zeros_like(x).uniform_(-rand_minmax, rand_minmax)
  else:
    eta = torch.zeros_like(x)

  # Clip eta
  eta = clip_eta(eta, norm, eps)
  adv_x = x + eta
  if clip_min is not None or clip_max is not None:
    adv_x = torch.clamp(adv_x, clip_min, clip_max)

  if y is None:
    # Using model predictions as ground truth to avoid label leaking
    _, y = torch.max(model_fn(x), 1)

  i = 0
  while i < nb_iter:
    adv_x = fast_gradient_method(model_fn, adv_x, eps_iter, norm, loss_fn=loss_fn,
                                 clip_min=clip_min, clip_max=clip_max, y=y, targeted=targeted)

    # Clipping perturbation eta to norm norm ball
    eta = adv_x - x
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta

    # Redo the clipping.
    # FGM already did it, but subtracting and re-adding eta can add some
    # small numerical error.
    if clip_min is not None or clip_max is not None:
      adv_x = torch.clamp(adv_x, clip_min, clip_max)
    i += 1

  asserts.append(eps_iter <= eps)
  if norm == np.inf and clip_min is not None:
    # TODO necessary to cast clip_min and clip_max to x.dtype?
    asserts.append(eps + clip_min <= clip_max)

  if sanity_checks:
    assert np.all(asserts), asserts
  return adv_x

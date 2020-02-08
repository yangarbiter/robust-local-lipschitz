from abc import abstractmethod

import numpy as np


class AttackModel():

    def __init__(self, norm):
        self.norm = norm
        super().__init__()

    def _pert_with_eps_constraint(self, pert_X, eps):
        if isinstance(eps, list):
            rret = []
            norms = np.linalg.norm(pert_X, axis=1, ord=self.norm)
            for ep in eps:
                t = np.copy(pert_X)
                t[norms > ep, :] = 0
                rret.append(t)
            return rret
        elif eps is not None:
            pert_X[np.linalg.norm(pert_X, axis=1, ord=self.norm) > eps, :] = 0
            return pert_X
        else:
            return pert_X

    @abstractmethod
    def perturb_ds(self, ds, eps):
        pass

    @abstractmethod
    def perturb(self, X, y, eps):
        pass

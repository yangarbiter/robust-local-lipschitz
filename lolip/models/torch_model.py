import gc
import os
from functools import partial
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.datasets import VisionDataset

import numpy as np
from sklearn.base import BaseEstimator
from .torch_utils import get_optimizer, get_loss, get_scheduler, CustomTensorDataset
from .torch_utils.archs import *
from ..attacks.torch.projected_gradient_descent import projected_gradient_descent
from .torch_utils.trades import trades_loss
from .torch_utils.llr import locally_linearity_regularization
from .torch_utils.tulip import tulip_loss
from .torch_utils import data_augs

DEBUG = int(os.getenv("DEBUG", 0))

class TorchModel(BaseEstimator):
    def __init__(self, lbl_enc, n_features, n_classes, loss_name='ce',
                n_channels=None, learning_rate=1e-4, momentum=0.0, batch_size=256,
                epochs=20, optimizer='sgd', architecture='arch_001', random_state=None,
                weight_decay=0.0, callbacks=None, train_type=None, eps:float=0.1, norm=np.inf,
                multigpu=False, dataaug=None, device=None, num_workers=4, trn_log_callbacks=None):
        print(f'lr: {learning_rate}, opt: {optimizer}, loss: {loss_name}, '
              f'arch: {architecture}, dataaug: {dataaug}, batch_size: {batch_size}, '
              f'momentum: {momentum}, weight_decay: {weight_decay}, eps: {eps}, '
              f'epochs: {epochs}')
        self.num_workers = num_workers
        self.n_features = n_features
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.epochs = epochs
        self.lbl_enc = lbl_enc
        self.loss_name = loss_name
        self.dataaug = dataaug
        self.trn_log_callbacks = trn_log_callbacks

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        arch_fn = globals()[self.architecture]
        if 'n_features' in inspect.getfullargspec(arch_fn)[0]:
            model = arch_fn(n_features=n_features, n_classes=self.n_classes, n_channels=n_channels)
        else:
            model = arch_fn(n_classes=self.n_classes, n_channels=n_channels)
        if self.device == 'cuda':
            model = model.cuda()

        self.multigpu = multigpu
        if self.multigpu:
            model = torch.nn.DataParallel(model, device_ids=[0, 1])

        self.optimizer = get_optimizer(model, optimizer, learning_rate, momentum, weight_decay)
        self.model = model

        self.random_state = random_state

        self.tst_ds = None
        self.start_epoch = 1

        ### Attack ####
        self.eps = eps
        self.norm = norm
        ###############

    def _get_dataset(self, X, y=None):
        X = self._preprocess_x(X)

        if self.dataaug is None:
            transform = None
        else:
            if y is None:
                transform = getattr(data_augs, self.dataaug)()[1]
            else:
                transform = getattr(data_augs, self.dataaug)()[0]

        if y is None:
            return CustomTensorDataset((torch.from_numpy(X).float(), ), transform=transform)
        dataset = CustomTensorDataset(
            (torch.from_numpy(X).float(), torch.from_numpy(y).long()), transform=transform)
        return dataset

    def _preprocess_x(self, X):
        if len(X.shape) ==4:
            return X.transpose(0, 3, 1, 2)
        else:
            return X

    def fit_dataset(self, dataset, verbose=None):
        if verbose is None:
            verbose = 0 if not DEBUG else 1
        log_interval = 1

        history = []
        if 'tulip' in self.loss_name:
            loss_fn = get_loss(self.loss_name, reduction="none")
        else:
            loss_fn = get_loss(self.loss_name, reduction="sum")
        scheduler = get_scheduler(self.optimizer, n_epochs=self.epochs, loss_name=self.loss_name)

        train_loader = torch.utils.data.DataLoader(dataset,
            batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        test_loader = None
        if self.tst_ds is not None:
            if isinstance(self.tst_ds, VisionDataset):
                ts_dataset = self.tst_ds
            else:
                tstX, tsty = self.tst_ds
                ts_dataset = self._get_dataset(tstX, tsty)

            test_loader = torch.utils.data.DataLoader(ts_dataset,
                batch_size=32, shuffle=False, num_workers=self.num_workers)

        for epoch in range(self.start_epoch, self.epochs+1):
            train_loss = 0.
            train_acc = 0.
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
                self.model.train()
                x, y = x.to(self.device), y.to(self.device)

                if 'trades' in self.loss_name:
                    if 'trades10' in self.loss_name:
                        beta = 10.0
                    elif 'trades20' in self.loss_name:
                        beta = 20.0
                    elif 'trades6' in self.loss_name:
                        beta = 6.0
                    elif 'trades3' in self.loss_name:
                        beta = 3.0
                    elif 'trades.5' in self.loss_name:
                        beta = 0.5
                    else:
                        beta = 1.0

                    if 'K20' in self.loss_name:
                        steps = 20
                    else:
                        steps = 10

                    version = None
                    if 'strades' in self.loss_name:
                        version = "sum"

                    outputs, loss = trades_loss(
                        self.model, loss_fn, x, y, norm=self.norm, optimizer=self.optimizer,
                        step_size=self.eps*2/steps, epsilon=self.eps, perturb_steps=steps, beta=beta,
                        version=version, device=self.device
                    )
                elif 'tulip' in self.loss_name:
                    if 'tulipem1' in self.loss_name:
                        lambd = 1e-1
                    elif 'tulipem2' in self.loss_name:
                        lambd = 1e-2
                    elif 'tulip0' in self.loss_name:
                        lambd = 0
                    else:
                        lambd = 1

                    if 'ssem1' in self.loss_name:
                        step_size = 1e-1
                    elif 'ssem2' in self.loss_name:
                        step_size = 1e-2
                    elif 'ssem3' in self.loss_name:
                        step_size = 1e-3
                    else:
                        step_size = 1e-0
                    self.optimizer.zero_grad()
                    outputs, loss = tulip_loss(self.model, loss_fn, x, y,
                            step_size=step_size, lambd=lambd)
                elif 'llr' in self.loss_name:
                    if 'llr65' in self.loss_name:
                        lambd, mu = 6.0, 5.0
                    elif 'llr36' in self.loss_name:
                        lambd, mu = 3.0, 6.0
                    else:
                        lambd, mu = 4.0, 3.0

                    if 'sllr' in self.loss_name:
                        version = "sum"
                    else:
                        version = None

                    epsilon = self.eps

                    outputs, loss = locally_linearity_regularization(
                        self.model, loss_fn, x, y, norm=self.norm, optimizer=self.optimizer,
                        step_size=epsilon/2, epsilon=epsilon, perturb_steps=2,
                        lambd=lambd, mu=mu, version=version
                    )
                elif 'advbeta' in self.loss_name:
                    self.model.train()
                    advx = projected_gradient_descent(self.model, x, y=y,
                            clip_min=0, clip_max=1,
                            eps_iter=self.eps/5,
                            eps=self.eps, norm=self.norm, nb_iter=10)

                    if 'beta.5' in self.loss_name:
                        beta = 0.5
                    elif 'beta8' in self.loss_name:
                        beta = 8.
                    elif 'beta4' in self.loss_name:
                        beta = 4.
                    elif 'beta2' in self.loss_name:
                        beta = 2.
                    else:
                        beta = 1.

                    self.optimizer.zero_grad()
                    outputs = self.model(advx)
                    adv_loss = loss_fn(outputs, y)
                    loss = loss_fn(self.model(x), y) + beta * adv_loss
                else:
                    self.model.train()
                    if 'adv' in self.loss_name:
                        x = projected_gradient_descent(self.model, x, y=y,
                                clip_min=0, clip_max=1,
                                eps_iter=self.eps/5,
                                eps=self.eps, norm=self.norm, nb_iter=10)
                    self.optimizer.zero_grad()
                    outputs = self.model(x)
                    loss = loss_fn(outputs, y)

                loss.backward()
                self.optimizer.step()

                if (epoch - 1) % log_interval == 0:
                    self.model.eval()
                    train_loss += loss.item()
                    train_acc += (outputs.argmax(dim=1)==y).sum().float().item()

                    if self.trn_log_callbacks is not None:
                        for callback_fn in self.trn_log_callbacks:
                            callback_fn(self, x, y, loss_fn)

            current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            scheduler.step()
            self.start_epoch = epoch

            if (epoch - 1) % log_interval == 0:
                print(f"current LR: {current_lr}")
                self.model.eval()
                history.append({
                    'epoch': epoch,
                    'lr': current_lr,
                    'trn_loss': train_loss / len(train_loader.dataset),
                    'trn_acc': train_acc / len(train_loader.dataset),
                })
                print('epoch: {}/{}, train loss: {:.3f}, train acc: {:.3f}'.format(
                    epoch, self.epochs, history[-1]['trn_loss'], history[-1]['trn_acc']))

                if self.tst_ds is not None:
                    tst_loss, tst_acc = 0., 0.
                    with torch.no_grad():
                        for tx, ty in test_loader:
                            tx, ty = tx.to(self.device), ty.to(self.device)
                            outputs = self.model(tx)
                            if loss_fn.reduction == 'none':
                                loss = torch.sum(loss_fn(outputs, ty))
                            else:
                                loss = loss_fn(outputs, ty)
                            tst_loss += loss.item()
                            tst_acc += (outputs.argmax(dim=1)==ty).sum().float().item()
                    history[-1]['tst_loss'] = tst_loss / len(test_loader.dataset)
                    history[-1]['tst_acc'] = tst_acc / len(test_loader.dataset)
                    print('             test loss: {:.3f}, test acc: {:.3f}'.format(
                          history[-1]['tst_loss'], history[-1]['tst_acc']))

        if test_loader is not None:
            del test_loader
        del train_loader
        gc.collect()

        return history

    def fit(self, X, y, sample_weight=None, verbose=None):
        dataset = self._get_dataset(X, y)
        return self.fit_dataset(dataset, verbose=verbose)

    def _prep_pred(self, X):
        if isinstance(X, VisionDataset):
            dataset = X
        else:
            if self.dataaug is None:
                transform = None
            else:
                transform = getattr(data_augs, self.dataaug)()[1]
            X = self._preprocess_x(X)
            self.model.eval()
            dataset = CustomTensorDataset((torch.from_numpy(X).float(), ), transform=transform)
        loader = torch.utils.data.DataLoader(dataset,
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader

    def predict_ds(self, ds):
        loader = torch.utils.data.DataLoader(ds,
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        ret = []
        for x in loader:
            x = x[0]
            ret.append(self.model(x.to(self.device)).argmax(1).cpu().numpy())
        del loader
        return np.concatenate(ret)

    def predict(self, X):
        loader = self._prep_pred(X)
        ret = []
        for [x] in loader:
            ret.append(self.model(x.to(self.device)).argmax(1).cpu().numpy())
        del loader
        return np.concatenate(ret)

    def predict_proba(self, X):
        loader = self._prep_pred(X)
        ret = []
        for [x] in loader:
            output = F.softmax(self.model(x.to(self.device)).detach())
            ret.append(output.cpu().numpy())
        del loader
        return np.concatenate(ret, axis=0)

    def predict_real(self, X):
        loader = self._prep_pred(X)
        ret = []
        for [x] in loader:
            ret.append(self.model(x.to(self.device)).detach().cpu().numpy())
        del loader
        return np.concatenate(ret, axis=0)

    def save(self, path):
        if self.multigpu:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        torch.save({
            'epoch': self.start_epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path % self.start_epoch)

    def load(self, path):
        loaded = torch.load(path)
        if 'epoch' in loaded:
            self.start_epoch = loaded['epoch']
            self.model.load_state_dict(loaded['model_state_dict'])
            self.optimizer.load_state_dict(loaded['optimizer_state_dict'])
        else:
            self.model.load_state_dict(loaded)
        self.model.eval()

import os

from autovar.base import RegisteringChoiceType, VariableClass, register_var
import numpy as np


DEBUG = int(os.getenv("DEBUG", 0))

def get_hyper(name, loss, arch, dataset_name):
    ret = {}
    ret['optimizer'] = 'sgd'

    if 'CNN' in arch and ('mnist' in dataset_name or 'fashion' in dataset_name):
        ret['epochs'] = 160
        ret['learning_rate'] = 1e-4
        ret['momentum'] = 0.9
        ret['batch_size'] = 64

    elif 'resImgnet112v3' in dataset_name:
        ret['epochs'] = 70
        ret['learning_rate'] = 1e-2
        ret['batch_size'] = 64

    elif 'ResNet' in arch or 'WRN' in arch:
        if 'svhn' in dataset_name:
            ret['epochs'] = 60
        elif 'cifar' in dataset_name:
            ret['epochs'] = 120
        else:
            ret['epochs'] = 200

        if 'adv' in loss:
            ret['learning_rate'] = 1e-3
        elif 'llr' in loss:
            ret['learning_rate'] = 1e-3
        else:
            ret['learning_rate'] = 1e-2
        ret['batch_size'] = 64
    else:
        ret['epochs'] = 500
        ret['learning_rate'] = 1e-1
        ret['batch_size'] = 128

    if DEBUG:
        ret['epochs'] = 2

    if name is not None:
        if 'nadam' in name:
            ret['optimizer'] = 'nadam'
        elif 'adam' in name:
            ret['optimizer'] = 'adam'

        if 'wd.9' in name:
            ret['weight_decay'] = 0.9

        if 'mo.9' in name:
            ret['momentum'] = 0.9
        elif 'mo0' in name:
            ret['momentum'] = 0

        if 'lrem4' in name:
            ret['learning_rate'] = 1e-4
        elif 'lrem3' in name:
            ret['learning_rate'] = 1e-3
        elif 'lrem2' in name:
            ret['learning_rate'] = 1e-2
        elif 'lrem1' in name:
            ret['learning_rate'] = 1e-1

        if 'bs256' in name:
            ret['batch_size'] = 256
        elif 'bs128' in name:
            ret['batch_size'] = 128
        elif 'bs32' in name:
            ret['batch_size'] = 32
        elif 'bs16' in name:
            ret['batch_size'] = 16

        if 'ep20' in name:
            ret['epochs'] = 20
        elif 'ep2' in name:
            ret['epochs'] = 2
        elif 'ep30' in name:
            ret['epochs'] = 30
        elif 'ep40' in name:
            ret['epochs'] = 40
        elif 'ep50' in name:
            ret['epochs'] = 50
        elif 'ep60' in name:
            ret['epochs'] = 60
        elif 'ep70' in name:
            ret['epochs'] = 70

    return ret

class ModelVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Model Variable Class"""
    var_name = "model"

    @register_var(argument=r'(?P<dataaug>[a-zA-Z0-9]+-)?(?P<loss>[a-zA-Z0-9\.]+)-tor-(?P<arch>[a-zA-Z0-9_]+)(?P<hyper>-[a-zA-Z0-9\.]+)?')
    @staticmethod
    def torch_model(auto_var, inter_var, dataaug, loss, arch, hyper, trnX, trny, n_channels, multigpu=False, trn_log_callbacks=None):
        from .torch_model import TorchModel

        dataaug = dataaug[:-1] if dataaug else None

        n_features = trnX.shape[1:]
        n_classes = len(np.unique(trny))
        dataset_name = auto_var.get_variable_name('dataset')

        params: dict = get_hyper(hyper, loss, arch, dataset_name)
        params['eps'] = auto_var.get_var("eps")
        params['norm'] = auto_var.get_var("norm")
        params['loss_name'] = loss
        params['n_features'] = n_features
        params['n_classes'] = n_classes
        params['train_type'] = None
        params['architecture'] = arch
        params['multigpu'] = multigpu
        params['n_channels'] = n_channels
        params['dataaug'] = dataaug

        model = TorchModel(
            lbl_enc=inter_var['lbl_enc'],
            **params,
        )
        return model

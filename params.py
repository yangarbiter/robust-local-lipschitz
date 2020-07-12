
from utils import ExpExperiments

random_seed = list(range(1))

class mnistLip(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "mnist"
        cls.experiment_fn = 'experiment01'
        grid_params = []
        arch = "CNN001"
        grid_params.append({
            'dataset': ['mnist'],
            'model': [
                f'advbeta2ce-tor-{arch}',
                f'advbetace-tor-{arch}',
                f'advbeta.5ce-tor-{arch}',
                f'strades6ce-tor-{arch}',
                f'strades3ce-tor-{arch}',
                f'stradesce-tor-{arch}',
                f'tulipce-tor-{arch}',
                f'ce-tor-{arch}',
                f'advce-tor-{arch}',
                f'sllrce-tor-{arch}',
            ],
            'eps': [0.1],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        arch = "CNN002"
        grid_params.append({
            'dataset': ['mnist'],
            'model': [
                f'advbeta2ce-tor-{arch}',
                f'advbetace-tor-{arch}',
                f'advbeta.5ce-tor-{arch}',
                f'strades6ce-tor-{arch}',
                f'strades3ce-tor-{arch}',
                f'stradesce-tor-{arch}',
                f'tulipce-tor-{arch}',
                f'ce-tor-{arch}',
                f'advce-tor-{arch}',
                f'sllrce-tor-{arch}',
            ],
            'eps': [0.1],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)

class mnistMT(mnistLip):
    def __new__(cls, *args, **kwargs):
        return mnistLip.__new__(cls, *args, **kwargs)
    def __init__(self):
        self.experiment_fn = 'experiment01'
        self.grid_params[0]['attack'].append('multitarget')
        self.grid_params[1]['attack'].append('multitarget')

class svhnLip(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "svhn"
        cls.experiment_fn = 'experiment01'
        grid_params = []
        grid_params.append({
            'dataset': ['svhn'],
            'model': [
                'ce-tor-WRN_40_10',
                'advbeta2ce-tor-WRN_40_10',
                'advbetace-tor-WRN_40_10',
                'advbeta.5ce-tor-WRN_40_10',
                'stradesce-tor-WRN_40_10',
                'strades3ce-tor-WRN_40_10',
                'strades6ce-tor-WRN_40_10',
                'tulipce-tor-WRN_40_10',
                #'advce-tor-WRN_40_10-lrem2',
                'advce-tor-WRN_40_10',
                'sllrce-tor-WRN_40_10',

                #'aug01-ce-tor-WRN_40_10',
                #'aug01-advbeta2ce-tor-WRN_40_10',
                #'aug01-advbeta2ce-tor-WRN_40_10-lrem2',
                #'aug01-advce-tor-WRN_40_10',
                #'aug01-advce-tor-WRN_40_10-lrem2',
                #'aug01-strades6ce-tor-WRN_40_10',

                'ce-tor-WRN_40_10_drop50',
                'advbeta2ce-tor-WRN_40_10_drop50',
                'advce-tor-WRN_40_10_drop50',
                'strades6ce-tor-WRN_40_10_drop50',
                'strades3ce-tor-WRN_40_10_drop50',

                #'aug01-ce-tor-WRN_40_10_drop50',
                #'aug01-advbeta2ce-tor-WRN_40_10_drop50',
                #'aug01-advce-tor-WRN_40_10_drop50',
                #'aug01-strades6ce-tor-WRN_40_10_drop50',
                #'aug01-strades3ce-tor-WRN_40_10_drop50',
            ],
            'eps': [0.031],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)

class svhnMT(svhnLip):
    def __new__(cls, *args, **kwargs):
        return svhnLip.__new__(cls, *args, **kwargs)
    def __init__(self):
        self.experiment_fn = 'experiment01'
        self.grid_params[0]['attack'].append('multitarget')


class cifarLip(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "cifar"
        cls.experiment_fn = 'experiment01'
        grid_params = []
        grid_params.append({
            'dataset': ['cifar10'],
            'model': [
                'ce-tor-WRN_40_10',
                #'advbeta2ce-tor-WRN_40_10-lrem2',
                #'advbetace-tor-WRN_40_10-lrem2',
                #'advbeta.5ce-tor-WRN_40_10-lrem2',
                'advbeta2ce-tor-WRN_40_10',
                'advbetace-tor-WRN_40_10',
                'advbeta.5ce-tor-WRN_40_10',
                'tulipce-tor-WRN_40_10',
                'stradesce-tor-WRN_40_10',
                'strades3ce-tor-WRN_40_10',
                'strades6ce-tor-WRN_40_10',
                'advce-tor-WRN_40_10-lrem2',
                'sllrce-tor-WRN_40_10-lrem2',
            ],
            'eps': [0.031],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        grid_params.append({
            'dataset': ['cifar10'],
            'model': [
                'aug01-strades3ce-tor-WRN_40_10_drop50',

                'aug01-ce-tor-WRN_40_10_drop20',
                'aug01-strades3ce-tor-WRN_40_10_drop20',
                'aug01-strades6ce-tor-WRN_40_10_drop20',
                'aug01-advce-tor-WRN_40_10_drop20-lrem2',
                'aug01-advbeta2ce-tor-WRN_40_10_drop20',
                'aug01-advbeta2ce-tor-WRN_40_10_drop20-lrem2',

                #'aug01-ce-tor-WRN_40_10_drop50',
                #'aug01-strades6ce-tor-WRN_40_10_drop50',
                #'aug01-advce-tor-WRN_40_10_drop50-lrem2',
                #'aug01-advbeta2ce-tor-WRN_40_10_drop50',
                #'aug01-advbeta2ce-tor-WRN_40_10_drop50-lrem2',

                'aug01-ce-tor-WRN_40_10',
                #'aug01-advbeta2ce-tor-WRN_40_10-lrem2',
                #'aug01-advbetace-tor-WRN_40_10-lrem2',
                #'aug01-advbeta.5ce-tor-WRN_40_10-lrem2',
                'aug01-advbeta2ce-tor-WRN_40_10',
                'aug01-advbetace-tor-WRN_40_10',
                'aug01-advbeta.5ce-tor-WRN_40_10',
                'aug01-tulipce-tor-WRN_40_10',
                'aug01-stradesce-tor-WRN_40_10',
                'aug01-strades3ce-tor-WRN_40_10',
                'aug01-strades6ce-tor-WRN_40_10',
                'aug01-advce-tor-WRN_40_10-lrem2',
                #'aug01-llrce-tor-WRN_40_10',
                'aug01-sllrce-tor-WRN_40_10-lrem2',
            ],
            'eps': [0.031],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)

class cifarMT(cifarLip):
    def __new__(cls, *args, **kwargs):
        return cifarLip.__new__(cls, *args, **kwargs)
    def __init__(self):
        self.experiment_fn = 'experiment01'
        self.grid_params[0]['attack'].append('multitarget')
        self.grid_params[1]['attack'].append('multitarget')

class resImgLips(ExpExperiments):
    def __new__(cls, *args, **kwargs):
        cls.name = "Restricted ImageNet"
        cls.experiment_fn = 'restrictedImgnet'
        grid_params = []
        arch = "ResNet50"
        grid_params.append({
            'dataset': ['resImgnet112v3'],
            'model': [
                f'ce-tor-{arch}-adambs128',
                f'advbeta2ce-tor-{arch}-adambs128',
                f'advbetace-tor-{arch}-adambs128',
                f'advbeta.5ce-tor-{arch}-adambs128',
                f'stradesce-tor-{arch}-adambs128',
                f'strades3ce-tor-{arch}-adambs128',
                f'strades6ce-tor-{arch}-adambs128',
                f'advce-tor-{arch}-adambs128',
                f'sllr36ce-tor-{arch}-adambs128',
                f'tulipce-tor-{arch}-adambs128',
            ],
            'eps': [0.005],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })

        arch = "ResNet50_drop50"
        grid_params.append({
            'dataset': ['resImgnet112v3'],
            'model': [
                f'ce-tor-{arch}-adambs128',
                f'advbeta2ce-tor-{arch}-adambs128',
                f'strades6ce-tor-{arch}-adambs128',
                f'strades3ce-tor-{arch}-adambs128',
                f'advce-tor-{arch}-adambs128',
            ],
            'eps': [0.005],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })

        arch = "ResNet50_drop20"
        grid_params.append({
            'dataset': ['resImgnet112v3'],
            'model': [
                f'ce-tor-{arch}-adambs128',
                f'advbeta2ce-tor-{arch}-adambs128',
                f'strades6ce-tor-{arch}-adambs128',
                f'strades3ce-tor-{arch}-adambs128',
                f'advce-tor-{arch}-adambs128',
            ],
            'eps': [0.005],
            'norm': ['inf'],
            'attack': ['pgd'],
            'random_seed': random_seed,
        })
        cls.grid_params = grid_params
        return ExpExperiments.__new__(cls, *args, **kwargs)

class resImgMT(resImgLips):
    def __new__(cls, *args, **kwargs):
        return resImgLips.__new__(cls, *args, **kwargs)
    def __init__(self):
        self.experiment_fn = 'restrictedImgnet'
        self.grid_params[0]['attack'].append('multitarget')

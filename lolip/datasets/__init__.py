import os

import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from autovar.base import RegisteringChoiceType, register_var, VariableClass


DEBUG = int(os.environ.get('DEBUG', 0))

class DatasetVarClass(VariableClass, metaclass=RegisteringChoiceType):
    """Defines the dataset to use"""
    var_name = 'dataset'

    @register_var(argument=r"mnist", shown_name="mnist")
    @staticmethod
    def mnist(auto_var, var_value, inter_var):
        from tensorflow.keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train[:, :, :, np.newaxis], x_test[:, :, :, np.newaxis]
        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255

        return x_train, y_train, x_test, y_test

    @register_var(argument=r"cifar10", shown_name="Cifar10")
    @staticmethod
    def cifar10(auto_var, var_value, inter_var):
        from tensorflow.keras.datasets import cifar10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train, y_test = y_train.reshape(-1), y_test.reshape(-1)
        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255

        return x_train, y_train, x_test, y_test

    @register_var(argument=r"svhn", shown_name="SVHN")
    @staticmethod
    def svhn(auto_var, var_value, inter_var):
        from torchvision.datasets import SVHN

        trn_svhn = SVHN("./data/", split='train', download=True)
        tst_svhn = SVHN("./data/", split='test', download=True)

        x_train, y_train, x_test, y_test = [], [], [], []
        for x, y in trn_svhn:
            x_train.append(np.array(x).reshape(32, 32, 3))
            y_train.append(y)
        for x, y in tst_svhn:
            x_test.append(np.array(x).reshape(32, 32, 3))
            y_test.append(y)
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        x_test, y_test = np.asarray(x_test), np.asarray(y_test)

        x_train, x_test = x_train.astype(np.float32) / 255, x_test.astype(np.float32) / 255

        return x_train, y_train, x_test, y_test

    @register_var(argument=r"resImgnet112v3", shown_name="Restricted ImageNet")
    @staticmethod
    def resImgnet112v3(auto_var, inter_var, eval_trn=False):
        if eval_trn:
            trn_ds = ImageFolder("./data/RestrictedImgNet/train",
                transform=transforms.Compose([
                    transforms.Resize(72),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                ]))
        else:
            trn_ds = ImageFolder("./data/RestrictedImgNet/train",
                transform=transforms.Compose([
                    transforms.Resize(72),
                    transforms.RandomCrop(64, padding=8),
                    transforms.ToTensor(),
                ]))
        tst_ds = ImageFolder("./data/RestrictedImgNet/val",
            transform=transforms.Compose([
                transforms.Resize(72),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]))
        return trn_ds, tst_ds

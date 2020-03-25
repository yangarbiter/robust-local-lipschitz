# Adversarial Robustness Through Local Lipschitzness

This repo contains the implementation of experiments in the paper

[Adversarial Robustness Through Local Lipschitzness](https://arxiv.org/abs/2003.02460)

Authors: [Yao-Yuan Yang](https://github.com/yangarbiter/)\*, [Cyrus Rashtchian](http://www.cyrusrashtchian.com)\*, Hongyang Zhang, Ruslan Salakhutdinov, Kamalika Chaudhuri (* equal contribution)

## Abstract

A standard method for improving the robustness of neural networks is adversarial training, where the network is trained on adversarial examples that are close to the training inputs. This produces classifiers that are robust, but it often decreases clean accuracy. Prior work even posits that the tradeoff between robustness and accuracy may be inevitable. We investigate this tradeoff in more depth through the lens of local Lipschitzness. In many image datasets, the classes are separated in the sense that images with different labels are not extremely close in <img src="https://render.githubusercontent.com/render/math?math=\ell_\infty"> distance. Using this separation as a starting point, we argue that it is possible to achieve both accuracy and robustness by encouraging the classifier to be locally smooth around the data. More precisely, we consider classifiers that are obtained by rounding locally Lipschitz functions. Theoretically, we show that such classifiers exist for any dataset such that there is a positive distance between the support of different classes. Empirically, we compare the local Lipschitzness of classifiers trained by several methods. Our results show that having a small Lipschitz constant correlates with achieving high clean and robust accuracy, and therefore, the smoothness of the classifier is an important property to consider in the context of adversarial examples.


## Setup

### Install requiremented libraries
```
pip install -r ./requirements.txt
```

### Install cleverhans from its github repository
```
pip install --upgrade git+https://github.com/tensorflow/cleverhans.git#egg=cleverhans
```

### Generate the Restricted ImageNet dataset
Use the script `./scripts/restrictedImgNet.py` to generate restrictedImgNet
dataset and put the data in `./data/RestrictedImgNet/` with torchvision
ImageFolder readable format. For more detail, please refer to
[lolip/datasets/\_\_init__.py](lolip/datasets/__init__.py).

## Repository structure

The spline example [notebooks/splines.ipynb](notebooks/splines.ipynb)

### Parameters

The default training parameters are set in [lolip/models/\_\_init__.py](lolip/models/__init__.py)

The network architecture defined in [lolip/models/torch_utils/archs.py](lolip/models/torch_utils/archs.py)

### Algorithm implementations

#### Defense Algorithms

- [TRADES](lolip/models/torch_utils/trades.py)
- [LLR](lolip/models/torch_utils/llr.py)
- [TULIP](lolip/models/torch_utils/tulip.py)
- [Adversarial Training](lolip/models/torch_model.py#L271)

#### Attack Algorithms

- [Projected Gradient Descent](lolip/attacks/torch/projected_gradient_descent.py)
- [Multi-targeted](lolip/attacks/torch/multi_target.py)

### Example options for model parameter

arch: ("CNN001", "CNN002", "WRN_40_10", "ResNet152", "ResNet50")

- ce-tor-{arch}
- strades6ce-tor-{arch}
- advce-tor-{arch}
- tulipce-tor-{arch}

## Examples

Run Natural training with CNN001 on the MNIST dataset
Perturbation distance is set to $0.1$ with L infinity norm.
Batch size is $64$ and using the SGD optimizer (default parameters).
```
python ./main.py --experiment experiment01 \
  --no-hooks \
  --norm inf --eps 0.1 \
  --dataset mnist \
  --model ce-tor-CNN001 \
  --attack pgd \
  --random_seed 0
```

Run TRADES (beta=6) with Wide ResNet 40-10 on the Cifar10 dataset
Perturbation distance is set to 0.031 with L infinity norm.
Batch size is $64$ and using the SGD optimizer
```
python ./main.py --experiment experiment01 \
  --no-hooks \
  --norm inf --eps 0.031 \
  --dataset cifar10 \
  --model strades6ce-tor-WRN_40_10 \
  --attack pgd \
  --random_seed 0
```

Run adversarial training with ResNet50 on the Restricted ImageNet dataset.
Perturbation distance is set to 0.005 with L infinity norm.
Attack with PGD attack.
Batch size is $128$ and using the Adam optimizer
```
python ./main.py --experiment restrictedImgnet \
  --no-hooks \
  --norm inf --eps 0.005 \
  --dataset resImgnet112v3 \
  --model advce-tor-ResNet50-adambs128 \
  --attack pgd \
  --random_seed 0
```

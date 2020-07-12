# A Closer Look at Accuracy vs. Robustness

This repo contains the implementation of experiments in the paper

[A Closer Look at Accuracy vs. Robustness](https://arxiv.org/abs/2003.02460)

Authors: [Yao-Yuan Yang](https://github.com/yangarbiter/)\*, [Cyrus Rashtchian](http://www.cyrusrashtchian.com)\*, Hongyang Zhang, Ruslan Salakhutdinov, Kamalika Chaudhuri (* equal contribution)

## Abstract

Current methods for training robust networks lead to a drop in test accuracy, which has led prior works to posit that a robustness-accuracy tradeoff may be inevitable in deep learning. We take a closer look at this phenomenon and first show that real image datasets are actually separated. With this property in mind, we then prove that robustness and accuracy should both be achievable for benchmark datasets through locally Lipschitz functions, and hence, there should be no inherent tradeoff between robustness and accuracy. Through extensive experiments with robustness methods, we argue that the gap between theory and practice arises from two limitations of current methods: either they fail to impose local Lipschitzness or they are insufficiently generalized. We explore combining dropout with robust training methods and obtain better generalization. We conclude that achieving robustness and accuracy in practice may require using methods that impose local Lipschitzness and augmenting them with deep learning generalization techniques.

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
[lolip/dataset/\_\_init\_\_.py](lolip/dataset/__init__.py).

## Repository structure

- The spline example [notebooks/splines.ipynb](notebooks/splines.ipynb)

- The code for getting train-train separation and test-train separation [notebooks/dataset_dist.ipynb](notebooks/dataset_dist.ipynb)

- The script to run proof-of-concept classifiers [scripts/run_proof_of_concept.sh](scripts/run_proof_of_concept.sh)

- The script to run separation for Restricted ImageNet  [scripts/restrictedImgNet_dists.py](scripts/restrictedImgNet_dists.py)

- The script to run separation for randomly labeled Restricted ImageNet
[scripts/random_restrictedImgNet_dists.py](scripts/random_restrictedImgNet_dists.py)

### Parameters

The default training parameters are set in [lolip/models/\_\_init\_\_.py](lolip/models/__init__.py)

The network architectures defined in [lolip/models/torch_utils/archs.py](lolip/models/torch_utils/archs.py)

### Algorithm implementations

#### Defense Algorithms

- [TRADES](lolip/models/torch_utils/trades.py)
- [LLR](lolip/models/torch_utils/llr.py)
- [TULIP](lolip/models/torch_utils/tulip.py)
- [Adversarial training](lolip/models/torch_model.py#L271)
- [Robust self training (RST)](lolip/models/torch_model.py#L271)

#### Attack Algorithms

- [Projected Gradient Descent (PGD)](lolip/attacks/projected_gradient_descent.py)
- [Multi-targeted Attack](lolip/attacks/multi_target.py)

### Example options for model parameter

arch: ("CNN001", "CNN002",
       "WRN_40_10", "WRN_40_10_drop20", "WRN_40_10_drop50",
       "ResNet50", "ResNet50_drop50")

- Natural: ce-tor-{arch}
- TRADES(beta=6): strades6ce-tor-{arch}
- adversarial training: advce-tor-{arch}
- RST(lambda=2): advbeta2ce-tor-{arch}
- TULIP(gradient regularization): tulipce-tor-{arch}
- LLR: sllrce-tor-{arch}

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

## Reproducing Results

### Scripts

- Table1 and Figures in Appendix D: [notebooks/dataset_dist.ipynb](notebooks/dataset_dist.ipynb)
- Table2: [scripts/Table2.sh](scripts/Table2.sh)
- Table3: [scripts/Table3.sh](scripts/Table3.sh)
- Table4: [scripts/Table4.sh](scripts/Table4.sh)

### Appendix C: Proof-of-concept classifier

Run Robust self training (lambda=2) with Wide ResNet 40-10 on the Cifar10 dataset
Perturbation distance is set to 0.031 with L infinity norm.
Batch size is $64$ and using the SGD optimizer
```
python ./main.py --experiment hypo \
  --no-hooks \
  --norm inf --eps 0.031 \
  --dataset cifar10 \
  --model advbeta2ce-tor-WRN_40_10 \
  --attack pgd \
  --random_seed 0
```

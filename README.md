# Adversarial Robustness Through Local Lipschitzness


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
[lolip/dataset/__init__.py](lolip/dataset/__init__.py).

## Repository structure

The spline example [notebooks/splines.ipynb](notebooks/splines.ipynb)

### Parameters

The default training parameters are set in [lolip/models/__init__.py](lolip/models/__init__.py)

The network architecture defined in [lolip/models/torch_utils/archs.py](lolip/models/torch_utils/archs.py)

### Algorithm implementations

#### Defense Algorithms

- [TRADES](lolip/models/torch_utils/trades.py)
- [LLR](lolip/models/torch_utils/llr.py)
- [TULIP](lolip/models/torch_utils/tulip.py)
- [Adversarial Training](lolip/models/torch_model.py#L271)

#### Attack Algorithms

- [Projected Gradient Descent](lolip/attacks/projected_gradient_descent.py)
- [Multi-targeted](lolip/attacks/multi_target.py)

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

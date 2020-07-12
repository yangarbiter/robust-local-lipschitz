python ./main.py --experiment hypo \
  --norm inf --eps 0.031 \
  --dataset mnist \
  --model advbeta4ce-tor-CNN002-bs128ep40 \
  --attack pgd \
  --random_seed 0

python ./main.py --experiment hypo \
  --norm inf --eps 0.031 \
  --dataset cifar10 \
  --model advbeta4ce-tor-WRN_40_10-adamlrem2ep70 \
  --attack pgd \
  --random_seed 0

python ./main.py --experiment hypo \
  --norm inf --eps 0.031 \
  --dataset svhn \
  --model advbeta4ce-tor-WRN_64_10-adamlrem2ep60 \
  --attack pgd \
  --random_seed 0

python ./main.py --experiment restrictedImgnetHypo \
  --norm inf --eps 0.005 \
  --dataset resImgnet112v3 \
  --model advbeta2ce-tor-ResNet50-adambs128ep30 \
  --attack pgd \
  --random_seed 0

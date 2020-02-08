import os
from os.path import join
from mkdir_p import mkdir_p

import torch

from lolip.datasets.imagenet import ImageNet


IMGNET_DIR_TRN = "/tmp2/imgnet/ILSVRC2012_img_train/ILSVRC2012_img_train/"
IMGNET_DIR_TST = "/tmp2/imgnet/ILSVRC2012_img_val/ILSVRC2012_img_val/"
TARGET_DIR_TRN = "/tmp2/RestrictedImageNet/train/"
TARGET_DIR_TST = "/tmp2/RestrictedImageNet/val/"

ds = ImageNet("/tmp2/ImageNet")
meta = torch.load("/tmp2/ImageNet/meta.bin")

inv_map = {v: k for k, v in meta[0].items()}

class_dict = {
    "dog": (151, 269),
    "cat": (281, 286),
    "frog": (30, 33),
    "turtle": (33, 38),
    "bird": (80, 101),
    "primate": (365, 383),
    "fish": (389, 398),
    "crab": (118, 122),
    "insect": (300, 320),
}

for class_name, idx in class_dict.items():
    mkdir_p(join(TARGET_DIR_TRN, class_name))
    mkdir_p(join(TARGET_DIR_TST, class_name))
    for k in ds.classes[idx[0]:idx[1]]:
        fn = inv_map[k]
        os.system(f"cp -rf {join(IMGNET_DIR_TRN, fn)}/* {join(TARGET_DIR_TRN, class_name)}")
        os.system(f"cp -rf {join(IMGNET_DIR_TST, fn)}/* {join(TARGET_DIR_TST, class_name)}")

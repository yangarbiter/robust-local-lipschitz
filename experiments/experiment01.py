import os

import torch
from bistiming import Stopwatch
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import set_random_seed
from lolip.utils import estimate_local_lip_v2
from lolip.variables import get_file_name


def run_experiment01(auto_var):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _ = set_random_seed(auto_var)
    norm = auto_var.get_var("norm")
    trnX, trny, tstX, tsty = auto_var.get_var("dataset")
    lbl_enc = OneHotEncoder(categories=[np.sort(np.unique(trny))], sparse=False).fit(trny.reshape(-1, 1))
    auto_var.set_intermidiate_variable("lbl_enc", lbl_enc)
    n_classes = len(np.unique(trny))
    n_channels = trnX.shape[-1]

    result = {}
    multigpu = False
    model = auto_var.get_var("model", trnX=trnX, trny=trny, multigpu=multigpu, n_channels=n_channels)
    model.tst_ds = (tstX, tsty)
    with Stopwatch("Fitting Model"):
        history = model.fit(trnX, trny)

    result['trn_acc'] = (model.predict(trnX) == trny).mean()
    result['tst_acc'] = (model.predict(tstX) == tsty).mean()
    print(f"train acc: {result['trn_acc']}")
    print(f"test acc: {result['tst_acc']}")

    attack_model = auto_var.get_var("attack", model=model, n_classes=n_classes,
                                    clip_min=0, clip_max=1,)
    with Stopwatch("Attacking"):
        adv_trnX = attack_model.perturb(trnX, trny)
        adv_tstX = attack_model.perturb(tstX, tsty)
    result['adv_trn_acc'] = (model.predict(adv_trnX) == trny).mean()
    result['adv_tst_acc'] = (model.predict(adv_tstX) == tsty).mean()
    print(f"adv trn acc: {result['adv_trn_acc']}")
    print(f"adv tst acc: {result['adv_tst_acc']}")
    del attack_model

    with Stopwatch("Estimating trn Lip"):
        trn_lip, _ = estimate_local_lip_v2(model.model, trnX, top_norm=1, btm_norm=norm,
                                    epsilon=auto_var.get_var("eps"), device=device)
    result['avg_trn_lip_1'] = trn_lip
    with Stopwatch("Estimating tst Lip"):
        tst_lip, _ = estimate_local_lip_v2(model.model, tstX, top_norm=1, btm_norm=norm,
                                     epsilon=auto_var.get_var("eps"), device=device)
    result['avg_tst_lip_1'] = tst_lip
    print(f"avg trn lip: {result['avg_trn_lip_1']}")
    print(f"avg tst lip: {result['avg_tst_lip_1']}")

    print(result)
    return result

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch
from tqdm import tqdm
import joblib

n_classes = 9
trn_ds = ImageFolder("./data/RestrictedImgNet/train",
    transform=transforms.Compose([
        transforms.Resize(72),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ]))
tst_ds = ImageFolder("./data/RestrictedImgNet/val",
    transform=transforms.Compose([
        transforms.Resize(72),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ]))

tst_dists = torch.ones((len(tst_ds), n_classes)).float()
batch_size = 256
############# for randomized labeling experiment
np.random.seed(0)
trn_ds.targets = np.random.choice(np.arange(9), size=len(trn_ds.targets))
for i in range(len(trn_ds.imgs)):
    trn_ds.imgs[i] = (trn_ds.imgs[i][0], trn_ds.targets[i])
tst_ds.targets = np.random.choice(np.arange(9), size=len(tst_ds.targets))
for i in range(len(tst_ds.imgs)):
    tst_ds.imgs[i] = (tst_ds.imgs[i][0], tst_ds.targets[i])
import ipdb; ipdb.set_trace()
############
trn_loader = torch.utils.data.DataLoader(trn_ds, batch_size=batch_size, shuffle=False, num_workers=16)
tst_loader = torch.utils.data.DataLoader(tst_ds, batch_size=batch_size, shuffle=False, num_workers=16)

for x, y in tqdm(trn_loader):
    y = y.numpy()
    x = x.flatten(1).cuda()
    for i, (xi, yi) in enumerate(tst_loader):
        yi = yi.numpy()
        xi = xi.flatten(1).cuda()

        cov = torch.norm(x.repeat((len(xi), 1)) - xi.repeat_interleave(len(x), dim=0), p=np.inf, dim=1)
        cov = cov.view(len(xi), len(x)).cpu()

        #cov = pairwise_distances(xi, x, metric='minkowski', n_jobs=16, p=np.inf)
        for j in range(n_classes):
            ty = (y == j)
            if ty.sum() >= 1:
                min_dist = cov[:, ty].min(axis=1)[0]
                tst_dists[i*batch_size: (i+1)*batch_size, j] = np.minimum(
                    tst_dists[i*batch_size: (i+1)*batch_size, j],
                    min_dist
                )
#joblib.dump(tst_dists.numpy(), "./restricted_tst_linf.pkl")
joblib.dump(tst_dists.numpy(), "./rand_restricted_tst_linf.pkl")

trn_dists = torch.ones((len(trn_ds), n_classes)).float()
batch_size = 256

############## for randomized labeling experiment
np.random.seed(0)
trn_ds.targets = np.random.choice(np.arange(9), size=len(trn_ds.targets))
for i in range(len(trn_ds.imgs)):
    trn_ds.imgs[i] = (trn_ds.imgs[i][0], trn_ds.targets[i])
#############
trn_loader = torch.utils.data.DataLoader(trn_ds, batch_size=batch_size, shuffle=False, num_workers=16)
tst_loader = torch.utils.data.DataLoader(trn_ds, batch_size=batch_size, shuffle=False, num_workers=16)

for x, y in tqdm(trn_loader):
    y = y.numpy()
    x = x.flatten(1).cuda()
    for i, (xi, yi) in enumerate(tst_loader):
        yi = yi.numpy()
        xi = xi.flatten(1).cuda()

        cov = torch.norm(x.repeat((len(xi), 1)) - xi.repeat_interleave(len(x), dim=0), p=np.inf, dim=1)
        cov = cov.view(len(xi), len(x)).cpu()

        for j in range(n_classes):
            ty = (y == j)
            if ty.sum() >= 1:
                min_dist = cov[:, ty].min(axis=1)[0]
                trn_dists[i*batch_size: (i+1)*batch_size, j] = np.minimum(
                    trn_dists[i*batch_size: (i+1)*batch_size, j],
                    min_dist
                )
import ipdb; ipdb.set_trace()
#joblib.dump(trn_dists.numpy(), "./restricted_trn_linf.pkl")
joblib.dump(trn_dists.numpy(), "./rand_restricted_trn_linf.pkl")

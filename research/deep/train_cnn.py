"""T35 histogram CNN, exactly as pre-declared in docs/CROP_NOWCAST_TRIALS.md (do not tune).

Runs the same on Kaggle (/kaggle/input -> /kaggle/working) and Colab (Drive), or locally.
Unit of work = (test year, seed). Each unit writes preds_<Y>_<seed>.csv; a unit whose file
already exists is skipped, so an interrupted run resumes on any platform without recomputation.
Usage: python train_cnn.py tensor.npz out_dir
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

TEST_YEARS, SEEDS = range(2013, 2022), range(5)
EPOCHS, PATIENCE, BATCH, LR, WD = 60, 10, 64, 1e-3, 1e-4


class Net(nn.Module):
    def __init__(self, bands=6):
        super().__init__()
        layers, c = [], bands
        for out in (32, 64, 128):
            layers += [nn.Conv2d(c, out, 3, stride=(1, 2), padding=1), nn.BatchNorm2d(out), nn.ReLU(), nn.Dropout(0.25)]
            c = out
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.body(x).mean(dim=(2, 3), keepdim=True)).squeeze(1)


def fit_predict(X, y, year, test_year, seed, dev):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = (year < test_year - 1) & np.isfinite(y)
    va = (year == test_year - 1) & np.isfinite(y)
    te = year == test_year
    mu = X[tr].mean(axis=(0, 2, 3), keepdims=True)  # per-band statistics from training years only
    sd = X[tr].std(axis=(0, 2, 3), keepdims=True) + 1e-6
    norm = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32, device=dev)
    xt, yt, xv, yv = norm(X[tr]), torch.tensor(y[tr], device=dev), norm(X[va]), torch.tensor(y[va], device=dev)
    net = Net(X.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    best, best_state, wait = np.inf, None, 0
    for _ in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(xt), device=dev)
        for i in range(0, len(xt), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            nn.functional.mse_loss(net(xt[b]), yt[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(net(xv), yv).item()
        if v < best:
            best, wait = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(norm(X[te])).cpu().numpy(), te, best


def main(tensor_npz, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(tensor_npz, allow_pickle=True)
    X, y, year, fips = z['X'], z['target'].astype(np.float32), z['year'], z['fips']
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', dev, 'X', X.shape, flush=True)
    for ty in TEST_YEARS:
        for seed in SEEDS:
            dest = out / f'preds_{ty}_{seed}.csv'
            if dest.exists():  # finished unit: never recomputed
                continue
            p, te, val = fit_predict(X, y, year, ty, seed, dev)
            tmp = dest.with_suffix('.part')
            with open(tmp, 'w') as fh:
                fh.write('fips,year,p\n')
                fh.writelines(f'{f},{ty},{v:.6f}\n' for f, v in zip(fips[te], p, strict=True))
            tmp.replace(dest)
            print(ty, seed, 'val_mse', round(val, 5), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:3])

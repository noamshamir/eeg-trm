# data/eeg.py
import math
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.model_selection import StratifiedShuffleSplit

class _Stream:
    def __init__(self, X, y, bs, shuffle, seed):
        self.X, self.y = X, y
        self.bs = len(X) if bs == -1 else bs
        self.shuffle, self.seed = shuffle, seed
        self.reset()
    def reset(self):
        self.idx = np.arange(len(self.y))
        if self.shuffle: np.random.default_rng(self.seed).shuffle(self.idx)
        self.ptr = 0
    def __iter__(self): 
        self.reset(); 
        return self
    def __next__(self):
        if self.ptr >= len(self.idx): raise StopIteration
        j = self.idx[self.ptr:self.ptr+self.bs]; self.ptr += self.bs
        return {"image": self.X[j], "label": self.y[j]}
    def __len__(self):
        return math.ceil(len(self.y) / self.bs)

def _ensure_3d(X):
    X = np.asarray(X, np.float32)
    if X.ndim == 1: return X[None, :, None]
    if X.ndim == 2: return X[None, :, :]
    return X

def _build_epochs(labels, data, fs=100, pre_seconds=4, channels=None):
    if channels is None: channels = list(range(data.shape[1]))
    win = int(pre_seconds * fs)
    n = data.shape[0]
    used = np.zeros(n, bool)
    Xp, Xn = [], []
    for i in range(len(labels)):
        end = int(labels.iloc[i, 0]); lab = int(labels.iloc[i, 1]); start = end - win
        if start < 0 or end > n: continue
        seg = data.iloc[start:end, channels].to_numpy(np.float32)
        if lab == 1: Xp.append(seg)
        elif lab == -1: Xn.append(seg)
        used[start:end] = True
    Xp = np.array(Xp) if len(Xp) else np.zeros((0, win, len(channels)), np.float32)
    Xn = np.array(Xn) if len(Xn) else np.zeros((0, win, len(channels)), np.float32)
    Xr = []
    step, cap, k = win, 200, 0
    for s in range(0, n - win + 1, step):
        if k >= cap: break
        e = s + win
        if not used[s:e].any():
            Xr.append(data.iloc[s:e, channels].to_numpy(np.float32)); k += 1
    Xr = np.array(Xr) if len(Xr) else np.zeros((0, win, len(channels)), np.float32)
    X = np.concatenate([Xn, Xr, Xp], 0)
    y = np.concatenate([
        np.zeros(len(Xn), int),        # -1 → 0
        np.ones(len(Xr), int),         # 0 → 1
        2*np.ones(len(Xp), int)        # +1 → 2
    ])    
    return X, y

def _psd_cube(X, fs, nperseg=None, noverlap=None, log=True, eps=1e-8):
    X = _ensure_3d(X)
    E, N, C = X.shape
    if nperseg is None: nperseg = int(min(N, fs))
    if noverlap is None: noverlap = nperseg // 2
    freqs = None
    P = np.empty((E, C, 0), np.float32)
    for e in range(E):
        Pc = []
        for c in range(C):
            f, pxx = welch(X[e, :, c], fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant", scaling="density")
            Pc.append(pxx.astype(np.float32))
        Pc = np.stack(Pc, 0)
        if freqs is None: freqs = f.astype(np.float32)
        if e == 0: P = np.empty((E, C, len(freqs)), np.float32)
        P[e] = Pc
    if log: P = np.log(P + eps)
    imgs = np.transpose(P, (0, 2, 1))  # (E, F, C)
    imgs = imgs[..., None]             # (E, F, C, 1)
    return imgs, freqs

def bcic_psd(batch_size, fs=100, pre_seconds=4, data_dir="data/concatenated_eeg", test_size=0.2, seed=0):
    data_path = f"{data_dir}/combined_eeg_data.csv"
    labels_path = f"{data_dir}/combined_labels.csv"

    training_data = pd.read_csv(data_path).astype(np.float32)
    labels = pd.read_csv(labels_path, dtype={0: np.int32, 1: np.int32})

    X, y = _build_epochs(labels, training_data, fs=fs, pre_seconds=pre_seconds)
    X, _ = _psd_cube(X, fs=fs, log=True)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    (tr_idx, te_idx), = sss.split(np.zeros(len(y)), y)
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    full_batch = batch_size == -1
    tr_iter = _Stream(Xtr, ytr, batch_size, shuffle=True, seed=seed)
    te_iter = _Stream(Xte, yte, batch_size, shuffle=False, seed=seed)

    meta = {
        "n_train": len(ytr),
        "n_test": len(yte),
        "steps_per_epoch": 1 if full_batch else len(tr_iter),
        "img_shape": (X.shape[1], X.shape[2], 1),  # (freq_bins, channels, depth)
        "n_classes": 3,
    }

    return tr_iter, te_iter, meta
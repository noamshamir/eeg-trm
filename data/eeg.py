# data/eeg.py
import os
import math
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from scipy.signal import welch, butter, iirnotch, sosfiltfilt
from sklearn.model_selection import StratifiedShuffleSplit


# ---------- I/O ----------
def _read_header_and_dims(data_path: str) -> Tuple[int, int]:
    one = pd.read_csv(data_path, nrows=1)
    with open(data_path, "r") as f:
        n_total_lines = sum(1 for _ in f)
    return max(0, n_total_lines - 1), one.shape[1]


def _read_labels(labels_path: str) -> pd.DataFrame:
    df = pd.read_csv(labels_path, dtype={0: np.int64, 1: np.int8})
    df = df.iloc[:, :2]
    df.columns = ["end", "label"]
    df = df[df["label"].isin([-1, 1])].copy()
    df.sort_values("end", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _label_to01(y_raw: np.ndarray) -> np.ndarray:
    return (y_raw.astype(np.int8) + 1) // 2


def _epoch_slice_from_csv(
    data_path: str,
    start: int,
    end: int,
    usecols: Optional[List[int]],
    dtype=np.float16,
) -> np.ndarray:
    nrows = end - start
    if nrows <= 0:
        raise ValueError("Invalid epoch slice: start >= end")
    skip = range(1, start + 1) if start > 0 else None
    df = pd.read_csv(
        data_path,
        dtype=dtype,
        nrows=nrows,
        skiprows=skip,
        usecols=usecols,
    )
    return df.to_numpy(dtype=dtype, copy=False)


# ---------- Preprocessing ----------
def _sos_bandpass(fs: int, lo: float, hi: float, order: int = 4):
    nyq = 0.5 * fs
    lo, hi = max(1e-6, lo / nyq), min(0.999999, hi / nyq)
    return butter(order, [lo, hi], btype="band", output="sos")


def _sos_notch(fs: int, freq: float, q: float = 30.0):
    if freq is None or freq <= 0:
        return None
    w0 = freq / (fs / 2.0)
    if w0 <= 0 or w0 >= 1:
        return None
    b, a = iirnotch(w0, q)
    # Convert (b,a) to sos using butter equivalence for stability
    # Minimal 2nd-order section:
    sos = np.zeros((1, 6), dtype=np.float64)
    sos[0, :3] = b
    sos[0, 3:] = a
    return sos


def _apply_filters(epoch_tc_f32: np.ndarray, fs: int,
                   bandpass: Optional[Tuple[float, float]],
                   notch_hz: Optional[float]) -> np.ndarray:
    X = epoch_tc_f32
    if notch_hz:
        sos_n = _sos_notch(fs, notch_hz)
        if sos_n is not None:
            for c in range(X.shape[1]):
                X[:, c] = sosfiltfilt(sos_n, X[:, c])
    if bandpass:
        lo, hi = bandpass
        sos_b = _sos_bandpass(fs, lo, hi, order=4)
        for c in range(X.shape[1]):
            X[:, c] = sosfiltfilt(sos_b, X[:, c])
    return X


def _normalize_epoch(epoch_tc_f32: np.ndarray, mode: str = "mad") -> np.ndarray:
    X = epoch_tc_f32
    if mode == "zscore":
        mu = np.mean(X, axis=0, dtype=np.float32)
        sd = np.std(X, axis=0, ddof=1, dtype=np.float32)
        sd[sd < 1e-6] = 1.0
        X = (X - mu) / sd
    else:  # "mad" (robust)
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0)
        mad[mad < 1e-6] = 1.0
        X = (X - med) / mad
    return X


def _artifact_ok(epoch_tc_f32: np.ndarray,
                 abs_max: Optional[float],
                 z_max: Optional[float]) -> bool:
    if abs_max is not None and np.max(np.abs(epoch_tc_f32)) > abs_max:
        return False
    if z_max is not None:
        z = (epoch_tc_f32 - epoch_tc_f32.mean(0)) / (epoch_tc_f32.std(0, ddof=1) + 1e-6)
        if np.max(np.abs(z)) > z_max:
            return False
    return True


# ---------- PSD ----------
def _psd_image_from_epoch(
    epoch_tc: np.ndarray,
    fs: int,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    log: bool = True,
    eps: float = 1e-8,
    out_dtype=np.float16,
) -> Tuple[np.ndarray, np.ndarray]:
    T, C = epoch_tc.shape
    nperseg = int(min(T, fs)) if nperseg is None else nperseg
    noverlap = nperseg // 2 if noverlap is None else noverlap

    freqs = None
    P = None
    for c in range(C):
        f, pxx = welch(
            epoch_tc[:, c],
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
        )
        if freqs is None:
            freqs = f.astype(np.float32, copy=False)
            P = np.empty((C, len(freqs)), dtype=np.float32)
        P[c] = pxx.astype(np.float32, copy=False)

    if log:
        P = np.log(P + eps, dtype=np.float32)

    img = np.transpose(P, (1, 0))[:, :, None].astype(out_dtype, copy=False)
    return img, freqs


# ---------- Streaming ----------
class _BatchStream:
    def __init__(
        self,
        data_path: str,
        labels_df: pd.DataFrame,
        indices: np.ndarray,
        batch_size: int,
        fs: int,
        pre_seconds: int,
        channels: Optional[List[int]],
        shuffle: bool,
        seed: int,
        psd_nperseg: Optional[int] = None,
        psd_noverlap: Optional[int] = None,
        bandpass: Optional[Tuple[float, float]] = None,
        notch_hz: Optional[float] = None,
        norm_mode: Optional[str] = None,
        artifact_abs_max: Optional[float] = None,
        artifact_z_max: Optional[float] = None,
    ):
        self.data_path = data_path
        self.labels_df = labels_df
        self.indices = np.array(indices, dtype=np.int64)
        self.batch_size = len(indices) if batch_size == -1 else int(batch_size)
        self.fs = fs
        self.pre_seconds = pre_seconds
        self.channels = channels
        self.shuffle = shuffle
        self.seed = seed
        self.psd_nperseg = psd_nperseg
        self.psd_noverlap = psd_noverlap
        self.bandpass = bandpass
        self.notch_hz = notch_hz
        self.norm_mode = norm_mode
        self.artifact_abs_max = artifact_abs_max
        self.artifact_z_max = artifact_z_max
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self._order = np.arange(len(self.indices))
        if self.shuffle:
            self._rng.shuffle(self._order)
        self._ptr = 0

    def __iter__(self):
        self.reset()
        return self

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __next__(self) -> Dict[str, np.ndarray]:
        if self._ptr >= len(self._order):
            raise StopIteration
        j = self._order[self._ptr : self._ptr + self.batch_size]
        self._ptr += self.batch_size
        take = self.indices[j]

        X_list, y_list = [], []
        win = int(self.pre_seconds * self.fs)

        for idx in take:
            end = int(self.labels_df.iloc[idx, 0])
            lab = int(self.labels_df.iloc[idx, 1])
            start = end - win
            if start < 0:
                continue

            epoch_tc = _epoch_slice_from_csv(
                self.data_path, start=start, end=end, usecols=self.channels, dtype=np.float16
            )
            if epoch_tc.shape[0] != win:
                continue

            Xf = epoch_tc.astype(np.float32, copy=False)
            if self.bandpass or self.notch_hz:
                Xf = _apply_filters(Xf, self.fs, self.bandpass, self.notch_hz)
            if self.norm_mode:
                Xf = _normalize_epoch(Xf, self.norm_mode)
            if not _artifact_ok(Xf, self.artifact_abs_max, self.artifact_z_max):
                continue

            img, _ = _psd_image_from_epoch(
                Xf, fs=self.fs,
                nperseg=self.psd_nperseg,
                noverlap=self.psd_noverlap,
                log=True, eps=1e-8, out_dtype=np.float16,
            )
            X_list.append(img)
            y_list.append(lab)

        if not X_list:
            return self.__next__()

        Xb = np.stack(X_list, axis=0)
        yb = _label_to01(np.array(y_list, dtype=np.int8))
        return {"image": Xb, "label": yb}


# ---------- Public API ----------
def bcic_psd(
    batch_size: int,
    fs: int = 100,
    pre_seconds: int = 4,
    data_dir: str = "data/concatenated_eeg",
    test_size: float = 0.2,
    seed: int = 0,
    channels: Optional[List[int]] = None,
    # new preprocessing toggles (all optional; default off/backward-compatible)
    bandpass: Optional[Tuple[float, float]] = None,   # e.g., (8, 30)
    notch_hz: Optional[float] = None,                 # e.g., 60 or 50
    norm_mode: Optional[str] = None,                  # "mad" | "zscore" | None
    artifact_abs_max: Optional[float] = None,         # e.g., 300.0 (µV-equivalent units)
    artifact_z_max: Optional[float] = None,           # e.g., 6.0
    psd_nperseg: Optional[int] = None,
    psd_noverlap: Optional[int] = None,
) -> Tuple["_BatchStream", "_BatchStream", Dict[str, object]]:
    data_path = os.path.join(data_dir, "combined_eeg_data.csv")
    labels_path = os.path.join(data_dir, "combined_labels.csv")

    labels_df = _read_labels(labels_path)
    if len(labels_df) == 0:
        raise ValueError("No {-1,+1} labels found.")

    _, n_channels = _read_header_and_dims(data_path)
    if channels is None:
        channels = list(range(n_channels))

    win = int(pre_seconds * fs)
    sim_epoch = np.zeros((win, len(channels)), dtype=np.float32)
    sim_img, freqs = _psd_image_from_epoch(sim_epoch, fs=fs, log=True)
    F, C, _ = sim_img.shape
    input_shape = (F, C, 1)

    y01 = _label_to01(labels_df["label"].to_numpy())
    (tr_idx, te_idx), = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    ).split(np.zeros_like(y01), y01)

    tr_stream = _BatchStream(
        data_path=data_path,
        labels_df=labels_df,
        indices=tr_idx,
        batch_size=batch_size,
        fs=fs,
        pre_seconds=pre_seconds,
        channels=channels,
        shuffle=True,
        seed=seed,
        psd_nperseg=psd_nperseg,
        psd_noverlap=psd_noverlap,
        bandpass=bandpass,
        notch_hz=notch_hz,
        norm_mode=norm_mode,
        artifact_abs_max=artifact_abs_max,
        artifact_z_max=artifact_z_max,
    )
    te_stream = _BatchStream(
        data_path=data_path,
        labels_df=labels_df,
        indices=te_idx,
        batch_size=batch_size,
        fs=fs,
        pre_seconds=pre_seconds,
        channels=channels,
        shuffle=False,
        seed=seed,
        psd_nperseg=psd_nperseg,
        psd_noverlap=psd_noverlap,
        bandpass=bandpass,
        notch_hz=notch_hz,
        norm_mode=norm_mode,
        artifact_abs_max=artifact_abs_max,
        artifact_z_max=artifact_z_max,
    )

    meta: Dict[str, object] = {
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "steps_per_epoch": 1 if batch_size == -1 else len(tr_stream),
        "n_classes": 2,
        "input_shape": input_shape,
        "fs": fs,
        "pre_seconds": pre_seconds,
        "channels_used": channels,
        "freq_bins": len(freqs),
    }
    return tr_stream, te_stream, meta
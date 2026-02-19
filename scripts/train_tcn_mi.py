#!/usr/bin/env python3
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]  # project root
"""Train TCN baseline on MI-friendly epoch CSVs and save results.

Expected dataset layout:
data/
  left/ right/ up/ down/ zoomIn/ zoomOut/
Each CSV: timestamp_sec, ch0..ch23 (4s @ 250Hz => 1000 rows)

Outputs:
result_tcn/
  confusion_matrix.png
  learning_curves.png
  best_model.pt
  metrics.json
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

LABELS = ["left","right","up","down","zoomIn","zoomOut"]
SR = 250
EPOCH_SEC = 4.0
N_SAMPLES = int(SR * EPOCH_SEC)
N_CH = 24

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def butter_bandpass_filtfilt(x: np.ndarray, sr: int, low: float, high: float, order: int = 4) -> np.ndarray:
    """Optional bandpass for MI (8–30Hz). If SciPy isn't available, returns input."""
    try:
        from scipy.signal import butter, filtfilt
    except Exception:
        return x
    nyq = 0.5 * sr
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    y = np.zeros_like(x)
    for c in range(x.shape[0]):
        y[c] = filtfilt(b, a, x[c]).astype(np.float32)
    return y

def per_channel_zscore(x: np.ndarray) -> np.ndarray:
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True) + 1e-8
    return ((x - m) / s).astype(np.float32)

def load_one_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    cols = [f"ch{i}" for i in range(N_CH)]
    X = df[cols].to_numpy(dtype=np.float32).T  # (ch, time)
    if X.shape[1] != N_SAMPLES:
        if X.shape[1] < N_SAMPLES:
            pad = N_SAMPLES - X.shape[1]
            X = np.pad(X, ((0,0),(0,pad)), mode="edge")
        else:
            X = X[:, :N_SAMPLES]
    return X

def build_index(data_root: Path):
    files, y = [], []
    for li, lab in enumerate(LABELS):
        folder = data_root / lab
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.csv")):
            files.append(p)
            y.append(li)
    return files, np.array(y, dtype=np.int64)

class EEGCSVSet(Dataset):
    def __init__(self, files, y, bandpass=True, zscore=True):
        self.files = files
        self.y = y
        self.bandpass = bandpass
        self.zscore = zscore

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        X = load_one_csv(self.files[idx])
        if self.bandpass:
            X = butter_bandpass_filtfilt(X, SR, 8.0, 30.0, order=4)
        if self.zscore:
            X = per_channel_zscore(X)
        Xt = torch.from_numpy(X)  # (ch, time) for TCN
        yt = torch.tensor(self.y[idx], dtype=torch.long)
        return Xt, yt

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation, dropout):
        super().__init__()
        pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop2 = nn.Dropout(dropout)

        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        y = self.conv1(x)
        y = self.chomp1(y)
        y = F.relu(y)
        y = self.bn1(y)
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.chomp2(y)
        y = F.relu(y)
        y = self.bn2(y)
        y = self.drop2(y)

        res = x if self.down is None else self.down(x)
        return F.relu(y + res)

class TCNClassifier(nn.Module):
    def __init__(self, n_ch=N_CH, n_classes=len(LABELS), channels=(64,64,64), kernel_size=5, dropout=0.3):
        super().__init__()
        layers = []
        in_ch = n_ch
        for i, out_ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, n_classes)

    def forward(self, x):  # x: (B,C,T)
        x = self.tcn(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

def plot_confusion(cm, out_path: Path):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick = np.arange(len(LABELS))
    plt.xticks(tick, LABELS, rotation=45, ha="right")
    plt.yticks(tick, LABELS)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_learning(history, out_path: Path):
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.legend()
    plt.title("Learning Curves")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def eval_loop(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    loss_sum, n = 0.0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = ce(logits, y)
            loss_sum += loss.item() * X.size(0)
            n += X.size(0)
            p = torch.argmax(logits, dim=1)
            y_true.append(y.cpu().numpy())
            y_pred.append(p.cpu().numpy())
    y_true = np.concatenate(y_true) if len(y_true) else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred) if len(y_pred) else np.array([], dtype=np.int64)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return float(loss_sum / max(n, 1)), float(acc), y_true, y_pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(ROOT/"data"))
    ap.add_argument("--out", type=str, default=str(ROOT/"result_tcn"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-bandpass", action="store_true")
    ap.add_argument("--no-zscore", action="store_true")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--channels", type=str, default="64,64,64")
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.3)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data)
    if not data_root.is_absolute():
        data_root = (ROOT / data_root).resolve()
    result_dir = Path(args.out)
    if not result_dir.is_absolute():
        result_dir = (ROOT / result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    files, y = build_index(data_root)
    if len(files) == 0:
        raise SystemExit(f"No CSV files found under: {data_root.resolve()}")

    X_tr, X_te, y_tr, y_te = train_test_split(files, y, test_size=0.15, random_state=args.seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.15/0.85, random_state=args.seed, stratify=y_tr)

    bandpass = not args.no_bandpass
    zscore = not args.no_zscore

    tr_set = EEGCSVSet(X_tr, y_tr, bandpass=bandpass, zscore=zscore)
    va_set = EEGCSVSet(X_va, y_va, bandpass=bandpass, zscore=zscore)
    te_set = EEGCSVSet(X_te, y_te, bandpass=bandpass, zscore=zscore)

    tr_loader = DataLoader(tr_set, batch_size=args.batch, shuffle=True, drop_last=False)
    va_loader = DataLoader(va_set, batch_size=args.batch, shuffle=False)
    te_loader = DataLoader(te_set, batch_size=args.batch, shuffle=False)

    channels = tuple(int(x.strip()) for x in args.channels.split(",") if x.strip())
    model = TCNClassifier(n_ch=N_CH, n_classes=len(LABELS), channels=channels, kernel_size=args.kernel, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    ce = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum, n, correct = 0.0, 0, 0
        for X, yy in tr_loader:
            X, yy = X.to(device), yy.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = ce(logits, yy)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * X.size(0)
            n += X.size(0)
            correct += (torch.argmax(logits, 1) == yy).sum().item()

        tr_loss = float(loss_sum / max(n, 1))
        tr_acc = float(correct / max(n, 1))
        va_loss, va_acc, _, _ = eval_loop(model, va_loader, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"[{ep:03d}/{args.epochs}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping. Best val_loss={best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_acc, y_true, y_pred = eval_loop(model, te_loader, device)
    cm = confusion_matrix(y_true, y_pred)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0

    print("\n=== TEST ===")
    print(f"test_loss={te_loss:.4f} test_acc={te_acc:.4f} macro_f1={macro_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4))

    plot_confusion(cm, result_dir / "confusion_matrix.png")
    plot_learning(history, result_dir / "learning_curves.png")

    best_path = result_dir / "best_model.pt"
    torch.save(model.state_dict(), best_path)

    metrics = {
        "labels": LABELS,
        "test_loss": float(te_loss),
        "test_acc": float(te_acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "train_size": int(len(tr_set)),
        "val_size": int(len(va_set)),
        "test_size": int(len(te_set)),
        "device": str(device),
        "bandpass_8_30": bool(bandpass),
        "zscore": bool(zscore),
        "seed": int(args.seed),
        "model": "TCN",
        "tcn_channels": list(channels),
        "kernel_size": int(args.kernel),
        "dropout": float(args.dropout),
    }
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(f"  {(result_dir/'confusion_matrix.png').resolve()}")
    print(f"  {(result_dir/'learning_curves.png').resolve()}")
    print(f"  {best_path.resolve()}")
    print(f"  {(result_dir/'metrics.json').resolve()}")

if __name__ == "__main__":
    main()
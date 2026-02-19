#!/usr/bin/env python3
"""Compare metrics.json outputs from EEGNet v2, ShallowConvNet, and TCN and save into result_compare/.

Reads:
  result_eegnet_v2/metrics.json
  result_shallowconvnet/metrics.json
  result_tcn/metrics.json

Writes to:
  result_compare/comparison.json
  result_compare/comparison.csv
  result_compare/comparison_acc.png
  result_compare/comparison_macro_f1.png
"""

from pathlib import Path
import json
import csv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]  # project root
OUTDIR = ROOT / "result_compare"
OUTDIR.mkdir(parents=True, exist_ok=True)

def load_metrics(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def save_bar(values, labels, title, ylabel, out_path: Path):
    # Do not set explicit colors (matplotlib defaults)
    plt.figure()
    x = list(range(len(labels)))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0.95, 1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    candidates = [
        ("EEGNet_v2", ROOT / "result_eegnet_v2" / "metrics.json"),
        ("ShallowConvNet", ROOT / "result_shallowconvnet" / "metrics.json"),
        ("TCN", ROOT / "result_tcn" / "metrics.json"),
    ]

    rows = []
    for name, p in candidates:
        m = load_metrics(p)
        if m is None:
            rows.append({"model": name, "found": False, "path": str(p)})
            continue
        rows.append({
            "model": name,
            "found": True,
            "path": str(p),
            "test_acc": m.get("test_acc"),
            "macro_f1": m.get("macro_f1"),
            "test_loss": m.get("test_loss"),
            "seed": m.get("seed"),
            "bandpass_8_30": m.get("bandpass_8_30"),
            "zscore": m.get("zscore"),
        })

    print("\n=== MODEL COMPARISON ===")
    print(f"{'Model':<16} {'Found':<6} {'Acc':<10} {'MacroF1':<10} {'Loss':<10} {'Seed':<6}")
    for r in rows:
        if not r.get("found"):
            print(f"{r['model']:<16} {'No':<6} {'-':<10} {'-':<10} {'-':<10} {'-':<6}  ({r['path']})")
        else:
            acc = float(r.get("test_acc", 0.0))
            mf1 = r.get("macro_f1")
            mf1 = float(mf1) if mf1 is not None else float("nan")
            loss = float(r.get("test_loss", 0.0))
            seed = r.get("seed")
            print(f"{r['model']:<16} {'Yes':<6} {acc:<10.4f} {mf1:<10.4f} {loss:<10.4f} {seed:<6}")

    (OUTDIR / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with open(OUTDIR / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model","found","path","test_acc","macro_f1","test_loss","seed","bandpass_8_30","zscore"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    found = [r for r in rows if r.get("found")]
    if found:
        labels = [r["model"] for r in found]
        accs = [float(r.get("test_acc", 0.0)) for r in found]
        f1s = [float(r.get("macro_f1", 0.0)) for r in found]
        save_bar(accs, labels, "Test Accuracy Comparison", "Accuracy", OUTDIR / "comparison_acc.png")
        save_bar(f1s, labels, "Macro-F1 Comparison", "Macro-F1", OUTDIR / "comparison_macro_f1.png")
        print(f"Saved: {(OUTDIR/'comparison_acc.png').resolve()}")
        print(f"Saved: {(OUTDIR/'comparison_macro_f1.png').resolve()}")

    print(f"Saved: {(OUTDIR/'comparison.json').resolve()}")
    print(f"Saved: {(OUTDIR/'comparison.csv').resolve()}")

if __name__ == "__main__":
    main()

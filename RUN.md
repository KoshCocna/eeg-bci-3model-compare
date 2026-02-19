# EEG BCI 3-Model Compare (EEGNet v2 / ShallowConvNet / TCN)

## Folder structure
- scripts/ : all python scripts
- data/ : generated (fake) MI epochs (CSV)
- result_eegnet_v2/ : EEGNet v2 outputs
- result_shallowconvnet/ : ShallowConvNet outputs
- result_tcn/ : TCN outputs
- result_compare/ : comparison outputs (json/csv + bar charts)

## 0) Environment (example)
```bash
pip install numpy pandas matplotlib scikit-learn torch
# optional for bandpass:
pip install scipy
```

## 1) Generate dataset (fake)
```bash
python scripts/generate_fake_mi_epochs_300.py --out data --per_label 300
```
> CSV files will be saved under `data/<label>/`.

## 2) Train models
```bash
python scripts/train_eegnet_mi.py
python scripts/train_shallowconvnet_mi.py
python scripts/train_tcn_mi.py
```

## 3) Compare
```bash
python scripts/compare_models.py
```
Outputs are saved to `result_compare/`.

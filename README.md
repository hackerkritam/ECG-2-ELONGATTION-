# PhysioDyn

PhysioDyn is a research code scaffold for a paper titled "Physiologically-Constrained Generative Models for Long-Duration ECG Synthesis and Robust HRV Reconstruction".

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Project layout

- `physiodyn/data/` – MIT-BIH download and preprocessing logic
- `physiodyn/models/` – BMM, IDM, and composer modules
- `physiodyn/losses/` – IDM objective functions
- `physiodyn/metrics/` – beat and HRV metrics
- `physiodyn/train/` – training pipelines
- `physiodyn/eval/` – evaluation scripts
- `configs/` – YAML hyperparameters
- `scripts/` – stage runners and evaluation scripts
- `outputs/` – checkpoints, figures, tables, logs
- `results/` – auto-generated result summary

## Order of commands

```bash
python scripts/run_all.py --stage download
python scripts/run_all.py --stage preprocess
python scripts/run_all.py --stage train_bmm
python scripts/run_all.py --stage eval_bmm
python scripts/run_all.py --stage train_idm
python scripts/run_all.py --stage eval_idm
python scripts/run_all.py --stage compose
python scripts/run_all.py --stage eval_composer
python scripts/run_all.py --stage make_tables
```

For a smoke test on CPU, run:

```bash
python scripts/run_all.py --quick --stage all
```

## Runtime notes

- Full training on the MIT-BIH Arrhythmia Database is computationally intensive.
- The repository is structured for CPU and GPU execution; CUDA is auto-detected.
- Expect longer runtimes on full-length IDM training and full ECG composition.

## Paper mapping

- Table 3 / Figure 2: `scripts/eval_bmm.py`
- Table 4 / Figures 3-5: `scripts/eval_idm.py`
- Figures 6-7: `scripts/eval_composer.py`
- Pipeline driver: `scripts/run_all.py`

## Notes

This project implements the requested scaffold and module interfaces, but actual paper reproduction requires the MIT-BIH dataset and a full training run. No numerical claims are reported here without code execution.

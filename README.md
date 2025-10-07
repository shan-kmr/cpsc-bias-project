# CPSC 4640: Multi-Objective Bias Analysis for E-Commerce Recommendations

This project audits visibility bias in e-commerce recommenders using the OTTO dataset and computes descriptive metrics (exposure disparity, coverage, position-bias) with a starter notebook and visualizations.

## Setup

1. Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure the Hugging Face token (never commit this):
```bash
export HF_TOKEN="<your_token>"
```

## Data Access

We use `datasets` and `huggingface_hub` to stream a subset of OTTO interactions. If Kaggle is preferred, download locally to `data/`.

## Run the prototype

```bash
jupyter lab
```
Open `notebooks/01_prototype_otto_bias.ipynb` and run all cells.

## Repo Structure
- `notebooks/` prototype and analysis notebooks
- `src/` reusable metrics code (future)
- `data/` raw or cached data (gitignored)
- `figures/` generated plots (gitignored)

## Notes
- Do not commit secrets. Use `HF_TOKEN` env var.
- Start with small samples to validate pipeline before scaling.

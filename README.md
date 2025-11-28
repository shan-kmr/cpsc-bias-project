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

Place small samples of OTTO sessions as JSONL in `data/` (e.g., `otto-recsys-train.jsonl`, `otto-recsys-test.jsonl`). Large JSONL files are gitignored.
Set an optional sample cap via `MAX_SESSIONS` (default 300):
```bash
export MAX_SESSIONS=500
```

## Run the prototype

```bash
jupyter lab
```
Open `notebooks/01_prototype_otto_bias.ipynb` and run all cells.

### Run the interactive dashboard

Install dependencies (once), then launch the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run src/dashboard_app.py
```
Controls are in the left sidebar (sample size, head fraction, early window, etc.). You can download CSV/HTML exports for figures and metrics.

### What the notebook does
- Loads a tiny sample from local JSONL files; falls back to synthetic if absent.
- Computes bias metrics: exposure Gini, long-tail (tail-share), position-bias proxy.
- Produces plots tied to research questions (RQ1–RQ3) with printed interpretations.

### How to interpret outputs
- Gini higher → stronger visibility concentration (bias). Tail-share lower → weaker long-tail exposure.
- Decile bars: rising top-decile share from clicks→orders suggests popularity amplification.
- Position curve: steeper negative slope → stronger position bias.
- Early→later trend: positive slope indicates compounding visibility advantages over time.

## Repo Structure
- `notebooks/` prototype and analysis notebooks
- `src/` reusable metrics code (future)
- `data/` raw or cached data (gitignored)
- `figures/` generated plots (gitignored)

## Notes
- Do not commit secrets. Use `HF_TOKEN` env var.
- Start with small samples to validate pipeline before scaling.

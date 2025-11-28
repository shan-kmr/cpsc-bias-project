# CPSC 4640: Multi-Objective Bias Analysis for E-Commerce Recommendations

A data science project that audits **visibility bias** in e-commerce recommender systems using the [OTTO dataset](https://www.kaggle.com/competitions/otto-recommender-system). We investigate how exposure inequality evolves across the customer funnel (clicks → carts → orders) and whether a "rich-get-richer" feedback loop amplifies popularity bias over time.

---

## Table of Contents

- [Motivation](#motivation)
- [Research Questions](#research-questions)
- [Methodology](#methodology)
- [Metrics](#metrics)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Privacy Considerations](#privacy-considerations)
- [References](#references)

---

## Motivation

Recommender systems power product discovery on e-commerce platforms, but they can inadvertently create **visibility bias**—a phenomenon where popular items receive disproportionate exposure, crowding out long-tail products. This creates a feedback loop:

```
Popular items → More exposure → More clicks → Even more popularity → ...
```

Understanding and measuring this bias is critical for:
- **Fairness**: Ensuring smaller sellers/niche products get reasonable visibility
- **Diversity**: Preventing filter bubbles and monotonous recommendations
- **Business health**: Long-tail items often have higher margins and better match niche user needs

---

## Research Questions

| RQ | Question |
|----|----------|
| **RQ1** | How does visibility bias evolve across clicks, carts, and orders? |
| **RQ2** | Which popularity segments (head vs. long-tail) are most impacted at each funnel stage? |
| **RQ3** | Does early session exposure compound into later-stage advantages (rich-get-richer)? |

---

## Methodology

### Data

We use the **OTTO Recommender System dataset**, which contains anonymized e-commerce session data with three event types:
- `click` — user viewed a product
- `cart` — user added product to cart
- `order` — user purchased the product

Each session is a sequence of timestamped events with item IDs (`aid`).

### Analysis Pipeline

1. **Load & preprocess** session data from JSONL files
2. **Compute exposure counts** per item for each event type
3. **Calculate inequality metrics** (Gini, HHI, tail-share)
4. **Bin items into popularity deciles** based on click counts
5. **Analyze position bias** using intra-session event positions
6. **Measure temporal compounding** by comparing early-exposure deciles to later-stage shares

---

## Metrics

### Gini Coefficient

Measures inequality in exposure distribution. Ranges from 0 (perfect equality) to 1 (maximum inequality).

$$
G = \frac{\sum_{i=1}^{n} \sum_{j=1}^{n} |x_i - x_j|}{2n^2 \bar{x}}
$$

where $x_i$ is the exposure count for item $i$ and $\bar{x}$ is the mean exposure.

### Herfindahl-Hirschman Index (HHI)

Measures market concentration. Higher values indicate more concentration.

$$
HHI = \sum_{i=1}^{n} s_i^2
$$

where $s_i = x_i / \sum_j x_j$ is the exposure share of item $i$.

### Tail-Share

Fraction of total exposure captured by the bottom 90% of items (long-tail). Lower values indicate stronger head dominance.

$$
TailShare_{10\%} = \frac{\sum_{i \in Tail} x_i}{\sum_{j} x_j}
$$

### Position-Bias Proxy

Engagement rate by intra-session position. A negative slope indicates items shown earlier receive more engagement.

### Early-Exposure Lift

Ratio of later-stage share for top early-exposure decile (E10) vs. bottom decile (E1):

$$
\text{Lift} = \frac{Share_{E10}}{Share_{E1}}
$$

Values > 1 suggest compounding visibility advantages.

---

## Project Structure

```
.
├── data/                          # Raw data (gitignored)
│   ├── otto-recsys-train.jsonl
│   └── otto-recsys-test.jsonl
├── figures/                       # Generated plots (gitignored)
│   └── flow.mmd
├── notebooks/
│   └── 01_prototype_otto_bias.ipynb   # Main analysis notebook
├── src/
│   ├── __init__.py
│   └── dashboard_app.py           # Streamlit interactive dashboard
├── requirements.txt
├── main.tex                       # LaTeX report (if applicable)
└── README.md
```

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
# Optional: Hugging Face token for future dataset access
export HF_TOKEN="<your_token>"

# Optional: Limit session sample size (default: 300)
export MAX_SESSIONS=500
```

### 4. Add data

Place OTTO session files in the `data/` directory:
- `otto-recsys-train.jsonl`
- `otto-recsys-test.jsonl`

If no data is present, the code falls back to synthetic data with similar statistical properties.

---

## Usage

### Run the Jupyter Notebook

```bash
jupyter lab
```

Open `notebooks/01_prototype_otto_bias.ipynb` and run all cells. The notebook:
- Loads session data and computes exposure counts
- Calculates Gini, tail-share, and other bias metrics
- Generates visualizations for each research question
- Prints interpretations of results

### Run the Interactive Dashboard

```bash
streamlit run src/dashboard_app.py
```

The dashboard provides:

| Feature | Description |
|---------|-------------|
| **Stage-wise KPIs** | Gini, HHI, and tail-share cards for each event type |
| **Exposure curves** | Sorted exposure distributions showing inequality |
| **Lorenz curves** | Cumulative exposure vs. cumulative items (inequality visualization) |
| **Position-bias plot** | Engagement rate by intra-session position |
| **Decile bar charts** | Head vs. tail exposure shares across funnel stages |
| **Conversion rates** | click→cart, click→order, cart→order by popularity decile |
| **Early→later trends** | Compounding analysis with slope metrics |
| **Temporal drift** | Gini over time (hourly/daily buckets) |
| **Export buttons** | Download CSV/HTML for all figures and metrics |

**Sidebar controls**: sample size, head fraction, early window, decile bins, scatter cap, time bucket.

---

## Key Findings

Based on our analysis of the OTTO dataset:

### RQ1: Visibility Bias Across the Funnel

| Stage | Gini | Tail-Share (90%) |
|-------|------|------------------|
| Click | ~0.37 | ~0.66 |
| Cart  | ~0.89 | ~0.18 |
| Order | ~0.95 | ~0.00 |

**Interpretation**: Exposure inequality **increases monotonically** across the funnel. By the order stage, nearly all exposure is concentrated in the head items.

### RQ2: Head vs. Long-Tail Concentration

- Top popularity decile (D10) captures ~34% of clicks, ~36% of carts, ~33% of orders
- Combined with rising Gini, this shows sharper concentration even when top-decile share appears stable

### RQ3: Rich-Get-Richer Compounding

- Items with high early-session exposure (E10) capture disproportionately more later-stage events
- Positive slope from E1→E10 for all event types, strongest for orders
- Early-exposure lift (E10/E1) > 1 confirms compounding advantages

**Conclusion**: The data supports the **rich-get-richer hypothesis**—popular items accumulate visibility advantages that compound through the funnel.

---

## Privacy Considerations

The OTTO dataset contains behavioral traces that could enable re-identification:

| Signal | Risk |
|--------|------|
| Session IDs | Linkability across events |
| Timestamps | Activity pattern fingerprinting |
| Behavioral sequences | Quasi-identifiers via rare paths |
| Item preferences | Sensitive interest inference |

### Mitigation Strategies (for production dashboards)

- **k-anonymity**: Suppress cells with n < k (e.g., k ≥ 50)
- **Differential privacy**: Add calibrated noise to public metrics
- **Aggregation**: Show only weekly, site-wide summaries
- **On-device computation**: User-specific views computed locally
- **Access controls**: Role-based access with audit logs

---

## References

- [OTTO Recommender System Competition (Kaggle)](https://www.kaggle.com/competitions/otto-recommender-system)
- Abdollahpouri, H., et al. (2020). *Multistakeholder Recommendation: Survey and Research Directions*
- Biega, A., et al. (2018). *Equity of Attention: Amortizing Individual Fairness in Rankings*
- Singh, A., & Joachims, T. (2018). *Fairness of Exposure in Rankings*
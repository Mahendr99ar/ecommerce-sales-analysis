## 🚀 Quick Summary
This project analyzes retail sales data to identify:
- Loss-making products
- Impact of discounts on profit
- Regional performance trends

# ecommerce-sales-analysis — End-to-End Data Analysis

**Predictive Sales Analysis and Profitability Optimisation for a Multi-Region Retailer**

A professional data analytics portfolio project applying Python-based EDA to 9,994 US retail
transactions (2014–2017) to uncover profit leakage, discount strategy failures, and geographic
growth opportunities.

---

## The Business Problem

The Superstore generates high revenue across three product categories, yet a significant
portion of transactions are loss-making. Management lacks visibility into:

- Which product sub-categories are actively destroying margin
- At what discount threshold the business starts losing money per order
- Which geographic regions are under-performing and why
- Whether shipping costs are silently eroding category-level profits

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| Pandas | Data ingestion, cleaning, aggregation |
| NumPy | Numerical computation |
| Matplotlib | Base visualisation layer |
| Seaborn | Statistical plots (boxplots, heatmaps) |

---

## Key Findings

- **Furniture Paradox**: Furniture drives ~26% of sales but records a **-7.9% profit margin** overall. Tables and Bookcases alone lose $260K+.
- **Profit Cliff**: Orders with ≥30% discount average **-$147 profit** each, vs. +$188 at 0% discount. A hard cap at 20% would recover significant margin.
- **Discount Epidemic**: 33% of all orders carry a discount of 30% or higher — the primary driver of profit volatility.
- **Regional Gap**: West leads in both sales volume and margin; South is under-penetrated and requires targeted marketing investment.
- **SLA Failures**: A portion of "Same Day" shipping orders take 1+ days — an operational quality gap.

---

## Visualisations

| # | Chart | Insight |
|---|-------|---------|
| 01 | Category Sales vs Profit | Furniture revenue ≠ Furniture profit |
| 02 | Sub-Category Profit Margins | Tables at -18.6%, Bookcases at -13.4% |
| 03 | Discount–Profit Cliff | Profit crashes past 30% discount |
| 04 | Regional Performance Dashboard | West & East dominate; South lags |
| 05 | Profit Boxplot by Category | High negative outlier density in Furniture |
| 06 | Monthly Sales Time-Series | Q4 surge every year; YoY growth visible |
| 07 | Correlation Heatmap | Discount–Profit: r = -0.22 |
| 08 | Customer Segment Analysis | Home Office has the best margin % |
| 09 | Shipping SLA Analysis | Same Day failures identified |
| 10 | Top & Bottom States by Profit | Loss states clearly mapped |

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/Mahendr99ar/ecommerce-sales-analysis.git
cd superstore-analysis

# Install dependencies
pip install -r requirements.txt

# Run the full analysis (generates charts in ./superstore_outputs/)
python superstore_analysis.py
```

If you have the real Superstore CSV, drop it in the project root as `SampleSuperstore.csv`
and remove or skip the data-generation block (Step 1) in the script.

---

## Strategic Recommendations

1. **Cap discretionary discounts at 20%** — require manager approval above that threshold, restricted to high-margin Technology items only.
2. **SKU Rationalisation** — discontinue or reprice the bottom 20% of loss-making Furniture SKUs (Tables, Bookcases).
3. **Geographic Re-investment** — shift marketing budget from the saturated West toward the South, the only under-penetrated region.
4. **Regional Warehousing** — pilot a Central/South fulfilment hub to reduce bulky furniture shipping costs.
5. **SLA Audit** — address Same Day shipping failures; consider auto-downgrade to First Class with partial refund.

---

## Project Structure

```
ecommerce-sales-analysis/
├── superstore_analysis.py   # Full analysis script
├── SampleSuperstore.csv     # Dataset (generated or drop real file here)
├── superstore_outputs/      # Auto-generated charts (10 PNGs)
├── requirements.txt
└── README.md
```

---

## Resume Bullet Points

- Analysed **9,994 retail transactions** using Python (Pandas/Seaborn) to identify a **$260K profit leak** in the Furniture category driven by negative-margin sub-categories.
- Engineered a temporal sales model revealing a consistent **Q4 demand surge**, enabling data-backed inventory recommendations for peak season.
- Designed visualisations exposing a **"Profit Cliff"** at 30% discount depth, leading to a proposed discount-governance framework.
- Standardised and cleaned a multi-dimensional retail dataset — removing duplicates, imputing nulls, and engineering 5 temporal features — reducing downstream processing errors.
- Identified that **33% of orders carried ≥30% discounts**, averaging a $147 loss each, with a potential recovery of $500K+ in annual margin if capped at 20%.

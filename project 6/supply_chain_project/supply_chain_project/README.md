# Project 6 — Supply Chain & Inventory Demand Analysis

## Quick Start (Windows)

```
# 1. Open Command Prompt or PowerShell
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 2. Run the full analysis
python main.py

# 3. (Optional) Open the notebook
jupyter notebook notebooks/Supply_Chain_Analysis.ipynb
```

---

## Project Structure

```
supply_chain_project/
├── main.py                        ← Entry point — run this
├── README.md
├── data/
│   ├── sales_inventory.csv        ← Generated: 32,880 records
│   └── products.csv               ← 30 products × 6 categories
├── src/
│   ├── generate_data.py           ← Synthetic dataset generator
│   ├── analysis.py                ← Core analytics module
│   └── visualizations.py          ← All charts (Matplotlib + Seaborn)
├── notebooks/
│   └── Supply_Chain_Analysis.ipynb ← Step-by-step Jupyter notebook
└── outputs/                        ← All PNG charts saved here
    ├── 01_dashboard_overview.png
    ├── 02_demand_trends_seasonality.png
    ├── 03_inventory_turnover.png
    ├── 04_stockout_overstock.png
    ├── 05_demand_forecast.png
    └── 06_eoq_optimization.png
```

---

## Pipeline (9 Steps)

| Step | Description |
|------|-------------|
| 1 | Synthetic dataset generation (3 years × 30 products) |
| 2 | Preprocessing & validation (null checks, feature engineering) |
| 3 | Demand trend & seasonality analysis |
| 4 | Inventory turnover (DSI, turnover ratio, fill rate) |
| 5 | Stockout & overstock pattern identification |
| 6 | Polynomial regression demand forecasting (12-month horizon) |
| 7 | EOQ (Economic Order Quantity) analysis |
| 8 | 6-panel visualization suite (dark theme) |
| 9 | Supply chain optimization insights |

---

## Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Transaction date |
| product_id | str | Unique product code |
| category | str | Electronics / Apparel / Home & Kitchen / Automotive / Sports / FMCG |
| units_demanded | int | Customer demand |
| units_sold | int | Actual units sold |
| lost_sales | int | Units lost due to stockout |
| inventory_level | int | End-of-day inventory |
| revenue | float | Daily revenue |
| holding_cost | float | Daily holding cost |
| stockout_flag | int | 1 = stockout event |
| fill_rate | float | units_sold / units_demanded |

---

## Key Metrics Produced

- **Inventory Turnover Ratio** — COGS / Avg Inventory
- **Days Sales of Inventory (DSI)** — 365 / Turnover
- **Seasonality Index** — Monthly demand vs. annual average
- **Fill Rate / Service Level** — % demand fulfilled
- **EOQ** — Optimal order quantity minimising total cost
- **Demand Forecast (R², MAE, RMSE)** — Polynomial regression model

---

## Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
jupyter (optional, for notebook)
```

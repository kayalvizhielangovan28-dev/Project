# 📊 Marketing Campaign Performance Analysis

Industry-oriented data science project measuring campaign ROI, reach,
engagement, and customer behaviour — built for Data Analyst / Marketing
Analyst / Business Analyst portfolios.

## Quick Start (Windows)

```
Double-click  RUN_PROJECT.bat
```

That single step installs dependencies, generates data, runs all analysis,
and opens the outputs folder automatically.

## Manual Steps

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset (5 000 rows, 24 columns)
python src/generate_data.py

# 3. Run full analysis (5 charts + insights printed to console)
python src/analysis.py

# 4. (Optional) Interactive notebook
jupyter notebook notebooks/marketing_analysis.ipynb
```

## Project Structure

```
marketing_campaign_analysis/
├── RUN_PROJECT.bat            # ← Windows one-click launcher
├── requirements.txt
├── data/
│   └── marketing_campaigns.csv   # auto-generated
├── src/
│   ├── generate_data.py          # synthetic dataset builder
│   └── analysis.py               # full analysis pipeline
├── notebooks/
│   └── marketing_analysis.ipynb  # interactive Jupyter notebook
└── outputs/
    ├── fig1_executive_dashboard.png
    ├── fig2_campaign_deepdive.png
    ├── fig3_customer_behaviour.png
    ├── fig4_ml_insights.png
    ├── fig5_budget_efficiency.png
    └── campaign_summary.csv
```

## What's Inside

| Figure | Content |
|--------|---------|
| Fig 1 | Executive Dashboard — revenue, ROI, funnel, monthly trend |
| Fig 2 | Campaign Deep-Dive — spend distribution, ROAS heatmap, CPA by device, ROI violin |
| Fig 3 | Customer Behaviour — segment, device, region, gender, bounce rate |
| Fig 4 | ML Insights — K-Means clusters (elbow + PCA), spend→revenue regression |
| Fig 5 | Budget Efficiency — utilisation vs ROI, quarterly P&L, correlation matrix |

## Dataset Features (24 columns)

`campaign_id`, `campaign_name`, `channel`, `date`, `month`, `quarter`,
`customer_segment`, `device_type`, `region`, `gender`, `impressions`,
`clicks`, `conversions`, `spend`, `revenue`, `ctr`, `cvr`, `roi`,
`cpa`, `roas`, `bounce_rate`, `avg_time_on_site_s`, `avg_pages_per_visit`,
`budget_allocated`

## Key KPIs Computed

- **CTR** — Click-Through Rate
- **CVR** — Conversion Rate
- **ROI** — Return on Investment
- **ROAS** — Return on Ad Spend
- **CPA** — Cost Per Acquisition
- **Profit** — Revenue minus Spend
- **Budget Utilisation** — Spend / Budget Allocated

## Skills Demonstrated

- Data cleaning & feature engineering (Pandas / NumPy)
- Exploratory data analysis & statistical summaries
- Business KPI calculation & interpretation
- Multi-campaign comparison with heatmaps & violin plots
- Customer segmentation (K-Means + PCA)
- Predictive regression (scikit-learn)
- Professional visualisation (Matplotlib + Seaborn)
- Actionable marketing insights derivation

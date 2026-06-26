# Project 7: Financial Data Analysis & Risk Assessment

## Overview
An end-to-end financial analytics system that ingests, cleans, analyses,
and classifies risk in a loan/credit portfolio using Python data science tools.

---

## Windows Quick Start

### 1. Install Python 3.10+ (if not already)
Download from https://www.python.org/downloads/ — tick **"Add Python to PATH"**.

### 2. Install dependencies
Open **Command Prompt** or **PowerShell** in this folder and run:

```cmd
pip install -r requirements.txt
```

### 3. Run the project

```cmd
python financial_risk_analysis.py
```

All outputs (charts + report) appear in the `outputs\` folder.

---

## Project Architecture

```
financial_risk_project/
│
├── financial_risk_analysis.py   ← MAIN SCRIPT (run this)
├── requirements.txt             ← Python dependencies
├── README.md                    ← This file
│
└── outputs/                     ← Auto-created on first run
    ├── 01_eda_dashboard.png
    ├── 02_correlation_matrix.png
    ├── 03_feature_risk_correlation.png
    ├── 04_trend_volatility.png
    ├── 05_model_confusion_matrices.png
    ├── 06_feature_importance.png
    ├── 07_risk_score_distribution.png
    ├── 08_sector_grade_risk.png
    ├── 09_sharpe_volatility.png
    ├── 10_pca_kmeans.png
    ├── financial_dataset_raw.csv
    └── financial_risk_report.txt
```

---

## Pipeline (8 Stages)

| Stage | Description |
|-------|-------------|
| 1 | **Dataset Generation** — 2,000 synthetic loans with 19 features |
| 2 | **Data Cleaning** — null handling, outlier capping, feature engineering |
| 3 | **EDA Dashboard** — 9-panel portfolio overview |
| 4 | **Risk Indicators** — correlation analysis (Pearson) |
| 5 | **Trend & Volatility** — 4-panel time-series analysis |
| 6 | **Risk Classification** — Logistic Regression, Random Forest, Gradient Boosting |
| 7 | **Portfolio Visuals** — sector heatmaps, Sharpe charts, PCA + K-Means |
| 8 | **Risk Report** — auto-generated text assessment with recommendations |

---

## Features Used

| Feature | Type | Description |
|---------|------|-------------|
| credit_score | numeric | FICO-style credit score (300–850) |
| debt_to_income | numeric | Monthly debt / monthly income |
| ltv_ratio | numeric | Loan amount / collateral value |
| interest_rate | numeric | Annual interest rate |
| employment_years | numeric | Years of employment |
| num_defaults_history | integer | Historical default count |
| volatility_30d | numeric | 30-day return standard deviation |
| sharpe_ratio | numeric | Risk-adjusted return metric |
| loan_to_income | derived | Loan amount / annual income |
| risk_premium | derived | Interest rate – risk-free rate |
| collateral_cover | derived | Collateral value / loan amount |
| grade_ordinal | encoded | Ordinal credit grade (1=AAA…7=CCC) |

---

## Model Results (5-Fold Cross-Validation)

| Model | CV Accuracy |
|-------|-------------|
| Logistic Regression | ~81.7% |
| Random Forest | ~83.8% |
| **Gradient Boosting** | **~84.0%** ← Best |

---

## Risk Classification Rules

| Risk Score | Label | Action |
|------------|-------|--------|
| 0 – 33 | Low | Standard monitoring |
| 34 – 66 | Medium | Enhanced review |
| 67 – 100 | High | Immediate escalation |

---

## Tools Used
- **Python** — Core language
- **Pandas** — Data ingestion, cleaning, feature engineering
- **NumPy** — Numerical computation, synthetic data generation
- **Matplotlib** — All charting and figure composition
- **Seaborn** — Heatmaps and KDE plots
- **Scikit-learn** — ML models, PCA, K-Means, scaling, evaluation

---

## Career Relevance
- **Data Analyst** — EDA, cleaning, visualization pipeline
- **Financial Analyst** — Exposure analysis, volatility metrics, sector risk
- **Risk Analyst** — Risk scoring, classification, assessment report

---

## Extending the Project
1. **Replace synthetic data** — point the script at a real CSV (Lending Club, etc.)
2. **Add a Jupyter Notebook** — drop `financial_risk_analysis.py` cells into a `.ipynb`
3. **Deploy the model** — export GBM with `joblib.dump()` and wrap in a Flask API
4. **Dashboard** — port charts to Streamlit for an interactive web app

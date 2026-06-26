# Industry-Oriented Fraud Detection Analysis
### Project 3 — Data Analytics Portfolio

---

## Overview
An end-to-end fraud detection system using real-world data science tools:
Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn

Simulates 50,000 banking/e-commerce transactions with realistic 3.5% fraud rate.

---

## Project Structure
```
fraud_detection/
├── fraud_detection_project.py   ← Main project script
├── requirements.txt             ← Python dependencies
├── run_project.bat              ← Windows one-click runner
├── data/
│   └── transactions.csv         ← Auto-generated dataset
└── outputs/
    ├── 1_eda_analysis.png       ← Exploratory Data Analysis (9 charts)
    ├── 2_anomaly_detection.png  ← Isolation Forest anomaly flags
    ├── 3_model_evaluation.png   ← ML model performance metrics
    └── 4_fraud_dashboard.png    ← Executive fraud dashboard
```

---

## How to Run (Windows)

**Option A — Double-click:**
```
run_project.bat
```

**Option B — Command Prompt / PowerShell:**
```bash
pip install -r requirements.txt
python fraud_detection_project.py
```

**Option C — Jupyter Notebook:**
Copy the script into a `.ipynb` notebook and run cell by cell.

---

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | Data Generation | 50,000 synthetic transactions with realistic fraud patterns |
| 2 | Data Cleaning | Handle nulls, duplicates, and outlier capping |
| 3 | Feature Engineering | 8 fraud indicator features + risk score |
| 4 | EDA | 9-panel visualization of transaction behavior |
| 5 | Anomaly Detection | Isolation Forest (unsupervised, no labels needed) |
| 6 | ML Classification | Logistic Regression + Random Forest with full evaluation |
| 7 | Dashboard | Executive KPI strip + fraud trend heatmaps |

---

## Fraud Indicators Engineered

| Feature | Logic |
|---------|-------|
| `is_odd_hour` | Transaction between 10 PM – 5 AM |
| `is_high_amount` | Above 90th percentile amount |
| `is_new_account` | Account age < 90 days |
| `is_low_txn_history` | Fewer than 3 prior transactions |
| `is_far_from_home` | Distance > 100 km |
| `is_international` | International transaction flag |
| `risk_score` | Weighted composite of all above |

---

## Models Used

- **Isolation Forest** — Unsupervised anomaly detection; no fraud labels needed. Achieves ~87% fraud overlap in flagged records.
- **Logistic Regression** — Baseline supervised classifier with class balancing.
- **Random Forest** — Ensemble classifier; provides feature importance ranking.

---

## Job Relevance
This project directly maps to real-world work in:
- **Fraud Analyst** — Pattern analysis, threshold tuning, anomaly investigation
- **Data Analyst** — EDA, trend dashboards, KPI reporting
- **Data Scientist** — Feature engineering, ML modeling, model evaluation

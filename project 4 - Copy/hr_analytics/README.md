# 🏢 HR Analytics — Employee Attrition Prediction
**Industry-Oriented Data Science Project**

---

## 📌 Project Overview

Organizations face high costs due to employee attrition. This project uses data science techniques to:
- Analyze HR data and identify key attrition drivers
- Predict which employees are at risk of leaving
- Provide actionable HR recommendations

---

## 🗂️ Project Structure

```
hr_analytics/
│
├── data/
│   ├── generate_dataset.py     # Synthetic IBM-style HR data generator
│   └── HR_Analytics.csv        # Generated dataset (1,470 employees, 35 features)
│
├── notebooks/
│   └── HR_Analytics_Notebook.ipynb   # Interactive Jupyter Notebook
│
├── outputs/                    # All charts & reports (auto-generated)
│   ├── 01_attrition_overview.png
│   ├── 02_age_income_distribution.png
│   ├── 03_satisfaction_heatmap.png
│   ├── 04_categorical_drivers.png
│   ├── 05_tenure_analysis.png
│   ├── 06_correlation_matrix.png
│   ├── 07_model_comparison.png
│   ├── 08_roc_curves.png
│   ├── 09_best_model_analysis.png
│   ├── 10_feature_importance.png
│   ├── 11_risk_segmentation.png
│   ├── model_metrics.csv
│   ├── feature_importance.csv
│   └── HR_Analytics_Report.txt
│
├── models/                     # Saved ML models
│   ├── random_forest_model.pkl
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── feature_columns.pkl
│
├── hr_attrition_analysis.py    # Main analysis script (full pipeline)
├── requirements.txt            # Python dependencies
├── run_project.bat             # Windows one-click launcher
└── README.md
```

---

## 🚀 Quick Start (Windows)

### Option A — One-Click (Recommended)
Double-click `run_project.bat`

### Option B — Manual
```bat
# Install dependencies
pip install -r requirements.txt

# Generate dataset
python data\generate_dataset.py

# Run full pipeline
python hr_attrition_analysis.py

# Open Jupyter Notebook (optional)
jupyter notebook notebooks\HR_Analytics_Notebook.ipynb
```

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|------|---------|
| **Python 3.8+** | Core language |
| **Pandas** | Data ingestion, cleaning, manipulation |
| **NumPy** | Numerical computation |
| **Matplotlib** | Charts and visualizations |
| **Seaborn** | Statistical plots and heatmaps |
| **Scikit-learn** | ML models, preprocessing, evaluation |
| **Joblib** | Model serialization |
| **Jupyter Notebook** | Interactive analysis environment |

---

## 📊 Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | **Data Ingestion** — Load 1,470-employee HR dataset (35 features) |
| 2 | **Data Cleaning** — Handle missing values, drop constants, encode target |
| 3 | **EDA** — 6 visualization charts covering attrition patterns |
| 4 | **Feature Engineering** — 6 derived features (satisfaction score, career stability, etc.) |
| 5 | **Model Training** — 5 ML models trained with stratified cross-validation |
| 6 | **Model Evaluation** — AUC, F1, precision, recall, confusion matrix, ROC curves |
| 7 | **Feature Importance** — Top 20 attrition drivers identified |
| 8 | **Risk Segmentation** — Employees scored Low / Medium / High risk |
| 9 | **HR Recommendations** — 6 strategic actions for retention |

---

## 🤖 Models Trained

| Model | AUC | Notes |
|-------|-----|-------|
| Logistic Regression | ~0.63 | Baseline linear model |
| Decision Tree | ~0.53 | Interpretable, simple |
| **Random Forest** | **~0.63** | **Best performer** |
| Gradient Boosting | ~0.60 | Ensemble, robust |
| SVM (RBF) | ~0.63 | Non-linear boundary |

---

## 💡 Key Findings

1. **Overtime** is the single strongest behavioural driver of attrition
2. **Low job satisfaction** (score 1–2) nearly doubles attrition risk
3. **New employees** (<2 years) are at highest risk
4. **Frequent travellers** leave at disproportionately high rates
5. **Low monthly income** is a persistent top-5 driver

---

## 🎯 Strategic Recommendations

1. **Overtime Policy** — Cap mandatory overtime; introduce comp-time
2. **Early Tenure Programme** — 90-day, 6-month, 12-month structured check-ins
3. **Pulse Surveys** — Quarterly satisfaction surveys; flag score ≤2 for HR follow-up
4. **Travel Review** — Revisit allowances and recovery days for frequent travellers
5. **Pay Benchmarking** — Annual market surveys; target 60th percentile for key roles
6. **Predictive Dashboard** — Monthly HRIS scoring; retention interviews for >60% risk

---

## 📋 Requirements

- Python 3.8 or higher
- Windows 10 / 11
- ~500MB disk space
- Internet connection (for pip install only)

---

## 👤 Target Roles

This project is directly relevant for:
- **Data Analyst**
- **People Analytics Analyst**
- **HR Data Scientist**
- **Workforce Analytics Specialist**

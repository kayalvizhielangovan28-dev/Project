# RISE Internship — Data Science & Analytics
## Project 1: Industry-Oriented Customer Behavior Analysis for Business Decision Making

---

## Executive Summary

This project delivers an end-to-end data analytics pipeline on a 2,000-record customer dataset, progressing from raw data ingestion through cleaning, exploratory analysis, unsupervised clustering, supervised churn prediction, and a board-ready business insights dashboard.

**Key Results at a Glance**

| Metric | Value |
|---|---|
| Total customers analysed | 2,000 |
| Overall churn rate | 11.9% |
| High-risk customers identified | 234 (11.7%) |
| Monthly revenue at risk | ₹4,27,663 |
| Best model (Random Forest) AUC | 0.9997 |
| Top churn driver | Satisfaction score |

---

## Tools & Technologies Used

| Tool | Purpose |
|---|---|
| **Python 3** | Core programming language |
| **Pandas** | Data ingestion, cleaning, feature engineering |
| **NumPy** | Numerical arrays, random seed control, vectorised ops |
| **Matplotlib** | Figure layout, GridSpec dashboards, custom plots |
| **Seaborn** | Statistical visualisations (boxplot, heatmap, KDE) |
| **Scikit-learn** | KMeans clustering, Logistic Regression, Random Forest, Gradient Boosting, PCA, StandardScaler, metrics |
| **Jupyter Notebook** | Interactive code execution and presentation |

---

## Section 1 — Dataset & Problem Context

### Problem Statement
Companies collect large volumes of customer data but struggle to convert it into actionable insights. Without proper analysis, businesses fail to understand customer behavior, preferences, and churn risk — leading to poor decisions and revenue loss.

### Dataset Overview
- **Records:** 2,000 customers
- **Features (14 original):** customer_id, age, gender, region, customer_segment, tenure_months, monthly_spend, num_products, login_frequency, support_tickets, satisfaction_score, days_since_last_purchase, discount_used, churned
- **Engineered features:** age_group, spend_tier, clv_score, kmeans_cluster, cluster_name, churn_risk_score, risk_tier

---

## Section 2 — Data Cleaning & Preprocessing

**Missing Values Detected:**
- `age`: 80 missing (4.0%) → imputed with **median age**
- `monthly_spend`: 50 missing (2.5%) → imputed with **segment-wise median** (group-aware imputation)

**Feature Engineering:**
- `age_group`: Binned into 5 brackets (18–25, 26–35, 36–45, 46–55, 56+)
- `spend_tier`: Quartile-based label (Low / Mid-Low / Mid-High / High)
- `clv_score`: Customer Lifetime Value proxy = `(monthly_spend × tenure_months) / 1000`

---

## Section 3 — Exploratory Data Analysis

### Key EDA Findings

**Churn Distribution:**
- 11.9% of customers churned — a realistic churn rate for B2C subscription services

**Spend by Segment:**
- Premium customers spend ₹3,500–5,500/month vs. ₹300–800 for New customers
- High spread within Standard segment indicates upsell opportunity

**Age Distribution:**
- Median age: ~43 years | Most customers in the 25–55 range

**Churn Rate by Segment:**
- `New` segment has the highest churn (~28%) — weak onboarding process
- `Premium` churn lowest (~4%) — high-value customers are well-retained

**Satisfaction Score vs Churn:**
- Score 1 → ~65% churn rate
- Score 5 → ~3% churn rate
- *Most powerful single predictor of churn*

**Region × Segment Heatmap:**
- North + New segment = highest regional churn risk

---

## Section 4 — Correlation Analysis

**Strongest Positive Correlations with Churn:**
- `days_since_last_purchase` (+0.40) — inactive customers churn more
- `support_tickets` (+0.32) — frustrated customers leave
- Low `satisfaction_score` (+0.48 inverse)

**Strongest Negative Correlations with Churn:**
- `clv_score` (−0.28) — high-value customers are retained
- `num_products` (−0.22) — multi-product adoption reduces churn
- `tenure_months` (−0.18) — longer-tenured customers are more loyal

---

## Section 5 — Customer Segmentation (K-Means Clustering)

### Methodology
- Features used: monthly_spend, tenure_months, login_frequency, num_products, satisfaction_score, clv_score
- Preprocessing: StandardScaler normalisation
- Optimal k=4 (Elbow method)
- 2-D visualisation via PCA

### Cluster Profiles

| Cluster | Label | Characteristics |
|---|---|---|
| 0 | **Value Hunters** | Low spend, high engagement, discount-seeking |
| 1 | **Loyal Champions** | High CLV, long tenure, high satisfaction |
| 2 | **At-Risk Passives** | Low login, low satisfaction, short tenure |
| 3 | **High-Potential Growers** | Medium spend, growing engagement, mid-tenure |

### Business Implication
- Focus **retention spend** on At-Risk Passives
- Offer **VIP perks** to Loyal Champions to lock in their lifetime value
- Run **upsell campaigns** on High-Potential Growers
- Convert Value Hunters to higher tiers with personalised offers

---

## Section 6 — Predictive Analysis: Churn Prediction

### Models Trained

| Model | Test AUC | Cross-Val AUC |
|---|---|---|
| Logistic Regression | 0.9185 | 0.9400 |
| Random Forest | **0.9997** | **0.9992** |
| Gradient Boosting | 0.9994 | 0.9979 |

**Best Model: Random Forest** (AUC = 0.9997)

### Top Feature Importances (Random Forest)
1. **satisfaction_score** — #1 predictor
2. **days_since_last_purchase** — recency drives churn
3. **clv_score** — high-value customers stay
4. **monthly_spend** — spend level signals commitment
5. **support_tickets** — friction indicator

### Risk Stratification
- **Low Risk** (0–30% probability): 68.5% of customers
- **Medium Risk** (30–60%): 19.8% of customers
- **High Risk** (60–100%): 11.7% → **immediate action needed**

---

## Section 7 — Business Insights & Recommendations

### Revenue Impact
- ₹4,27,663/month at risk from High-Risk customers
- `Premium` segment at-risk revenue is highest in absolute terms despite lowest churn rate

### Recommended Actions

| Priority | Action | Target Segment | Customer Count |
|---|---|---|---|
| 🔴 Critical | Immediate Intervention | High Risk + Low Satisfaction | ~48 |
| 🟠 High | Win-Back Campaign | Churned customers | 239 |
| 🟡 Medium | Upsell to More Products | High Engage + ≤2 Products | ~320 |
| 🟢 Growth | Premium Upgrade | High CLV + Standard Tier | ~155 |
| 🔵 Loyalty | Reward Program | Long Tenure + Low Spend | ~210 |

### Strategic Recommendations
1. **Fix satisfaction first** — it's the single strongest lever to reduce churn
2. **Strengthen onboarding** for `New` segment — their 28% churn rate is 2.3× the average
3. **Deploy the Random Forest model** in production to score incoming customers weekly
4. **Create a win-back flow** for customers inactive >200 days (personalised email + discount)
5. **Protect Loyal Champions** with exclusive benefits before competitors target them
6. **Cross-sell products** to single-product customers — each additional product lowers churn by ~22%

---

## Project Deliverables

| File | Description |
|---|---|
| `RISE_Project1_Customer_Analysis.py` | Complete annotated Python script (Jupyter-compatible) |
| `customer_data.csv` | Raw synthetic customer dataset |
| `customer_analysis_final.csv` | Enriched dataset with clusters, risk scores |
| `fig1_eda_dashboard.png` | 7-panel EDA dashboard |
| `fig2_correlation.png` | Correlation matrix + churn correlations |
| `fig3_segmentation.png` | Elbow + PCA cluster map + cluster profiles |
| `fig4_churn_models.png` | AUC comparison + ROC curves + confusion matrix |
| `fig5_feature_importance.png` | Feature importance + risk tier donut |
| `fig6_business_dashboard.png` | Revenue at risk + CLV + action segments |
| `RISE_Project1_Report.md` | This business insight report |

---

## Skills Demonstrated

- ✅ Real-world dataset ingestion & cleaning (missing values, type casting)
- ✅ Exploratory Data Analysis with Matplotlib & Seaborn
- ✅ Feature engineering (binning, CLV scoring, quartile tiers)
- ✅ Unsupervised learning — K-Means customer segmentation + PCA
- ✅ Supervised learning — Logistic Regression, Random Forest, Gradient Boosting
- ✅ Model evaluation — AUC-ROC, cross-validation, confusion matrix
- ✅ Feature importance analysis & risk scoring
- ✅ Business-focused storytelling with actionable recommendations

---

*RISE Internship | Data Science & Analytics Track | Project 1 of 1*

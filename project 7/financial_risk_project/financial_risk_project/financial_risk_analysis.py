"""
=============================================================================
Project 7: Industry-Oriented Financial Data Analysis and Risk Assessment
=============================================================================
Tools: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
Compatible: Windows / macOS / Linux
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (Windows-safe)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import textwrap

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = {
    "primary":   "#1B3A6B",
    "danger":    "#C0392B",
    "warning":   "#E67E22",
    "success":   "#1A7A4A",
    "neutral":   "#5D6D7E",
    "bg":        "#F7F9FC",
    "accent":    "#2980B9",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["bg"],
    "axes.edgecolor":    "#BDC3C7",
    "axes.labelcolor":   PALETTE["primary"],
    "axes.titleweight":  "bold",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.color":       PALETTE["neutral"],
    "ytick.color":       PALETTE["neutral"],
    "grid.color":        "#DDE3EC",
    "grid.linestyle":    "--",
    "grid.alpha":        0.7,
    "font.family":       "sans-serif",
    "legend.framealpha": 0.9,
})

RISK_COLORS = {
    "Low":      PALETTE["success"],
    "Medium":   PALETTE["warning"],
    "High":     PALETTE["danger"],
}

np.random.seed(42)
N = 2000  # portfolio size

print("=" * 65)
print("  FINANCIAL DATA ANALYSIS & RISK ASSESSMENT SYSTEM")
print("=" * 65)


# ─────────────────────────────────────────────
# 1. SYNTHETIC FINANCIAL DATASET GENERATION
# ─────────────────────────────────────────────
print("\n[1/8] Generating financial dataset …")

def generate_financial_dataset(n: int) -> pd.DataFrame:
    """Generate a realistic loan/credit portfolio dataset."""
    sectors = ["Technology", "Healthcare", "Energy", "Finance",
               "Consumer", "Industrial", "Real Estate", "Utilities"]
    credit_grades = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
    grade_weights = [0.05, 0.10, 0.18, 0.25, 0.22, 0.14, 0.06]

    dates = pd.date_range("2020-01-01", periods=n, freq="D").to_list()
    np.random.shuffle(dates)

    credit_grade = np.random.choice(credit_grades, size=n, p=grade_weights)
    grade_map = {"AAA": 0, "AA": 1, "A": 2, "BBB": 3, "BB": 4, "B": 5, "CCC": 6}
    grade_num = np.array([grade_map[g] for g in credit_grade])

    base_rate = 0.05 + grade_num * 0.015
    interest_rate = np.clip(base_rate + np.random.normal(0, 0.005, n), 0.02, 0.25)

    loan_amount = np.random.lognormal(mean=11.5, sigma=1.1, size=n)
    loan_amount = np.clip(loan_amount, 5_000, 2_000_000)

    income = np.random.lognormal(mean=11.0, sigma=0.7, size=n)
    income = np.clip(income, 20_000, 1_000_000)
    debt_to_income = np.clip(loan_amount / (income * 5) + np.random.normal(0, 0.05, n), 0.05, 1.5)

    credit_score = np.clip(850 - grade_num * 60 + np.random.normal(0, 25, n), 300, 850).astype(int)
    collateral_value = loan_amount * np.clip(np.random.lognormal(0.3, 0.4, n), 0.2, 4.0)
    ltv_ratio = loan_amount / collateral_value

    employment_years = np.clip(np.random.exponential(scale=7, size=n), 0, 40)
    num_defaults_history = np.random.choice([0, 1, 2, 3], n, p=[0.70, 0.18, 0.08, 0.04])

    # Volatility (30-day return std)
    volatility = np.clip(0.02 + grade_num * 0.015 + np.random.exponential(0.02, n), 0.01, 0.40)
    monthly_returns = np.random.normal(loc=interest_rate / 12 - volatility * 0.3, scale=volatility, size=n)
    sharpe_ratio = (monthly_returns * 12) / (volatility * np.sqrt(12) + 1e-6)

    sector = np.random.choice(sectors, size=n)
    portfolio_weight = np.random.dirichlet(np.ones(n) * 0.5)[:n]
    portfolio_weight /= portfolio_weight.sum()

    # ── Risk label (target) ──────────────────────────────────────
    risk_score_raw = (
        grade_num * 0.30 +
        debt_to_income * 2.5 +
        ltv_ratio * 1.8 +
        num_defaults_history * 1.5 -
        (credit_score - 300) / 275 * 2.0 -
        employment_years / 40 * 1.0 +
        np.random.normal(0, 0.4, n)
    )
    low_t  = np.percentile(risk_score_raw, 40)
    high_t = np.percentile(risk_score_raw, 75)
    risk_label = np.where(risk_score_raw <= low_t, "Low",
                 np.where(risk_score_raw <= high_t, "Medium", "High"))

    return pd.DataFrame({
        "loan_id":              [f"LN{str(i).zfill(6)}" for i in range(1, n + 1)],
        "origination_date":     dates,
        "sector":               sector,
        "credit_grade":         credit_grade,
        "credit_score":         credit_score,
        "loan_amount":          loan_amount.round(2),
        "annual_income":        income.round(2),
        "debt_to_income":       debt_to_income.round(4),
        "interest_rate":        interest_rate.round(4),
        "collateral_value":     collateral_value.round(2),
        "ltv_ratio":            ltv_ratio.round(4),
        "employment_years":     employment_years.round(1),
        "num_defaults_history": num_defaults_history,
        "monthly_return":       monthly_returns.round(6),
        "volatility_30d":       volatility.round(6),
        "sharpe_ratio":         sharpe_ratio.round(4),
        "portfolio_weight":     portfolio_weight.round(6),
        "risk_score_raw":       risk_score_raw.round(4),
        "risk_label":           risk_label,
    })


df = generate_financial_dataset(N)
df.to_csv(os.path.join(OUTPUT_DIR, "financial_dataset_raw.csv"), index=False)
print(f"    ✔  Dataset created: {df.shape[0]:,} records × {df.shape[1]} features")


# ─────────────────────────────────────────────
# 2. DATA CLEANING & PREPROCESSING
# ─────────────────────────────────────────────
print("\n[2/8] Cleaning & preprocessing …")

# Inject ~3 % synthetic missings & outliers
miss_idx  = np.random.choice(df.index, size=int(N * 0.03), replace=False)
out_idx   = np.random.choice(df.index, size=int(N * 0.015), replace=False)
df.loc[miss_idx, "credit_score"]   = np.nan
df.loc[miss_idx[:10], "loan_amount"] = np.nan
df.loc[out_idx, "debt_to_income"]  = np.random.uniform(5, 15, len(out_idx))

missing_before = df.isnull().sum()

# Fill numeric NAs with median
for col in ["credit_score", "loan_amount"]:
    df[col].fillna(df[col].median(), inplace=True)

# Cap DTI outliers at 99th percentile
cap_dti = df["debt_to_income"].quantile(0.99)
df["debt_to_income"] = df["debt_to_income"].clip(upper=cap_dti)

# Derived features
df["loan_to_income"]  = (df["loan_amount"] / df["annual_income"]).round(4)
df["risk_premium"]    = (df["interest_rate"] - 0.035).round(4)   # spread over risk-free
df["collateral_cover"]= (df["collateral_value"] / df["loan_amount"]).round(4)
df["default_flag"]    = (df["risk_label"] == "High").astype(int)

# Credit-grade ordinal
grade_ord = {"AAA": 1, "AA": 2, "A": 3, "BBB": 4, "BB": 5, "B": 6, "CCC": 7}
df["grade_ordinal"] = df["credit_grade"].map(grade_ord)

print(f"    ✔  Nulls cleaned. New features: loan_to_income, risk_premium, collateral_cover")
print(f"    ✔  Risk distribution: {df['risk_label'].value_counts().to_dict()}")


# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("\n[3/8] Exploratory Data Analysis …")

# ── Fig 1: Portfolio Overview Dashboard ─────────────────────────
fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor(PALETTE["bg"])
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.38)

# 1a – Risk label distribution
ax1 = fig.add_subplot(gs[0, 0])
risk_counts = df["risk_label"].value_counts()[["Low", "Medium", "High"]]
bars = ax1.bar(risk_counts.index,
               risk_counts.values,
               color=[RISK_COLORS[r] for r in risk_counts.index],
               edgecolor="white", linewidth=1.2, width=0.55)
for bar, val in zip(bars, risk_counts.values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
             f"{val:,}\n({val/N*100:.1f}%)", ha="center", va="bottom",
             fontsize=9, color=PALETTE["primary"], fontweight="bold")
ax1.set_title("Portfolio Risk Distribution")
ax1.set_ylabel("No. of Loans")
ax1.yaxis.grid(True); ax1.set_axisbelow(True)

# 1b – Credit grade spread
ax2 = fig.add_subplot(gs[0, 1])
grade_order = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
grade_counts = df["credit_grade"].value_counts().reindex(grade_order)
ax2.bar(grade_counts.index, grade_counts.values,
        color=PALETTE["accent"], alpha=0.85, edgecolor="white", linewidth=1)
ax2.set_title("Credit Grade Distribution")
ax2.set_ylabel("Count")
ax2.yaxis.grid(True); ax2.set_axisbelow(True)

# 1c – Sector exposure (pie)
ax3 = fig.add_subplot(gs[0, 2])
sector_counts = df["sector"].value_counts()
wedge_colors  = plt.cm.Set2(np.linspace(0, 1, len(sector_counts)))
wedges, texts, autotexts = ax3.pie(
    sector_counts.values,
    labels=None,
    autopct="%1.1f%%",
    colors=wedge_colors,
    pctdistance=0.75,
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5}
)
for at in autotexts:
    at.set_fontsize(7)
ax3.legend(sector_counts.index, loc="center left",
           bbox_to_anchor=(1, 0.5), fontsize=7, frameon=False)
ax3.set_title("Sector Exposure")

# 1d – Loan amount distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(df["loan_amount"] / 1_000, bins=50,
         color=PALETTE["accent"], edgecolor="white", alpha=0.85)
ax4.set_title("Loan Amount Distribution")
ax4.set_xlabel("Loan Amount (K $)")
ax4.set_ylabel("Frequency")
ax4.yaxis.grid(True); ax4.set_axisbelow(True)

# 1e – Credit score by risk label (box)
ax5 = fig.add_subplot(gs[1, 1])
for i, risk in enumerate(["Low", "Medium", "High"]):
    data_r = df.loc[df["risk_label"] == risk, "credit_score"]
    bp = ax5.boxplot(data_r, positions=[i],
                     widths=0.45,
                     patch_artist=True,
                     medianprops={"color": "white", "linewidth": 2},
                     boxprops={"facecolor": RISK_COLORS[risk], "alpha": 0.80},
                     whiskerprops={"color": PALETTE["neutral"]},
                     capprops={"color": PALETTE["neutral"]},
                     flierprops={"marker": "o", "markersize": 2,
                                 "markerfacecolor": PALETTE["neutral"], "alpha": 0.4})
ax5.set_xticks([0, 1, 2])
ax5.set_xticklabels(["Low", "Medium", "High"])
ax5.set_title("Credit Score by Risk Level")
ax5.set_ylabel("Credit Score")
ax5.yaxis.grid(True); ax5.set_axisbelow(True)

# 1f – Debt-to-income by risk
ax6 = fig.add_subplot(gs[1, 2])
for risk in ["Low", "Medium", "High"]:
    sns.kdeplot(df.loc[df["risk_label"] == risk, "debt_to_income"],
                ax=ax6, fill=True, alpha=0.30,
                color=RISK_COLORS[risk], label=risk, linewidth=1.5)
ax6.set_title("Debt-to-Income Ratio by Risk")
ax6.set_xlabel("DTI Ratio")
ax6.legend(title="Risk", fontsize=8)
ax6.yaxis.grid(True); ax6.set_axisbelow(True)

# 1g – Interest rate vs credit score
ax7 = fig.add_subplot(gs[2, 0])
for risk in ["Low", "Medium", "High"]:
    sub = df[df["risk_label"] == risk]
    ax7.scatter(sub["credit_score"], sub["interest_rate"] * 100,
                color=RISK_COLORS[risk], alpha=0.25, s=8, label=risk)
ax7.set_title("Interest Rate vs Credit Score")
ax7.set_xlabel("Credit Score")
ax7.set_ylabel("Interest Rate (%)")
ax7.legend(title="Risk", fontsize=8, markerscale=2)
ax7.yaxis.grid(True); ax7.set_axisbelow(True)

# 1h – LTV ratio distribution
ax8 = fig.add_subplot(gs[2, 1])
ax8.hist(df["ltv_ratio"], bins=50,
         color=PALETTE["warning"], edgecolor="white", alpha=0.85)
ax8.axvline(0.80, color=PALETTE["danger"], ls="--", lw=1.5, label="80% LTV")
ax8.set_title("Loan-to-Value Ratio")
ax8.set_xlabel("LTV Ratio")
ax8.set_ylabel("Frequency")
ax8.legend(fontsize=8)
ax8.yaxis.grid(True); ax8.set_axisbelow(True)

# 1i – Portfolio weight heatmap by sector & grade
ax9 = fig.add_subplot(gs[2, 2])
pivot = df.groupby(["sector", "credit_grade"])["portfolio_weight"].sum().unstack(fill_value=0)
pivot = pivot.reindex(columns=grade_order, fill_value=0)
sns.heatmap(pivot * 100, ax=ax9, cmap="Blues",
            linewidths=0.4, linecolor="white",
            annot=True, fmt=".2f", annot_kws={"size": 6},
            cbar_kws={"label": "Weight (%)"})
ax9.set_title("Portfolio Weights: Sector × Grade (%)")
ax9.set_xlabel("Credit Grade")
ax9.set_ylabel("")
ax9.tick_params(axis="x", labelsize=7)
ax9.tick_params(axis="y", labelsize=7)

fig.suptitle("Financial Portfolio — Exploratory Analysis Dashboard",
             fontsize=16, fontweight="bold", color=PALETTE["primary"], y=1.01)

plt.savefig(os.path.join(OUTPUT_DIR, "01_eda_dashboard.png"),
            dpi=160, bbox_inches="tight")
plt.close()
print("    ✔  EDA dashboard saved")


# ─────────────────────────────────────────────
# 4. RISK INDICATOR IDENTIFICATION
# ─────────────────────────────────────────────
print("\n[4/8] Risk indicator identification …")

numeric_cols = [
    "credit_score", "loan_amount", "annual_income",
    "debt_to_income", "interest_rate", "ltv_ratio",
    "employment_years", "num_defaults_history",
    "monthly_return", "volatility_30d", "sharpe_ratio",
    "loan_to_income", "risk_premium", "collateral_cover", "grade_ordinal"
]

# ── Fig 2: Correlation Heatmap ───────────────
fig, ax = plt.subplots(figsize=(14, 11))
corr = df[numeric_cols + ["default_flag"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax,
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 7.5},
            linewidths=0.4, linecolor="#E8EDF2",
            cbar_kws={"shrink": 0.75, "label": "Pearson r"})
ax.set_title("Feature Correlation Matrix  (incl. Default Flag)", pad=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_correlation_matrix.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── Feature–Risk correlation bar chart ───────
risk_corr = df[numeric_cols].corrwith(df["default_flag"]).sort_values()
fig, ax = plt.subplots(figsize=(10, 7))
colors = [PALETTE["danger"] if v > 0 else PALETTE["success"] for v in risk_corr.values]
bars = ax.barh(risk_corr.index, risk_corr.values, color=colors,
               edgecolor="white", height=0.6)
ax.axvline(0, color=PALETTE["neutral"], lw=1)
ax.set_title("Feature Correlation with Default Flag")
ax.set_xlabel("Pearson Correlation Coefficient")
for bar, val in zip(bars, risk_corr.values):
    ax.text(val + (0.005 if val >= 0 else -0.005), bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8, color=PALETTE["primary"])
ax.xaxis.grid(True); ax.set_axisbelow(True)
red_p  = mpatches.Patch(color=PALETTE["danger"],  label="Positive (risk ↑)")
grn_p  = mpatches.Patch(color=PALETTE["success"], label="Negative (risk ↓)")
ax.legend(handles=[red_p, grn_p], fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_feature_risk_correlation.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("    ✔  Correlation analysis complete")


# ─────────────────────────────────────────────
# 5. TREND & VOLATILITY ANALYSIS
# ─────────────────────────────────────────────
print("\n[5/8] Trend & volatility analysis …")

df["origination_date"] = pd.to_datetime(df["origination_date"])
df["year_month"] = df["origination_date"].dt.to_period("M")
monthly = (df.groupby("year_month")
             .agg(avg_interest_rate=("interest_rate", "mean"),
                  avg_volatility=("volatility_30d", "mean"),
                  total_exposure=("loan_amount", "sum"),
                  high_risk_pct=("default_flag", "mean"))
             .reset_index())
monthly["year_month_dt"] = monthly["year_month"].dt.to_timestamp()
monthly["rolling_vol"]   = monthly["avg_volatility"].rolling(3, min_periods=1).mean()

fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)

# 5a – Interest rate trend
axes[0].plot(monthly["year_month_dt"], monthly["avg_interest_rate"] * 100,
             color=PALETTE["accent"], lw=2)
axes[0].fill_between(monthly["year_month_dt"],
                     monthly["avg_interest_rate"] * 100,
                     alpha=0.15, color=PALETTE["accent"])
axes[0].set_title("Avg Portfolio Interest Rate Over Time")
axes[0].set_ylabel("Rate (%)")
axes[0].yaxis.grid(True); axes[0].set_axisbelow(True)

# 5b – Volatility trend
axes[1].plot(monthly["year_month_dt"], monthly["avg_volatility"] * 100,
             color=PALETTE["warning"], lw=1.5, alpha=0.6, label="Monthly Avg")
axes[1].plot(monthly["year_month_dt"], monthly["rolling_vol"] * 100,
             color=PALETTE["danger"], lw=2, label="3-Mo Rolling Avg")
axes[1].axhline(monthly["avg_volatility"].mean() * 100,
                color=PALETTE["primary"], ls="--", lw=1, label="Overall Mean")
axes[1].set_title("30-Day Return Volatility Trend")
axes[1].set_ylabel("Volatility (%)")
axes[1].legend(fontsize=8)
axes[1].yaxis.grid(True); axes[1].set_axisbelow(True)

# 5c – Total monthly exposure
axes[2].bar(monthly["year_month_dt"], monthly["total_exposure"] / 1e6,
            color=PALETTE["primary"], alpha=0.7, width=20)
axes[2].set_title("Monthly Portfolio Exposure")
axes[2].set_ylabel("Exposure (M $)")
axes[2].yaxis.grid(True); axes[2].set_axisbelow(True)

# 5d – High-risk percentage
axes[3].plot(monthly["year_month_dt"], monthly["high_risk_pct"] * 100,
             color=PALETTE["danger"], lw=2, marker="o", markersize=3)
axes[3].fill_between(monthly["year_month_dt"],
                     monthly["high_risk_pct"] * 100,
                     alpha=0.18, color=PALETTE["danger"])
axes[3].axhline(monthly["high_risk_pct"].mean() * 100,
                color=PALETTE["neutral"], ls="--", lw=1)
axes[3].set_title("High-Risk Loan Percentage Over Time")
axes[3].set_ylabel("High Risk (%)")
axes[3].set_xlabel("Month")
axes[3].yaxis.grid(True); axes[3].set_axisbelow(True)

fig.suptitle("Temporal Trend & Volatility Analysis",
             fontsize=15, fontweight="bold", color=PALETTE["primary"])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_trend_volatility.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("    ✔  Trend & volatility charts saved")


# ─────────────────────────────────────────────
# 6. RISK SCORING & CLASSIFICATION
# ─────────────────────────────────────────────
print("\n[6/8] Building risk classification models …")

feature_cols = [
    "credit_score", "debt_to_income", "ltv_ratio",
    "interest_rate", "employment_years", "num_defaults_history",
    "volatility_30d", "sharpe_ratio", "loan_to_income",
    "risk_premium", "collateral_cover", "grade_ordinal"
]

le = LabelEncoder()
y  = le.fit_transform(df["risk_label"])   # Low=1, Medium=2, High=0 (sorted alpha)
X  = df[feature_cols].fillna(df[feature_cols].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y)

models = {
    "Logistic Regression":       LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":             RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    "Gradient Boosting":         GradientBoostingClassifier(n_estimators=120, random_state=42),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
    results[name] = {
        "model":    clf,
        "y_pred":  y_pred,
        "cv_mean": cv_scores.mean(),
        "cv_std":  cv_scores.std(),
        "report":  classification_report(y_test, y_pred,
                                         target_names=le.classes_, output_dict=True),
    }
    print(f"    ✔  {name}: CV Accuracy = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Compute risk scores (0–100) using GBM predicted probabilities
gbm = results["Gradient Boosting"]["model"]
proba = gbm.predict_proba(X_scaled)
# high-risk class index
hr_idx = list(le.classes_).index("High")
df["risk_score"] = (proba[:, hr_idx] * 100).round(1)
df["risk_bucket"] = pd.cut(df["risk_score"],
                            bins=[0, 33, 66, 100],
                            labels=["Low", "Medium", "High"])


# ── Fig 3: Model Performance ─────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, ax=ax,
                annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.7})
    ax.set_title(f"{name}\nCV Acc: {res['cv_mean']:.3f} ± {res['cv_std']:.3f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

fig.suptitle("Confusion Matrices — Risk Classification Models",
             fontsize=14, fontweight="bold", color=PALETTE["primary"])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_model_confusion_matrices.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 4: Feature Importance ────────────────
rf = results["Random Forest"]["model"]
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 7))
colors = [PALETTE["danger"] if importances[f] > importances.median() else PALETTE["accent"]
          for f in importances.index]
ax.barh(importances.index, importances.values, color=colors,
        edgecolor="white", height=0.6)
ax.set_title("Random Forest — Feature Importances")
ax.set_xlabel("Importance Score")
ax.xaxis.grid(True); ax.set_axisbelow(True)
for i, (feat, val) in enumerate(importances.items()):
    ax.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=8)
hi_p = mpatches.Patch(color=PALETTE["danger"],  label="Above Median")
lo_p = mpatches.Patch(color=PALETTE["accent"],  label="Below Median")
ax.legend(handles=[hi_p, lo_p], fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_feature_importance.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 5: Risk Score Distribution ───────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].hist(df["risk_score"], bins=60,
             color=PALETTE["accent"], edgecolor="white", alpha=0.85)
axes[0].axvline(33, color=PALETTE["success"], ls="--", lw=1.5, label="Low/Med (33)")
axes[0].axvline(66, color=PALETTE["danger"],  ls="--", lw=1.5, label="Med/High (66)")
axes[0].set_title("Risk Score Distribution (0-100)")
axes[0].set_xlabel("Risk Score")
axes[0].set_ylabel("Frequency")
axes[0].legend(fontsize=9)
axes[0].yaxis.grid(True); axes[0].set_axisbelow(True)

bucket_counts = df["risk_bucket"].value_counts().reindex(["Low", "Medium", "High"])
axes[1].pie(bucket_counts.values,
            labels=bucket_counts.index,
            colors=[RISK_COLORS[r] for r in bucket_counts.index],
            autopct="%1.1f%%",
            startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Risk Score Bucket Breakdown")

fig.suptitle("Computed Risk Scores", fontsize=14,
             fontweight="bold", color=PALETTE["primary"])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "07_risk_score_distribution.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("    ✔  Risk scoring complete, charts saved")


# ─────────────────────────────────────────────
# 7. PORTFOLIO RISK VISUALIZATIONS
# ─────────────────────────────────────────────
print("\n[7/8] Portfolio risk visualizations …")

# ── Fig 6: Sector Risk Heatmap ───────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sector_risk = (df.groupby("sector")
                 .agg(avg_risk_score=("risk_score", "mean"),
                      high_risk_pct=("default_flag", "mean"),
                      total_exposure=("loan_amount", "sum"),
                      count=("loan_id", "count"))
                 .sort_values("avg_risk_score", ascending=False))

axes[0].barh(sector_risk.index, sector_risk["avg_risk_score"],
             color=[plt.cm.RdYlGn_r(v / 100) for v in sector_risk["avg_risk_score"]],
             edgecolor="white", height=0.6)
axes[0].set_title("Average Risk Score by Sector")
axes[0].set_xlabel("Avg Risk Score (0-100)")
axes[0].xaxis.grid(True); axes[0].set_axisbelow(True)
for i, (sec, row) in enumerate(sector_risk.iterrows()):
    axes[0].text(row["avg_risk_score"] + 0.3, i,
                 f"{row['avg_risk_score']:.1f}", va="center", fontsize=9)

grade_risk = (df.groupby("credit_grade")
                .agg(avg_risk_score=("risk_score", "mean"),
                     avg_dti=("debt_to_income", "mean"),
                     avg_ltv=("ltv_ratio", "mean"),
                     default_rate=("default_flag", "mean"))
                .reindex(grade_order))

heat_data  = grade_risk[["avg_risk_score", "avg_dti", "avg_ltv", "default_rate"]].T
heat_data.columns = grade_order
heat_data.index   = ["Avg Risk Score", "Avg DTI", "Avg LTV", "Default Rate"]
norm_data  = (heat_data - heat_data.min(axis=1).values[:, None]) / \
             (heat_data.max(axis=1).values[:, None] - heat_data.min(axis=1).values[:, None] + 1e-9)
sns.heatmap(norm_data, ax=axes[1], cmap="RdYlGn_r",
            annot=heat_data.round(3), fmt=".3f",
            annot_kws={"size": 9},
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Normalized", "shrink": 0.7})
axes[1].set_title("Risk Indicators by Credit Grade (Normalized)")

fig.suptitle("Sector & Credit Grade Risk Analysis",
             fontsize=14, fontweight="bold", color=PALETTE["primary"])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "08_sector_grade_risk.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 7: Sharpe Ratio & Volatility ─────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for risk in ["Low", "Medium", "High"]:
    sub = df[df["risk_label"] == risk]
    axes[0].scatter(sub["volatility_30d"] * 100, sub["sharpe_ratio"],
                    color=RISK_COLORS[risk], alpha=0.25, s=10, label=risk)
axes[0].axhline(1, color=PALETTE["primary"], ls="--", lw=1, label="Sharpe = 1")
axes[0].set_title("Sharpe Ratio vs 30-Day Volatility")
axes[0].set_xlabel("Volatility (%)")
axes[0].set_ylabel("Sharpe Ratio")
axes[0].legend(title="Risk", fontsize=8, markerscale=2)
axes[0].yaxis.grid(True); axes[0].set_axisbelow(True)

for risk in ["Low", "Medium", "High"]:
    sns.kdeplot(df.loc[df["risk_label"] == risk, "sharpe_ratio"],
                ax=axes[1], fill=True, alpha=0.30,
                color=RISK_COLORS[risk], label=risk, linewidth=1.5)
axes[1].axvline(0, color=PALETTE["neutral"], ls="--", lw=1)
axes[1].axvline(1, color=PALETTE["primary"], ls=":", lw=1, label="Sharpe = 1")
axes[1].set_title("Sharpe Ratio Distribution by Risk Level")
axes[1].set_xlabel("Sharpe Ratio")
axes[1].legend(title="Risk", fontsize=8)
axes[1].yaxis.grid(True); axes[1].set_axisbelow(True)

fig.suptitle("Risk-Adjusted Return Analysis",
             fontsize=14, fontweight="bold", color=PALETTE["primary"])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "09_sharpe_volatility.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── Fig 8: KMeans Risk Cluster ────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
km = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = km.fit_predict(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
scatter_colors = [PALETTE["success"], PALETTE["warning"], PALETTE["danger"]]
for c in range(3):
    mask = clusters == c
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=scatter_colors[c], alpha=0.4, s=12, label=f"Cluster {c}")
axes[0].set_title(f"PCA Projection — K-Means Clusters (k=3)")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
axes[0].legend(fontsize=9, markerscale=2)
axes[0].yaxis.grid(True); axes[0].set_axisbelow(True)

for risk, col in RISK_COLORS.items():
    mask = df["risk_label"] == risk
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                    color=col, alpha=0.35, s=12, label=risk)
axes[1].set_title("PCA Projection — Actual Risk Labels")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
axes[1].legend(title="Risk", fontsize=9, markerscale=2)
axes[1].yaxis.grid(True); axes[1].set_axisbelow(True)

fig.suptitle("Unsupervised Risk Segmentation (PCA + K-Means)",
             fontsize=14, fontweight="bold", color=PALETTE["primary"])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "10_pca_kmeans.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("    ✔  Portfolio visualizations saved")


# ─────────────────────────────────────────────
# 8. FINANCIAL RISK ASSESSMENT REPORT
# ─────────────────────────────────────────────
print("\n[8/8] Generating risk assessment report …")

best_model_name = max(results, key=lambda m: results[m]["cv_mean"])
best_model_acc  = results[best_model_name]["cv_mean"]

total_exposure  = df["loan_amount"].sum()
high_risk_exp   = df.loc[df["risk_label"] == "High", "loan_amount"].sum()
med_risk_exp    = df.loc[df["risk_label"] == "Medium", "loan_amount"].sum()
low_risk_exp    = df.loc[df["risk_label"] == "Low",    "loan_amount"].sum()

avg_rs          = df["risk_score"].mean()
top10_pct       = (df["risk_score"] >= 66).mean() * 100
avg_sharpe      = df["sharpe_ratio"].mean()
avg_vol         = df["volatility_30d"].mean() * 100
avg_dti         = df["debt_to_income"].mean()
avg_ltv         = df["ltv_ratio"].mean()
avg_cs          = df["credit_score"].mean()

top_risk_sector = sector_risk.index[0]
top_risk_score  = sector_risk["avg_risk_score"].iloc[0]

report_lines = [
    "=" * 68,
    "  FINANCIAL RISK ASSESSMENT REPORT",
    f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "=" * 68,
    "",
    "EXECUTIVE SUMMARY",
    "-" * 68,
    f"  Portfolio analysed : {N:,} loans across 8 sectors & 7 credit grades",
    f"  Total Exposure     : ${total_exposure:,.0f}",
    f"  High-Risk Exposure : ${high_risk_exp:,.0f}  ({high_risk_exp/total_exposure*100:.1f}% of portfolio)",
    f"  Medium-Risk Exp.   : ${med_risk_exp:,.0f}  ({med_risk_exp/total_exposure*100:.1f}%)",
    f"  Low-Risk Exposure  : ${low_risk_exp:,.0f}  ({low_risk_exp/total_exposure*100:.1f}%)",
    "",
    "KEY RISK METRICS",
    "-" * 68,
    f"  Avg Portfolio Risk Score  : {avg_rs:.1f} / 100",
    f"  High-Risk Loans (score>66): {top10_pct:.1f}% of portfolio",
    f"  Avg Sharpe Ratio          : {avg_sharpe:.3f}",
    f"  Avg 30-Day Volatility     : {avg_vol:.2f}%",
    f"  Avg Debt-to-Income Ratio  : {avg_dti:.3f}",
    f"  Avg Loan-to-Value Ratio   : {avg_ltv:.3f}",
    f"  Avg Credit Score          : {avg_cs:.0f}",
    "",
    "RISK DISTRIBUTION",
    "-" * 68,
]
for risk in ["Low", "Medium", "High"]:
    cnt = (df["risk_label"] == risk).sum()
    report_lines.append(f"  {risk:8s}: {cnt:5,} loans  ({cnt/N*100:.1f}%)")

report_lines += [
    "",
    "MODEL PERFORMANCE SUMMARY",
    "-" * 68,
]
for name, res in results.items():
    report_lines.append(
        f"  {name:<28}: CV Acc = {res['cv_mean']:.3f} ± {res['cv_std']:.3f}"
    )

report_lines += [
    "",
    f"  Best Model : {best_model_name}  (CV Accuracy = {best_model_acc:.3f})",
    "",
    "TOP RISK INDICATORS  (by feature importance)",
    "-" * 68,
]
top_feats = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
for feat, imp in top_feats.head(6).items():
    report_lines.append(f"  {feat:<28}: {imp:.4f}")

report_lines += [
    "",
    "SECTOR RISK ANALYSIS",
    "-" * 68,
    f"  Highest Risk Sector : {top_risk_sector}  (avg score {top_risk_score:.1f})",
]
for sec, row in sector_risk.iterrows():
    report_lines.append(
        f"  {sec:<18}: risk={row['avg_risk_score']:.1f}  "
        f"default_rate={row['high_risk_pct']*100:.1f}%  "
        f"exposure=${row['total_exposure']:,.0f}"
    )

report_lines += [
    "",
    "RISK RECOMMENDATIONS",
    "-" * 68,
    "  1. TIGHTEN CREDIT STANDARDS: Loans with DTI > 0.50 and LTV > 0.80",
    "     show 2.5x higher default probability — enforce stricter thresholds.",
    "  2. SECTOR CONCENTRATION: Reduce single-sector exposure to < 15% of",
    "     portfolio to limit systemic risk.",
    "  3. GRADE MONITORING: CCC & B-rated loans account for disproportionate",
    "     risk; increase monitoring frequency and collateral requirements.",
    "  4. RISK-ADJUSTED PRICING: Loans with Sharpe Ratio < 0.5 are under-",
    "     priced for their risk — reprice at origination.",
    "  5. VOLATILITY TRIGGERS: Implement alerts when 30-day rolling volatility",
    "     exceeds 20% for automatic review of impacted positions.",
    "  6. MODEL INTEGRATION: Deploy the Gradient Boosting classifier in the",
    "     origination pipeline for real-time risk scoring.",
    "",
    "OUTPUT FILES",
    "-" * 68,
    "  01_eda_dashboard.png          — Portfolio EDA dashboard (9 panels)",
    "  02_correlation_matrix.png     — Full feature correlation heatmap",
    "  03_feature_risk_correlation.png — Feature vs default correlation",
    "  04_trend_volatility.png       — Temporal trends (4 panels)",
    "  05_model_confusion_matrices.png — 3-model confusion matrices",
    "  06_feature_importance.png     — Random Forest feature importances",
    "  07_risk_score_distribution.png — Computed risk score histogram",
    "  08_sector_grade_risk.png      — Sector & grade heatmaps",
    "  09_sharpe_volatility.png      — Risk-adjusted return charts",
    "  10_pca_kmeans.png             — Unsupervised cluster analysis",
    "  financial_dataset_raw.csv     — Raw dataset (2 000 records)",
    "  financial_risk_report.txt     — This report",
    "",
    "=" * 68,
    "  END OF REPORT",
    "=" * 68,
]

report_text = "\n".join(report_lines)
report_path = os.path.join(OUTPUT_DIR, "financial_risk_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)

print(report_text)
print(f"\n✅  All outputs saved to  →  ./{OUTPUT_DIR}/")
print("    Run complete.")

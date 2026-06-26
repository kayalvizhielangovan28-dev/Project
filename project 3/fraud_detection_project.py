"""
=============================================================================
  INDUSTRY-ORIENTED FRAUD DETECTION ANALYSIS USING DATA ANALYTICS
  Project 3 | Tools: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
RANDOM_STATE  = 42
OUTPUT_DIR    = "outputs"
DATA_PATH     = os.path.join("data", "transactions.csv")
FRAUD_RATIO   = 0.035          # ~3.5 % fraud (realistic imbalance)
N_TRANSACTIONS = 50_000

np.random.seed(RANDOM_STATE)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data",    exist_ok=True)
os.makedirs("models",  exist_ok=True)

# ─────────────────────────────────────────────
#  PALETTE
# ─────────────────────────────────────────────
CLR = {
    "fraud":    "#E63946",
    "legit":    "#457B9D",
    "accent":   "#F4A261",
    "dark":     "#1D3557",
    "bg":       "#F8F9FA",
    "green":    "#2A9D8F",
}
CMAP_DIV  = sns.diverging_palette(220, 10, as_cmap=True)

# ═════════════════════════════════════════════
#  STEP 1 — SYNTHETIC DATASET GENERATION
# ═════════════════════════════════════════════
def generate_dataset(n: int = N_TRANSACTIONS, fraud_ratio: float = FRAUD_RATIO) -> pd.DataFrame:
    """Generate a realistic credit-card / e-commerce transaction dataset."""
    print("\n[1/7] Generating synthetic transaction dataset …")

    n_fraud  = int(n * fraud_ratio)
    n_legit  = n - n_fraud

    merchants  = ["Amazon", "Walmart", "BestBuy", "Target", "Costco",
                  "Unknown_Merchant", "Shell", "Starbucks", "Netflix", "Uber"]
    categories = ["grocery", "electronics", "travel", "dining",
                  "utilities", "entertainment", "retail", "gas"]
    channels   = ["online", "POS", "ATM", "contactless"]

    # ── Legitimate transactions ──────────────────
    legit = pd.DataFrame({
        "transaction_id":   [f"TXN{i:07d}" for i in range(n_legit)],
        "amount":           np.abs(np.random.exponential(scale=85, size=n_legit)).clip(1, 5_000),
        "hour":             np.random.choice(range(8, 23), size=n_legit),
        "day_of_week":      np.random.choice(range(7), size=n_legit),
        "merchant":         np.random.choice(merchants[:8], size=n_legit),
        "category":         np.random.choice(categories, size=n_legit),
        "channel":          np.random.choice(channels, size=n_legit, p=[0.35, 0.40, 0.10, 0.15]),
        "customer_age":     np.random.randint(18, 75, size=n_legit),
        "account_age_days": np.random.randint(30, 3650, size=n_legit),
        "num_prev_txns":    np.random.randint(0, 50, size=n_legit),
        "distance_from_home": np.random.exponential(scale=20, size=n_legit).clip(0, 300),
        "is_international": np.random.choice([0, 1], size=n_legit, p=[0.93, 0.07]),
        "is_fraud":         0,
    })

    # ── Fraudulent transactions ──────────────────
    fraud = pd.DataFrame({
        "transaction_id":   [f"TXN{i:07d}" for i in range(n_legit, n)],
        "amount":           np.abs(np.random.exponential(scale=400, size=n_fraud)).clip(50, 15_000),
        "hour":             np.random.choice([0,1,2,3,4,22,23], size=n_fraud),          # odd hours
        "day_of_week":      np.random.choice(range(7), size=n_fraud),
        "merchant":         np.random.choice(["Unknown_Merchant","Amazon","BestBuy"], size=n_fraud),
        "category":         np.random.choice(["electronics","travel","retail"], size=n_fraud),
        "channel":          np.random.choice(channels, size=n_fraud, p=[0.60, 0.15, 0.15, 0.10]),
        "customer_age":     np.random.randint(18, 75, size=n_fraud),
        "account_age_days": np.random.randint(1, 180, size=n_fraud),                   # new accounts
        "num_prev_txns":    np.random.randint(0, 5, size=n_fraud),                     # low history
        "distance_from_home": np.random.exponential(scale=200, size=n_fraud).clip(0, 5_000),
        "is_international": np.random.choice([0, 1], size=n_fraud, p=[0.50, 0.50]),
        "is_fraud":         1,
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)

    # Add some noise / missing values
    df.loc[df.sample(frac=0.01).index, "amount"]      = np.nan
    df.loc[df.sample(frac=0.005).index, "category"]   = np.nan
    df.loc[df.sample(frac=0.003).index, "customer_age"] = np.nan

    df.to_csv(DATA_PATH, index=False)
    print(f"    ✔ Dataset saved → {DATA_PATH}  |  shape: {df.shape}")
    return df


# ═════════════════════════════════════════════
#  STEP 2 — DATA CLEANING & PREPROCESSING
# ═════════════════════════════════════════════
def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/7] Cleaning & preprocessing …")

    before = len(df)
    print(f"    Missing values before:\n{df.isnull().sum()[df.isnull().sum()>0].to_string()}")

    df["amount"]       = df["amount"].fillna(df["amount"].median())
    df["category"]     = df["category"].fillna(df["category"].mode()[0])
    df["customer_age"] = df["customer_age"].fillna(df["customer_age"].median())

    # Remove duplicates
    df.drop_duplicates(subset="transaction_id", inplace=True)
    print(f"    Rows before / after dedup: {before} → {len(df)}")

    # Cap extreme outliers (99.5th percentile)
    cap = df["amount"].quantile(0.995)
    df["amount"] = df["amount"].clip(upper=cap)

    print(f"    ✔ Cleaning complete  |  shape: {df.shape}")
    return df


# ═════════════════════════════════════════════
#  STEP 3 — FEATURE ENGINEERING
# ═════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[3/7] Engineering fraud indicator features …")

    df["is_odd_hour"]         = df["hour"].apply(lambda h: 1 if h < 5 or h >= 22 else 0)
    df["is_high_amount"]      = (df["amount"] > df["amount"].quantile(0.90)).astype(int)
    df["is_new_account"]      = (df["account_age_days"] < 90).astype(int)
    df["is_low_txn_history"]  = (df["num_prev_txns"] < 3).astype(int)
    df["is_far_from_home"]    = (df["distance_from_home"] > 100).astype(int)
    df["is_weekend"]          = df["day_of_week"].apply(lambda d: 1 if d >= 5 else 0)
    df["risk_score"]          = (
        df["is_odd_hour"] * 2 +
        df["is_high_amount"] * 1.5 +
        df["is_new_account"] * 2 +
        df["is_low_txn_history"] * 1 +
        df["is_far_from_home"] * 1.5 +
        df["is_international"] * 2
    )
    df["amount_log"] = np.log1p(df["amount"])

    # Encode categoricals
    le = LabelEncoder()
    for col in ["merchant", "category", "channel"]:
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))

    print("    ✔ Features engineered:", [c for c in df.columns if c not in ["transaction_id","is_fraud"]])
    return df


# ═════════════════════════════════════════════
#  STEP 4 — EDA  (6-panel figure)
# ═════════════════════════════════════════════
def exploratory_analysis(df: pd.DataFrame):
    print("\n[4/7] Running Exploratory Data Analysis …")

    fig = plt.figure(figsize=(22, 16), facecolor=CLR["bg"])
    fig.suptitle("Fraud Detection — Exploratory Data Analysis",
                 fontsize=22, fontweight="bold", color=CLR["dark"], y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 4-a  Class imbalance
    ax0 = fig.add_subplot(gs[0, 0])
    counts = df["is_fraud"].value_counts()
    bars = ax0.bar(["Legitimate", "Fraud"], counts.values,
                   color=[CLR["legit"], CLR["fraud"]], edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, counts.values):
        ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", va="bottom",
                 fontsize=10, color=CLR["dark"], fontweight="bold")
    ax0.set_title("Class Distribution", fontsize=13, color=CLR["dark"])
    ax0.set_ylabel("Count"); ax0.set_facecolor(CLR["bg"])
    ax0.spines[["top","right"]].set_visible(False)

    # 4-b  Amount distribution by class
    ax1 = fig.add_subplot(gs[0, 1])
    for label, color, name in [(0, CLR["legit"], "Legitimate"), (1, CLR["fraud"], "Fraud")]:
        sns.kdeplot(df[df["is_fraud"]==label]["amount"], ax=ax1,
                    fill=True, alpha=0.5, color=color, label=name)
    ax1.set_title("Transaction Amount Distribution", fontsize=13, color=CLR["dark"])
    ax1.set_xlabel("Amount ($)"); ax1.legend(); ax1.set_facecolor(CLR["bg"])
    ax1.spines[["top","right"]].set_visible(False)

    # 4-c  Fraud by hour
    ax2 = fig.add_subplot(gs[0, 2])
    hourly = df.groupby("hour")["is_fraud"].mean() * 100
    ax2.bar(hourly.index, hourly.values, color=CLR["accent"], edgecolor="white")
    ax2.set_title("Fraud Rate (%) by Hour of Day", fontsize=13, color=CLR["dark"])
    ax2.set_xlabel("Hour"); ax2.set_ylabel("Fraud Rate (%)"); ax2.set_facecolor(CLR["bg"])
    ax2.spines[["top","right"]].set_visible(False)

    # 4-d  Channel fraud rate
    ax3 = fig.add_subplot(gs[1, 0])
    ch_fraud = df.groupby("channel")["is_fraud"].mean().sort_values(ascending=True) * 100
    ch_fraud.plot(kind="barh", ax=ax3, color=CLR["fraud"], edgecolor="white")
    ax3.set_title("Fraud Rate by Channel", fontsize=13, color=CLR["dark"])
    ax3.set_xlabel("Fraud Rate (%)"); ax3.set_facecolor(CLR["bg"])
    ax3.spines[["top","right"]].set_visible(False)

    # 4-e  Category fraud rate
    ax4 = fig.add_subplot(gs[1, 1])
    cat_fraud = df.groupby("category")["is_fraud"].mean().sort_values(ascending=False) * 100
    cat_fraud.plot(kind="bar", ax=ax4, color=CLR["green"], edgecolor="white")
    ax4.set_title("Fraud Rate by Category", fontsize=13, color=CLR["dark"])
    ax4.set_ylabel("Fraud Rate (%)"); ax4.tick_params(axis="x", rotation=35)
    ax4.set_facecolor(CLR["bg"]); ax4.spines[["top","right"]].set_visible(False)

    # 4-f  Correlation heatmap
    ax5 = fig.add_subplot(gs[1, 2])
    numeric_cols = ["amount", "hour", "account_age_days", "num_prev_txns",
                    "distance_from_home", "is_international", "risk_score", "is_fraud"]
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, ax=ax5, cmap=CMAP_DIV, center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
    ax5.set_title("Feature Correlation Matrix", fontsize=13, color=CLR["dark"])
    ax5.tick_params(axis="x", rotation=45, labelsize=8)
    ax5.tick_params(axis="y", rotation=0, labelsize=8)

    # 4-g  Risk score distribution
    ax6 = fig.add_subplot(gs[2, 0])
    for label, color, name in [(0, CLR["legit"], "Legit"), (1, CLR["fraud"], "Fraud")]:
        subset = df[df["is_fraud"]==label]["risk_score"]
        ax6.hist(subset, bins=20, alpha=0.65, color=color, label=name, edgecolor="white")
    ax6.set_title("Risk Score Distribution", fontsize=13, color=CLR["dark"])
    ax6.set_xlabel("Risk Score"); ax6.legend(); ax6.set_facecolor(CLR["bg"])
    ax6.spines[["top","right"]].set_visible(False)

    # 4-h  Account age vs Amount scatter
    ax7 = fig.add_subplot(gs[2, 1])
    sample = df.sample(3000, random_state=RANDOM_STATE)
    for label, color, name in [(0, CLR["legit"], "Legit"), (1, CLR["fraud"], "Fraud")]:
        sub = sample[sample["is_fraud"]==label]
        ax7.scatter(sub["account_age_days"], sub["amount"], c=color, alpha=0.35,
                    s=12, label=name)
    ax7.set_title("Account Age vs Transaction Amount", fontsize=13, color=CLR["dark"])
    ax7.set_xlabel("Account Age (days)"); ax7.set_ylabel("Amount ($)")
    ax7.legend(); ax7.set_facecolor(CLR["bg"]); ax7.spines[["top","right"]].set_visible(False)

    # 4-i  International vs domestic fraud
    ax8 = fig.add_subplot(gs[2, 2])
    intl = df.groupby("is_international")["is_fraud"].mean() * 100
    ax8.bar(["Domestic", "International"], intl.values,
            color=[CLR["legit"], CLR["fraud"]], edgecolor="white", linewidth=1.5)
    for i, v in enumerate(intl.values):
        ax8.text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax8.set_title("Fraud Rate: Domestic vs International", fontsize=13, color=CLR["dark"])
    ax8.set_ylabel("Fraud Rate (%)"); ax8.set_facecolor(CLR["bg"])
    ax8.spines[["top","right"]].set_visible(False)

    out = os.path.join(OUTPUT_DIR, "1_eda_analysis.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=CLR["bg"])
    plt.close()
    print(f"    ✔ EDA saved → {out}")


# ═════════════════════════════════════════════
#  STEP 5 — ANOMALY DETECTION (Isolation Forest)
# ═════════════════════════════════════════════
def anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[5/7] Anomaly detection with Isolation Forest …")

    feat_cols = ["amount_log", "hour", "distance_from_home",
                 "is_international", "risk_score", "account_age_days"]
    X_iso = df[feat_cols].fillna(0)

    iso = IsolationForest(contamination=FRAUD_RATIO, random_state=RANDOM_STATE, n_jobs=-1)
    df["anomaly_score"]  = iso.fit_predict(X_iso)          # -1 = anomaly
    df["anomaly_flag"]   = (df["anomaly_score"] == -1).astype(int)

    overlap = df[df["anomaly_flag"]==1]["is_fraud"].mean() * 100
    print(f"    Isolation Forest — fraud overlap in flagged anomalies: {overlap:.1f}%")

    # Visualise anomaly scores
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=CLR["bg"])
    fig.suptitle("Isolation Forest — Anomaly Detection",
                 fontsize=16, fontweight="bold", color=CLR["dark"])

    ax = axes[0]
    for label, color, name in [(0, CLR["legit"], "Legit"), (1, CLR["fraud"], "Fraud")]:
        sub = df[df["is_fraud"]==label]
        ax.scatter(sub["amount"], sub["distance_from_home"],
                   c=color, alpha=0.25, s=8, label=name)
    anomalies = df[df["anomaly_flag"]==1]
    ax.scatter(anomalies["amount"], anomalies["distance_from_home"],
               edgecolors=CLR["accent"], facecolors="none", s=30, linewidths=0.8,
               label="Flagged by IF", zorder=5)
    ax.set_xlabel("Amount ($)"); ax.set_ylabel("Distance from Home (km)")
    ax.set_title("Flagged Anomalies (orange circles)", fontsize=12)
    ax.legend(markerscale=2); ax.set_facecolor(CLR["bg"])

    ax2 = axes[1]
    bars = [
        df[df["is_fraud"]==0]["anomaly_flag"].mean() * 100,
        df[df["is_fraud"]==1]["anomaly_flag"].mean() * 100,
    ]
    ax2.bar(["Legitimate", "Fraud"], bars, color=[CLR["legit"], CLR["fraud"]], edgecolor="white")
    for i, v in enumerate(bars):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax2.set_title("% Flagged as Anomaly by True Class", fontsize=12)
    ax2.set_ylabel("% Flagged"); ax2.set_facecolor(CLR["bg"])
    ax2.spines[["top","right"]].set_visible(False)

    out = os.path.join(OUTPUT_DIR, "2_anomaly_detection.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=CLR["bg"])
    plt.close()
    print(f"    ✔ Anomaly detection saved → {out}")
    return df


# ═════════════════════════════════════════════
#  STEP 6 — ML CLASSIFICATION
# ═════════════════════════════════════════════
def train_and_evaluate(df: pd.DataFrame):
    print("\n[6/7] Training ML classifiers …")

    FEATURES = [
        "amount_log", "hour", "day_of_week", "customer_age",
        "account_age_days", "num_prev_txns", "distance_from_home",
        "is_international", "is_odd_hour", "is_high_amount",
        "is_new_account", "is_low_txn_history", "is_far_from_home",
        "is_weekend", "risk_score",
        "merchant_enc", "category_enc", "channel_enc",
    ]

    X = df[FEATURES].fillna(0)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        Xtr = X_train_s if name == "Logistic Regression" else X_train
        Xte = X_test_s  if name == "Logistic Regression" else X_test
        model.fit(Xtr, y_train)
        y_pred  = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        avg_prec = average_precision_score(y_test, y_proba)
        results[name] = {
            "model": model, "y_pred": y_pred, "y_proba": y_proba,
            "roc_auc": roc_auc, "avg_prec": avg_prec,
            "report": classification_report(y_test, y_pred, target_names=["Legit","Fraud"])
        }
        print(f"\n    ── {name} ──")
        print(f"    ROC-AUC : {roc_auc:.4f}   |   Avg Precision : {avg_prec:.4f}")
        print(results[name]["report"])

    # ── Evaluation plots ─────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(22, 13), facecolor=CLR["bg"])
    fig.suptitle("Model Evaluation — Fraud Detection",
                 fontsize=18, fontweight="bold", color=CLR["dark"])

    model_colors = [CLR["legit"], CLR["fraud"]]

    for idx, (name, res) in enumerate(results.items()):
        # Confusion matrix
        ax = axes[0, idx]
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"],
                    linewidths=1, cbar=False)
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=12, color=CLR["dark"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

        # ROC curve
        ax2 = axes[1, idx]
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax2.plot(fpr, tpr, color=model_colors[idx], lw=2.5,
                 label=f"AUC = {res['roc_auc']:.3f}")
        ax2.plot([0,1],[0,1], "k--", lw=1)
        ax2.fill_between(fpr, tpr, alpha=0.12, color=model_colors[idx])
        ax2.set_title(f"{name}\nROC Curve", fontsize=12, color=CLR["dark"])
        ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
        ax2.legend(loc="lower right"); ax2.set_facecolor(CLR["bg"])
        ax2.spines[["top","right"]].set_visible(False)

    # Feature importance (Random Forest)
    ax_fi = axes[0, 2]
    rf = results["Random Forest"]["model"]
    importances = pd.Series(rf.feature_importances_, index=FEATURES).nlargest(12)
    importances.sort_values().plot(kind="barh", ax=ax_fi, color=CLR["accent"], edgecolor="white")
    ax_fi.set_title("Random Forest\nTop 12 Feature Importances", fontsize=12, color=CLR["dark"])
    ax_fi.set_facecolor(CLR["bg"]); ax_fi.spines[["top","right"]].set_visible(False)

    # PR curve comparison
    ax_pr = axes[1, 2]
    for idx, (name, res) in enumerate(results.items()):
        prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
        ax_pr.plot(rec, prec, color=model_colors[idx], lw=2.5,
                   label=f"{name} (AP={res['avg_prec']:.3f})")
    ax_pr.set_title("Precision-Recall Curves", fontsize=12, color=CLR["dark"])
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.legend(); ax_pr.set_facecolor(CLR["bg"])
    ax_pr.spines[["top","right"]].set_visible(False)

    out = os.path.join(OUTPUT_DIR, "3_model_evaluation.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=CLR["bg"])
    plt.close()
    print(f"\n    ✔ Model evaluation saved → {out}")
    return results


# ═════════════════════════════════════════════
#  STEP 7 — FRAUD TREND DASHBOARD
# ═════════════════════════════════════════════
def fraud_trend_dashboard(df: pd.DataFrame):
    print("\n[7/7] Building fraud trend dashboard …")

    fig = plt.figure(figsize=(22, 14), facecolor=CLR["bg"])
    fig.suptitle("Fraud Trend & Risk Intelligence Dashboard",
                 fontsize=20, fontweight="bold", color=CLR["dark"], y=0.99)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # KPI summary strip
    total      = len(df)
    n_fraud    = df["is_fraud"].sum()
    fraud_pct  = n_fraud / total * 100
    avg_fraud_amt  = df[df["is_fraud"]==1]["amount"].mean()
    avg_legit_amt  = df[df["is_fraud"]==0]["amount"].mean()
    high_risk  = (df["risk_score"] >= 6).sum()

    kpis = [
        ("Total Transactions", f"{total:,}",    CLR["dark"]),
        ("Fraud Cases",        f"{n_fraud:,}",  CLR["fraud"]),
        ("Fraud Rate",         f"{fraud_pct:.2f}%", CLR["fraud"]),
        ("Avg Fraud Amount",   f"${avg_fraud_amt:,.0f}", CLR["accent"]),
        ("Avg Legit Amount",   f"${avg_legit_amt:,.0f}", CLR["legit"]),
        ("High-Risk Txns",     f"{high_risk:,}",CLR["accent"]),
    ]
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.set_facecolor(CLR["dark"])
    ax_kpi.set_xlim(0, len(kpis)); ax_kpi.set_ylim(0, 1)
    ax_kpi.axis("off")
    for i, (label, value, color) in enumerate(kpis):
        ax_kpi.text(i + 0.5, 0.65, value, ha="center", va="center",
                    fontsize=20, fontweight="bold", color=color)
        ax_kpi.text(i + 0.5, 0.25, label, ha="center", va="center",
                    fontsize=9, color="white")
    ax_kpi.set_title("Executive KPI Summary", fontsize=13, color="white",
                     loc="left", pad=8)

    # Fraud by day of week
    ax1 = fig.add_subplot(gs[1, 0])
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    daily = df.groupby("day_of_week")["is_fraud"].mean() * 100
    ax1.bar(days, daily.values, color=CLR["fraud"], edgecolor="white")
    ax1.set_title("Fraud Rate by Day of Week", fontsize=12, color=CLR["dark"])
    ax1.set_ylabel("Fraud Rate (%)"); ax1.set_facecolor(CLR["bg"])
    ax1.spines[["top","right"]].set_visible(False)

    # Top merchant fraud
    ax2 = fig.add_subplot(gs[1, 1])
    merch = (df.groupby("merchant")["is_fraud"]
               .agg(["sum","count"])
               .assign(rate=lambda x: x["sum"]/x["count"]*100)
               .sort_values("rate", ascending=True)
               .tail(8))
    merch["rate"].plot(kind="barh", ax=ax2, color=CLR["accent"], edgecolor="white")
    ax2.set_title("Top Merchant Fraud Rates", fontsize=12, color=CLR["dark"])
    ax2.set_xlabel("Fraud Rate (%)"); ax2.set_facecolor(CLR["bg"])
    ax2.spines[["top","right"]].set_visible(False)

    # Cumulative fraud amount heatmap hour × day
    ax3 = fig.add_subplot(gs[1, 2])
    pivot = df[df["is_fraud"]==1].pivot_table(
        values="amount", index="hour", columns="day_of_week",
        aggfunc="sum", fill_value=0)
    pivot.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    sns.heatmap(pivot, ax=ax3, cmap="YlOrRd", fmt=".0f",
                linewidths=0.3, cbar_kws={"label": "Total Fraud $"})
    ax3.set_title("Fraud $ Heatmap — Hour × Day", fontsize=12, color=CLR["dark"])
    ax3.set_xlabel("Day of Week"); ax3.set_ylabel("Hour of Day")

    out = os.path.join(OUTPUT_DIR, "4_fraud_dashboard.png")
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=CLR["bg"])
    plt.close()
    print(f"    ✔ Dashboard saved → {out}")


# ═════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  FRAUD DETECTION ANALYSIS — Industry Data Science Project")
    print("=" * 65)

    df = generate_dataset()
    df = clean_and_preprocess(df)
    df = engineer_features(df)
    exploratory_analysis(df)
    df = anomaly_detection(df)
    train_and_evaluate(df)
    fraud_trend_dashboard(df)

    print("\n" + "=" * 65)
    print("  PROJECT COMPLETE — All outputs saved in /outputs/")
    print("=" * 65)
    print("""
  Output Files:
    outputs/1_eda_analysis.png       — Exploratory Data Analysis
    outputs/2_anomaly_detection.png  — Isolation Forest Anomaly Flags
    outputs/3_model_evaluation.png   — ML Model Performance & Metrics
    outputs/4_fraud_dashboard.png    — Executive Fraud Trend Dashboard
    data/transactions.csv            — Synthetic Dataset (50,000 rows)
    """)


if __name__ == "__main__":
    main()

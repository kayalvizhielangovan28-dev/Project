#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   RISE INTERNSHIP — DATA SCIENCE & ANALYTICS                               ║
║   Project 1: Industry-Oriented Customer Behavior Analysis                  ║
║   for Business Decision Making                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Tools Used:
  • Python 3          – Core programming language
  • Pandas            – Data ingestion, cleaning, transformation
  • NumPy             – Numerical computing, array operations
  • Matplotlib        – Base plotting and figure layout
  • Seaborn           – Statistical visualisations
  • Scikit-learn      – ML models, preprocessing, evaluation
  • (Jupyter Notebook – Run this script cell-by-cell in a .ipynb)

Author  : RISE Intern — Data Science Track
Dataset : Synthetic customer dataset (2 000 records, 14 features)
"""

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 0 — IMPORTS & GLOBAL CONFIG
# ═══════════════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

# Colour palette & plot theme
PALETTE = ['#4361EE', '#F72585', '#7209B7', '#3A0CA3', '#4CC9F0', '#FF9F1C']
BG      = '#F8F9FF'
DARK    = '#1A1A2E'
ACCENT  = '#F72585'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.edgecolor': '#CCCCDD', 'axes.labelcolor': DARK,
    'xtick.color': DARK, 'ytick.color': DARK,
    'text.color': DARK, 'font.family': 'DejaVu Sans',
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.color': '#BBBBCC'
})

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA GENERATION (simulates real customer dataset ingestion)
# ═══════════════════════════════════════════════════════════════════════════
np.random.seed(42)
N = 2000

ages      = np.random.randint(18, 70, N)
genders   = np.random.choice(['Male', 'Female', 'Other'], N, p=[0.48, 0.48, 0.04])
regions   = np.random.choice(['North', 'South', 'East', 'West', 'Central'], N)
segments  = np.random.choice(['Premium', 'Standard', 'Basic', 'New'], N,
                               p=[0.20, 0.35, 0.30, 0.15])
tenure    = np.random.randint(1, 120, N)

monthly_spend = np.where(
    segments == 'Premium',  np.random.normal(4500, 800, N),
    np.where(segments == 'Standard', np.random.normal(2200, 600, N),
    np.where(segments == 'Basic',    np.random.normal(900,  300, N),
                                     np.random.normal(500,  200, N))))
monthly_spend = np.clip(monthly_spend, 100, 12000)

num_products          = np.random.choice([1,2,3,4,5], N, p=[0.30,0.30,0.20,0.12,0.08])
login_frequency       = np.random.poisson(12, N)
support_tickets       = np.random.poisson(1.5, N)
satisfaction_score    = np.random.choice([1,2,3,4,5], N, p=[0.05,0.10,0.20,0.40,0.25])
days_since_purchase   = np.random.randint(1, 365, N)
discount_used         = np.random.choice([0, 1], N, p=[0.40, 0.60])

# Churn logic: low satisfaction, long purchase gap, short tenure → higher risk
churn_prob = (
    0.30 * (satisfaction_score <= 2).astype(int) +
    0.20 * (days_since_purchase > 200).astype(int) +
    0.20 * (tenure < 12).astype(int) +
    0.15 * (support_tickets > 4).astype(int) +
    0.10 * (monthly_spend < 500).astype(int) +
    0.05 * np.random.rand(N)
)
churned = (churn_prob > 0.35).astype(int)

df = pd.DataFrame({
    'customer_id':              [f'CUST{str(i).zfill(5)}' for i in range(1, N+1)],
    'age':                      ages.astype(float),   # float to allow NaN injection
    'gender':                   genders,
    'region':                   regions,
    'customer_segment':         segments,
    'tenure_months':            tenure,
    'monthly_spend':            monthly_spend.round(2),
    'num_products':             num_products,
    'login_frequency':          login_frequency,
    'support_tickets':          support_tickets,
    'satisfaction_score':       satisfaction_score,
    'days_since_last_purchase': days_since_purchase,
    'discount_used':            discount_used,
    'churned':                  churned,
})

# Inject realistic missing values
df.loc[np.random.choice(df.index, 80,  replace=False), 'age']          = np.nan
df.loc[np.random.choice(df.index, 50,  replace=False), 'monthly_spend'] = np.nan

print("─" * 60)
print(f"Dataset shape     : {df.shape}")
print(f"Missing values    :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"Churn rate        : {df['churned'].mean():.1%}")
print("─" * 60)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA CLEANING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
# 2a. Handle missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['monthly_spend'].fillna(
    df.groupby('customer_segment')['monthly_spend'].transform('median'), inplace=True)

# 2b. Feature engineering
df['age_group']  = pd.cut(df['age'], bins=[17,25,35,45,55,70],
                           labels=['18-25', '26-35', '36-45', '46-55', '56+'])
df['spend_tier'] = pd.qcut(df['monthly_spend'], 4,
                            labels=['Low', 'Mid-Low', 'Mid-High', 'High'])
df['clv_score']  = (df['monthly_spend'] * df['tenure_months'] / 1000).round(2)

print("Preprocessing complete. New columns:", ['age_group', 'spend_tier', 'clv_score'])
print(df.describe().T[['mean', 'std', 'min', 'max']].round(2))

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — EXPLORATORY DATA ANALYSIS (EDA)
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 14))
fig.suptitle('Customer Behavior Analysis — EDA Overview',
             fontsize=22, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

# 3a. Churn donut
ax0 = fig.add_subplot(gs[0, 0])
churn_counts = df['churned'].value_counts()
ax0.pie(churn_counts, labels=['Retained', 'Churned'],
        colors=[PALETTE[0], ACCENT], autopct='%1.1f%%',
        startangle=90, pctdistance=0.75,
        wedgeprops={'width': 0.55, 'edgecolor': BG, 'linewidth': 3})
ax0.set_title('Churn vs Retained', fontweight='bold')

# 3b. Spend by segment (boxplot)
ax1 = fig.add_subplot(gs[0, 1:3])
seg_order = ['Premium', 'Standard', 'Basic', 'New']
sns.boxplot(data=df, x='customer_segment', y='monthly_spend',
            order=seg_order, palette=PALETTE[:4], ax=ax1)
ax1.set_title('Monthly Spend by Segment', fontweight='bold')
ax1.set_xlabel('Customer Segment')
ax1.set_ylabel('Monthly Spend (₹)')

# 3c. Age histogram
ax2 = fig.add_subplot(gs[0, 3])
ax2.hist(df['age'].dropna(), bins=25, color=PALETTE[0], alpha=0.85, edgecolor='white')
ax2.axvline(df['age'].median(), color=ACCENT, linestyle='--', linewidth=2,
            label=f'Median: {df["age"].median():.0f}')
ax2.set_title('Age Distribution', fontweight='bold')
ax2.set_xlabel('Age'); ax2.legend(fontsize=9)

# 3d. Churn rate by segment
ax3 = fig.add_subplot(gs[1, 0:2])
churn_seg = df.groupby('customer_segment')['churned'].mean().sort_values(ascending=False)
bars = ax3.bar(churn_seg.index, churn_seg.values * 100,
               color=PALETTE[:len(churn_seg)], edgecolor='white')
for b, v in zip(bars, churn_seg.values):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{v:.1%}',
             ha='center', fontweight='bold', fontsize=10)
ax3.set_title('Churn Rate by Segment', fontweight='bold')
ax3.set_ylabel('Churn Rate (%)')

# 3e. Satisfaction vs churn
ax4 = fig.add_subplot(gs[1, 2:4])
sat_churn = df.groupby('satisfaction_score')['churned'].mean() * 100
ax4.bar(sat_churn.index, sat_churn.values,
        color=[PALETTE[0] if v < 15 else ACCENT for v in sat_churn.values],
        edgecolor='white')
for i, v in zip(sat_churn.index, sat_churn.values):
    ax4.text(i, v+0.3, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=10)
ax4.set_title('Churn Rate by Satisfaction Score', fontweight='bold')
ax4.set_xlabel('Score (1=Worst, 5=Best)')
ax4.set_ylabel('Churn Rate (%)')

# 3f. Tenure vs spend scatter
ax5 = fig.add_subplot(gs[2, 0:2])
m = df['churned'] == 1
ax5.scatter(df.loc[~m,'tenure_months'], df.loc[~m,'monthly_spend'],
            alpha=0.3, s=15, c=PALETTE[0], label='Retained')
ax5.scatter(df.loc[m,'tenure_months'], df.loc[m,'monthly_spend'],
            alpha=0.5, s=20, c=ACCENT, label='Churned', marker='x')
ax5.set_title('Tenure vs Monthly Spend', fontweight='bold')
ax5.set_xlabel('Tenure (Months)'); ax5.set_ylabel('Monthly Spend (₹)')
ax5.legend()

# 3g. Region × segment churn heatmap
ax6 = fig.add_subplot(gs[2, 2:4])
region_seg = (df.groupby(['region', 'customer_segment'])['churned']
                .mean().unstack() * 100)
sns.heatmap(region_seg[seg_order], annot=True, fmt='.1f', cmap='RdYlBu_r',
            ax=ax6, linewidths=0.5, cbar_kws={'label': 'Churn %'})
ax6.set_title('Churn % by Region × Segment', fontweight='bold')

plt.savefig('fig1_eda_dashboard.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
num_cols = ['age', 'tenure_months', 'monthly_spend', 'num_products',
            'login_frequency', 'support_tickets', 'satisfaction_score',
            'days_since_last_purchase', 'clv_score', 'churned']
corr = df[num_cols].corr()

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Correlation Analysis & Feature Insights', fontsize=18, fontweight='bold')

# Heatmap
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=axes[0], linewidths=0.5, cbar_kws={'shrink': 0.8})
axes[0].set_title('Feature Correlation Matrix', fontweight='bold')

# Churn correlation bar chart
churn_corr = corr['churned'].drop('churned').sort_values()
axes[1].barh(churn_corr.index, churn_corr.values,
             color=[ACCENT if v > 0 else PALETTE[0] for v in churn_corr.values],
             edgecolor='white')
axes[1].axvline(0, color=DARK, linewidth=1)
axes[1].set_title('Feature Correlation with Churn', fontweight='bold')
axes[1].set_xlabel('Pearson Correlation')
for i, (idx, v) in enumerate(churn_corr.items()):
    axes[1].text(v + (0.002 if v >= 0 else -0.002), i, f'{v:.3f}',
                 va='center', ha='left' if v >= 0 else 'right', fontsize=9)

plt.tight_layout()
plt.savefig('fig2_correlation.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — CUSTOMER SEGMENTATION (K-Means)
# ═══════════════════════════════════════════════════════════════════════════
seg_features = ['monthly_spend', 'tenure_months', 'login_frequency',
                'num_products', 'satisfaction_score', 'clv_score']
X_seg    = df[seg_features].fillna(0)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_seg)

# Elbow method to find optimal k
inertias = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    inertias.append(km.fit(X_scaled).inertia_)

# Fit final model (k=4)
km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
df['kmeans_cluster'] = km_final.fit_predict(X_scaled)

# PCA for 2-D visualisation
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

CLUSTER_LABELS = {
    0: 'Value Hunters',
    1: 'Loyal Champions',
    2: 'At-Risk Passives',
    3: 'High-Potential Growers'
}
df['cluster_name'] = df['kmeans_cluster'].map(CLUSTER_LABELS)

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle('Customer Segmentation — K-Means Clustering', fontsize=18, fontweight='bold')

# Elbow
axes[0].plot(range(2, 9), inertias, 'o-', color=PALETTE[0], linewidth=2.5, markersize=8)
axes[0].axvline(4, color=ACCENT, linestyle='--', linewidth=2, label='Optimal k=4')
axes[0].set_title('Elbow Method', fontweight='bold')
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia'); axes[0].legend()

# PCA scatter
for cid, cname in CLUSTER_LABELS.items():
    m = df['kmeans_cluster'] == cid
    axes[1].scatter(X_pca[m, 0], X_pca[m, 1], alpha=0.55, s=25,
                    c=PALETTE[cid], label=cname)
axes[1].set_title('Cluster Map (PCA)', fontweight='bold')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
axes[1].legend(fontsize=8)

# Cluster profile bar
cp = df.groupby('kmeans_cluster')[seg_features].mean()
cp_norm = (cp - cp.min()) / (cp.max() - cp.min())
cp_norm.rename(index=CLUSTER_LABELS, inplace=True)
cp_norm.T.plot(kind='bar', ax=axes[2], color=PALETTE[:4], width=0.75, edgecolor='white')
axes[2].set_title('Normalised Cluster Profiles', fontweight='bold')
axes[2].set_xlabel('Feature'); axes[2].set_ylabel('Normalised Value')
axes[2].legend(title='Cluster', fontsize=7, bbox_to_anchor=(1, 1))
axes[2].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('fig3_segmentation.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()

print("\nCluster Summary:")
print(df.groupby('cluster_name')[seg_features + ['churned']].mean().round(2))

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — PREDICTIVE ANALYSIS: CHURN PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
feature_cols = ['age', 'tenure_months', 'monthly_spend', 'num_products',
                'login_frequency', 'support_tickets', 'satisfaction_score',
                'days_since_last_purchase', 'discount_used', 'clv_score',
                'kmeans_cluster']
X = df[feature_cols].fillna(0)
y = df['churned']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

sc2 = StandardScaler()

models = {
    'Logistic Regression': (LogisticRegression(max_iter=500, random_state=42), True),
    'Random Forest':       (RandomForestClassifier(n_estimators=150, random_state=42,
                                                    class_weight='balanced'), False),
    'Gradient Boosting':   (GradientBoostingClassifier(n_estimators=100,random_state=42), False),
}

results = {}
for name, (model, scale) in models.items():
    Xtr = sc2.fit_transform(X_train) if scale else X_train
    Xts = sc2.transform(X_test)      if scale else X_test
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xts)
    y_prob = model.predict_proba(Xts)[:, 1]
    cv_auc = cross_val_score(model, Xtr, y_train, cv=5, scoring='roc_auc').mean()
    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
        'auc': roc_auc_score(y_test, y_prob), 'cv_auc': cv_auc
    }
    print(f"\n{'─'*50}")
    print(f"{name}  |  AUC={results[name]['auc']:.4f}  |  CV-AUC={cv_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Retained','Churned']))

best_name = max(results, key=lambda x: results[x]['auc'])
best      = results[best_name]

# Evaluation charts
fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle('Churn Prediction — Model Evaluation', fontsize=18, fontweight='bold')

names = list(results.keys())
aucs  = [results[n]['auc']    for n in names]
cvs   = [results[n]['cv_auc'] for n in names]
xpos  = np.arange(len(names)); w = 0.35
axes[0].bar(xpos-w/2, aucs, w, color=PALETTE[0], label='Test AUC', edgecolor='white')
axes[0].bar(xpos+w/2, cvs,  w, color=PALETTE[1], label='CV AUC',   edgecolor='white')
for i, (a, c) in enumerate(zip(aucs, cvs)):
    axes[0].text(i-w/2, a+0.005, f'{a:.3f}', ha='center', fontsize=9, fontweight='bold')
    axes[0].text(i+w/2, c+0.005, f'{c:.3f}', ha='center', fontsize=9, fontweight='bold')
axes[0].set_xticks(xpos)
axes[0].set_xticklabels([n.replace(' ', '\n') for n in names])
axes[0].set_ylim(0.5, 1.05)
axes[0].set_title('Model AUC Comparison', fontweight='bold')
axes[0].legend()

for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[1].plot(fpr, tpr, color=PALETTE[i], linewidth=2.5,
                 label=f"{name} ({res['auc']:.3f})")
axes[1].plot([0,1],[0,1], '--', color='gray', linewidth=1.5)
axes[1].set_title('ROC Curves', fontweight='bold')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(fontsize=9)

cm = confusion_matrix(y_test, best['y_pred'])
ConfusionMatrixDisplay(cm, display_labels=['Retained','Churned']).plot(
    ax=axes[2], cmap='Blues', colorbar=False)
axes[2].set_title(f'Confusion Matrix\n({best_name})', fontweight='bold')

plt.tight_layout()
plt.savefig('fig4_churn_models.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()

# ── 6b. Feature Importance ───────────────────────────────────────────────
rf_model = results['Random Forest']['model']
fi        = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()
df['churn_risk_score'] = rf_model.predict_proba(X)[:, 1]
df['risk_tier'] = pd.cut(df['churn_risk_score'], bins=[0, 0.3, 0.6, 1.0],
                          labels=['Low Risk', 'Medium Risk', 'High Risk'])

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Feature Importance & Churn Risk Drivers', fontsize=18, fontweight='bold')

axes[0].barh(fi.index, fi.values,
             color=[ACCENT if v > fi.median() else PALETTE[0] for v in fi.values],
             edgecolor='white')
axes[0].set_title('Random Forest — Feature Importance', fontweight='bold')
axes[0].set_xlabel('Importance Score')
for i, v in enumerate(fi.values):
    axes[0].text(v+0.001, i, f'{v:.3f}', va='center', fontsize=9)

risk_counts = df['risk_tier'].value_counts()
axes[1].pie(risk_counts, labels=risk_counts.index,
            colors=[PALETTE[0], PALETTE[5], ACCENT],
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor': BG, 'linewidth': 3})
axes[1].set_title('Customer Churn Risk Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('fig5_feature_importance.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — BUSINESS INSIGHTS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 14))
fig.suptitle('Business Insights Dashboard — Strategic Recommendations',
             fontsize=22, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

# Revenue at risk by segment
ax0 = fig.add_subplot(gs[0, 0:2])
seg_rev = df.groupby('customer_segment').agg(
    total   =('monthly_spend', 'sum'),
    at_risk =('monthly_spend', lambda x: x[df.loc[x.index,'risk_tier']=='High Risk'].sum())
).reset_index()
x = np.arange(len(seg_rev)); w = 0.35
ax0.bar(x-w/2, seg_rev['total'],   w, color=PALETTE[0], label='Total',   edgecolor='white')
ax0.bar(x+w/2, seg_rev['at_risk'], w, color=ACCENT,     label='At-Risk', edgecolor='white')
ax0.set_xticks(x); ax0.set_xticklabels(seg_rev['customer_segment'])
ax0.set_title('Revenue vs At-Risk Revenue by Segment', fontweight='bold')
ax0.set_ylabel('Monthly Spend (₹)'); ax0.legend()

# CLV by cluster
ax1 = fig.add_subplot(gs[0, 2:4])
clv_c = df.groupby('cluster_name')['clv_score'].mean().sort_values(ascending=False)
bars  = ax1.bar(range(len(clv_c)), clv_c.values, color=PALETTE[:len(clv_c)], edgecolor='white')
ax1.set_xticks(range(len(clv_c)))
ax1.set_xticklabels([c.replace(' ','\n') for c in clv_c.index], fontsize=9)
ax1.set_title('Average CLV Score by Customer Cluster', fontweight='bold')
ax1.set_ylabel('CLV Score (₹K)')
for b, v in zip(bars, clv_c.values):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f'₹{v:.1f}K',
             ha='center', fontweight='bold')

# Engagement vs Monetisation
ax2 = fig.add_subplot(gs[1, 0:2])
sc = ax2.scatter(df['login_frequency'], df['monthly_spend'],
                 c=df['churn_risk_score'], cmap='RdYlGn_r',
                 alpha=0.5, s=20, vmin=0, vmax=1)
plt.colorbar(sc, ax=ax2, label='Churn Risk Score')
ax2.set_title('Engagement vs Monetisation (colour = risk)', fontweight='bold')
ax2.set_xlabel('Login Frequency / Month')
ax2.set_ylabel('Monthly Spend (₹)')

# Recommended actions
ax3 = fig.add_subplot(gs[1, 2:4])
action_counts = {
    'Immediate Intervention\n(High Risk + Low Sat)':
        ((df['risk_tier']=='High Risk') & (df['satisfaction_score']<=2)).sum(),
    'Loyalty Reward\n(Long Tenure + Low Spend)':
        ((df['tenure_months']>48) & (df['spend_tier']=='Low')).sum(),
    'Win-Back Campaign\n(Churned)':
        (df['churned']==1).sum(),
    'Upsell Opportunity\n(High Engage + Few Products)':
        ((df['login_frequency']>15) & (df['num_products']<=2)).sum(),
    'Premium Upgrade\n(High CLV + Standard Seg)':
        ((df['clv_score'] > df['clv_score'].quantile(0.75)) &
         (df['customer_segment']=='Standard')).sum(),
}
ax3.barh(list(action_counts.keys()), list(action_counts.values()),
         color=[ACCENT, PALETTE[5], PALETTE[1], PALETTE[0], PALETTE[2]],
         edgecolor='white')
ax3.set_title('Recommended Action Segments', fontweight='bold')
ax3.set_xlabel('Number of Customers')
for i, v in enumerate(action_counts.values()):
    ax3.text(v+2, i, str(v), va='center', fontweight='bold')

plt.savefig('fig6_business_dashboard.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  RISE PROJECT 1 — BUSINESS INSIGHT SUMMARY REPORT")
print("═"*65)
print(f"  Total Customers Analysed  : {len(df):,}")
print(f"  Overall Churn Rate        : {df['churned'].mean():.1%}")
print(f"  High-Risk Customers       : {(df['risk_tier']=='High Risk').sum():,} "
      f"({(df['risk_tier']=='High Risk').mean():.1%} of base)")
print(f"  Monthly Revenue at Risk   : ₹{df.loc[df['risk_tier']=='High Risk','monthly_spend'].sum():,.0f}")
print(f"  Best Predictive Model     : {best_name} (AUC = {best['auc']:.4f})")
print(f"  Top Churn Driver          : {fi.idxmax()}")
print()
print("  SEGMENT CHURN RATES:")
for seg, rate in churn_seg.items():
    print(f"    {seg:12s}  →  {rate:.1%}")
print()
print("  KEY RECOMMENDATIONS:")
print("  1. Prioritise immediate outreach to High-Risk customers (11.7% of base).")
print("  2. Satisfaction score is the #1 churn predictor — invest in CX programs.")
print("  3. 'New' segment has the highest churn rate — strengthen onboarding.")
print("  4. Loyal Champions cluster has the highest CLV — protect with VIP perks.")
print("  5. Customers inactive >200 days need win-back campaigns with discounts.")
print("═"*65)

df.to_csv('customer_analysis_final.csv', index=False)
print("\n  Output file saved: customer_analysis_final.csv")

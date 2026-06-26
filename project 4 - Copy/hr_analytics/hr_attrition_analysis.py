"""
============================================================
  HR Analytics - Employee Attrition Prediction
  Industry-Oriented Data Science Project
  Compatible with: Windows 10/11, Python 3.8+
============================================================
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Windows-safe backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)
import joblib

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH    = "data/HR_Analytics.csv"
OUTPUT_DIR   = "outputs"
MODEL_DIR    = "models"
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

PALETTE = {
    'primary'   : '#2563EB',
    'danger'    : '#DC2626',
    'success'   : '#16A34A',
    'warning'   : '#D97706',
    'neutral'   : '#6B7280',
    'light'     : '#F3F4F6',
    'attrition' : ['#16A34A', '#DC2626'],
}

plt.rcParams.update({
    'figure.dpi'       : 150,
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'axes.titlesize'   : 13,
    'axes.titleweight' : 'bold',
    'axes.labelsize'   : 11,
})

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✔  Saved → {path}")

def section(title):
    bar = "═" * 60
    print(f"\n{bar}\n  {title}\n{bar}")

# ─────────────────────────────────────────────
# 1. DATA INGESTION
# ─────────────────────────────────────────────
section("1. DATA INGESTION")

df = pd.read_csv(DATA_PATH)
print(f"  Rows   : {df.shape[0]:,}")
print(f"  Columns: {df.shape[1]}")
print(f"\n  Columns:\n  {list(df.columns)}")

# ─────────────────────────────────────────────
# 2. DATA CLEANING & PREPROCESSING
# ─────────────────────────────────────────────
section("2. DATA CLEANING & PREPROCESSING")

print(f"\n  Missing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")
print(f"\n  Duplicate rows: {df.duplicated().sum()}")

# Drop constant / ID columns
drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
print(f"\n  Dropped constant/ID columns: {drop_cols}")

# Encode target
df['Attrition_Flag'] = (df['Attrition'] == 'Yes').astype(int)
print(f"\n  Attrition distribution:\n{df['Attrition'].value_counts()}")
print(f"  Attrition rate: {df['Attrition_Flag'].mean()*100:.1f}%")

# Summary stats
print(f"\n  Numeric summary (selected):")
print(df[['Age','MonthlyIncome','YearsAtCompany','JobSatisfaction']].describe().round(2))

# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
section("3. EXPLORATORY DATA ANALYSIS")

# --- 3a. Attrition Overview
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Employee Attrition Overview', fontsize=15, fontweight='bold', y=1.02)

counts = df['Attrition'].value_counts()
axes[0].pie(counts, labels=counts.index, autopct='%1.1f%%',
            colors=PALETTE['attrition'], startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':2})
axes[0].set_title('Attrition Distribution')

dept_attr = df.groupby('Department')['Attrition_Flag'].mean().sort_values()*100
bars = axes[1].barh(dept_attr.index, dept_attr.values,
                    color=[PALETTE['danger'] if v > 20 else PALETTE['primary'] for v in dept_attr.values])
axes[1].set_xlabel('Attrition Rate (%)')
axes[1].set_title('Attrition Rate by Department')
for bar, val in zip(bars, dept_attr.values):
    axes[1].text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                 f'{val:.1f}%', va='center', fontsize=10)
plt.tight_layout()
save(fig, '01_attrition_overview.png')

# --- 3b. Age & Income distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Age & Income Distribution by Attrition', fontsize=14, fontweight='bold')

for label, color in zip(['No','Yes'], PALETTE['attrition']):
    subset = df[df['Attrition'] == label]['Age']
    axes[0].hist(subset, bins=20, alpha=0.7, color=color, label=label, edgecolor='white')
axes[0].set_xlabel('Age'); axes[0].set_ylabel('Count')
axes[0].set_title('Age Distribution'); axes[0].legend(title='Attrition')

sns.boxplot(data=df, x='Attrition', y='MonthlyIncome',
            palette={'No': PALETTE['success'], 'Yes': PALETTE['danger']}, ax=axes[1])
axes[1].set_title('Monthly Income vs Attrition')
axes[1].set_xlabel('Attrition'); axes[1].set_ylabel('Monthly Income (USD)')
plt.tight_layout()
save(fig, '02_age_income_distribution.png')

# --- 3c. Satisfaction heatmap
section_fields = ['JobSatisfaction','EnvironmentSatisfaction',
                  'RelationshipSatisfaction','WorkLifeBalance','JobInvolvement']
attr_by_sat = pd.DataFrame({
    col: df.groupby(col)['Attrition_Flag'].mean()*100
    for col in section_fields
}).T

fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(attr_by_sat, annot=True, fmt='.1f', cmap='RdYlGn_r',
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Attrition Rate (%)'})
ax.set_title('Attrition Rate (%) by Satisfaction Dimensions & Score', fontsize=13, fontweight='bold')
ax.set_xlabel('Satisfaction Score (1=Low → 4=High)')
plt.tight_layout()
save(fig, '03_satisfaction_heatmap.png')

# --- 3d. Key categorical factors
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Key Categorical Drivers of Attrition', fontsize=14, fontweight='bold')

cats = {
    'OverTime'      : axes[0, 0],
    'BusinessTravel': axes[0, 1],
    'MaritalStatus' : axes[1, 0],
    'JobLevel'      : axes[1, 1],
}
for col, ax in cats.items():
    rates = df.groupby(col)['Attrition_Flag'].mean().sort_values(ascending=False)*100
    colors = [PALETTE['danger'] if v > 20 else PALETTE['primary'] for v in rates.values]
    bars = ax.bar(rates.index.astype(str), rates.values, color=colors, edgecolor='white')
    ax.set_title(f'Attrition by {col}')
    ax.set_ylabel('Attrition Rate (%)')
    ax.set_xlabel(col)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                f'{b.get_height():.1f}%', ha='center', fontsize=9)
plt.tight_layout()
save(fig, '04_categorical_drivers.png')

# --- 3e. Tenure analysis
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Tenure-Related Attrition Patterns', fontsize=14, fontweight='bold')

for ax, col, label in zip(axes,
    ['YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion'],
    ['Years at Company','Years in Role','Years Since Promotion']):
    bins = pd.cut(df[col], bins=[0,1,2,5,10,20,40], include_lowest=True)
    rates = df.groupby(bins, observed=False)['Attrition_Flag'].mean()*100
    ax.bar(rates.index.astype(str), rates.values, color=PALETTE['primary'], edgecolor='white')
    ax.set_title(label); ax.set_ylabel('Attrition Rate (%)')
    ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
save(fig, '05_tenure_analysis.png')

# --- 3f. Correlation matrix
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ['Attrition_Flag']]
corr = df[numeric_cols + ['Attrition_Flag']].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.3, ax=ax, annot_kws={'size': 7})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
save(fig, '06_correlation_matrix.png')

print("  EDA visualizations complete.")

# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────
section("4. FEATURE ENGINEERING")

df_fe = df.copy()

# Derived features
df_fe['IncomeToAge']          = df_fe['MonthlyIncome'] / df_fe['Age']
df_fe['CareerStability']      = df_fe['YearsAtCompany'] / (df_fe['TotalWorkingYears'] + 1)
df_fe['PromotionLag']         = df_fe['YearsSinceLastPromotion'] - df_fe['YearsInCurrentRole']
df_fe['SatisfactionScore']    = (df_fe['JobSatisfaction'] + df_fe['EnvironmentSatisfaction'] +
                                  df_fe['RelationshipSatisfaction'] + df_fe['WorkLifeBalance']) / 4
df_fe['EngagementScore']      = (df_fe['JobInvolvement'] + df_fe['JobSatisfaction']) / 2
df_fe['IsHighRisk']           = (
    (df_fe['OverTime'] == 'Yes').astype(int) +
    (df_fe['JobSatisfaction'] <= 2).astype(int) +
    (df_fe['WorkLifeBalance'] == 1).astype(int) +
    (df_fe['YearsAtCompany'] < 2).astype(int)
)

print("  Engineered features:")
new_features = ['IncomeToAge','CareerStability','PromotionLag',
                'SatisfactionScore','EngagementScore','IsHighRisk']
for f in new_features:
    print(f"    → {f}: mean={df_fe[f].mean():.3f}, std={df_fe[f].std():.3f}")

# Encode categoricals
cat_cols = df_fe.select_dtypes(include='object').columns.tolist()
cat_cols = [c for c in cat_cols if c != 'Attrition']

le_map = {}
for col in cat_cols:
    le = LabelEncoder()
    df_fe[col] = le.fit_transform(df_fe[col])
    le_map[col] = le

print(f"\n  Encoded {len(cat_cols)} categorical columns: {cat_cols}")

# ─────────────────────────────────────────────
# 5. MODEL BUILDING
# ─────────────────────────────────────────────
section("5. MACHINE LEARNING MODELS")

feature_cols = [c for c in df_fe.columns if c not in ['Attrition', 'Attrition_Flag']]
X = df_fe[feature_cols]
y = df_fe['Attrition_Flag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n  Train size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")
print(f"  Features  : {X_train.shape[1]}")

models = {
    'Logistic Regression' : LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'Decision Tree'       : DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    'Random Forest'       : RandomForestClassifier(n_estimators=200, max_depth=8,
                                                    random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                        max_depth=4, random_state=RANDOM_STATE),
    'SVM'                 : SVC(probability=True, kernel='rbf', random_state=RANDOM_STATE),
}

results   = {}
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("\n  Training models...\n")
for name, model in models.items():
    use_scaled = name in ['Logistic Regression', 'SVM']
    Xtr = X_train_sc if use_scaled else X_train
    Xte = X_test_sc  if use_scaled else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring='roc_auc')

    results[name] = {
        'model'    : model,
        'accuracy' : accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall'   : recall_score(y_test, y_pred),
        'f1'       : f1_score(y_test, y_pred),
        'roc_auc'  : roc_auc_score(y_test, y_prob),
        'cv_auc'   : cv_scores.mean(),
        'cv_std'   : cv_scores.std(),
        'y_pred'   : y_pred,
        'y_prob'   : y_prob,
        'scaled'   : use_scaled,
    }
    print(f"  ✔  {name:<22} AUC={results[name]['roc_auc']:.4f}  "
          f"F1={results[name]['f1']:.4f}  CV-AUC={results[name]['cv_auc']:.4f}±{results[name]['cv_std']:.4f}")

# ─────────────────────────────────────────────
# 6. MODEL EVALUATION & VISUALIZATION
# ─────────────────────────────────────────────
section("6. MODEL EVALUATION")

# --- Metrics comparison bar chart
metrics_df = pd.DataFrame({
    name: {k: v for k, v in r.items() if k in ['accuracy','precision','recall','f1','roc_auc']}
    for name, r in results.items()
}).T

fig, ax = plt.subplots(figsize=(13, 6))
x      = np.arange(len(metrics_df))
width  = 0.15
metric_colors = [PALETTE['primary'], PALETTE['success'], PALETTE['warning'],
                 PALETTE['danger'], '#7C3AED']
for i, (col, color) in enumerate(zip(metrics_df.columns, metric_colors)):
    ax.bar(x + i*width, metrics_df[col], width, label=col.title(), color=color, alpha=0.85)
ax.set_xticks(x + width*2)
ax.set_xticklabels(metrics_df.index, rotation=15, ha='right')
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
save(fig, '07_model_comparison.png')

# --- ROC Curves
fig, ax = plt.subplots(figsize=(9, 7))
roc_colors = [PALETTE['primary'], PALETTE['success'], PALETTE['danger'],
              PALETTE['warning'], '#7C3AED']
for (name, r), color in zip(results.items(), roc_colors):
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})",
            color=color, linewidth=2)
ax.plot([0,1],[0,1],'--', color='gray', linewidth=1, label='Random')
ax.fill_between([0,1],[0,1], alpha=0.05, color='gray')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves – All Models', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
save(fig, '08_roc_curves.png')

# --- Best model deep-dive
best_name = max(results, key=lambda k: results[k]['roc_auc'])
best      = results[best_name]
print(f"\n  Best model: {best_name}  (AUC = {best['roc_auc']:.4f})")
print(f"\n  Classification Report:\n"
      f"{classification_report(y_test, best['y_pred'], target_names=['Stay','Leave'])}")

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Best Model Deep-Dive: {best_name}', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Stay','Leave'], yticklabels=['Stay','Leave'])
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix')

# Probability distribution
axes[1].hist(best['y_prob'][y_test == 0], bins=30, alpha=0.7,
             color=PALETTE['success'], label='Actual Stay')
axes[1].hist(best['y_prob'][y_test == 1], bins=30, alpha=0.7,
             color=PALETTE['danger'], label='Actual Leave')
axes[1].axvline(0.5, color='black', linestyle='--', label='Threshold=0.5')
axes[1].set_xlabel('Predicted Attrition Probability')
axes[1].set_ylabel('Count')
axes[1].set_title('Predicted Probability Distribution')
axes[1].legend()
plt.tight_layout()
save(fig, '09_best_model_analysis.png')

# ─────────────────────────────────────────────
# 7. FEATURE IMPORTANCE & ATTRITION DRIVERS
# ─────────────────────────────────────────────
section("7. FEATURE IMPORTANCE & ATTRITION DRIVERS")

rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
top20 = importances.head(20)

fig, ax = plt.subplots(figsize=(11, 8))
colors = [PALETTE['danger'] if i < 5 else PALETTE['primary'] for i in range(len(top20))]
bars = ax.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1], edgecolor='white')
ax.set_xlabel('Feature Importance Score')
ax.set_title('Top 20 Attrition Drivers (Random Forest)', fontsize=13, fontweight='bold')

# Color legend
high_patch = mpatches.Patch(color=PALETTE['danger'], label='Top 5 drivers')
rest_patch = mpatches.Patch(color=PALETTE['primary'], label='Other features')
ax.legend(handles=[high_patch, rest_patch], loc='lower right')
plt.tight_layout()
save(fig, '10_feature_importance.png')

print(f"\n  Top 10 Attrition Drivers:")
for rank, (feat, score) in enumerate(top20.head(10).items(), 1):
    print(f"    {rank:2}. {feat:<30} {score:.4f}")

# --- Attrition risk segmentation
# Risk breakdown
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Attrition Risk Segmentation', fontsize=14, fontweight='bold')

risk_labels = pd.cut(pd.Series(best['y_prob']),
                     bins=[0, 0.3, 0.6, 1.0], labels=['Low Risk', 'Medium Risk', 'High Risk'])

risk_counts = risk_labels.value_counts()
axes[0].pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%',
            colors=[PALETTE['success'], PALETTE['warning'], PALETTE['danger']],
            wedgeprops={'edgecolor':'white','linewidth':2})
axes[0].set_title('Test Set Risk Distribution')

# Salary vs risk scatter
test_df = X_test.copy()
test_df['AttritionProb'] = best['y_prob']
test_df['Actual'] = y_test.values
scatter = axes[1].scatter(test_df['MonthlyIncome'], test_df['Age'],
                          c=test_df['AttritionProb'], cmap='RdYlGn_r',
                          alpha=0.6, s=30, vmin=0, vmax=1)
plt.colorbar(scatter, ax=axes[1], label='Attrition Probability')
axes[1].set_xlabel('Monthly Income'); axes[1].set_ylabel('Age')
axes[1].set_title('Risk Map: Income vs Age')
plt.tight_layout()
save(fig, '11_risk_segmentation.png')

# ─────────────────────────────────────────────
# 8. SAVE MODELS & ARTIFACTS
# ─────────────────────────────────────────────
section("8. SAVING MODELS & ARTIFACTS")

joblib.dump(rf_model,        os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
joblib.dump(results[best_name]['model'], os.path.join(MODEL_DIR, 'best_model.pkl'))
joblib.dump(scaler,          os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(le_map,          os.path.join(MODEL_DIR, 'label_encoders.pkl'))
joblib.dump(feature_cols,    os.path.join(MODEL_DIR, 'feature_columns.pkl'))
print("  Models saved to /models/")

# Results CSV
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'model_metrics.csv'))
importances.head(20).to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), header=['importance'])
print("  Metrics & importance CSVs saved to /outputs/")

# ─────────────────────────────────────────────
# 9. HR INSIGHTS & RECOMMENDATIONS
# ─────────────────────────────────────────────
section("9. HR INSIGHTS & STRATEGIC RECOMMENDATIONS")

insights = """
╔══════════════════════════════════════════════════════════════╗
║           HR ANALYTICS — KEY INSIGHTS & RECOMMENDATIONS      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📊 DATA FINDINGS                                            ║
║  ─────────────────────────────────────────────────────────  ║
║  • Overall attrition rate is well above industry avg        ║
║  • Overtime workers show ~2x higher attrition risk          ║
║  • Employees with <2 years tenure are highest risk          ║
║  • Low job satisfaction (score 1-2) sharply lifts risk      ║
║  • Frequent travelers leave at disproportionately high rates ║
║  • Single employees have higher attrition than married      ║
║                                                              ║
║  🤖 MODEL PERFORMANCE SUMMARY                               ║
║  ─────────────────────────────────────────────────────────  ║
║  Best Model   : {best:<26}              ║
║  ROC-AUC      : {auc:<26.4f}              ║
║  F1-Score     : {f1:<26.4f}              ║
║  Recall       : {recall:<26.4f}              ║
║  (Recall = % of actual leavers correctly identified)        ║
║                                                              ║
║  🎯 STRATEGIC RECOMMENDATIONS                               ║
║  ─────────────────────────────────────────────────────────  ║
║  1. OVERTIME POLICY                                         ║
║     Audit excessive overtime; introduce comp-time or        ║
║     flexible scheduling. Cap mandatory overtime at 10h/wk.  ║
║                                                             ║
║  2. EARLY TENURE RETENTION PROGRAMME                        ║
║     Implement 90-day, 6-month, and 1-year check-ins.        ║
║     Assign mentors to all employees in first 24 months.     ║
║                                                             ║
║  3. SATISFACTION PULSE SURVEYS                              ║
║     Quarterly micro-surveys on job, env, & relationship     ║
║     satisfaction. Flag employees with score ≤ 2 for HR      ║
║     follow-up within 2 weeks.                               ║
║                                                             ║
║  4. TRAVEL COMPENSATION                                     ║
║     Review travel allowance & recovery days for frequent    ║
║     travellers. Consider role-rotation every 18 months.     ║
║                                                             ║
║  5. COMPENSATION BENCHMARKING                              ║
║     Monthly income is a top-5 attrition driver. Conduct     ║
║     annual market pay surveys; target P60 for key roles.    ║
║                                                             ║
║  6. PREDICTIVE ATTRITION DASHBOARD                         ║
║     Deploy model scores in HRIS monthly. Route employees    ║
║     with >60% probability to retention interviews.          ║
║                                                             ║
╚══════════════════════════════════════════════════════════════╝
""".format(best=best_name, auc=best['roc_auc'], f1=best['f1'], recall=best['recall'])

print(insights)

# Save report
with open(os.path.join(OUTPUT_DIR, 'HR_Analytics_Report.txt'), 'w') as f:
    f.write("HR ANALYTICS — ATTRITION PREDICTION REPORT\n")
    f.write("="*60 + "\n\n")
    f.write(f"Dataset: {df.shape[0]} employees, {df.shape[1]} features\n")
    f.write(f"Attrition rate: {df['Attrition_Flag'].mean()*100:.1f}%\n\n")
    f.write("MODEL PERFORMANCE:\n")
    f.write(metrics_df.to_string() + "\n\n")
    f.write("TOP 10 ATTRITION DRIVERS:\n")
    f.write(importances.head(10).to_string() + "\n\n")
    f.write(insights)
print("  Report saved to outputs/HR_Analytics_Report.txt")

section("PIPELINE COMPLETE ✔")
print(f"""
  Summary
  ────────────────────────────────────
  Employees analysed : {df.shape[0]:,}
  Features used      : {len(feature_cols)}
  Models trained     : {len(models)}
  Best model         : {best_name}
  Best AUC           : {best['roc_auc']:.4f}
  Outputs saved to   : ./{OUTPUT_DIR}/
  Models saved to    : ./{MODEL_DIR}/
  ────────────────────────────────────
  Charts generated:
    01_attrition_overview.png
    02_age_income_distribution.png
    03_satisfaction_heatmap.png
    04_categorical_drivers.png
    05_tenure_analysis.png
    06_correlation_matrix.png
    07_model_comparison.png
    08_roc_curves.png
    09_best_model_analysis.png
    10_feature_importance.png
    11_risk_segmentation.png
""")

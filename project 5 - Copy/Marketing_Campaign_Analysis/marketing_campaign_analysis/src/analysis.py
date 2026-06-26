"""
analysis.py
Full marketing campaign performance analysis.
Run: python src/analysis.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os, textwrap

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/marketing_campaigns.csv"
OUT_DIR     = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────
PALETTE = ["#2563EB","#7C3AED","#059669","#DC2626","#D97706",
           "#0891B2","#BE185D","#65A30D"]
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8FAFC",
    "axes.edgecolor":   "#CBD5E1",
    "grid.color":       "#E2E8F0",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

money  = FuncFormatter(lambda x, _: f"${x:,.0f}")
pct    = FuncFormatter(lambda x, _: f"{x:.1f}%")
comma  = FuncFormatter(lambda x, _: f"{x:,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  MARKETING CAMPAIGN PERFORMANCE ANALYSIS")
print("=" * 65)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
print(f"\n▶ Loaded  : {len(df):,} rows × {len(df.columns)} columns")

# Basic cleaning
before = len(df)
df.drop_duplicates(inplace=True)
df.dropna(subset=["campaign_name","impressions","clicks","conversions","spend"], inplace=True)
df = df[(df["impressions"] > 0) & (df["spend"] > 0) & (df["clicks"] <= df["impressions"])]
print(f"▶ Cleaned : {len(df):,} rows retained  ({before - len(df)} removed)")

# Derived metrics (recalculate cleanly)
df["ctr"]  = (df["clicks"]      / df["impressions"] * 100).round(2)
df["cvr"]  = (df["conversions"] / df["clicks"]      * 100).round(2)
df["roi"]  = ((df["revenue"] - df["spend"]) / df["spend"] * 100).round(2)
df["roas"] = (df["revenue"] / df["spend"]).round(2)
df["cpa"]  = (df["spend"]   / df["conversions"].replace(0, np.nan)).round(2)
df["profit"]         = df["revenue"] - df["spend"]
df["budget_util_pct"]= (df["spend"] / df["budget_allocated"] * 100).round(2)
df["month_num"]      = df["date"].dt.month

# ══════════════════════════════════════════════════════════════════════════════
# 2. SUMMARY STATS
# ══════════════════════════════════════════════════════════════════════════════
total = {
    "Total Spend":        df["spend"].sum(),
    "Total Revenue":      df["revenue"].sum(),
    "Total Profit":       df["profit"].sum(),
    "Total Impressions":  df["impressions"].sum(),
    "Total Clicks":       df["clicks"].sum(),
    "Total Conversions":  df["conversions"].sum(),
    "Overall ROI (%)":    df["roi"].mean(),
    "Overall ROAS":       df["roas"].mean(),
    "Avg CTR (%)":        df["ctr"].mean(),
    "Avg CVR (%)":        df["cvr"].mean(),
    "Avg CPA ($)":        df["cpa"].mean(),
    "Budget Utilisation": df["budget_util_pct"].mean(),
}
print("\n── Overall KPIs ────────────────────────────────────────────")
for k, v in total.items():
    if "%" in k:    print(f"   {k:<28}: {v:>9.2f}%")
    elif "$" in k:  print(f"   {k:<28}: ${v:>8.2f}")
    elif isinstance(v, float): print(f"   {k:<28}: {v:>9.2f}")
    else:           print(f"   {k:<28}: {v:>12,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — EXECUTIVE DASHBOARD (2 × 3 grid)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 13))
fig.suptitle("Marketing Campaign — Executive Dashboard", fontsize=18, fontweight="bold", y=0.98)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

camp_kpi = df.groupby("campaign_name").agg(
    spend=("spend","sum"), revenue=("revenue","sum"),
    conversions=("conversions","sum"), impressions=("impressions","sum"),
    clicks=("clicks","sum")).reset_index()
camp_kpi["roi"]  = ((camp_kpi["revenue"] - camp_kpi["spend"]) / camp_kpi["spend"] * 100).round(1)
camp_kpi["ctr"]  = (camp_kpi["clicks"] / camp_kpi["impressions"] * 100).round(2)
camp_kpi["roas"] = (camp_kpi["revenue"] / camp_kpi["spend"]).round(2)
camp_kpi.sort_values("revenue", ascending=False, inplace=True)

# 1-a  Revenue by campaign
ax = fig.add_subplot(gs[0, 0])
bars = ax.barh(camp_kpi["campaign_name"], camp_kpi["revenue"]/1e6,
               color=PALETTE[:len(camp_kpi)])
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.1f}M"))
ax.set_title("Total Revenue by Campaign", fontweight="bold")
ax.set_xlabel("Revenue (USD)")
for bar in bars:
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"${bar.get_width():.1f}M", va="center", fontsize=8)

# 1-b  ROI by campaign
ax = fig.add_subplot(gs[0, 1])
sorted_roi = camp_kpi.sort_values("roi")
colors_roi = ["#DC2626" if v < 100 else "#059669" for v in sorted_roi["roi"]]
ax.barh(sorted_roi["campaign_name"], sorted_roi["roi"], color=colors_roi)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.xaxis.set_major_formatter(pct)
ax.set_title("ROI by Campaign", fontweight="bold")

# 1-c  Spend vs Revenue scatter
ax = fig.add_subplot(gs[0, 2])
for i, row in camp_kpi.iterrows():
    ax.scatter(row["spend"]/1e3, row["revenue"]/1e3,
               s=row["conversions"]/5, alpha=0.85,
               color=PALETTE[list(camp_kpi.index).index(i) % len(PALETTE)],
               label=row["campaign_name"])
ax.set_xlabel("Total Spend ($K)")
ax.set_ylabel("Total Revenue ($K)")
ax.set_title("Spend vs Revenue\n(bubble = conversions)", fontweight="bold")
ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_: f"${x:.0f}K"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"${x:.0f}K"))
ax.legend(fontsize=6, loc="upper left")

# 1-d  Monthly revenue trend
monthly = df.groupby("month_num").agg(revenue=("revenue","sum"),
                                      spend=("spend","sum")).reset_index()
ax = fig.add_subplot(gs[1, 0])
ax.fill_between(monthly["month_num"], monthly["revenue"]/1e6, alpha=0.25, color=PALETTE[0])
ax.plot(monthly["month_num"], monthly["revenue"]/1e6, marker="o", color=PALETTE[0], label="Revenue")
ax.fill_between(monthly["month_num"], monthly["spend"]/1e6, alpha=0.15, color=PALETTE[3])
ax.plot(monthly["month_num"], monthly["spend"]/1e6, marker="s", color=PALETTE[3], label="Spend")
ax.set_xticks(range(1,13))
ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"${x:.1f}M"))
ax.set_title("Monthly Revenue vs Spend", fontweight="bold")
ax.legend()

# 1-e  CTR by channel
chan_ctr = df.groupby("channel").agg(ctr=("ctr","mean"),
                                      cvr=("cvr","mean")).reset_index()
x = np.arange(len(chan_ctr))
ax = fig.add_subplot(gs[1, 1])
w = 0.35
ax.bar(x - w/2, chan_ctr["ctr"], w, label="CTR", color=PALETTE[0])
ax.bar(x + w/2, chan_ctr["cvr"], w, label="CVR", color=PALETTE[2])
ax.set_xticks(x); ax.set_xticklabels(chan_ctr["channel"], rotation=20, ha="right")
ax.yaxis.set_major_formatter(pct)
ax.set_title("Avg CTR & CVR by Channel", fontweight="bold")
ax.legend()

# 1-f  Conversion funnel
ax = fig.add_subplot(gs[1, 2])
funnel_vals = [df["impressions"].sum(), df["clicks"].sum(), df["conversions"].sum()]
funnel_labels = [f"Impressions\n{funnel_vals[0]/1e6:.1f}M",
                 f"Clicks\n{funnel_vals[1]/1e3:.0f}K",
                 f"Conversions\n{funnel_vals[2]/1e3:.0f}K"]
widths = [v / funnel_vals[0] for v in funnel_vals]
heights = [0.25] * 3
ys      = [0.62, 0.35, 0.08]
for i, (w2, y, lbl) in enumerate(zip(widths, ys, funnel_labels)):
    ax.barh(y, w2, height=0.22, color=PALETTE[i], alpha=0.85)
    ax.text(w2/2, y, lbl, ha="center", va="center", fontweight="bold",
            fontsize=9, color="white")
ax.set_xlim(0, 1.05); ax.axis("off")
ax.set_title("Conversion Funnel", fontweight="bold")

plt.savefig(f"{OUT_DIR}/fig1_executive_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅  Fig 1 — Executive Dashboard saved.")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — CAMPAIGN DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Campaign Deep-Dive Analysis", fontsize=16, fontweight="bold")

# 2-a  Spend distribution
ax = axes[0, 0]
for i, (cname, grp) in enumerate(df.groupby("campaign_name")):
    ax.hist(grp["spend"], bins=30, alpha=0.55, label=cname, color=PALETTE[i % len(PALETTE)])
ax.xaxis.set_major_formatter(money)
ax.set_title("Spend Distribution by Campaign", fontweight="bold")
ax.set_xlabel("Spend per Record"); ax.set_ylabel("Frequency")
ax.legend(fontsize=7)

# 2-b  ROAS heatmap (campaign × quarter)
ax = axes[0, 1]
pivot_roas = df.pivot_table(values="roas", index="campaign_name",
                             columns="quarter", aggfunc="mean")
sns.heatmap(pivot_roas, annot=True, fmt=".1f", cmap="RdYlGn",
            linewidths=0.5, ax=ax, cbar_kws={"label":"ROAS"})
ax.set_title("ROAS Heatmap — Campaign × Quarter", fontweight="bold")
ax.set_xlabel(""); ax.set_ylabel("")

# 2-c  CPA by campaign & device
ax = axes[1, 0]
pivot_cpa = df.pivot_table(values="cpa", index="campaign_name",
                            columns="device_type", aggfunc="mean")
pivot_cpa.plot(kind="bar", ax=ax, color=PALETTE[:3], width=0.7, edgecolor="white")
ax.yaxis.set_major_formatter(money)
ax.set_title("Avg CPA — Campaign × Device", fontweight="bold")
ax.set_xlabel(""); ax.tick_params(axis="x", rotation=25)
ax.legend(title="Device")

# 2-d  Violin: ROI distributions
ax = axes[1, 1]
order = camp_kpi["campaign_name"].tolist()
sns.violinplot(data=df, x="campaign_name", y="roi", ax=ax,
               palette=PALETTE, order=order, cut=0, inner="quartile")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(pct)
ax.axhline(0, linestyle="--", color="red", linewidth=0.8)
ax.set_title("ROI Distribution by Campaign (violin)", fontweight="bold")
ax.set_xlabel("")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig2_campaign_deepdive.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅  Fig 2 — Campaign Deep-Dive saved.")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — CUSTOMER BEHAVIOUR
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle("Customer Response & Behaviour Analysis", fontsize=16, fontweight="bold")

# 3-a  Segment revenue
seg = df.groupby("customer_segment").agg(revenue=("revenue","sum"),
                                          conversions=("conversions","sum"),
                                          spend=("spend","sum")).reset_index()
seg["roi"] = ((seg["revenue"] - seg["spend"]) / seg["spend"] * 100).round(1)
ax = axes[0, 0]
bars = ax.bar(seg["customer_segment"], seg["revenue"]/1e6, color=PALETTE[:4])
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"${x:.1f}M"))
ax.set_title("Revenue by Customer Segment", fontweight="bold")
ax.set_xticklabels(seg["customer_segment"], rotation=15, ha="right", fontsize=8)
for bar, roi in zip(bars, seg["roi"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"ROI:{roi:.0f}%", ha="center", va="bottom", fontsize=7.5, color="#374151")

# 3-b  Device split (pie)
ax = axes[0, 1]
dev = df.groupby("device_type")["revenue"].sum()
wedges, texts, autotexts = ax.pie(dev, labels=dev.index, autopct="%1.1f%%",
                                   colors=PALETTE[:3], startangle=140,
                                   wedgeprops={"edgecolor":"white","linewidth":1.5})
for at in autotexts: at.set_fontsize(9)
ax.set_title("Revenue Share by Device", fontweight="bold")

# 3-c  Region heatmap
ax = axes[0, 2]
reg_pivot = df.pivot_table(values="conversions", index="region",
                            columns="campaign_name", aggfunc="sum")
sns.heatmap(reg_pivot, annot=True, fmt=".0f", cmap="Blues",
            linewidths=0.4, ax=ax)
ax.set_title("Conversions — Region × Campaign", fontweight="bold")
ax.set_xlabel(""); ax.set_ylabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)

# 3-d  Gender ROI
ax = axes[1, 0]
gen = df.groupby("gender").agg(roi=("roi","mean"),
                                revenue=("revenue","sum")).reset_index()
ax.bar(gen["gender"], gen["roi"], color=PALETTE[4:7])
ax.yaxis.set_major_formatter(pct)
ax.set_title("Avg ROI by Gender", fontweight="bold")

# 3-e  Time on site vs CVR
ax = axes[1, 1]
bucket = pd.cut(df["avg_time_on_site_s"], bins=6)
time_cvr = df.groupby(bucket, observed=True)["cvr"].mean().reset_index()
time_cvr["label"] = time_cvr["avg_time_on_site_s"].astype(str).str[:12]
ax.plot(time_cvr["label"], time_cvr["cvr"], marker="o", color=PALETTE[0], linewidth=2)
ax.yaxis.set_major_formatter(pct)
ax.set_title("Avg Time on Site vs CVR", fontweight="bold")
ax.set_xlabel("Time on Site (seconds, bucketed)")
ax.tick_params(axis="x", rotation=20)

# 3-f  Bounce rate by channel
ax = axes[1, 2]
bounce = df.groupby("channel")["bounce_rate"].mean().sort_values()
ax.barh(bounce.index, bounce.values, color=PALETTE[:len(bounce)])
ax.xaxis.set_major_formatter(pct)
ax.set_title("Avg Bounce Rate by Channel", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig3_customer_behaviour.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅  Fig 3 — Customer Behaviour saved.")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — ML: CLUSTERING + REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
features = ["spend","revenue","ctr","cvr","roi","roas","cpa","bounce_rate",
            "avg_time_on_site_s","avg_pages_per_visit"]
ml_df = df[features].copy().dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(ml_df)

# K-Means clustering
inertias = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

best_k = 4
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
ml_df["cluster"] = km_final.fit_predict(X_scaled)

pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

# Linear regression: spend → revenue
lr = LinearRegression()
lr.fit(df[["spend"]], df["revenue"])
y_pred = lr.predict(df[["spend"]])
r2 = r2_score(df["revenue"], y_pred)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("ML-Driven Campaign Insights", fontsize=16, fontweight="bold")

# 4-a  Elbow
ax = axes[0, 0]
ax.plot(list(K_range), inertias, marker="o", color=PALETTE[0], linewidth=2)
ax.axvline(best_k, linestyle="--", color=PALETTE[3], linewidth=1.2,
           label=f"Chosen k={best_k}")
ax.set_title("K-Means Elbow Curve", fontweight="bold")
ax.set_xlabel("Number of Clusters"); ax.set_ylabel("Inertia")
ax.legend()

# 4-b  PCA cluster scatter
ax = axes[0, 1]
for c in range(best_k):
    mask = ml_df["cluster"] == c
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], alpha=0.5, s=15,
               color=PALETTE[c], label=f"Cluster {c}")
ax.set_title("Campaign Clusters (PCA 2D projection)", fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.legend()

# 4-c  Cluster profiles
ax = axes[1, 0]
profile_cols = ["roi","roas","ctr","cvr"]
cluster_profile = ml_df.groupby("cluster")[profile_cols].mean()
cluster_profile_norm = (cluster_profile - cluster_profile.min()) / \
                       (cluster_profile.max() - cluster_profile.min())
cluster_profile_norm.T.plot(kind="bar", ax=ax, color=PALETTE[:best_k],
                             width=0.7, edgecolor="white")
ax.set_title("Cluster Profiles (normalised)", fontweight="bold")
ax.set_xlabel("Metric"); ax.set_ylabel("Normalised Score")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title="Cluster", fontsize=8)

# 4-d  Linear regression: spend → revenue
ax = axes[1, 1]
sample = df.sample(min(800, len(df)), random_state=42)
ax.scatter(sample["spend"], sample["revenue"], alpha=0.25, s=12, color=PALETTE[0])
spend_line = np.linspace(df["spend"].min(), df["spend"].max(), 200)
ax.plot(spend_line, lr.predict(spend_line.reshape(-1,1)),
        color=PALETTE[3], linewidth=2, label=f"Linear fit  R²={r2:.3f}")
ax.xaxis.set_major_formatter(money); ax.yaxis.set_major_formatter(money)
ax.set_title("Spend → Revenue Regression", fontweight="bold")
ax.set_xlabel("Spend ($)"); ax.set_ylabel("Revenue ($)")
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4_ml_insights.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅  Fig 4 — ML Insights saved.")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — BUDGET EFFICIENCY & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Budget Efficiency & Optimisation Signals", fontsize=16, fontweight="bold")

# 5-a  Budget utilisation vs ROI
ax = axes[0]
bu_roi = df.groupby("campaign_name").agg(
    budget_util=("budget_util_pct","mean"),
    roi=("roi","mean")).reset_index()
for i, row in bu_roi.iterrows():
    ax.scatter(row["budget_util"], row["roi"], s=140,
               color=PALETTE[i % len(PALETTE)], zorder=3)
    ax.annotate(row["campaign_name"].replace("_"," "),
                (row["budget_util"], row["roi"]), fontsize=6.5,
                xytext=(4, 4), textcoords="offset points")
ax.axhline(bu_roi["roi"].mean(), linestyle="--", color="grey", linewidth=0.8, label="Avg ROI")
ax.axvline(80, linestyle=":", color="orange", linewidth=0.8, label="80% util line")
ax.set_xlabel("Avg Budget Utilisation (%)"); ax.set_ylabel("Avg ROI (%)")
ax.set_title("Budget Utilisation vs ROI", fontweight="bold")
ax.legend(fontsize=8)

# 5-b  Quarterly spend & profit
ax = axes[1]
q_data = df.groupby("quarter").agg(spend=("spend","sum"),
                                    profit=("profit","sum")).reset_index()
x = np.arange(len(q_data))
bars1 = ax.bar(x - 0.18, q_data["spend"]/1e6,  0.35, label="Spend",  color=PALETTE[3], alpha=0.85)
bars2 = ax.bar(x + 0.18, q_data["profit"]/1e6, 0.35, label="Profit", color=PALETTE[2], alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(q_data["quarter"])
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"${x:.1f}M"))
ax.set_title("Quarterly Spend & Profit", fontweight="bold")
ax.legend()

# 5-c  Correlation heatmap
ax = axes[2]
corr_cols = ["spend","revenue","impressions","clicks","conversions",
             "ctr","cvr","roi","roas","cpa","bounce_rate"]
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, linewidths=0.4, ax=ax,
            annot_kws={"size": 6}, cbar_kws={"shrink": 0.8})
ax.set_title("Metric Correlation Matrix", fontweight="bold")
ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig5_budget_efficiency.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅  Fig 5 — Budget Efficiency saved.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PRINT ACTIONABLE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  ACTIONABLE INSIGHTS & RECOMMENDATIONS")
print("=" * 65)

best_roi_camp  = camp_kpi.sort_values("roi",  ascending=False).iloc[0]
worst_roi_camp = camp_kpi.sort_values("roi",  ascending=True).iloc[0]
best_rev_camp  = camp_kpi.sort_values("revenue", ascending=False).iloc[0]
best_roas_camp = camp_kpi.sort_values("roas", ascending=False).iloc[0]

insights = [
    ("1. Top Revenue Generator",
     f"{best_rev_camp['campaign_name'].replace('_',' ')} generated the highest revenue "
     f"(${best_rev_camp['revenue']:,.0f}) and should receive increased budget allocation."),
    ("2. Best ROI Campaign",
     f"{best_roi_camp['campaign_name'].replace('_',' ')} delivered the best ROI at "
     f"{best_roi_camp['roi']:.1f}%. Prioritise scaling this campaign."),
    ("3. Under-performer",
     f"{worst_roi_camp['campaign_name'].replace('_',' ')} has the lowest ROI "
     f"({worst_roi_camp['roi']:.1f}%). Review creative assets and targeting or reallocate budget."),
    ("4. Best ROAS",
     f"{best_roas_camp['campaign_name'].replace('_',' ')} has the highest ROAS "
     f"({best_roas_camp['roas']:.2f}×), meaning every $1 spent returns "
     f"${best_roas_camp['roas']:.2f} in revenue."),
    ("5. Device Strategy",
     f"Analyse device-level CPA: allocate more budget toward the device with the "
     f"lowest CPA to maximise conversion efficiency."),
    ("6. Seasonal Opportunities",
     f"Monthly trend reveals peak revenue months. Pre-load budget in those months "
     f"and reduce spend during troughs."),
    ("7. Audience Insight",
     f"Segment with highest ROI should be targeted with personalised messaging "
     f"across top-performing channels."),
    ("8. Retargeting Signal",
     f"High bounce-rate channels indicate poor landing page alignment. "
     f"A/B test landing pages for those channels to reduce drop-off."),
    ("9. Cluster Targeting",
     f"K-Means clustering identified {best_k} distinct campaign performance segments. "
     f"Model 'high-ROI cluster' attributes and replicate them in future campaigns."),
    ("10. Budget Utilisation",
     f"Campaigns with <80% budget utilisation still delivered strong ROI — "
     f"avoid over-spending beyond the efficiency threshold."),
]

for title, body in insights:
    print(f"\n  {title}")
    for line in textwrap.wrap(body, width=60):
        print(f"    {line}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. SAVE SUMMARY REPORT CSV
# ══════════════════════════════════════════════════════════════════════════════
camp_kpi["profit"] = camp_kpi["revenue"] - camp_kpi["spend"]
camp_kpi.to_csv(f"{OUT_DIR}/campaign_summary.csv", index=False)
print(f"\n✅  Summary CSV saved → {OUT_DIR}/campaign_summary.csv")
print("\n🎉  Analysis complete. All outputs in ./outputs/")

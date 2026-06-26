"""
Supply Chain Visualization Module
All Matplotlib / Seaborn charts saved to outputs/
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE   = ['#2196F3','#FF5722','#4CAF50','#FF9800','#9C27B0','#00BCD4']
BG        = '#0D1117'
CARD      = '#161B22'
TEXT      = '#E6EDF3'
ACCENT    = '#58A6FF'
GRID      = '#21262D'

def _style():
    plt.rcParams.update({
        'figure.facecolor':  BG,
        'axes.facecolor':    CARD,
        'axes.edgecolor':    GRID,
        'axes.labelcolor':   TEXT,
        'axes.titlecolor':   TEXT,
        'xtick.color':       TEXT,
        'ytick.color':       TEXT,
        'text.color':        TEXT,
        'grid.color':        GRID,
        'grid.linestyle':    '--',
        'grid.alpha':        0.5,
        'legend.facecolor':  CARD,
        'legend.edgecolor':  GRID,
        'legend.labelcolor': TEXT,
        'font.family':       'DejaVu Sans',
    })

def _save(name: str):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close('all')
    print(f"  Saved: {name}")
    return path

def fmt_inr(x, _=None):
    if x >= 1e7:  return f'₹{x/1e7:.1f}Cr'
    if x >= 1e5:  return f'₹{x/1e5:.1f}L'
    if x >= 1e3:  return f'₹{x/1e3:.0f}K'
    return f'₹{x:.0f}'

# ── 1. Dashboard Overview ────────────────────────────────────────────────────
def plot_dashboard(df: pd.DataFrame, turnover_df: pd.DataFrame,
                   stockout_df: pd.DataFrame):
    _style()
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Supply Chain & Inventory Intelligence Dashboard',
                 fontsize=22, fontweight='bold', color=ACCENT, y=0.98)

    total_rev      = df['revenue'].sum()
    total_lost     = (df['lost_sales'] * df['unit_price']).sum()
    avg_fill       = df['fill_rate'].mean() * 100
    total_stockout = df['stockout_flag'].sum()

    # KPI cards
    kpis = [
        ('Total Revenue',    fmt_inr(total_rev),     '#2196F3'),
        ('Lost Revenue',     fmt_inr(total_lost),    '#FF5722'),
        ('Avg Fill Rate',    f'{avg_fill:.1f}%',      '#4CAF50'),
        ('Stockout Events',  f'{total_stockout:,}',   '#FF9800'),
    ]
    for i, (label, val, col) in enumerate(kpis):
        ax = fig.add_axes([0.02 + i * 0.245, 0.81, 0.22, 0.13])
        ax.set_facecolor(CARD)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')
        ax.add_patch(mpatches.FancyBboxPatch((0.05,0.05), 0.9, 0.9,
            boxstyle='round,pad=0.02', facecolor=CARD, edgecolor=col, lw=2))
        ax.text(0.5, 0.65, val, ha='center', va='center',
                fontsize=22, fontweight='bold', color=col)
        ax.text(0.5, 0.25, label, ha='center', va='center',
                fontsize=11, color=TEXT)

    # Monthly revenue
    ax1 = fig.add_axes([0.02, 0.44, 0.44, 0.33])
    monthly_rev = df.groupby(df['date'].dt.to_period('M'))['revenue'].sum()
    ax1.fill_between(range(len(monthly_rev)), monthly_rev.values,
                     alpha=0.3, color='#2196F3')
    ax1.plot(range(len(monthly_rev)), monthly_rev.values,
             color='#2196F3', lw=2)
    tick_step = max(1, len(monthly_rev) // 8)
    ax1.set_xticks(range(0, len(monthly_rev), tick_step))
    ax1.set_xticklabels([str(monthly_rev.index[i])
                         for i in range(0, len(monthly_rev), tick_step)],
                        rotation=45, fontsize=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_inr))
    ax1.set_title('Monthly Revenue Trend', fontweight='bold', color=ACCENT)
    ax1.grid(True, alpha=0.3)

    # Category revenue pie
    ax2 = fig.add_axes([0.50, 0.44, 0.22, 0.33])
    cat_rev = df.groupby('category')['revenue'].sum()
    wedges, _, autotexts = ax2.pie(
        cat_rev.values, labels=None,
        autopct='%1.1f%%', colors=PALETTE,
        pctdistance=0.75, startangle=90)
    for at in autotexts:
        at.set_fontsize(8); at.set_color(TEXT)
    ax2.legend(cat_rev.index, loc='lower center', fontsize=7,
               bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax2.set_title('Revenue by Category', fontweight='bold', color=ACCENT)

    # Fill rate by category
    ax3 = fig.add_axes([0.75, 0.44, 0.23, 0.33])
    fill = df.groupby('category')['fill_rate'].mean().sort_values()
    bars = ax3.barh(fill.index, fill.values * 100, color=PALETTE[:len(fill)])
    for bar, val in zip(bars, fill.values):
        ax3.text(bar.get_width() - 2, bar.get_y() + bar.get_height() / 2,
                 f'{val*100:.1f}%', va='center', ha='right', fontsize=8, color=BG)
    ax3.set_xlim(0, 105)
    ax3.axvline(95, color='#FF5722', lw=1.5, linestyle='--', alpha=0.8, label='95% target')
    ax3.set_title('Fill Rate by Category', fontweight='bold', color=ACCENT)
    ax3.legend(fontsize=8)

    # Turnover ratio
    ax4 = fig.add_axes([0.02, 0.04, 0.30, 0.34])
    cat_turn = turnover_df.groupby('category')['turnover_ratio'].mean().sort_values()
    bars4 = ax4.bar(cat_turn.index, cat_turn.values, color=PALETTE[:len(cat_turn)])
    for bar, val in zip(bars4, cat_turn.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.1f}x', ha='center', fontsize=9, color=TEXT)
    ax4.set_xticklabels(cat_turn.index, rotation=30, fontsize=8)
    ax4.set_title('Avg Inventory Turnover Ratio', fontweight='bold', color=ACCENT)
    ax4.set_ylabel('Turnover (×/year)', color=TEXT)

    # Stockout vs Overstock
    ax5 = fig.add_axes([0.36, 0.04, 0.30, 0.34])
    x   = np.arange(len(stockout_df))
    w   = 0.35
    b1  = ax5.bar(x - w/2, stockout_df['stockout_pct'],  w, label='Stockout %',  color='#FF5722')
    b2  = ax5.bar(x + w/2, stockout_df['overstock_pct'], w, label='Overstock %', color='#FF9800')
    ax5.set_xticks(x)
    ax5.set_xticklabels(stockout_df['category'], rotation=30, fontsize=8)
    ax5.set_title('Stockout vs Overstock (%)', fontweight='bold', color=ACCENT)
    ax5.legend(fontsize=8)
    ax5.set_ylabel('% of Days', color=TEXT)

    # Heatmap: category × month
    ax6 = fig.add_axes([0.70, 0.04, 0.29, 0.34])
    pivot = df.groupby(['category','month'])['units_sold'].sum().unstack()
    pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                     'Jul','Aug','Sep','Oct','Nov','Dec']
    sns.heatmap(pivot, ax=ax6, cmap='YlOrRd',
                annot=False, fmt='.0f', linewidths=0.3,
                cbar_kws={'shrink': 0.8})
    ax6.set_title('Sales Volume Heatmap\n(Category × Month)',
                  fontweight='bold', color=ACCENT, fontsize=10)
    ax6.tick_params(axis='x', labelsize=7, rotation=45)
    ax6.tick_params(axis='y', labelsize=8, rotation=0)

    _save('01_dashboard_overview.png')

# ── 2. Demand Trend & Seasonality ───────────────────────────────────────────
def plot_demand_trends(df: pd.DataFrame, season_df: pd.DataFrame,
                       cat_season_df: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Demand Trend & Seasonality Analysis',
                 fontsize=18, fontweight='bold', color=ACCENT, y=1.01)

    # Overall monthly
    ax = axes[0, 0]
    monthly = df.groupby(df['date'].dt.to_period('M'))['units_demanded'].sum()
    ax.plot(range(len(monthly)), monthly.values, color='#2196F3', lw=2)
    ax.fill_between(range(len(monthly)), monthly.values, alpha=0.2, color='#2196F3')
    step = max(1, len(monthly) // 8)
    ax.set_xticks(range(0, len(monthly), step))
    ax.set_xticklabels([str(monthly.index[i]) for i in range(0, len(monthly), step)],
                       rotation=45, fontsize=8)
    ax.set_title('Overall Monthly Demand', fontweight='bold', color=ACCENT)
    ax.set_ylabel('Units Demanded'); ax.grid(True, alpha=0.3)

    # Seasonality index
    ax = axes[0, 1]
    months = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']
    colors = ['#FF5722' if v > 1.05 else ('#4CAF50' if v < 0.95 else '#2196F3')
              for v in season_df['seasonality_index']]
    bars = ax.bar(months, season_df['seasonality_index'], color=colors)
    ax.axhline(1.0, color='white', lw=1.5, linestyle='--', alpha=0.7, label='Baseline')
    ax.set_title('Seasonality Index (Overall)', fontweight='bold', color=ACCENT)
    ax.set_ylabel('Index (1.0 = avg)'); ax.legend(fontsize=9)
    for bar, val in zip(bars, season_df['seasonality_index']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', fontsize=8, color=TEXT)

    # Category demand over time
    ax = axes[1, 0]
    for i, cat in enumerate(df['category'].unique()):
        cat_monthly = (df[df['category'] == cat]
                         .groupby(df['date'].dt.to_period('M'))['units_demanded'].sum())
        ax.plot(range(len(cat_monthly)), cat_monthly.values,
                label=cat, color=PALETTE[i % len(PALETTE)], lw=1.8)
    ax.set_xticks([])
    ax.set_title('Monthly Demand by Category', fontweight='bold', color=ACCENT)
    ax.set_ylabel('Units Demanded')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

    # Category seasonality heatmap
    ax = axes[1, 1]
    pivot = cat_season_df.pivot(index='category', columns='month',
                                values='seasonality_index')
    pivot.columns = months
    sns.heatmap(pivot, ax=ax, cmap='RdYlGn', center=1.0, annot=True,
                fmt='.2f', linewidths=0.5, cbar_kws={'label': 'Seasonality Index'})
    ax.set_title('Category-wise Seasonality Index', fontweight='bold', color=ACCENT)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    _save('02_demand_trends_seasonality.png')

# ── 3. Inventory Turnover Analysis ──────────────────────────────────────────
def plot_inventory_turnover(turnover_df: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Inventory Turnover Analysis',
                 fontsize=18, fontweight='bold', color=ACCENT)

    cat_turn = turnover_df.groupby('category').agg(
        avg_turnover =('turnover_ratio','mean'),
        avg_dos      =('days_on_shelf','mean'),
        avg_fill     =('fill_rate','mean'),
    ).reset_index()

    # Box plot: turnover by category
    ax = axes[0, 0]
    cats = turnover_df['category'].unique()
    data = [turnover_df[turnover_df['category'] == c]['turnover_ratio'].values for c in cats]
    bp   = ax.boxplot(data, labels=cats, patch_artist=True, notch=True)
    for patch, col in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    for element in ['whiskers','caps','fliers','medians']:
        plt.setp(bp[element], color=TEXT)
    ax.set_title('Turnover Ratio Distribution by Category',
                 fontweight='bold', color=ACCENT)
    ax.set_ylabel('Turnover Ratio (×/year)')
    ax.tick_params(axis='x', rotation=30, labelsize=8)

    # Scatter: turnover vs fill rate
    ax = axes[0, 1]
    for i, cat in enumerate(cats):
        sub = turnover_df[turnover_df['category'] == cat]
        ax.scatter(sub['turnover_ratio'], sub['fill_rate'] * 100,
                   color=PALETTE[i], alpha=0.6, label=cat, s=60)
    ax.set_xlabel('Inventory Turnover Ratio')
    ax.set_ylabel('Fill Rate (%)')
    ax.set_title('Turnover vs Fill Rate', fontweight='bold', color=ACCENT)
    ax.axhline(95, color='#FF5722', lw=1.5, linestyle='--', alpha=0.8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Days on shelf
    ax = axes[1, 0]
    sorted_df = cat_turn.sort_values('avg_dos')
    bars = ax.barh(sorted_df['category'], sorted_df['avg_dos'], color=PALETTE)
    ax.axvline(30, color='#4CAF50', lw=2, linestyle='--', label='30-day target')
    ax.axvline(60, color='#FF5722', lw=2, linestyle='--', label='60-day warning')
    for bar, val in zip(bars, sorted_df['avg_dos']):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}d', va='center', fontsize=9, color=TEXT)
    ax.set_title('Avg Days on Shelf (DSI)', fontweight='bold', color=ACCENT)
    ax.set_xlabel('Days'); ax.legend(fontsize=9)

    # Quadrant: high/low turnover vs fill rate
    ax = axes[1, 1]
    med_turn = cat_turn['avg_turnover'].median()
    med_fill = cat_turn['avg_fill'].median() * 100
    for i, row in cat_turn.iterrows():
        ax.scatter(row['avg_turnover'], row['avg_fill'] * 100,
                   s=200, color=PALETTE[i % len(PALETTE)], zorder=3)
        ax.annotate(row['category'], (row['avg_turnover'], row['avg_fill'] * 100),
                    textcoords='offset points', xytext=(8, 4), fontsize=9, color=TEXT)
    ax.axvline(med_turn, color='white', lw=1, linestyle='--', alpha=0.5)
    ax.axhline(med_fill, color='white', lw=1, linestyle='--', alpha=0.5)
    ax.text(med_turn * 1.05, med_fill * 1.005, 'Stars ✓', color='#4CAF50', fontsize=9)
    ax.text(0.1, med_fill * 1.005, 'Slow movers', color='#FF9800', fontsize=9)
    ax.set_title('Turnover vs Fill Rate Quadrant', fontweight='bold', color=ACCENT)
    ax.set_xlabel('Avg Turnover Ratio'); ax.set_ylabel('Avg Fill Rate (%)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save('03_inventory_turnover.png')

# ── 4. Stockout & Overstock Analysis ────────────────────────────────────────
def plot_stockout_overstock(df: pd.DataFrame, stockout_df: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Stockout & Overstock Pattern Analysis',
                 fontsize=18, fontweight='bold', color=ACCENT)

    # Stockout frequency monthly
    ax = axes[0, 0]
    df2 = df.copy()
    df2['ym'] = df2['date'].dt.to_period('M')
    monthly_so = df2.groupby('ym')['stockout_flag'].sum()
    ax.bar(range(len(monthly_so)), monthly_so.values,
           color=['#FF5722' if v > monthly_so.mean() else '#FF9800'
                  for v in monthly_so.values], alpha=0.8)
    step = max(1, len(monthly_so) // 8)
    ax.set_xticks(range(0, len(monthly_so), step))
    ax.set_xticklabels([str(monthly_so.index[i]) for i in range(0, len(monthly_so), step)],
                       rotation=45, fontsize=8)
    ax.axhline(monthly_so.mean(), color='white', lw=1.5, linestyle='--',
               label=f'Avg: {monthly_so.mean():.0f}')
    ax.set_title('Monthly Stockout Events', fontweight='bold', color=ACCENT)
    ax.set_ylabel('Stockout Events'); ax.legend(fontsize=9)

    # Lost vs actual revenue
    ax = axes[0, 1]
    df2_lost = df.copy()
    df2_lost['lost_revenue'] = df2_lost['lost_sales'] * df2_lost['unit_price']
    cat_lost = df2_lost.groupby('category').agg(
        actual_rev=('revenue','sum'),
        lost_rev  =('lost_revenue','sum'),
    ).reset_index()
    x = np.arange(len(cat_lost)); w = 0.4
    ax.bar(x - w/2, cat_lost['actual_rev'] / 1e7, w, label='Actual Revenue (Cr)', color='#4CAF50')
    ax.bar(x + w/2, cat_lost['lost_rev']   / 1e7, w, label='Lost Revenue (Cr)',   color='#FF5722')
    ax.set_xticks(x); ax.set_xticklabels(cat_lost['category'], rotation=30, fontsize=8)
    ax.set_title('Actual vs Lost Revenue by Category', fontweight='bold', color=ACCENT)
    ax.set_ylabel('Revenue (₹ Crore)'); ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'₹{v:.1f}Cr'))

    # Inventory level distribution
    ax = axes[1, 0]
    for i, cat in enumerate(df['category'].unique()):
        sub = df[df['category'] == cat]['inventory_level']
        ax.hist(sub, bins=40, alpha=0.5, color=PALETTE[i], label=cat, density=True)
    ax.set_title('Inventory Level Distribution by Category',
                 fontweight='bold', color=ACCENT)
    ax.set_xlabel('Inventory Level (units)'); ax.set_ylabel('Density')
    ax.legend(fontsize=8)

    # Stockout risk heatmap: DOW × Month
    ax = axes[1, 1]
    pivot_so = df.groupby(['day_of_week','month'])['stockout_flag'].mean().unstack()
    pivot_so.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    pivot_so.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec']
    sns.heatmap(pivot_so, ax=ax, cmap='Reds', annot=True,
                fmt='.3f', linewidths=0.3, cbar_kws={'label': 'Stockout Rate'})
    ax.set_title('Stockout Risk: Day of Week × Month', fontweight='bold', color=ACCENT)
    ax.tick_params(axis='x', rotation=45, labelsize=8)

    plt.tight_layout()
    _save('04_stockout_overstock.png')

# ── 5. Demand Forecast ──────────────────────────────────────────────────────
def plot_forecast(forecast_results: dict):
    _style()
    cats = list(forecast_results.keys())
    n    = len(cats)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Demand Forecasting (Polynomial Regression)',
                 fontsize=18, fontweight='bold', color=ACCENT)

    for idx, cat in enumerate(cats):
        ax  = axes[idx // 3][idx % 3]
        res = forecast_results[cat]
        hist = res['historical']
        fore = res['forecast']
        x_h  = range(len(hist))
        x_f  = range(len(hist), len(hist) + len(fore))

        ax.plot(x_h, hist['units_demanded'], color='#2196F3', lw=2, label='Historical')
        ax.plot(x_f, fore['forecast_demand'], color='#FF9800',
                lw=2.5, linestyle='--', label='Forecast')
        lo = fore['forecast_demand'] * 0.85
        hi = fore['forecast_demand'] * 1.15
        ax.fill_between(x_f, lo, hi, alpha=0.2, color='#FF9800', label='±15% CI')
        ax.axvline(len(hist) - 0.5, color='white', lw=1, linestyle=':', alpha=0.6)
        ax.set_title(f'{cat}\nMAE={res["mae"]:,.0f}  R²={res["r2"]:.3f}',
                     fontweight='bold', color=ACCENT, fontsize=10)
        ax.set_ylabel('Units Demanded'); ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save('05_demand_forecast.png')

# ── 6. EOQ & Optimization ───────────────────────────────────────────────────
def plot_eoq_optimization(eoq_df: pd.DataFrame):
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle('EOQ & Supply Chain Optimization Insights',
                 fontsize=18, fontweight='bold', color=ACCENT)

    # EOQ by category
    ax = axes[0, 0]
    cat_eoq = eoq_df.groupby('category')['eoq'].mean().sort_values(ascending=False)
    bars = ax.bar(cat_eoq.index, cat_eoq.values, color=PALETTE)
    for bar, val in zip(bars, cat_eoq.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}', ha='center', fontsize=9, color=TEXT)
    ax.set_title('Economic Order Quantity (EOQ) by Category',
                 fontweight='bold', color=ACCENT)
    ax.set_ylabel('EOQ (units)'); ax.tick_params(axis='x', rotation=30, labelsize=8)

    # Cost breakdown
    ax = axes[0, 1]
    cat_cost = eoq_df.groupby('category').agg(
        order_cost  =('annual_order_cost','mean'),
        holding_cost=('annual_holding_cost','mean'),
    ).reset_index()
    x = np.arange(len(cat_cost)); w = 0.4
    ax.bar(x - w/2, cat_cost['order_cost']   / 1e3, w, label='Ordering Cost (K)', color='#2196F3')
    ax.bar(x + w/2, cat_cost['holding_cost'] / 1e3, w, label='Holding Cost (K)',  color='#FF9800')
    ax.set_xticks(x); ax.set_xticklabels(cat_cost['category'], rotation=30, fontsize=8)
    ax.set_title('Annual Cost Breakdown at EOQ', fontweight='bold', color=ACCENT)
    ax.set_ylabel('Cost (₹ Thousands)'); ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'₹{v:.0f}K'))

    # Annual demand vs EOQ scatter
    ax = axes[1, 0]
    for i, cat in enumerate(eoq_df['category'].unique()):
        sub = eoq_df[eoq_df['category'] == cat]
        ax.scatter(sub['annual_demand'], sub['eoq'],
                   color=PALETTE[i], alpha=0.7, label=cat, s=80)
    ax.set_title('Annual Demand vs EOQ', fontweight='bold', color=ACCENT)
    ax.set_xlabel('Annual Demand (units)'); ax.set_ylabel('EOQ (units)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Total cost curve (illustrative)
    ax = axes[1, 1]
    q_range = np.linspace(10, 1000, 300)
    D, S, H = 5000, 500, 250
    tc = (D / q_range) * S + (q_range / 2) * H
    oc = (D / q_range) * S
    hc = (q_range / 2) * H
    q_star = np.sqrt(2 * D * S / H)
    ax.plot(q_range, tc, color='#2196F3', lw=2.5, label='Total Cost')
    ax.plot(q_range, oc, color='#4CAF50', lw=1.5, linestyle='--', label='Ordering Cost')
    ax.plot(q_range, hc, color='#FF9800', lw=1.5, linestyle='--', label='Holding Cost')
    ax.axvline(q_star, color='#FF5722', lw=2, linestyle=':', label=f'EOQ = {q_star:.0f}')
    ax.scatter([q_star], [2 * (D / q_star) * S], color='#FF5722', s=100, zorder=5)
    ax.set_title('EOQ Cost Curve (Illustrative)', fontweight='bold', color=ACCENT)
    ax.set_xlabel('Order Quantity (units)'); ax.set_ylabel('Annual Cost (₹)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'₹{v:,.0f}'))

    plt.tight_layout()
    _save('06_eoq_optimization.png')

"""
Project 6: Industry-Oriented Supply Chain & Inventory Demand Analysis
========================================================================
Run with:  python main.py
All outputs saved to  outputs/
"""

import os, sys, time, warnings
warnings.filterwarnings('ignore')

# ── make src importable ────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from generate_data   import generate_supply_chain_data
from analysis        import (preprocess, monthly_demand_trend, seasonality_index,
                              category_seasonality, inventory_turnover,
                              stockout_overstock_summary, forecast_demand, eoq_analysis)
from visualizations  import (plot_dashboard, plot_demand_trends,
                              plot_inventory_turnover, plot_stockout_overstock,
                              plot_forecast, plot_eoq_optimization)

import pandas as pd

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║   Supply Chain & Inventory Demand Analysis  |  Project 6        ║
║   Tools: Python · Pandas · NumPy · Matplotlib · Seaborn · SKL   ║
╚══════════════════════════════════════════════════════════════════╝
"""

def hr(title=''):
    print('\n' + '─' * 60 + (f'  {title}' if title else ''))

def main():
    print(BANNER)
    t0 = time.time()

    # ── 1. Data Generation ────────────────────────────────────────
    hr('STEP 1 · Generating Dataset')
    sales_df, products_df = generate_supply_chain_data()
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    sales_df.to_csv(os.path.join(data_dir, 'sales_inventory.csv'), index=False)
    products_df.to_csv(os.path.join(data_dir, 'products.csv'), index=False)
    print(f'  Sales records : {len(sales_df):,}')
    print(f'  Products      : {len(products_df)}')
    print(f'  Date range    : {sales_df["date"].min().date()} → {sales_df["date"].max().date()}')
    print(f'  Categories    : {", ".join(sales_df["category"].unique())}')

    # ── 2. Preprocessing ─────────────────────────────────────────
    hr('STEP 2 · Preprocessing & Validation')
    df = preprocess(sales_df)

    nulls = df.isnull().sum().sum()
    print(f'  Null values   : {nulls}')
    print(f'  Columns       : {list(df.columns)}')
    print(f'  Fill rate     : {df["fill_rate"].mean()*100:.2f}% (avg)')
    print(f'  Revenue total : ₹{df["revenue"].sum()/1e7:.2f} Cr')

    # ── 3. Demand Analytics ───────────────────────────────────────
    hr('STEP 3 · Demand Trend & Seasonality')
    monthly_demand = monthly_demand_trend(df)
    season_idx     = seasonality_index(df)
    cat_season     = category_seasonality(df)

    peak = season_idx.loc[season_idx['seasonality_index'].idxmax()]
    trough = season_idx.loc[season_idx['seasonality_index'].idxmin()]
    months = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']
    print(f'  Peak month    : {months[int(peak["month"])-1]}  (index {peak["seasonality_index"]:.2f})')
    print(f'  Trough month  : {months[int(trough["month"])-1]}  (index {trough["seasonality_index"]:.2f})')

    # ── 4. Inventory Turnover ─────────────────────────────────────
    hr('STEP 4 · Inventory Turnover Analysis')
    turnover_df = inventory_turnover(df)
    best  = turnover_df.groupby('category')['turnover_ratio'].mean().idxmax()
    worst = turnover_df.groupby('category')['turnover_ratio'].mean().idxmin()
    print(f'  Best turnover  : {best}')
    print(f'  Worst turnover : {worst}')
    print('\n  Category Turnover Summary:')
    cat_sum = turnover_df.groupby('category').agg(
        Avg_Turnover=('turnover_ratio','mean'),
        Avg_DSI     =('days_on_shelf','mean'),
        Avg_Fill    =('fill_rate','mean'),
    ).round(2)
    print(cat_sum.to_string())

    # ── 5. Stockout / Overstock ───────────────────────────────────
    hr('STEP 5 · Stockout & Overstock Patterns')
    stockout_df = stockout_overstock_summary(df)
    total_lost  = (df['lost_sales'] * df['unit_price']).sum()
    print(f'  Total lost revenue : ₹{total_lost/1e7:.2f} Cr')
    print('\n  Stockout / Overstock by Category:')
    print(stockout_df[['category','stockout_pct','overstock_pct','lost_revenue']].to_string(index=False))

    # ── 6. Demand Forecasting ─────────────────────────────────────
    hr('STEP 6 · Demand Forecasting')
    categories = df['category'].unique().tolist()
    forecast_results = {}
    for cat in categories:
        res = forecast_demand(df, cat, periods=12)
        forecast_results[cat] = res
        print(f'  {cat:<18} MAE={res["mae"]:>8,.0f}  RMSE={res["rmse"]:>8,.0f}  R²={res["r2"]:.4f}')

    # ── 7. EOQ Analysis ───────────────────────────────────────────
    hr('STEP 7 · EOQ & Optimization')
    eoq_df = eoq_analysis(products_df, df)
    print('  EOQ Summary by Category:')
    print(eoq_df.groupby('category')[['eoq','annual_order_cost','annual_holding_cost',
                                       'total_annual_cost']].mean().round(0).to_string())

    # ── 8. Visualizations ─────────────────────────────────────────
    hr('STEP 8 · Generating Visualizations')
    print('  Rendering charts...')
    plot_dashboard(df, turnover_df, stockout_df)
    plot_demand_trends(df, season_idx, cat_season)
    plot_inventory_turnover(turnover_df)
    plot_stockout_overstock(df, stockout_df)
    plot_forecast(forecast_results)
    plot_eoq_optimization(eoq_df)

    # ── 9. Insights Summary ───────────────────────────────────────
    hr('STEP 9 · Supply Chain Optimization Insights')
    insights = [
        ('Revenue',       f'Total ₹{df["revenue"].sum()/1e7:.1f} Cr across 3 years'),
        ('Lost Revenue',  f'₹{total_lost/1e7:.1f} Cr lost due to stockouts — optimize reorder points'),
        ('Best Turnover', f'{best} shows highest inventory efficiency'),
        ('Worst Turnover',f'{worst} needs demand forecasting & lean ordering'),
        ('Peak Season',   f'{months[int(peak["month"])-1]} is peak — pre-stock by {months[int(peak["month"])-2]}'),
        ('EOQ',           f'Avg EOQ ranges from {eoq_df.groupby("category")["eoq"].mean().min():.0f} to '
                          f'{eoq_df.groupby("category")["eoq"].mean().max():.0f} units by category'),
        ('Fill Rate',     f'Avg {df["fill_rate"].mean()*100:.1f}% — target ≥ 95% for key categories'),
    ]
    for label, text in insights:
        print(f'  ▸ {label:<14} {text}')

    # ── Done ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    hr()
    print(f'\n  ✅  Analysis complete in {elapsed:.1f}s')
    print(f'  📁  Charts saved to:  outputs/')
    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    for f in sorted(os.listdir(out_dir)):
        if f.endswith('.png'):
            print(f'     • {f}')
    print()


if __name__ == '__main__':
    main()

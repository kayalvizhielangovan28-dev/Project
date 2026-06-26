"""
Supply Chain Analysis Module
Core analytics: demand trends, seasonality, turnover, stock-out/overstock
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# ── helpers ──────────────────────────────────────────────────────────────────

def load_data(sales_path: str, products_path: str):
    sales_df    = pd.read_csv(sales_path, parse_dates=['date'])
    products_df = pd.read_csv(products_path)
    return sales_df, products_df


def preprocess(sales_df: pd.DataFrame) -> pd.DataFrame:
    df = sales_df.copy()
    df['date']          = pd.to_datetime(df['date'])
    df['year']          = df['date'].dt.year
    df['month']         = df['date'].dt.month
    df['quarter']       = df['date'].dt.quarter
    df['week']          = df['date'].dt.isocalendar().week.astype(int)
    df['day_of_week']   = df['date'].dt.dayofweek
    df['month_name']    = df['date'].dt.strftime('%b')
    df['fill_rate']     = (df['units_sold'] / df['units_demanded'].replace(0, np.nan)).fillna(1.0)
    df['service_level'] = df['fill_rate'].clip(0, 1)
    return df


# ── demand analytics ──────────────────────────────────────────────────────────

def monthly_demand_trend(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(['year', 'month', 'category'])
              .agg(total_demand=('units_demanded','sum'),
                   total_sold  =('units_sold','sum'),
                   revenue     =('revenue','sum'))
              .reset_index())


def seasonality_index(df: pd.DataFrame) -> pd.DataFrame:
    monthly  = df.groupby('month')['units_demanded'].mean()
    overall  = df['units_demanded'].mean()
    idx      = (monthly / overall).reset_index()
    idx.columns = ['month', 'seasonality_index']
    return idx


def category_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    cat_month = (df.groupby(['category','month'])['units_demanded']
                   .mean().reset_index())
    cat_avg   = df.groupby('category')['units_demanded'].mean().reset_index()
    cat_avg.columns = ['category','avg_demand']
    merged = cat_month.merge(cat_avg, on='category')
    merged['seasonality_index'] = merged['units_demanded'] / merged['avg_demand']
    return merged


# ── inventory turnover ────────────────────────────────────────────────────────

def inventory_turnover(df: pd.DataFrame) -> pd.DataFrame:
    result = []
    for (prod_id, cat), grp in df.groupby(['product_id','category']):
        cogs        = grp['units_sold'].sum() * grp['unit_price'].mean()
        avg_inv     = grp['inventory_level'].mean() * grp['unit_price'].mean()
        turnover    = cogs / avg_inv if avg_inv > 0 else 0
        dos         = 365 / turnover if turnover > 0 else np.inf
        stockout_days = grp['stockout_flag'].sum()
        overstock_days= (grp['inventory_level'] > grp['reorder_point'] * 3).sum()
        result.append({
            'product_id':     prod_id,
            'category':       cat,
            'cogs':           round(cogs, 2),
            'avg_inventory':  round(avg_inv, 2),
            'turnover_ratio': round(turnover, 2),
            'days_on_shelf':  round(dos, 1),
            'stockout_days':  stockout_days,
            'overstock_days': overstock_days,
            'fill_rate':      round(grp['fill_rate'].mean(), 3),
        })
    return pd.DataFrame(result)


# ── stockout / overstock ──────────────────────────────────────────────────────

def stockout_overstock_summary(df: pd.DataFrame) -> pd.DataFrame:
    result = []
    for cat, grp in df.groupby('category'):
        total_days        = len(grp)
        stockout_days     = grp['stockout_flag'].sum()
        overstock_days    = (grp['inventory_level'] > grp['reorder_point'] * 3).sum()
        lost_revenue      = (grp['lost_sales'] * grp['unit_price']).sum()
        excess_holding    = grp['holding_cost'][grp['inventory_level'] >
                                                grp['reorder_point'] * 3].sum()
        result.append({
            'category':          cat,
            'total_records':     total_days,
            'stockout_days':     int(stockout_days),
            'overstock_days':    int(overstock_days),
            'stockout_pct':      round(stockout_days / total_days * 100, 2),
            'overstock_pct':     round(overstock_days / total_days * 100, 2),
            'lost_revenue':      round(lost_revenue, 2),
            'excess_holding_cost':round(excess_holding, 2),
        })
    return pd.DataFrame(result)


# ── demand forecasting ────────────────────────────────────────────────────────

def forecast_demand(df: pd.DataFrame, category: str, periods: int = 12):
    cat_df = (df[df['category'] == category]
                .groupby(['year','month'])['units_demanded']
                .sum().reset_index())
    cat_df['time_idx'] = range(len(cat_df))

    X = cat_df[['time_idx']].values
    y = cat_df['units_demanded'].values

    poly = PolynomialFeatures(degree=2)
    X_p  = poly.fit_transform(X)
    model = LinearRegression().fit(X_p, y)

    future_idx  = np.arange(len(cat_df), len(cat_df) + periods).reshape(-1, 1)
    future_pred = model.predict(poly.transform(future_idx))

    mae  = mean_absolute_error(y, model.predict(X_p))
    rmse = np.sqrt(mean_squared_error(y, model.predict(X_p)))

    return {
        'historical':    cat_df,
        'forecast':      pd.DataFrame({'period': range(1, periods + 1),
                                       'forecast_demand': future_pred.clip(0)}),
        'mae':           round(mae, 2),
        'rmse':          round(rmse, 2),
        'r2':            round(model.score(X_p, y), 4),
    }


# ── EOQ helper ────────────────────────────────────────────────────────────────

def eoq_analysis(products_df: pd.DataFrame, sales_df: pd.DataFrame,
                  holding_rate: float = 0.25, order_cost: float = 500) -> pd.DataFrame:
    annual = (sales_df.groupby('product_id')['units_sold']
                      .sum().reset_index()
                      .rename(columns={'units_sold':'annual_demand'}))
    merged = products_df.merge(annual, on='product_id')

    merged['holding_cost_unit'] = merged['unit_price'] * holding_rate
    merged['eoq'] = np.sqrt(
        (2 * merged['annual_demand'] * order_cost) / merged['holding_cost_unit']
    ).round(0)
    merged['annual_order_cost']   = (merged['annual_demand'] / merged['eoq']) * order_cost
    merged['annual_holding_cost'] = (merged['eoq'] / 2) * merged['holding_cost_unit']
    merged['total_annual_cost']   = merged['annual_order_cost'] + merged['annual_holding_cost']
    return merged[['product_id','product_name','category','unit_price',
                   'annual_demand','eoq','annual_order_cost',
                   'annual_holding_cost','total_annual_cost']].round(2)

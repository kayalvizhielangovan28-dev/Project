"""
Supply Chain Dataset Generator
Generates realistic synthetic inventory and sales data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

def generate_supply_chain_data():
    # Configuration
    start_date = datetime(2022, 1, 1)
    end_date   = datetime(2024, 12, 31)
    date_range = pd.date_range(start_date, end_date, freq='D')

    categories = {
        'Electronics':    {'base_demand': 120, 'price_range': (500, 5000),  'lead_time': (7, 14)},
        'Apparel':        {'base_demand': 200, 'price_range': (200, 2000),  'lead_time': (5, 10)},
        'Home & Kitchen': {'base_demand': 150, 'price_range': (300, 3000),  'lead_time': (4, 8)},
        'Automotive':     {'base_demand': 80,  'price_range': (800, 8000),  'lead_time': (10, 21)},
        'Sports':         {'base_demand': 100, 'price_range': (400, 4000),  'lead_time': (5, 12)},
        'FMCG':           {'base_demand': 300, 'price_range': (50,  500),   'lead_time': (2, 5)},
    }

    products = []
    pid = 1001
    for cat, cfg in categories.items():
        for i in range(1, 6):
            price = random.randint(*cfg['price_range'])
            products.append({
                'product_id':    f'P{pid}',
                'product_name':  f'{cat} Product {i}',
                'category':      cat,
                'unit_price':    price,
                'reorder_point': random.randint(50, 150),
                'reorder_qty':   random.randint(200, 500),
                'lead_time_days':random.randint(*cfg['lead_time']),
                'base_demand':   cfg['base_demand'],
            })
            pid += 1

    products_df = pd.DataFrame(products)

    # ---------- Sales records ----------
    sales_records = []
    for _, prod in products_df.iterrows():
        inventory = random.randint(300, 800)
        for date in date_range:
            # Seasonality
            month = date.month
            dow   = date.dayofweek
            season_factor = 1.0
            if prod['category'] == 'Electronics':
                season_factor = 1.5 if month in [11, 12] else (0.7 if month in [1, 2] else 1.0)
            elif prod['category'] == 'Apparel':
                season_factor = 1.4 if month in [3, 4, 10, 11] else (0.8 if month in [6, 7] else 1.0)
            elif prod['category'] == 'Sports':
                season_factor = 1.3 if month in [4, 5, 6] else (0.7 if month in [11, 12] else 1.0)
            elif prod['category'] == 'FMCG':
                season_factor = 1.2 if month in [10, 11, 12] else 1.0

            weekend_factor = 1.2 if dow >= 5 else 1.0
            trend_factor   = 1 + (date - start_date).days / (365 * 5)
            demand_noise   = np.random.normal(1.0, 0.15)
            demand = max(0, int(prod['base_demand'] / 30 * season_factor *
                                weekend_factor * trend_factor * demand_noise))

            actual_sales = min(demand, inventory)
            lost_sales   = max(0, demand - actual_sales)
            inventory   -= actual_sales

            # Reorder
            reorder_placed = False
            if inventory <= prod['reorder_point']:
                inventory    += prod['reorder_qty']
                reorder_placed= True

            holding_cost  = inventory * prod['unit_price'] * 0.0003
            stockout_flag = 1 if inventory <= 0 else 0

            sales_records.append({
                'date':           date,
                'product_id':     prod['product_id'],
                'product_name':   prod['product_name'],
                'category':       prod['category'],
                'units_demanded': demand,
                'units_sold':     actual_sales,
                'lost_sales':     lost_sales,
                'inventory_level':inventory,
                'unit_price':     prod['unit_price'],
                'revenue':        actual_sales * prod['unit_price'],
                'holding_cost':   round(holding_cost, 2),
                'reorder_placed': reorder_placed,
                'stockout_flag':  stockout_flag,
                'reorder_point':  prod['reorder_point'],
                'lead_time_days': prod['lead_time_days'],
            })

    sales_df = pd.DataFrame(sales_records)
    return sales_df, products_df


if __name__ == '__main__':
    print("Generating supply chain dataset...")
    sales_df, products_df = generate_supply_chain_data()
    out = os.path.join(os.path.dirname(__file__), '..', 'data')
    sales_df.to_csv(os.path.join(out, 'sales_inventory.csv'), index=False)
    products_df.to_csv(os.path.join(out, 'products.csv'), index=False)
    print(f"Sales records  : {len(sales_df):,}")
    print(f"Products       : {len(products_df)}")
    print("Saved to data/")

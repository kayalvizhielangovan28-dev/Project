"""
generate_data.py
Generates a realistic synthetic marketing campaign dataset.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

# ── Configuration ────────────────────────────────────────────────────────────
N_RECORDS = 5000

CAMPAIGNS = {
    "Email_Newsletter":      {"base_ctr": 0.22, "base_cvr": 0.08, "cpc": 0.10, "channel": "Email"},
    "Google_Search_Ads":     {"base_ctr": 0.35, "base_cvr": 0.12, "cpc": 1.80, "channel": "Search"},
    "Facebook_Social":       {"base_ctr": 0.18, "base_cvr": 0.06, "cpc": 0.90, "channel": "Social"},
    "Instagram_Stories":     {"base_ctr": 0.25, "base_cvr": 0.07, "cpc": 1.10, "channel": "Social"},
    "YouTube_Video_Ads":     {"base_ctr": 0.12, "base_cvr": 0.04, "cpc": 0.40, "channel": "Video"},
    "LinkedIn_B2B":          {"base_ctr": 0.14, "base_cvr": 0.09, "cpc": 3.50, "channel": "Social"},
    "Influencer_Marketing":  {"base_ctr": 0.30, "base_cvr": 0.05, "cpc": 2.00, "channel": "Influencer"},
    "Retargeting_Display":   {"base_ctr": 0.40, "base_cvr": 0.15, "cpc": 0.60, "channel": "Display"},
}

SEGMENTS   = ["Young Adults (18-24)", "Millennials (25-34)",
               "Gen X (35-44)", "Seniors (45+)"]
DEVICES    = ["Mobile", "Desktop", "Tablet"]
REGIONS    = ["North", "South", "East", "West", "Central"]
GENDERS    = ["Male", "Female", "Non-binary"]
START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2024, 12, 31)


def random_date(start, end):
    return start + timedelta(days=random.randint(0, (end - start).days))


# ── Build rows ───────────────────────────────────────────────────────────────
rows = []
for _ in range(N_RECORDS):
    campaign_name = random.choice(list(CAMPAIGNS))
    cfg           = CAMPAIGNS[campaign_name]

    impressions = int(np.random.lognormal(mean=9, sigma=1.2))
    impressions = max(500, min(impressions, 500_000))

    ctr_noise = np.random.normal(0, 0.05)
    ctr       = max(0.01, min(cfg["base_ctr"] + ctr_noise, 0.70))
    clicks    = int(impressions * ctr)

    cvr_noise   = np.random.normal(0, 0.03)
    cvr         = max(0.005, min(cfg["base_cvr"] + cvr_noise, 0.40))
    conversions = int(clicks * cvr)

    # spend / revenue
    spend       = round(clicks * cfg["cpc"] * np.random.uniform(0.85, 1.15), 2)
    avg_order   = np.random.uniform(50, 400)
    revenue     = round(conversions * avg_order, 2)
    roi         = round(((revenue - spend) / spend * 100) if spend > 0 else 0, 2)
    cpa         = round(spend / conversions if conversions > 0 else 0, 2)
    roas        = round(revenue / spend if spend > 0 else 0, 2)

    segment = random.choice(SEGMENTS)
    device  = random.choice(DEVICES)
    region  = random.choice(REGIONS)
    gender  = random.choice(GENDERS)
    date    = random_date(START_DATE, END_DATE)

    bounce_rate = round(np.random.uniform(0.20, 0.75), 2)
    time_site   = round(np.random.uniform(30, 600), 1)   # seconds
    pages_visit = round(np.random.uniform(1, 10), 1)

    rows.append({
        "campaign_id":        f"C{random.randint(1000, 9999)}",
        "campaign_name":      campaign_name,
        "channel":            cfg["channel"],
        "date":               date,
        "month":              date.strftime("%B"),
        "quarter":            f"Q{(date.month - 1) // 3 + 1}",
        "customer_segment":   segment,
        "device_type":        device,
        "region":             region,
        "gender":             gender,
        "impressions":        impressions,
        "clicks":             clicks,
        "conversions":        conversions,
        "spend":              spend,
        "revenue":            revenue,
        "ctr":                round(ctr * 100, 2),          # %
        "cvr":                round(cvr * 100, 2),          # %
        "roi":                roi,                           # %
        "cpa":                cpa,
        "roas":               roas,
        "bounce_rate":        round(bounce_rate * 100, 2),  # %
        "avg_time_on_site_s": time_site,
        "avg_pages_per_visit":pages_visit,
        "budget_allocated":   round(spend * np.random.uniform(1.0, 1.5), 2),
    })

df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
os.makedirs("data", exist_ok=True)
df.to_csv("data/marketing_campaigns.csv", index=False)
print(f"✅  Dataset saved: {len(df):,} rows × {len(df.columns)} columns")
print(df.head(3).to_string())

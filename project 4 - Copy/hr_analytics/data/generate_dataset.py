"""
HR Analytics Dataset Generator
Generates a realistic IBM-style HR dataset for attrition analysis
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 1470

departments = np.random.choice(['Sales', 'Research & Development', 'Human Resources'], n, p=[0.30, 0.60, 0.10])
job_roles = {
    'Sales': ['Sales Executive', 'Sales Representative', 'Manager'],
    'Research & Development': ['Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Research Director'],
    'Human Resources': ['Human Resources', 'Manager']
}

age = np.random.randint(18, 61, n)
gender = np.random.choice(['Male', 'Female'], n, p=[0.60, 0.40])
education = np.random.choice([1, 2, 3, 4, 5], n, p=[0.12, 0.19, 0.39, 0.22, 0.08])
education_field = np.random.choice(['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'], n, p=[0.41, 0.27, 0.15, 0.09, 0.04, 0.04])
marital_status = np.random.choice(['Single', 'Married', 'Divorced'], n, p=[0.32, 0.46, 0.22])

monthly_income_base = np.random.randint(1009, 20000, n)
job_level = np.random.choice([1, 2, 3, 4, 5], n, p=[0.26, 0.37, 0.21, 0.11, 0.05])
monthly_income = (monthly_income_base * (0.5 + job_level * 0.3)).astype(int).clip(1009, 19999)

p_yac = np.array([0.12] + [0.04]*10 + [0.02]*10 + [0.01]*10 + [0.005]*9 + [0.005])
p_yac = p_yac / p_yac.sum()
years_at_company = np.random.choice(range(0, 41), n, p=p_yac)
total_working_years = np.clip(years_at_company + np.random.randint(0, 20, n), 0, 40)
years_in_current_role = np.clip(np.random.randint(0, years_at_company + 1, n), 0, 18)
years_since_last_promotion = np.clip(np.random.randint(0, years_in_current_role + 1, n), 0, 15)
years_with_curr_manager = np.clip(np.random.randint(0, years_at_company + 1, n), 0, 17)

distance_from_home = np.random.choice(range(1, 30), n)
num_companies_worked = np.clip(np.random.randint(0, 10, n), 0, 9)
training_times_last_year = np.random.choice([0, 1, 2, 3, 4, 5, 6], n, p=[0.06, 0.16, 0.27, 0.26, 0.14, 0.08, 0.03])
percent_salary_hike = np.random.randint(11, 26, n)

job_satisfaction = np.random.choice([1, 2, 3, 4], n, p=[0.20, 0.20, 0.30, 0.30])
environment_satisfaction = np.random.choice([1, 2, 3, 4], n, p=[0.20, 0.25, 0.28, 0.27])
relationship_satisfaction = np.random.choice([1, 2, 3, 4], n, p=[0.16, 0.19, 0.34, 0.31])
work_life_balance = np.random.choice([1, 2, 3, 4], n, p=[0.05, 0.23, 0.61, 0.11])
job_involvement = np.random.choice([1, 2, 3, 4], n, p=[0.07, 0.20, 0.56, 0.17])
performance_rating = np.random.choice([3, 4], n, p=[0.85, 0.15])

overtime = np.random.choice(['Yes', 'No'], n, p=[0.29, 0.71])
business_travel = np.random.choice(['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'], n, p=[0.19, 0.71, 0.10])
stock_option_level = np.random.choice([0, 1, 2, 3], n, p=[0.47, 0.36, 0.12, 0.05])

# Attrition logic based on realistic risk factors
attrition_prob = 0.16 * np.ones(n)
attrition_prob += (overtime == 'Yes') * 0.12
attrition_prob += (job_satisfaction <= 2) * 0.10
attrition_prob += (environment_satisfaction <= 2) * 0.06
attrition_prob += (work_life_balance == 1) * 0.10
attrition_prob += (distance_from_home > 20) * 0.05
attrition_prob += (years_at_company < 3) * 0.08
attrition_prob += (stock_option_level == 0) * 0.04
attrition_prob += (business_travel == 'Travel_Frequently') * 0.06
attrition_prob += (marital_status == 'Single') * 0.04
attrition_prob -= (job_level >= 3) * 0.06
attrition_prob -= (years_at_company > 10) * 0.05
attrition_prob = np.clip(attrition_prob, 0.02, 0.85)
attrition_raw = np.random.binomial(1, attrition_prob, n)
attrition = np.where(attrition_raw == 1, 'Yes', 'No')

job_role_list = [np.random.choice(job_roles[dept]) for dept in departments]

df = pd.DataFrame({
    'Age': age,
    'Attrition': attrition,
    'BusinessTravel': business_travel,
    'DailyRate': np.random.randint(102, 1500, n),
    'Department': departments,
    'DistanceFromHome': distance_from_home,
    'Education': education,
    'EducationField': education_field,
    'EmployeeCount': 1,
    'EmployeeNumber': range(1, n + 1),
    'EnvironmentSatisfaction': environment_satisfaction,
    'Gender': gender,
    'HourlyRate': np.random.randint(30, 101, n),
    'JobInvolvement': job_involvement,
    'JobLevel': job_level,
    'JobRole': job_role_list,
    'JobSatisfaction': job_satisfaction,
    'MaritalStatus': marital_status,
    'MonthlyIncome': monthly_income,
    'MonthlyRate': np.random.randint(2094, 26000, n),
    'NumCompaniesWorked': num_companies_worked,
    'Over18': 'Y',
    'OverTime': overtime,
    'PercentSalaryHike': percent_salary_hike,
    'PerformanceRating': performance_rating,
    'RelationshipSatisfaction': relationship_satisfaction,
    'StandardHours': 80,
    'StockOptionLevel': stock_option_level,
    'TotalWorkingYears': total_working_years,
    'TrainingTimesLastYear': training_times_last_year,
    'WorkLifeBalance': work_life_balance,
    'YearsAtCompany': years_at_company,
    'YearsInCurrentRole': years_in_current_role,
    'YearsSinceLastPromotion': years_since_last_promotion,
    'YearsWithCurrManager': years_with_curr_manager
})

df.to_csv('/home/claude/hr_analytics/data/HR_Analytics.csv', index=False)
print(f"Dataset generated: {df.shape}")
print(f"Attrition rate: {df['Attrition'].value_counts(normalize=True)['Yes']*100:.1f}%")

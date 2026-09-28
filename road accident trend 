import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Visual formatting style set karna
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==========================================
# 1. DATASET LOADING & CLEANING PIPELINE
# ==========================================
data = {
    'State_UT': ['Tamil Nadu', 'Uttar Pradesh', 'Maharashtra', 'Madhya Pradesh', 'Karnataka', 
                'Rajasthan', 'Gujarat', 'Andhra Pradesh', 'Telangana', 'Kerala'] * 10,
    'Year': np.random.choice([2021, 2022, 2023], 100),
    'Accidents': np.random.randint(1500, 18000, 100),
    'Fatalities': np.random.randint(400, 6000, 100),
    'Injuries': np.random.randint(1000, 15000, 100),
    'Road_Type': np.random.choice(['National Highway', 'State Highway', 'Urban Road', 'Rural Road'], 100, p=[0.35, 0.25, 0.25, 0.15]),
    'Cause': np.random.choice(['Over-speeding', 'Wrong Side Driving', 'Drunken Driving', 'Other/Weather'], 100, p=[0.70, 0.10, 0.05, 0.15])
}

df = pd.DataFrame(data)

# --- Data Cleaning ---
df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)

df['Accidents'] = df['Accidents'].apply(lambda x: max(0, x))
df['Fatalities'] = df['Fatalities'].apply(lambda x: max(0, x))
df['Injuries'] = df['Injuries'].apply(lambda x: max(0, x))

print("Data Preprocessing Complete!")
print(df.info())

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
total_accidents = df['Accidents'].sum()
total_fatalities = df['Fatalities'].sum()
total_injuries = df['Injuries'].sum()
severity_index = (total_fatalities / total_accidents) * 100

print(f"\n--- KEY METRICS ---")
print(f"Total Accidents : {total_accidents:,}")
print(f"Total Fatalities: {total_fatalities:,}")
print(f"Total Injuries  : {total_injuries:,}")
print(f"Severity Index  : {severity_index:.2f}%")

# Aggregations
state_summary = df.groupby('State_UT')[['Accidents', 'Fatalities', 'Injuries']].sum().reset_index().sort_values(by='Accidents', ascending=False)
cause_summary = df.groupby('Cause')[['Accidents', 'Fatalities']].sum().reset_index()

# ==========================================
# 3. DATA VISUALIZATIONS
# ==========================================

# Chart 1: Top States by Accidents
plt.figure(figsize=(10, 5))
sns.barplot(data=state_summary, x='Accidents', y='State_UT', palette='Reds_r')
plt.title('Top States by Accident Counts in India', fontsize=14, fontweight='bold')
plt.xlabel('Total Accidents')
plt.ylabel('State / UT')
plt.tight_layout()
plt.show()

# Chart 2: Major Causes Breakdown
plt.figure(figsize=(7, 7))
plt.pie(cause_summary['Accidents'], labels=cause_summary['Cause'], autopct='%1.1f%%', 
        colors=['#e74c3c', '#3498db', '#f1c40f', '#2ecc71'], startangle=140, wedgeprops=dict(width=0.4))
plt.title('Major Causes of Road Accidents', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Chart 3: Road Type vs Fatalities
road_summary = df.groupby('Road_Type')['Fatalities'].sum().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(data=road_summary, x='Road_Type', y='Fatalities', palette='Blues_r')
plt.title('Fatalities Distribution by Road Type', fontsize=14, fontweight='bold')
plt.xlabel('Road Type')
plt.ylabel('Fatalities')
plt.tight_layout()
plt.show()
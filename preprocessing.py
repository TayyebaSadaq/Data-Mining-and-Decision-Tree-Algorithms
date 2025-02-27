# IMPORTS
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mstats

## LOAD DATASET
data = pd.read_csv('data/appointments.csv')
print(data.head())

## CHECK DATA INFORMATION
# data.info()

## CHECK FOR NULL VALUES
is_null = data.isnull().sum()
# print(is_null) # some null values - for now we leave them since they're there for unattended/cancelled appointments so they're needed
## check if attended appointment has null values - so we can drop them
# print(data[data['status'] == 'No'].isnull().sum())

## CONVERTING CATEGORICAL DATA TO NUMERICAL
## first handle the 'sex' column since it's just 1 and 0 (male and female)
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])
# print(data.head())
## hot encoding the 'status' column
data = pd.get_dummies(data, columns=['status'], drop_first=False)
print(data.head())

## HANDLING OUTLIERS

# Waiting Time
# Winsorize to cap extreme values at the 1st and 99th percentiles
data['waiting_time'] = mstats.winsorize(data['waiting_time'], limits=[0.01, 0.01])
# Categorize waiting times into bins
data['waiting_time_category'] = pd.cut(data['waiting_time'], bins=[-1, 5, 15, np.inf], labels=['Short', 'Moderate', 'Long'])

# Appointment Duration
data['unusually_short_appointment'] = np.where(data['appointment_duration'] < 5, 1, 0)
data['appointment_duration'] = np.where(data['appointment_duration'] == 0, data['appointment_duration'].median(), data['appointment_duration'])
data['appointment_duration'] = np.log1p(data['appointment_duration'])  # Apply log transformation

# Scheduling Interval
data['scheduling_interval'] = mstats.winsorize(data['scheduling_interval'], limits=[0.01, 0.01])
data['scheduling_interval_category'] = pd.cut(data['scheduling_interval'], bins=[-1, 0, 7, 30, np.inf], labels=['Same-day', '1 week', '1 month+', 'Long-term'])

print(data.head())

## VISUALISATIONS AFTER HANDLING OUTLIERS
## BOX PLOTS
fig, axs = plt.subplots(len(data.select_dtypes(include=[np.number]).columns), 1, dpi=95, figsize=(7, 17))
fig.suptitle('Boxplots for Outlier Detection After Handling', fontsize=16)
i = 0
for col in data.select_dtypes(include=[np.number]).columns:
    axs[i].boxplot(data[col].dropna(), vert=False)  # Drop NA values for plotting
    axs[i].set_ylabel(col)
    i += 1
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title
plt.show()

## CORRELATION MATRIX
numeric_data = data.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix', fontsize=16)
plt.show()
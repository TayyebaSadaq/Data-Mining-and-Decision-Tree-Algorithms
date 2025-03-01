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
print(is_null)

## CONVERTING CATEGORICAL DATA TO NUMERICAL
## first handle the 'sex' column since it's just 1 and 0 (male and female)
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])
print(data.head())

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

# CHECK FOR DUPLICATES
data = data.drop_duplicates()

# Check for remaining missing values
print(data.isnull().sum())

# Remove duplicate rows
data = data.drop_duplicates()

# Descriptive statistics
print(data.describe())

# Print column names to debug ValueError
print("Column names:", data.columns)

# Correlation matrix
numeric_data = data.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix', fontsize=16)
plt.show()

# Histograms
data.hist(bins=30, figsize=(15, 10))
plt.suptitle('Histograms of Numerical Columns', fontsize=16)
plt.show()

# Box plots for numerical data
numerical_columns = data.select_dtypes(include=[np.number]).columns
num_numerical = len(numerical_columns)
num_cols = 3  # Number of columns in the grid
num_rows = (num_numerical + num_cols - 1) // num_cols  # Calculate the number of rows needed

fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
fig.suptitle('Box Plots of Numerical Columns', fontsize=16)

for i, col in enumerate(numerical_columns):
    row = i // num_cols
    col_idx = i % num_cols
    sns.boxplot(x=data[col], ax=axs[row, col_idx])
    axs[row, col_idx].set_title(f'Box Plot of {col}')

# Remove any empty subplots
for j in range(i + 1, num_rows * num_cols):
    fig.delaxes(axs.flatten()[j])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Bar plots for categorical data
categorical_columns = ['sex', 'waiting_time_category', 'scheduling_interval_category', 'status_attended', 'status_cancelled', 'status_did not attend', 'status_scheduled', 'status_unknown']
num_categorical = len(categorical_columns)
num_cols = 2  # Number of columns in the grid
num_rows = (num_categorical + num_cols - 1) // num_cols  # Calculate the number of rows needed

fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
fig.suptitle('Bar Plots of Categorical Columns', fontsize=16)

for i, col in enumerate(categorical_columns):
    row = i // num_cols
    col_idx = i % num_cols
    sns.countplot(x=col, data=data, ax=axs[row, col_idx])
    axs[row, col_idx].set_title(f'Bar Plot of {col}')

# Remove any empty subplots
for j in range(i + 1, num_rows * num_cols):
    fig.delaxes(axs.flatten()[j])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Bar chart for appointment status
status_columns = ['status_attended', 'status_cancelled', 'status_did not attend', 'status_unknown']
status_counts = data[status_columns].sum()

plt.figure(figsize=(10, 6))
sns.barplot(x=status_counts.index, y=status_counts.values, palette=['green', 'orange', 'red', 'grey'])
plt.title('Appointment Status Distribution')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()


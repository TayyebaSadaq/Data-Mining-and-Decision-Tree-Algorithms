# IMPORTS
import pandas as pd
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

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

# Commented out the part that fills NaN values
# Fill NaN values for specific columns when the appointment was cancelled or not attended
# columns_to_fill = ['check_in_time', 'appointment_duration', 'start_time', 'end_time', 'waiting_time']
# data[columns_to_fill] = data[columns_to_fill].fillna(0)  # Fill with 0 or any other placeholder value

print(data.head())

## CONVERTING CATEGORICAL DATA TO NUMERICAL
## first handle the 'sex' column since it's just 1 and 0 (male and female)
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])
# print(data.head())
## hot encoding the 'status' column
data = pd.get_dummies(data, columns=['status'], drop_first=False)
print(data.head())

## HISTOGRAMS
data.hist(bins=30, figsize=(15, 10))
plt.suptitle('Histograms of Numerical Columns', fontsize=16)
plt.show()

## BOX PLOTS
fig, axs = plt.subplots(len(data.select_dtypes(include=[np.number]).columns), 1, dpi=95, figsize=(7, 17))
fig.suptitle('Boxplots for Outlier Detection', fontsize=16)
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
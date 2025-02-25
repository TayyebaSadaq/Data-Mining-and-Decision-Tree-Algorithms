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

# Fill NaN values for specific columns when the appointment was cancelled or not attended
columns_to_fill = ['check_in_time', 'appointment_duration', 'start_time', 'end_time', 'waiting_time']
data[columns_to_fill] = data[columns_to_fill].fillna(0)  # Fill with 0 or any other placeholder value

print(data.head())

## CONVERTING CATEGORICAL DATA TO NUMERICAL
## first handle the 'sex' column since it's just 1 and 0 (male and female)
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])
# print(data.head())
## hot encoding the 'status' column
data = pd.get_dummies(data, columns=['status'], drop_first=True)
print(data.head())

## Boxplot for outlier detection
# fig, axs = plt.subplots(len(data.select_dtypes(include=[np.number]).columns), 1, dpi=95, figsize=(7, 17))
# fig.suptitle('Boxplots for Outlier Detection', fontsize=16)
# i = 0
# for col in data.select_dtypes(include=[np.number]).columns:
#     axs[i].boxplot(data[col].dropna(), vert=False)  # Drop NA values for plotting
#     axs[i].set_ylabel(col)
#     i += 1
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title
# plt.show()

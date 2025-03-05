## import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import scipy as sp

## load data
data = pd.read_csv('data/diabetes.csv')

## inspect data
# print(data.head())
# data.info()
# print(data.describe())

## check for missing values
# print(data.isnull().sum())
## check for 0 values
# print((data == 0).sum())

## check for categorical variables
categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
# print("Categorical Columns: ", categorical_columns)
numerical_columns = [col for col in data.columns if data[col].dtype != 'object']
# print("Numerical Columns: ", numerical_columns)

## checking distribution - histograms and boxplots
columns_to_plot = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
# Histograms
plt.figure(figsize=(20, 15))
for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(4, 2, i)
    sns.histplot(data[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
# plt.show()
# Boxplots
plt.figure(figsize=(20, 15))
for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(4, 2, i)
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
plt.tight_layout()
# plt.show()
zero_values = (data == 0).sum()
# print("Count of 0 values in each column:")
# print(zero_values)

## handling missing values
# Replacing 0 values in skewed data with median
skewed_columns = ['Insulin', 'DiabetesPedigreeFunction', 'SkinThickness']
for col in skewed_columns:
    data[col] = data[col].replace(0, data[col].median())

# Replacing 0 values in normally distributed data with mean
normal_columns = ['BloodPressure', 'Glucose', 'BMI']
for col in normal_columns:
    data[col] = data[col].replace(0, data[col].mean())

# Final check - pregnancies and outcome can be 0 but the others are handled
zero_values = (data == 0).sum()
print("Count of 0 values in each column:")
print(zero_values)

## handling outliers for each feature
# Glucose: Cap values at 1st and 99th percentiles (Winsorization)
data["Glucose"] = data["Glucose"].clip(data["Glucose"].quantile(0.01), data["Glucose"].quantile(0.99))

# Blood Pressure: Replace 0 values with the median and cap extreme outliers
data["BloodPressure"] = data["BloodPressure"].replace(0, data["BloodPressure"].median())
data["BloodPressure"] = data["BloodPressure"].clip(data["BloodPressure"].quantile(0.01), data["BloodPressure"].quantile(0.99))

# Skin Thickness: Cap extreme values at the 99th percentile
data["SkinThickness"] = data["SkinThickness"].clip(data["SkinThickness"].quantile(0.01), data["SkinThickness"].quantile(0.99))

# Insulin: Use log transformation to reduce skewness
data["Insulin"] = np.log1p(data["Insulin"])

# BMI: Winsorization to cap extreme values
data["BMI"] = data["BMI"].clip(data["BMI"].quantile(0.01), data["BMI"].quantile(0.99))

# Diabetes Pedigree Function (DPF): Log transform to reduce the impact of large values
data["DiabetesPedigreeFunction"] = np.log1p(data["DiabetesPedigreeFunction"])

# Check for any more outliers for each feature
outliers = {}
for col in columns_to_plot:
    outliers[col] = detect_outliers_iqr(data, col)
    print(f"Outliers in {col}:")
    print(outliers[col])
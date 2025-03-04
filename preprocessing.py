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
print(data.head())
data.info()
print(data.describe())

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
plt.show()
# Boxplots
plt.figure(figsize=(20, 15))
for i, col in enumerate(columns_to_plot, 1):
    plt.subplot(4, 2, i)
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
plt.tight_layout()
plt.show()

## handling missing values
## Dropping rows where Glucose and BMI are 0 (since there's not many of them)
# data = data.drop(data[data['Glucose'] == 0].index)
# data = data.drop(data[data['BMI'] == 0].index)

# ## Replacing 0 values in BloodPressure, SkinThickness, and Insulin with their median
# data['BloodPressure'] = data['BloodPressure'].replace(0, data['BloodPressure'].median())
# data['SkinThickness'] = data['SkinThickness'].replace(0, data['SkinThickness'].median())
# data['Insulin'] = data['Insulin'].replace(0, data['Insulin'].median())

# ## Final check - pregnancies and outcome can be 0 but the others are handled
# print(data.isnull().sum())



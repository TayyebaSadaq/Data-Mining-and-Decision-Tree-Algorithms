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

## handling missing values
## Dropping rows where Glucose and BMI are 0 (since there's not many of them)
data = data.drop(data[data['Glucose'] == 0].index)
data = data.drop(data[data['BMI'] == 0].index)

## Replacing 0 values in BloodPressure, SkinThickness, and Insulin with their median
data['BloodPressure'] = data['BloodPressure'].replace(0, data['BloodPressure'].median())
data['SkinThickness'] = data['SkinThickness'].replace(0, data['SkinThickness'].median())
data['Insulin'] = data['Insulin'].replace(0, data['Insulin'].median())

## Final check - pregnancies and outcome can be 0 but the others are handled
print(data.isnull().sum())  

## REMOVING OUTLIERS
# check outliers for pregnancies, bmi, insulin, blood pressure
fig, axs = plt.subplots(8, 1, dpi = 95, figsize = (7,17))
i = 0
for col in ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']:
    axs[i].boxplot(data[col], vert = False)
    axs[i].set_ylabel(col)
    i += 1
plt.show()

# Function to remove outliers based on IQR
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

data = remove_outliers(data, ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
## visualize the data after removing outliers   
fig, axs = plt.subplots(8, 1, dpi = 95, figsize = (7,17))
i = 0
for col in ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']:
    axs[i].boxplot(data[col], vert = False)
    axs[i].set_ylabel(col)
    i += 1
plt.show()

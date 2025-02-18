# IMPORTS
import pandas as pd
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

## LOAD DATASET
data = pd.read_csv('data/appointments.csv')
# print(data.head())

## CHECK DATA INFORMATION
# data.info()

## CHECK FOR NULL VALUES
is_null = data.isnull().sum()
# print(is_null) # some null values

## STATISTICAL ANALYSIS
# mean, standard deviation, count, min, 25%, 50%, 75% and max values for each column
stat_analysis = data.describe()
# print(stat_analysis)

## CHECK OUTLIERS
fig, axs = plt.subplots(16,1,dpi=95, figsize=(7,17))
i = 0
for col in data.columns:
    axs[i].boxplot(data[col], vert=False)
    axs[i].set_ylabel(col)
    i+=1
plt.show
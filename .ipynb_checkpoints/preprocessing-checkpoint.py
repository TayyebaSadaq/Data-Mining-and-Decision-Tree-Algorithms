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
stat_analysis = data.describe()
print(stat_analysis)
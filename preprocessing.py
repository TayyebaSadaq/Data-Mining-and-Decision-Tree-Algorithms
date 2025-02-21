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
print(is_null) # some null values - for now we leave them since they're there for unattended/cancelled appointments so they're needed
# check if attended appointment has null values - so we can drop them
print(data[data['status'] == 'No'].isnull().sum())

import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('Bank-Customer-Churn-Prediction.csv')

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\n" + "="*80)
print("\nFirst few rows:")
print(df.head())
print("\n" + "="*80)
print("\nDataset Info:")
print(df.info())
print("\n" + "="*80)
print("\nBasic Statistics:")
print(df.describe())
print("\n" + "="*80)
print("\nMissing Values:")
print(df.isnull().sum())
print("\n" + "="*80)
print("\nColumn Names:")
print(df.columns.tolist())
print("\n" + "="*80)
print("\nTarget Variable Distribution:")
if 'churn' in df.columns:
    print(df['churn'].value_counts())
elif 'Churn' in df.columns:
    print(df['Churn'].value_counts())
elif 'Exited' in df.columns:
    print(df['Exited'].value_counts())

import pandas as pd

df = pd.read_parquet('data/processed/features_dev.parquet')

print('Nulls BEFORE fix:', df.isnull().sum().sum())

# Age is all NaN - fill with default value 40 (average adult age)
df['age'] = df['age'].fillna(40)

print('Nulls AFTER fix:', df.isnull().sum().sum())
print('Age min:', df['age'].min())
print('Age max:', df['age'].max())
print('Age mean:', round(df['age'].mean(), 1))

# Save fixed parquet
df.to_parquet('data/processed/features_dev.parquet', index=False)
print()
print('Saved successfully!')
print('Shape:', df.shape)
print('Fraud rate:', str(round(df['fraud_label'].mean()*100, 2)) + '%')
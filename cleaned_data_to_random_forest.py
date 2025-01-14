import pandas as pd
import numpy as np

# Load dataset
file_path = 'cleaned_job_posting_3.csv'
import io

# Open the file and decode manually while ignoring errors
with open('cleaned_job_posting_3.csv', 'rb') as f:
    content = f.read().decode('utf-8', errors='ignore')

# Use pandas to read from the cleaned content
df = pd.read_csv(io.StringIO(content))

# type of columns
text_columns = ['title', 'location', 'department', 'company_profile', 'description','requirements','benefits','employment_type','required_experience','required_education','industry','function']
numeric_columns = ['salary_range', 'telecommuting', 'has_company_logo', 'has_questions']

# Check for invalid entries (non-numeric values) in these columns
for col in numeric_columns:
    invalid_rows = df[~df[col].apply(lambda x: str(x).replace('.', '', 1).isdigit())]
    if not invalid_rows.empty:
        print(f"Invalid values detected in column '{col}':")
        print(invalid_rows)

# Replace non-numeric values with NaN for numeric columns
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calculate the percentage of missing values in each column
missing_percentage = df.isna().mean() * 100

# Identify columns with more than 50% missing values
columns_to_drop = missing_percentage[missing_percentage > 50].index
print(columns_to_drop)

# Drop columns with more than 50% missing values
df_cleaned = df.drop(columns=columns_to_drop)

# Now you can proceed with filling the remaining missing values in the cleaned dataset
from sklearn.impute import SimpleImputer

# Impute numerical columns with the mean
numeric_columns = ['telecommuting', 'has_company_logo', 'has_questions']  # Replace with your actual numerical columns
imputer = SimpleImputer(strategy='mean')
df_cleaned[numeric_columns] = imputer.fit_transform(df_cleaned[numeric_columns])

# Impute categorical columns with the most frequent value
categorical_columns = ['title', 'location', 'company_profile', 'description','requirements','benefits','employment_type','required_experience','required_education','industry','function']  # Replace with your actual categorical columns
imputer_cat = SimpleImputer(strategy='most_frequent')
df_cleaned[categorical_columns] = imputer_cat.fit_transform(df_cleaned[categorical_columns])

# Verify that missing values have been handled
print(df_cleaned.isna().sum())

# Save the cleaned DataFrame to a new CSV file
df.to_csv('cleaned_job_posting_4.csv', index=False)
print("Data cleaned and saved to 'cleaned_job_posting_4.csv'")
# 5th Q
import pandas as pd
import numpy as np

# Create the dataset
data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'Age': [21, 35, 26, 45, 67, np.nan, 32, 31, np.nan, 42, np.nan, 32, 35, 35],
    'Income': ['1L', '1,00,000', '45000', '', '10,000', '10000', '5$', '5 Dollars', '10,000', '15000', '25,000', '35000', '150000', '35000'],
    'Gender': ['Male', 'Male', 'Male', '', 'Female', 'Female', 'Female', 'Male', 'Female', 'Female', 'Female', 'Male', 'Female', 'Male'],
    'DoB': ['31.05.1992', '10-05-2002', 'Aug 5, 2000', '', '31.03.1986', '10/5/1987', '31.05.1992', '10-05-2002', 'Aug 5, 2000', 'Sep 12’2000', '31.03.1986', '10/5/1987', 'Sep 12’2000', '31.03.1986'],
    'Buys': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)

# Handle missing data and formatting issues
# Replace '1L' with '100000' and '5$'/'5 Dollars' with '5000'
df['Income'] = df['Income'].replace(['1L', '5$', '5 Dollars'], ['100000', '5000', '5000'])

# Convert 'Income' to numeric, forcing errors to NaN and then filling missing values with the median
df['Income'] = pd.to_numeric(df['Income'].str.replace(',', ''), errors='coerce')
df['Income'].fillna(df['Income'].median(), inplace=True)

# Fill missing 'Age' values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing 'Gender' values with the mode
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)

# Convert 'DoB' to datetime, forcing errors to NaT and then filling missing values with the mode
df['DoB'] = pd.to_datetime(df['DoB'], errors='coerce')
df['DoB'].fillna(df['DoB'].mode()[0], inplace=True)

print("\nCleaned Dataset:")
print(df)

# Apply five statistical measures
mean_age = df['Age'].mean()
median_income = df['Income'].median()
std_income = df['Income'].std()
count_gender = df['Gender'].value_counts()
mode_buys = df['Buys'].mode()[0]
print(mean_age)
print(median_income)
print(std_income)
print(count_gender)
print(mode_buys)
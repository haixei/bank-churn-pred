import pandas as pd
from sklearn.preprocessing import power_transform
from eda import data

# Explore the information about the data set
print(data.info())
print(data.shape)

# Null values by feature in percentage
print(round(data.isnull().mean() * 100, 3))

# Check the sanity
sanity = True
positive_cols = ['Balance', 'CreditScore', 'EstimatedSalary', 'Tenure', 'NumOfProducts']
binary_cols = ['Exited', 'Gender', 'HasCrCard', 'IsActiveMember']

# Loop trough the columns
for col in positive_cols:
    sanity = data[data[col] < 0].shape[0] == 0
    print('Column ' + col + ' is sane: ' + str(sanity))

for col in binary_cols:
    sanity = len(data[col].value_counts().values) == 2
    print('Column ' + col + ' is sane: ' + str(sanity))

# Remove outliers
def findThresholds(data, col, low_quantile=0.10, up_quantile=0.90):
    q_low = data[col].quantile(low_quantile)
    q_high = data[col].quantile(up_quantile)

    # Get the range
    range = q_high - q_low
    high_limit = q_high + 1.5 * range
    low_limit = q_low - 1.5 * range

    return low_limit, high_limit

# Are there any outliers in the variables
def has_outliers(data, numeric_columns):
    variable_names = []
    for col in numeric_columns:
        low_limit, high_limit = findThresholds(data, col)
        # Check every column but NumOfProducts, it has only distrete values
        if data[(data[col] > high_limit) | (data[col] < low_limit)].any(axis=None) and col != 'NumOfProducts':
            number_of_outliers = data[(data[col] > high_limit) | (data[col] < low_limit)].shape[0]
            if number_of_outliers > 10:
                variable_names.append(col)

    if len(variable_names) > 0:
        print(variable_names)
    else:
        print('No outliers found.')

# Get numeric columns names
num_col_names = data.select_dtypes(include=['int64', 'float64']).columns.values
has_outliers(data, num_col_names)


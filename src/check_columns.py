import pandas as pd

# Load the dataset
data_file = "data/cleaned_calls.csv"
data = pd.read_csv(data_file)

# Print the column names
print("\nColumn names in the dataset:")
print(data.columns)
print("\nFirst few rows of the dataset:")
print(data.head())

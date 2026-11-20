import pandas as pd

# Load the CSV
df = pd.read_csv("data.csv")

# Print the names of all columns
print("--- Column Names ---")
print(df.columns.tolist())

# Print the first 5 rows to see the data
print("\n--- First 5 Rows ---")
print(df.head())
import pandas as pd

# Read the CSV files
file1 = pd.read_csv('train.csv')
file2 = pd.read_csv('test.csv')
file3 = pd.read_csv('val.csv')

# Merge the files
merged_data = pd.concat([file1, file2, file3], axis=0)

# Save the merged data to a new CSV file
merged_data.to_csv('merged_file.csv', index=False)

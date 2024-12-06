#%%
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('../Data/provided_data.csv')

# Display the first 5 rows
print(df.head())

# Display basic information about the dataset
print(df.info())
#%%
# Calculate and print summary statistics
print(df.describe())
#%%
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df.iloc[:, 0], df.iloc[:, 1])
plt.xlabel('Frame Number')
plt.ylabel('Value')
plt.title('Second Column vs Frame Number')
plt.grid(True)
plt.savefig('plot.png')
plt.show()

# %%

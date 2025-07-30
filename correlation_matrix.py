import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 

FILE_NAME="train.tsv"
df = pd.read_csv(FILE_NAME, sep='\t')

df_processed = df.copy()
numerical_cols = ['Cena', 'Godina proizvodnje', 'Zapremina motora', 'Kilometraza', 'Konjske snage']

for col in numerical_cols:
    if df_processed[col].isnull().any():
        df_processed[col].fillna(df_processed[col].median(), inplace=True)

current_year = 2025
df_processed['Age'] = current_year - df_processed['Godina proizvodnje']
df_processed.drop('Godina proizvodnje', axis=1, inplace=True)
numerical_cols.remove('Godina proizvodnje')
numerical_cols.append('Age')

# Select only the numerical columns for the correlation matrix
# This includes 'Price' and the numerical features (like 'Car_Age', 'Kilometraza', etc.)
numerical_df = df_processed.select_dtypes(include=np.number)

# Calculate the correlation matrix
correlation_matrix = numerical_df.corr()

# Set up the matplotlib figure for better visualization
plt.figure(figsize=(10, 8)) # Adjust size as needed for readability

# Create a heatmap using seaborn
# annot=True displays the correlation values on the heatmap
# cmap='coolwarm' sets the color scheme (red for positive, blue for negative)
# fmt=".2f" formats the annotation values to two decimal places
# linewidths=.5 adds lines between cells for better separation
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

# Add a title to the plot
plt.title('Correlation Matrix of Numerical Features and Price', fontsize=16)

# Display the plot
plt.show()

print("\nCorrelation matrix values:")
print(correlation_matrix)
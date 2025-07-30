import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd # Ensure pandas is imported

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

print(df_processed["Age"].max())

# List of numerical columns to visualize
columns_to_visualize = ['Cena', 'Kilometraza', 'Zapremina motora', 'Konjske snage', 'Age']

# Set a style for the plots for better aesthetics
sns.set_style("whitegrid")

print("Generating Histograms and Box Plots for key numerical features...")

# Loop through each column and create a histogram and a box plot
for col in columns_to_visualize:
    if col in df_processed.columns:
        print(f"\n--- Visualizing '{col}' ---")

        # Create a figure with two subplots (one for histogram, one for box plot)
        plt.figure(figsize=(14, 6)) # Adjust figure size as needed

        # Subplot 1: Histogram
        plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
        sns.histplot(df_processed[col], kde=True, bins=50) # kde=True for density curve
        plt.title(f'Distribution of {col}', fontsize=14)
        plt.xlabel(col)
        plt.ylabel('Frequency')

        # Subplot 2: Box Plot
        plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
        sns.boxplot(y=df_processed[col])
        plt.title(f'Box Plot of {col}', fontsize=14)
        plt.ylabel(col)

        plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
        plt.show()
    else:
        print(f"Warning: Column '{col}' not found in df_processed. Skipping visualization for this column.")

print("\nVisualization complete. Analyze the plots for outliers and distribution shapes.")
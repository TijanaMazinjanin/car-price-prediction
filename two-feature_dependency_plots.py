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
df_processed['Car_Age'] = current_year - df_processed['Godina proizvodnje']
df_processed.drop('Godina proizvodnje', axis=1, inplace=True)
numerical_cols.remove('Godina proizvodnje')
numerical_cols.append('Car_Age')

# Set a style for the plots for better aesthetics
sns.set_style("whitegrid")

print("Generating Scatter Plots for key feature dependencies...")

# Plot 1: Cena vs. Kilometraza
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Kilometraza', y='Cena', data=df_processed, alpha=0.6)
plt.title('Cena vs. Kilometraza', fontsize=16)
plt.xlabel('Kilometraza')
plt.ylabel('Cena')
plt.xscale('log') # Often helpful for skewed features like mileage
plt.yscale('log') # Often helpful for skewed target like price
plt.grid(True, which="both", ls="--", c='0.7') # Add grid for log scales
plt.show()
print("Interpretation: Look for trends. Does price decrease as mileage increases? Are there clusters or outliers?")


# Plot 2: Cena vs. Car_Age
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Car_Age', y='Cena', data=df_processed, alpha=0.6)
plt.title('Cena vs. Car_Age', fontsize=16)
plt.xlabel('Car Age (Years)')
plt.ylabel('Cena')
# plt.xscale('log') # Age might not need log scale, depends on distribution
plt.yscale('log') # Price might benefit from log scale
plt.grid(True, which="both", ls="--", c='0.7')
plt.show()
print("Interpretation: Does price decrease as age increases? Are there very old, expensive cars (classics)?")


# Plot 3: Cena vs. Konjske snage
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Konjske snage', y='Cena', data=df_processed, alpha=0.6)
plt.title('Cena vs. Konjske snage', fontsize=16)
plt.xlabel('Konjske snage (Horsepower)')
plt.ylabel('Cena')
plt.xscale('log') # Horsepower might benefit from log scale
plt.yscale('log') # Price might benefit from log scale
plt.grid(True, which="both", ls="--", c='0.7')
plt.show()
print("Interpretation: Does price increase with horsepower? The weak negative correlation was surprising, look for non-linear patterns.")


# Plot 4: Zapremina motora vs. Konjske snage (to visualize multicollinearity)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Zapremina motora', y='Konjske snage', data=df_processed, alpha=0.6)
plt.title('Zapremina motora vs. Konjske snage (High Multicollinearity)', fontsize=16)
plt.xlabel('Zapremina motora')
plt.ylabel('Konjske snage')
plt.grid(True)
plt.show()
print("Interpretation: Expect to see a clear upward trend, confirming the strong positive correlation. This shows why they are multicollinear.")


print("\nScatter plot visualization complete. Analyze the plots for patterns, outliers, and non-linear relationships.")
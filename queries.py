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
# df_processed.drop('Godina proizvodnje', axis=1, inplace=True)
# numerical_cols.remove('Godina proizvodnje')
numerical_cols.append('Age')

print("\nNajstariji automobili\n")
print(df_processed['Age'].max())
youngest_cars = df_processed[ (df_processed['Age'] > 25) & (df_processed['Age'] < 51) ]
print(youngest_cars)

high_mileage_threshold = 60000 # Kilometraza > 100,000
high_price_threshold = 600000   # Cena > 600,000 (adjust this if 100,000 is more appropriate for "luxury")

# Query the DataFrame to find cars that meet both criteria
high_price_high_mileage_cars = df_processed[
    (df_processed['Kilometraza'] > high_mileage_threshold) &
    (df_processed['Cena'] > high_price_threshold)
]

print(f"--- Cars with Kilometraza > {high_mileage_threshold} and Cena > {high_price_threshold} ---")
if not high_price_high_mileage_cars.empty:
    print(f"Found {len(high_price_high_mileage_cars)} such cars.")
    print("\nDetails of these cars (Marka, Model, Cena, Kilometraza, Konjske snage, Age):")
    # Display relevant columns for inspection
    print(high_price_high_mileage_cars[['Marka', 'Karoserija', 'Cena', 'Kilometraza', 'Konjske snage', 'Age', 'Godina proizvodnje']].sort_values(by='Cena', ascending=False))
else:
    print("No cars found matching these criteria with the current thresholds.")

print("\n--- Summary of Brands and Types for these cars ---")
if not high_price_high_mileage_cars.empty:
    print("\nTop Brands:")
    print(high_price_high_mileage_cars['Marka'].value_counts())
    print("\nTop Types:")
    print(high_price_high_mileage_cars['Karoserija'].value_counts())
else:
    print("No summary available as no cars matched the criteria.")


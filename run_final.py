import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

FILE_NAME="train.tsv"

TARGET_COL = 'Cena'
YEAR_COL = 'Godina proizvodnje'
ENGINE_VOLUME_COL = 'Zapremina motora'
MILEAGE_COL = 'Kilometraza'
HP_COL = 'Konjske snage'
BRAND_COL = 'Marka'
TOWN_COL = 'Grad'
TYPE_COL = 'Karoserija'
FUEL_COL = 'Gorivo'
GEARBOX_COL = 'Menjac'

try:
    df = pd.read_csv(FILE_NAME, sep='\t')
except FileNotFoundError:
    exit()

df_processed = df.copy()

numerical_cols_initial = [
    col for col in df_processed.select_dtypes(include=np.number).columns.tolist()
    if col != TARGET_COL
]
categorical_cols_initial = df_processed.select_dtypes(include='object').columns.tolist()

for col in numerical_cols_initial:
    if df_processed[col].isnull().any():
        median_val = df_processed[col].median()
        df_processed[col].fillna(median_val, inplace=True)

for col in categorical_cols_initial:
    if df_processed[col].isnull().any():
        mode_val = df_processed[col].mode()[0]
        df_processed[col].fillna(mode_val, inplace=True)

current_year = 2025
df_processed['Car_Age'] = current_year - df_processed[YEAR_COL]

age_cap_threshold_upper = 20
age_cap_threshold_lower = 0
df_processed['Car_Age'] = np.clip(df_processed['Car_Age'], a_min=age_cap_threshold_lower, a_max=age_cap_threshold_upper)

cols_to_cap_iqr = [ENGINE_VOLUME_COL, MILEAGE_COL, HP_COL, TARGET_COL]

for col in cols_to_cap_iqr:
    if col in df_processed.columns:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        lower_bound = max(0, lower_bound)

        df_processed[col] = np.clip(df_processed[col], a_min=lower_bound, a_max=upper_bound)

df_processed[MILEAGE_COL] = np.clip(df_processed[MILEAGE_COL], a_min=df_processed[MILEAGE_COL].min(), a_max=700000)

df_processed['HP_per_EngineVolume'] = df_processed[HP_COL] / (df_processed[ENGINE_VOLUME_COL] + 1e-6)
df_processed['Mileage_per_Age'] = df_processed[MILEAGE_COL] / (df_processed['Car_Age'] + 1e-6)

df_processed.drop(YEAR_COL, axis=1, inplace=True)

# Z-score transformation (StandardScaler) for target and key numerical features
price_scaler = StandardScaler()
mileage_scaler = StandardScaler()
engine_volume_scaler = StandardScaler()
hp_scaler = StandardScaler()

df_processed[TARGET_COL] = price_scaler.fit_transform(df_processed[[TARGET_COL]])
df_processed[MILEAGE_COL] = mileage_scaler.fit_transform(df_processed[[MILEAGE_COL]])
df_processed[ENGINE_VOLUME_COL] = engine_volume_scaler.fit_transform(df_processed[[ENGINE_VOLUME_COL]])
df_processed[HP_COL] = hp_scaler.fit_transform(df_processed[[HP_COL]])

X = df_processed.drop([TARGET_COL, ENGINE_VOLUME_COL], axis=1)
y = df_processed[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

global_mean_y_train = y_train.mean()
town_encoding_map = y_train.groupby(X_train[TOWN_COL]).mean()

X_train[TOWN_COL] = X_train[TOWN_COL].map(town_encoding_map).fillna(global_mean_y_train)
X_test[TOWN_COL] = X_test[TOWN_COL].map(town_encoding_map).fillna(global_mean_y_train)

numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features_for_onehot = [col for col in X_train.select_dtypes(include='object').columns.tolist() if col != TOWN_COL]

numerical_transformer = StandardScaler()
one_hot_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', one_hot_transformer, categorical_features_for_onehot)
    ],
    remainder='passthrough'
)

model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', model)])

pipeline.fit(X_train, y_train)

y_pred_scaled = pipeline.predict(X_test)

y_pred = price_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_actual = price_scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()

mse = mean_squared_error(y_actual, y_pred)
rmse = np.sqrt(mse)

print(rmse)
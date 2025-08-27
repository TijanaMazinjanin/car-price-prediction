import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import joblib

FILE_NAME = "train.tsv"

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


def evaluate_model(model, X_test, y_test, price_scaler, model_name="Model"):
    y_pred_scaled = model.predict(X_test)

    # Inverse transform predictions and actuals to original scale
    y_pred = price_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = price_scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_pred)

    r2 = r2_score(y_actual, y_pred)

    print(f"\n--- {model_name} Performance ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared (R2): {r2:.4f}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_actual, y=y_pred, alpha=0.6)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
    plt.title(f'{model_name}: Actual vs. Predicted Prices')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.grid(True)
    plt.show()

    return {'rmse': rmse, 'mae': mae, 'r2': r2}


# -------------------
# Load dataset
# -------------------
try:
    df = pd.read_csv(FILE_NAME, sep='\t')
    print(f"Shape of the dataset: {df.shape}")
    print(df.head())
except FileNotFoundError:
    print("Error: train.tsv not found.")
    exit()

df_processed = df.copy()

# -------------------
# Missing values
# -------------------
for col in df_processed.select_dtypes(include=np.number).columns:
    if df_processed[col].isnull().any():
        df_processed[col].fillna(df_processed[col].median(), inplace=True)

for col in df_processed.select_dtypes(include='object').columns:
    if df_processed[col].isnull().any():
        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

# -------------------
# Feature engineering
# -------------------
current_year = 2025
df_processed['Car_Age'] = np.clip(current_year - df_processed[YEAR_COL], 0, 20)
df_processed['HP_per_EngineVolume'] = df_processed[HP_COL] / (df_processed[ENGINE_VOLUME_COL] + 1e-6)
df_processed['Mileage_per_Age'] = df_processed[MILEAGE_COL] / (df_processed['Car_Age'] + 1e-6)

# -------------------
# Outlier capping (IQR)
# -------------------
cols_to_cap = [ENGINE_VOLUME_COL, MILEAGE_COL, HP_COL, TARGET_COL]
for col in cols_to_cap:
    Q1 = df_processed[col].quantile(0.25)
    Q3 = df_processed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - 1.5 * IQR)
    upper_bound = Q3 + 1.5 * IQR
    df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)

df_processed[MILEAGE_COL] = np.clip(df_processed[MILEAGE_COL],
                                    a_min=df_processed[MILEAGE_COL].min(),
                                    a_max=700000)

# -------------------
# Scale target
# -------------------
price_scaler = StandardScaler()
df_processed[TARGET_COL] = price_scaler.fit_transform(df_processed[[TARGET_COL]])

# -------------------
# Train/test split
# -------------------
X = df_processed.drop([TARGET_COL, ENGINE_VOLUME_COL, YEAR_COL], axis=1)
y = df_processed[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------
# Manual target encoding for "Grad"
# -------------------
global_mean_y_train = y_train.mean()
town_encoding_map = y_train.groupby(X_train[TOWN_COL]).mean()

X_train[TOWN_COL] = X_train[TOWN_COL].map(town_encoding_map).fillna(global_mean_y_train)
X_test[TOWN_COL] = X_test[TOWN_COL].map(town_encoding_map).fillna(global_mean_y_train)

# -------------------
# Define features for ColumnTransformer
# -------------------
numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features_for_onehot = [col for col in X_train.select_dtypes(include='object').columns if col != TOWN_COL]

numerical_transformer = StandardScaler()
one_hot_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', one_hot_transformer, categorical_features_for_onehot)
    ],
    remainder='passthrough'
)

# -------------------
# Train and evaluate models
# -------------------
models = {
    "Lasso": Lasso(alpha=0.1, max_iter=2000),
    "Ridge": Ridge(alpha=1.0),
    "Elastic Net": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000),
    "Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    "SVR (RBF)": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    try:
        pipeline.fit(X_train, y_train)
        results[name] = evaluate_model(pipeline, X_test, y_test, price_scaler, name)
    except Exception as e:
        print(f"Error training {name}: {e}")
        results[name] = {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}

# -------------------
# Best model selection
# -------------------
best_model_name = min(
    (name for name in results if not np.isnan(results[name]['rmse'])),
    key=lambda n: results[n]['rmse'],
    default=None
)

if best_model_name:
    print(f"\nBest performing model: {best_model_name} (RMSE: {results[best_model_name]['rmse']:.2f})")
    X_full_copy = df_processed.drop([TARGET_COL, ENGINE_VOLUME_COL, YEAR_COL], axis=1).copy()
    y_full_copy = df_processed[TARGET_COL].copy()

    global_mean_y_full = y_full_copy.mean()
    full_town_encoding_map = y_full_copy.groupby(X_full_copy[TOWN_COL]).mean()
    X_full_copy[TOWN_COL] = X_full_copy[TOWN_COL].map(full_town_encoding_map).fillna(global_mean_y_full)

    numerical_features_final = X_full_copy.select_dtypes(include=np.number).columns.tolist()
    categorical_features_final_onehot = [col for col in X_full_copy.select_dtypes(include='object').columns if col != TOWN_COL]

    preprocessor_final = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features_final),
            ('cat', one_hot_transformer, categorical_features_final_onehot)
        ],
        remainder='passthrough'
    )

    best_pipeline_final = Pipeline(steps=[('preprocessor', preprocessor_final),
                                          ('regressor', models[best_model_name])])
    best_pipeline_final.fit(X_full_copy, y_full_copy)

    joblib.dump(best_pipeline_final, 'car_price_prediction_pipeline.pkl')
    joblib.dump({'town_encoding_map': full_town_encoding_map, 'global_mean': global_mean_y_full,
                 'price_scaler': price_scaler}, 'preprocessing_objects.pkl')

    print("Model and preprocessing objects saved.")
else:
    print("No valid model found.")

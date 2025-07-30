import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import joblib

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

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred_log = model.predict(X_test)

    # Inverse transform predictions and actuals to original scale for meaningful MAE/RMSE
    y_pred = np.expm1(y_pred_log)
    y_actual = np.expm1(y_test)

    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_pred)

    r2_log = r2_score(y_test, y_pred_log)

    print(f"\n--- {model_name} Performance ---")
    print(f"Root Mean Squared Error (RMSE) (Original Scale): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE) (Original Scale): {mae:.2f}")
    print(f"R-squared (R2) (Log Scale): {r2_log:.4f}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_actual, y=y_pred, alpha=0.6)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
    plt.title(f'{model_name}: Actual vs. Predicted Prices (Original Scale)')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    return {'rmse': rmse, 'mae': mae, 'r2_log': r2_log}


try:
    df = pd.read_csv(FILE_NAME, sep='\t')
    print(f"Shape of the dataset: {df.shape}")
    print(df.head())
except FileNotFoundError:
    print("Error: train.tsv not found. Please make sure the file is in the correct directory.")
    exit()

df_processed = df.copy()

numerical_cols_initial = [
    col for col in df_processed.select_dtypes(include=np.number).columns.tolist()
    if col != TARGET_COL
]
categorical_cols_initial = df_processed.select_dtypes(include='object').columns.tolist()

print("\nInitial Numerical Columns (excluding target):", numerical_cols_initial)
print("Initial Categorical Columns:", categorical_cols_initial)

# --- 4. Handle Missing Values ---
print("\n--- Handling Missing Values ---")
for col in numerical_cols_initial:
    if df_processed[col].isnull().any():
        median_val = df_processed[col].median()
        df_processed[col].fillna(median_val, inplace=True)
        print(f"Missing values in numerical column '{col}' imputed with median: {median_val}")

for col in categorical_cols_initial:
    if df_processed[col].isnull().any():
        mode_val = df_processed[col].mode()[0]
        df_processed[col].fillna(mode_val, inplace=True)
        print(f"Missing values in categorical column '{col}' imputed with mode: '{mode_val}'")

print("\nMissing values after imputation:")
print(df_processed.isnull().sum())

# --- 5. Feature Engineering: Car Age & Outlier Handling for Age ---
print("\n--- Feature Engineering & Age Outlier Handling ---")
current_year = 2025
df_processed['Car_Age'] = current_year - df_processed[YEAR_COL]

age_cap_threshold = 40
df_processed['Car_Age'] = np.where(df_processed['Car_Age'] > age_cap_threshold,
                                   age_cap_threshold,
                                   df_processed['Car_Age'])
print(f"Capped 'Car_Age' at {age_cap_threshold} years for values above this threshold.")
print(f"Min Car_Age: {df_processed['Car_Age'].min()}, Max Car_Age: {df_processed['Car_Age'].max()}")

df_processed['HP_per_EngineVolume'] = df_processed[HP_COL] / (df_processed[ENGINE_VOLUME_COL] + 1e-6)
df_processed['Mileage_per_Age'] = df_processed[MILEAGE_COL] / (df_processed['Car_Age'] + 1e-6)

df_processed.drop(YEAR_COL, axis=1, inplace=True)

# --- 6. Apply Log Transformations to Skewed Numerical Features and Target ---
print("\n--- Applying Log Transformations ---")
df_processed[TARGET_COL] = np.log1p(df_processed[TARGET_COL])
print(f"Applied np.log1p to target column '{TARGET_COL}'.")

df_processed[MILEAGE_COL] = np.log1p(df_processed[MILEAGE_COL])
print(f"Applied np.log1p to feature column '{MILEAGE_COL}'.")

# --- 7. Define Features (X) and Target (y) for Modeling ---
X = df_processed.drop(TARGET_COL, axis=1)
y = df_processed[TARGET_COL]

print(f"\nShape of X (features): {X.shape}")
print(f"Shape of y (target): {y.shape}")

# --- 8. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nShape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")


# --- MANUAL TARGET ENCODING FOR 'GRAD' ---
print(f"\n--- Applying Manual Target Encoding for '{TOWN_COL}' ---")

# Calculate global mean of the target from the training set
global_mean_y_train = y_train.mean()
print(f"Global mean of target (log-transformed) in training set: {global_mean_y_train:.4f}")

# Calculate target means for each 'Grad' category in the training set
# CORRECTED LINE: Group y_train by X_train[TOWN_COL] to get the mapping
town_encoding_map = y_train.groupby(X_train[TOWN_COL]).mean()

# Apply target encoding to X_train
X_train[TOWN_COL] = X_train[TOWN_COL].map(town_encoding_map).fillna(global_mean_y_train)
print(f"'{TOWN_COL}' in X_train encoded using training data means.")

# Apply target encoding to X_test using the mapping from the training set
# For unseen categories in X_test, use the global mean from y_train
X_test[TOWN_COL] = X_test[TOWN_COL].map(town_encoding_map).fillna(global_mean_y_train)
print(f"'{TOWN_COL}' in X_test encoded using training data means (unseen values filled with global mean).")

# Now 'Grad' is a numerical feature, so remove it from categorical_features
# and ensure it's in numerical_features for StandardScaler later.
numerical_features = X_train.select_dtypes(include=np.number).columns.tolist() # Re-select numerical features after encoding 'Grad'
# Filter out 'Grad' from categorical_features that will go into OneHotEncoder.
categorical_features_for_onehot = [col for col in X_train.select_dtypes(include='object').columns.tolist() if col != TOWN_COL]


# --- 9. Define Preprocessing Pipelines with ColumnTransformer ---
print(f"\nFeatures for ColumnTransformer (after manual Target Encoding for '{TOWN_COL}'):")
print(f"Numerical (StandardScaler): {numerical_features}")
print(f"Categorical (OneHotEncoder): {categorical_features_for_onehot}")

numerical_transformer = StandardScaler()
one_hot_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', one_hot_transformer, categorical_features_for_onehot)
    ],
    remainder='passthrough'
)

# --- 10. Model Training and Evaluation ---
print("\n--- Training and Evaluating Models ---")

models = {
    "Lasso (Coordinate Descent)": Lasso(alpha=0.1, max_iter=2000),
    "Lasso (SGD - GD)": SGDRegressor(loss='huber', penalty='l1', alpha=0.0001, max_iter=2000, random_state=42, early_stopping=True),
    "Ridge (Closed Form)": Ridge(alpha=1.0),
    "Ridge (SGD - GD)": SGDRegressor(loss='squared_error', penalty='l2', alpha=0.0001, max_iter=2000, random_state=42, early_stopping=True),
    "Elastic Net": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=100000),
    "Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    "Kernel Regression (SVR - RBF)": SVR(kernel='rbf', C=1.0, epsilon=0.1)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])

    try:
        pipeline.fit(X_train, y_train)
        results[name] = evaluate_model(pipeline, X_test, y_test, name)
    except Exception as e:
        print(f"Error training {name}: {e}")
        results[name] = {'rmse': np.nan, 'mae': np.nan, 'r2_log': np.nan}


print("\n--- Summary of Model Performance (RMSE and R2 Score) ---")
sorted_results = sorted(results.items(), key=lambda item: item[1]['rmse'] if not np.isnan(item[1]['rmse']) else float('inf'))

for model_name, metrics in sorted_results:
    if not np.isnan(metrics['rmse']):
        print(f"{model_name}:")
        print(f"  RMSE (Original Scale) = {metrics['rmse']:.2f}")
        print(f"  MAE (Original Scale) = {metrics['mae']:.2f}")
        print(f"  R2 (Log Scale) = {metrics['r2_log']:.4f}\n")
    else:
        print(f"{model_name}: Training failed or metrics not available.\n")

# --- 11. Saving the Best Model and Preprocessor ---
best_model_name = None
min_rmse = float('inf')

for name, metrics in results.items():
    if not np.isnan(metrics['rmse']) and metrics['rmse'] < min_rmse:
        min_rmse = metrics['rmse']
        best_model_name = name

if best_model_name:
    print(f"\nBest performing model based on RMSE: {best_model_name} (RMSE: {min_rmse:.2f})")

    # Retrain the best model's pipeline on the full dataset for deployment
    # This requires re-doing the target encoding for the full dataset.
    print(f"Retraining {best_model_name} on the full dataset with global encoding...")

    X_full_copy = df_processed.drop(TARGET_COL, axis=1).copy() # Get fresh X from df_processed
    y_full_copy = df_processed[TARGET_COL].copy() # Get fresh y from df_processed

    # Global mean for the entire dataset
    global_mean_y_full = y_full_copy.mean()
    full_town_encoding_map = y_full_copy.groupby(X_full_copy[TOWN_COL]).mean()

    # Apply encoding to the full dataset copy
    X_full_copy[TOWN_COL] = X_full_copy[TOWN_COL].map(full_town_encoding_map).fillna(global_mean_y_full)

    # Update numerical and categorical feature lists based on the full encoded data
    numerical_features_final = X_full_copy.select_dtypes(include=np.number).columns.tolist()
    categorical_features_final_onehot = [col for col in X_full_copy.select_dtypes(include='object').columns.tolist() if col != TOWN_COL]

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

    # Save the pipeline and the encoding map
    joblib.dump(best_pipeline_final, 'car_price_prediction_pipeline_manual_target_encoded_grad.pkl')
    joblib.dump({'town_encoding_map': full_town_encoding_map, 'global_mean': global_mean_y_full},
                'grad_target_encoding_map.pkl')

    print(f"Full pipeline (preprocessor + {best_model_name}) saved as 'car_price_prediction_pipeline_manual_target_encoded_grad.pkl'")
    print(f"Manual Target Encoding map for '{TOWN_COL}' saved as 'grad_target_encoding_map.pkl'")
else:
    print("\nNo best model found, likely due to all models failing or having NaN RMSE.")

# --- 12. Considerations for 'Grad' High Cardinality (Post-Analysis) ---
print("\n--- Note on 'Grad' (Town) High Cardinality ---")
num_unique_towns = df_processed[TOWN_COL].nunique()
print(f"Number of unique towns in '{TOWN_COL}': {num_unique_towns}")
print(f"'{TOWN_COL}' is now handled using a manual Target Encoding strategy. This prevents data leakage during training/testing.")
print("When deploying this model, you will need to load 'grad_target_encoding_map.pkl' and apply the same encoding transformation to new incoming data before passing it to the loaded pipeline.")
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


TARGET_COL = 'Cena'
YEAR_COL = 'Godina proizvodnje'
ENGINE_VOLUME_COL = 'Zapremina motora'
MILEAGE_COL = 'Kilometraza'
HP_COL = 'Konjske snage'
TOWN_COL = 'Grad'
BRAND_COL = 'Marka'
TYPE_COL = 'Karoserija'
FUEL_COL = 'Gorivo' 
GEARBOX_COL = 'Menjac' 

def load_data(file_path):
    
    try:
        df = pd.read_csv(file_path, sep='\t')
        return df
    except FileNotFoundError:
        sys.exit(1)

def preprocess_data(df, training=True, town_encoding_map=None, global_mean=None, price_scaler=None):
   
    df_processed = df.copy()
    
    
    for col in df_processed.select_dtypes(include=np.number).columns.tolist():
        if df_processed[col].isnull().any():
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)

    for col in df_processed.select_dtypes(include='object').columns.tolist():
        if df_processed[col].isnull().any():
            mode_val = df_processed[col].mode()[0]
            df_processed[col].fillna(mode_val, inplace=True)


    current_year = 2025
    df_processed['Car_Age'] = current_year - df_processed[YEAR_COL]
    df_processed['Car_Age'] = np.clip(df_processed['Car_Age'], 0, 20)
    df_processed['HP_per_EngineVolume'] = df_processed[HP_COL] / (df_processed[ENGINE_VOLUME_COL] + 1e-6)
    df_processed['Mileage_per_Age'] = df_processed[MILEAGE_COL] / (df_processed['Car_Age'] + 1e-6)

    cols_to_cap = [ENGINE_VOLUME_COL, MILEAGE_COL, HP_COL]
    if training:
        cols_to_cap.append(TARGET_COL)

    for col in cols_to_cap:
        if col in df_processed.columns:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = max(0, Q1 - 1.5 * IQR)
            upper_bound = Q3 + 1.5 * IQR
            df_processed[col] = np.clip(df_processed[col], lower_bound, upper_bound)
    
    df_processed[MILEAGE_COL] = np.clip(df_processed[MILEAGE_COL], a_min=df_processed[MILEAGE_COL].min(), a_max=700000)
    
    if training:
        price_scaler = StandardScaler()
        df_processed[TARGET_COL] = price_scaler.fit_transform(df_processed[[TARGET_COL]])
    else:
        df_processed[TARGET_COL] = price_scaler.transform(df_processed[[TARGET_COL]])

    if training:
        y_train_temp = df_processed[TARGET_COL]
        town_encoding_map = y_train_temp.groupby(df_processed[TOWN_COL]).mean()
        global_mean = y_train_temp.mean()
        df_processed[TOWN_COL] = df_processed[TOWN_COL].map(town_encoding_map).fillna(global_mean)
    else:
        df_processed[TOWN_COL] = df_processed[TOWN_COL].map(town_encoding_map).fillna(global_mean)

    
    df_processed.drop([YEAR_COL, ENGINE_VOLUME_COL], axis=1, inplace=True)
    
   
    X = df_processed.drop(TARGET_COL, axis=1)
    y = df_processed[TARGET_COL]

    if training:
        return X, y, price_scaler, town_encoding_map, global_mean
    else:
        return X, y

def train(X_train, y_train):
    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features_for_onehot = [col for col in X_train.select_dtypes(include='object').columns.tolist() if col != TOWN_COL]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_for_onehot)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', SVR(kernel='rbf'))])

    param_grid = {
        'regressor__C': [0.1, 1, 10, 100],
        'regressor__epsilon': [0.01, 0.1, 0.5, 1.0]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, price_scaler):
    y_pred_scaled = model.predict(X_test)
    
    y_pred = price_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = price_scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)

    print(rmse)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(1)

    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]

    df_train = load_data(train_file_path)
    df_test = load_data(test_file_path)

    X_train, y_train, price_scaler, town_encoding_map, global_mean = preprocess_data(df_train, training=True)
    X_test, y_test = preprocess_data(df_test, training=False, 
                                     town_encoding_map=town_encoding_map, 
                                     global_mean=global_mean, 
                                     price_scaler=price_scaler)
    
    model = train(X_train, y_train)

    evaluate_model(model, X_test, y_test, price_scaler)
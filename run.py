import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

FILE_NAME="train.tsv"

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"\n--- {model_name} Performance ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    return rmse

def draw_correlation_matrix(df_processed):

    numerical_df = df_processed.select_dtypes(include=np.number)

    correlation_matrix = numerical_df.corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

    plt.title('Correlation Matrix of Numerical Features and Price', fontsize=16)

    plt.show()

    print("\nCorrelation matrix values:")
    print(correlation_matrix)

def draw(df):
    pass

try:
    df = pd.read_csv(FILE_NAME, sep='\t')
    print(f"Shape of the dataset: {df.shape}")
    print(df.head())

    # data processing

    df_processed = df.copy()

    # numerical_cols = ['Cena', 'Godina proizvodnje', 'Zapremina motora', 'Kilometraza', 'Konjske snage']
    # categorical_cols = ['Marka', 'Grad', 'Karoserija', 'Gorivo', 'Menjac']
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    try:
        for col in numerical_cols:
            if df_processed[col].isnull().any():
                df_processed[col].fillna(df_processed[col].median(), inplace=True)

        for col in categorical_cols:
            if df_processed[col].isnull().any():
                #value that appears most often
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        
        current_year = 2025
        df_processed['Age'] = current_year - df_processed['Godina proizvodnje']
        df_processed.drop('Godina proizvodnje', axis=1, inplace=True)
        numerical_cols.remove('Godina proizvodnje')
        numerical_cols.append('Age')


        num_unique_brands = df_processed['Grad'].nunique()
        print(num_unique_brands)

        encoder = OneHotEncoder(sparse_output=False)

        encoded = encoder.fit_transform(df_processed[categorical_cols])

        one_hot_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

        df_encoded = pd.concat([df, one_hot_df], axis=1)

        df_encoded = df_encoded.drop(categorical_cols, axis=1)

        print(df_encoded.head())

        X = df_processed.drop('Cena', axis=1)
        y = df_processed['Cena']

        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include='object').columns.tolist()

        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
        preprocessor = ColumnTransformer(
            transformers=[
                ('numerical', numerical_transformer, numerical_features),
                ('categorical', categorical_transformer, categorical_features)
            ])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"\nShape of X_train: {X_train.shape}")
        print(f"Shape of X_test: {X_test.shape}")
        print(f"Shape of y_train: {y_train.shape}")
        print(f"Shape of y_test: {y_test.shape}")

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        linear_model = LinearRegression()
        linear_model.fit(X_train_processed, y_train)
        linear_metrics = evaluate_model(linear_model, X_test_processed, y_test, "Linear Regression")

        print("\n--- Training Ridge Regression ---")
        ridge_model = Ridge(alpha=1.0) # You can tune alpha
        ridge_model.fit(X_train_processed, y_train)
        ridge_metrics = evaluate_model(ridge_model, X_test_processed, y_test, "Ridge Regression")

        print("\n--- Training Lasso Regression ---")
        lasso_model = Lasso(alpha=0.1) # You can tune alpha
        lasso_model.fit(X_train_processed, y_train)
        lasso_metrics = evaluate_model(lasso_model, X_test_processed, y_test, "Lasso Regression")


        print("\n--- Training Random Forest Regressor ---")
        # RandomForest generally performs well. Consider tuning n_estimators and max_depth.
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_processed, y_train)
        rf_metrics = evaluate_model(rf_model, X_test_processed, y_test, "Random Forest Regressor")




    except Exception as e:
        print(e)

except:
    print("Error: train.tsv not found. Please make sure the file is in the correct directory.")
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TOP_COUNTRIES = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']

def group_countries(country: str) -> str:
    if country in TOP_COUNTRIES:
        return country
    return 'Other'

def train_and_evaluate(data_path: str = "backend/data/dataset.csv", model_dir: str = "backend/model"):
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    # Feature engineering: Country grouping
    df['Grouped_country'] = df['Country'].apply(group_countries)

    # Column categorisation
    skewed_col = ['Study_Hours']
    other_numeric_cols = [
        "Age",
        "Avg_Daily_Usage_Hours",
        "Daily_Unlocks",
        "Physical_Activity_Hours",
        "Sleep_Hours_Per_Night"
    ]
    ordinal_col = ['Stress_Level']
    normal_col = [
        "Gender",
        "Academic_Level",
        "Most_Used_Platform",
        "Purpose_Of_Use",
        "Grouped_country"
    ]

    feature_cols = skewed_col + other_numeric_cols + ordinal_col + normal_col
    target_col = 'Mental_Health_Score'

    X = df[feature_cols]
    y = df[target_col]

    # Train / Test split (70% train, 30% test with fixed random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    print(f"Training set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows")

    # Preprocessing pipelines
    skew_pipeline = Pipeline(steps=[
        ('log_transform', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('scale', StandardScaler())
    ])

    plain_numeric_pipeline = Pipeline(steps=[
        ('scale', StandardScaler())
    ])

    ordinal_pipeline = Pipeline(steps=[
        ('encode', OrdinalEncoder(categories=[['Low', 'Medium', 'High', 'Very High']]))
    ])

    nominal_pipeline = Pipeline(steps=[
        ('encode', OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("Skewed_Pipeline", skew_pipeline, skewed_col),
        ("Plain_Numeric", plain_numeric_pipeline, other_numeric_cols),
        ("Ordinal", ordinal_pipeline, ordinal_col),
        ("Nominal", nominal_pipeline, normal_col)
    ])

    # 1. Baseline: Linear Regression
    print("\n--- Training Linear Regression Baseline ---")
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)

    lr_train_preds = lr_pipeline.predict(X_train)
    lr_test_preds = lr_pipeline.predict(X_test)

    lr_metrics = {
        "train_r2": float(r2_score(y_train, lr_train_preds)),
        "test_r2": float(r2_score(y_test, lr_test_preds)),
        "train_mae": float(mean_absolute_error(y_train, lr_train_preds)),
        "test_mae": float(mean_absolute_error(y_test, lr_test_preds)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, lr_train_preds))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, lr_test_preds)))
    }
    print("Linear Regression Results:", lr_metrics)

    # 2. Random Forest Regressor (Default)
    print("\n--- Training Random Forest Regressor (Default) ---")
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    rf_pipeline.fit(X_train, y_train)

    rf_train_preds = rf_pipeline.predict(X_train)
    rf_test_preds = rf_pipeline.predict(X_test)

    rf_metrics = {
        "train_r2": float(r2_score(y_train, rf_train_preds)),
        "test_r2": float(r2_score(y_test, rf_test_preds)),
        "train_mae": float(mean_absolute_error(y_train, rf_train_preds)),
        "test_mae": float(mean_absolute_error(y_test, rf_test_preds)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, rf_train_preds))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, rf_test_preds)))
    }
    print("Random Forest (Default) Results:", rf_metrics)

    # 3. Hyperparameter Tuning for Random Forest
    print("\n--- Hyperparameter Tuning Random Forest (RandomizedSearchCV) ---")
    param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [5, 10, 15, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }

    random_search = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=param_grid,
        n_iter=15,
        cv=5,
        scoring='r2',
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    best_pipeline = random_search.best_estimator_
    best_train_preds = best_pipeline.predict(X_train)
    best_test_preds = best_pipeline.predict(X_test)

    tuned_metrics = {
        "best_params": random_search.best_params_,
        "train_r2": float(r2_score(y_train, best_train_preds)),
        "test_r2": float(r2_score(y_test, best_test_preds)),
        "train_mae": float(mean_absolute_error(y_train, best_train_preds)),
        "test_mae": float(mean_absolute_error(y_test, best_test_preds)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, best_train_preds))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, best_test_preds)))
    }
    print("Tuned Random Forest Results:", tuned_metrics)

    # Determine best model based on test R2
    models_evaluated = {
        "Linear Regression": lr_metrics,
        "Random Forest (Default)": rf_metrics,
        "Random Forest (Tuned)": tuned_metrics
    }

    # Save best pipeline
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, "mental_health_pipeline.pkl")
    joblib.dump(best_pipeline, model_save_path)
    print(f"\nSaved best pipeline to: {model_save_path}")

    # Also save metadata and evaluation metrics to JSON
    summary = {
        "dataset_shape": df.shape,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_columns": feature_cols,
        "target_column": target_col,
        "top_countries": TOP_COUNTRIES,
        "models": models_evaluated,
        "selected_model": "Random Forest (Tuned)",
        "test_metrics": {
            "MAE": round(tuned_metrics["test_mae"], 4),
            "RMSE": round(tuned_metrics["test_rmse"], 4),
            "R2": round(tuned_metrics["test_r2"], 4)
        }
    }

    metrics_save_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_save_path}")

    # Test an inference sample to ensure it works end-to-end
    sample_row = pd.DataFrame([{
        'Study_Hours': 4.0,
        'Age': 21,
        'Avg_Daily_Usage_Hours': 5.5,
        'Daily_Unlocks': 60,
        'Physical_Activity_Hours': 1.0,
        'Sleep_Hours_Per_Night': 7.0,
        'Stress_Level': 'Medium',
        'Gender': 'Male',
        'Academic_Level': 'Undergraduate',
        'Most_Used_Platform': 'Instagram',
        'Purpose_Of_Use': 'Entertainment',
        'Grouped_country': 'India'
    }])
    sample_pred = best_pipeline.predict(sample_row)[0]
    print(f"Sample test prediction for Indian 21yo male: {sample_pred:.2f}")

    return summary

if __name__ == "__main__":
    train_and_evaluate()

# %%
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import BayesianRidge
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

import mlflow 
import mlflow.sklearn, mlflow.xgboost
import os
import pickle

# Load the data
df = pd.read_csv('../data/train.csv')
df['SalePrice_log'] = np.log1p(df['SalePrice'])

# Identify numeric columns and remove 'SalePrice'
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'SalePrice' in numeric_columns:
    numeric_columns.remove('SalePrice')
    numeric_columns.remove('SalePrice_log')

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Build a pipeline for numeric features: fill missing values with mean and scale.
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Build a pipeline for categorical features: fill missing values with mode and convert to dummy variables.
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

# Combine both pipelines using ColumnTransformer.
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_columns),
    ('cat', categorical_pipeline, categorical_columns)
])

# --- Initial model pipeline with Bayesian Ridge (for reference) ---
initial_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('br', BayesianRidge())
])

# Define features (X) and target (y)
X = df.drop(['SalePrice','SalePrice_log'], axis=1)
y = df['SalePrice_log']

# Split data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the initial pipeline.
initial_pipeline.fit(X_train, y_train)

# --- Feature selection using SelectFromModel ---
# First, transform the training data using the preprocessor.
X_train_trans = preprocessor.transform(X_train)

# Use a Bayesian Ridge model with a threshold based on the median absolute coefficient.
selector = SelectFromModel(BayesianRidge(), threshold='median')
selector.fit(X_train_trans, y_train)

# Get the indices of the selected features.
selected_indices = selector.get_support(indices=True)

#Encoding categorical feature names and combining with numeric feature names
cat_ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_ohe.get_feature_names_out(categorical_columns)

# Combine numeric and categorical feature names.
all_feature_names = numeric_columns + list(cat_feature_names)
selected_feature_names = [all_feature_names[i] for i in selected_indices]
print("Selected features:")
print(selected_feature_names)

# --- New pipeline that includes feature selection ---
new_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectFromModel(BayesianRidge(), threshold='median')),
    ('br', BayesianRidge())
])

print("Pipelines created")
# Define the parameter grid
param_grid = {
    # Tuning hyperparameters for the base estimators:
    'br__max_iter': [100, 300, 500],
    'br__alpha_1': [1e-6, 1e-5, 1e-4],
    'br__alpha_2': [1e-6, 1e-5, 1e-4],
    'br__lambda_1': [1e-6, 1e-5, 1e-4],
    'br__lambda_2': [1e-6, 1e-5, 1e-4],
}
# Enable autolog before model training
mlflow.sklearn.autolog(
    log_models=True,
    log_input_examples=True,
    log_model_signatures=True
)


grid_search = GridSearchCV(
    estimator=new_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring=['neg_mean_squared_error', 'r2'],
    refit='neg_mean_squared_error',  # Specify which metric to use for best model
    n_jobs=-1,
    verbose=2,
    return_train_score=True  # Added to get training scores
)

print("GridSearchCV created")
# Create model directory
MODEL_DIR = "../saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)
# print(os.getcwd())

# Train and log with MLflow

mlflow.set_experiment("dsia/salesPrice")
print("Experiment set")
with mlflow.start_run(run_name="BayesianRidge_Regressor-GridSearchCV"):
    mlflow.sklearn.autolog(log_models=True, log_input_examples=True, log_model_signatures=True)

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Log best parameters explicitly
    mlflow.log_params(grid_search.best_params_)
    
    # Predictions and Metrics
    y_pred = best_model.predict(X_test)
    # Convert back to original scale
    y_test_orig = np.expm1(y_test)  
    y_pred_orig = np.expm1(y_pred)

    mse = mean_squared_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_orig, y_pred_orig)

    # Explicit logging of main metrics
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2_Score", r2)

    # Save and log model
    pickle_path = os.path.join(MODEL_DIR, "BR_regressor.pkl")
    with open(pickle_path, 'wb') as file:
        pickle.dump(best_model, file)
    
    mlflow.log_artifact(pickle_path)
    mlflow.sklearn.log_model(best_model, "model")

    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"RMSE: {rmse:.2f}, R2 Score: {r2:.4f}")
    # Log additional artifacts
    mlflow.log_artifact(pickle_path)
    

# Display plots
# plt.show()
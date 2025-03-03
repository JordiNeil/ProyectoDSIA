# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

import mlflow 
import mlflow.sklearn, mlflow.xgboost
import os
import pickle

# %%
# Your selected columns
selected_num_columns = ['LotFrontage', 'LotArea', 'TotalBsmtSF', 'GrLivArea', 'SalePrice']
categorical_columns_filtered = ['Street', 'LandContour', 'LandSlope', 'Utilities', 'Neighborhood',
                                  'Condition1', 'Condition2', 'HouseStyle', 'BldgType', 'OverallQual',
                                  'OverallCond', 'RoofStyle', 'Exterior1st', 'ExterCond', 'BsmtCond',
                                  'BsmtFinType1', 'CentralAir', 'Heating', 'KitchenQual',
                                  'TotRmsAbvGrd', 'GarageType', 'GarageCond', 'PavedDrive',
                                  'SaleType', 'SaleCondition', 'Fireplaces', 'GarageCars']

categorical_columns_filtered = [ 'LandSlope',  'Neighborhood',
                                  'Condition1',  'HouseStyle', 'BldgType', 'OverallQual',
                                  'OverallCond', 'RoofStyle', 'Exterior1st', 'ExterCond', 'BsmtCond',
                                  'BsmtFinType1', 'KitchenQual',
                                  'TotRmsAbvGrd', 'GarageType',  'PavedDrive',
                                  'SaleType', 'SaleCondition', 'Fireplaces', 'GarageCars']

# %%
# Load the data
df = pd.read_csv('../data/train.csv')

numeric_features = [col for col in selected_num_columns if col != 'SalePrice']
X = df[[col for col in selected_num_columns if col != 'SalePrice'] + categorical_columns_filtered]
y = df['SalePrice']

print("Dataset read. Shape:", df.shape)
# %%
# Convert categorical variables to dummy variables
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformations using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_columns_filtered)
    ]
)

# %%
gbr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('gbr', GradientBoostingRegressor(random_state=42))
])

print("Pipelines created")
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'gbr__n_estimators': [100, 200, 300],
    'gbr__learning_rate': [0.01, 0.05, 0.1],
    'gbr__max_depth': [3, 5, 7]
}
# Enable autolog before model training
mlflow.sklearn.autolog(
    log_models=True,
    log_input_examples=True,
    log_model_signatures=True
)


grid_search = GridSearchCV(
    estimator=gbr_pipeline,
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
with mlflow.start_run(run_name="GradientBoostingRegressor-GridSearchCV"):
    mlflow.sklearn.autolog(log_models=True, log_input_examples=True, log_model_signatures=True)

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Log best parameters explicitly
    mlflow.log_params(grid_search.best_params_)
    
    # Predictions and Metrics
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Explicit logging of main metrics
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2_Score", r2)

    # Save and log model
    pickle_path = os.path.join(MODEL_DIR, "gbr.pkl")
    with open(pickle_path, 'wb') as file:
        pickle.dump(best_model, file)
    
    mlflow.log_artifact(pickle_path)

    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"RMSE: {rmse:.2f}, R2 Score: {r2:.4f}")
    
    # Create and save feature importance plot
    
    # Log additional artifacts
    mlflow.log_artifact(pickle_path)
    

# Display plots
# plt.show()
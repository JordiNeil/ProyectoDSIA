# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow 
import mlflow.sklearn
import os
import pickle

# %%
selected_num_columns = ['LotFrontage','LotArea','TotalBsmtSF','GrLivArea','SalePrice']
##Se selecciona porque garage cars porque tiene mejor correlacion con saleprice y ambas estaban muy correlacionadas
categorical_columns_filtered =  ['Street','LandContour',
                    'LandSlope','Utilities','Neighborhood','Condition1',
                    'Condition2','HouseStyle','BldgType','OverallQual',
                    'OverallCond','RoofStyle','Exterior1st',
                    'ExterCond', 'BsmtCond','BsmtFinType1','CentralAir',
                    'Heating','KitchenQual','TotRmsAbvGrd', 'GarageType',
                    'GarageCond','PavedDrive',
                    'SaleType','SaleCondition','Fireplaces',
                    'GarageCars',
                    ]

# %%
# Load the data
df = pd.read_csv('../data/train.csv')

df = df[selected_num_columns + categorical_columns_filtered]

# %%
# Basic data exploration
print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])

# %%
# Handle missing values
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)

categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)


# %%
# Convert categorical variables to dummy variables
df_encoded = pd.get_dummies(df, drop_first=True)

# %%
print([column for column in df_encoded.columns])
print(len(df_encoded.columns))

# %%
# Split the data
X = df_encoded.drop(['SalePrice'], axis=1)
y = df_encoded['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Train Random Forest model
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train_scaled, y_train)

# %%
# len(rf_model.estimators_)

# Enable autolog before model training
mlflow.sklearn.autolog(
    log_models=True,
    log_input_examples=True,
    log_model_signatures=True
)

# Create model directory
MODEL_DIR = "../saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)
# print(os.getcwd())

# Train and log with MLflow
run_name = "random-forest-complete"
mlflow.set_experiment("dsia/salesPrice/JordiModelsRF")
with mlflow.start_run(run_name=run_name):
    # Train the model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions      
    y_pred = rf_model.predict(X_test_scaled)
    
    # Save model with pickle
    pickle_path = os.path.join(MODEL_DIR, "random_forest_model.pkl")
    with open(pickle_path, 'wb') as file:
        pickle.dump(rf_model, file)
    
    # Create and save feature importance plot
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    feature_imp_path = os.path.join(MODEL_DIR, "feature_importance.png")
    plt.savefig(feature_imp_path)
    
    # Create and save predictions plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Sale Price')
    plt.ylabel('Predicted Sale Price')
    plt.title('Actual vs Predicted Sale Prices')
    plt.tight_layout()
    predictions_path = os.path.join(MODEL_DIR, "predictions.png")
    plt.savefig(predictions_path)
    
    # Log additional artifacts
    mlflow.log_artifact(pickle_path)
    mlflow.log_artifact(feature_imp_path)
    mlflow.log_artifact(predictions_path)
    
    # Print metrics (these are automatically logged by autolog)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print("\nModel Performance:")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R2 Score: {r2:.4f}")

# Display plots
# plt.show()
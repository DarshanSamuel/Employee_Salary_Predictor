import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib

# Load dataset
df = pd.read_csv("salary_data.csv")
df_clean = df.dropna().copy()

# Feature Engineering
df_clean["Age_Exp_Interaction"] = df_clean["Age"] * df_clean["Years of Experience"]
df_clean["Experience_Level"] = pd.cut(df_clean["Years of Experience"],
                                      bins=[0, 5, 10, 20, 100],
                                      labels=["Junior", "Mid", "Senior", "Expert"])
df_clean["Predicted Salary"] = df_clean["Current Salary"] * (1.1 ** df_clean["Years of Experience"])

# Target and features
target = "Predicted Salary"
y = df_clean[target].values
X = df_clean.drop(columns=[target])

# Feature groups
categorical_features = ["Gender", "Education Level", "Job Title", "Experience_Level"]
numerical_features = ["Age", "Years of Experience", "Age_Exp_Interaction", "Current Salary"]

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])

# Pipeline with XGBoost
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.7,
    colsample_bytree=0.8,
    reg_alpha=1,  # L1 regularization
    reg_lambda=1, # L2 regularization
    min_child_weight=1,
    random_state=42
))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)

# Train model
model_pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = model_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Metrics
print(f"R² Score: {r2:.4f}")
print(f"Model Accuracy: {r2 * 100:.2f}%")
print(f"RMSE: INR {rmse:,.2f}")
print(f"MAE: INR {mae:,.2f}")

# Save model pipeline using joblib
joblib.dump(model_pipeline, "xgboost_salary_model.pkl")
print("Model saved as 'xgboost_salary_model.pkl'")

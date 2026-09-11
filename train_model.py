import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset (local CSV in same folder)
DATA_DIR = "BostonHousingData.csv"
df = pd.read_csv(DATA_DIR)

print("Dataset Shape:", df.shape)
print(df.info())

# 2. Separate features and target
TARGET_COL = "MEDV"
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 4. Pipeline: imputer + RandomForest
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

# 5. Train
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)

# 7. Save the trained pipeline
joblib.dump(model, "house_price_model.pkl")
print("\nModel saved as house_price_model.pkl")
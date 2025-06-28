import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load datasets
calories = pd.read_csv(r'C:\Users\Khawaish Jindal\Downloads\calories.csv')
exercise = pd.read_csv(r'C:\Users\Khawaish Jindal\Downloads\exercise.csv')

# Merge datasets on ID
calories_new = pd.concat([exercise, calories["Calories"]], axis=1)

# Check for missing values
if calories_new.isnull().sum().sum() > 0:
    print("Warning: Missing values detected! Filling with mean values.")
    calories_new.fillna(calories_new.mean(), inplace=True)

# Select relevant features
X = calories_new[['Age', 'Weight', 'Height', 'Heart_Rate', 'Duration']]  # Use correct column names
y = calories_new['Calories']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Important for models like Linear Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model (Better alternative: RandomForestRegressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

with open("c_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("c_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅  Model & scaler saved!")

def predict_calories(features):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    return model.predict(features_scaled)[0]


print("Model training complete. Model saved as model.pkl")
print("Scaler saved as scaler.pkl")

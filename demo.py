# train_workout_recommender.py
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ────────────────────────────────────────────────────────────────
# 1. LOAD DATA ─ change the path to your actual CSV
# ────────────────────────────────────────────────────────────────
df = pd.read_csv("health_dataset.csv")   # Age,BMI,Exercise_Frequency, ...

# ────────────────────────────────────────────────────────────────
# 2. LABELING FUNCTION  → adds Workout_Type column
#    (tweak thresholds to your liking)
# ────────────────────────────────────────────────────────────────
def recommend_workout(row):
    bmi            = row['BMI']
    exercise_days  = row['Exercise_Frequency']
    diet           = row['Diet_Quality']
    
    # Extremely low activity + very high BMI
    if bmi >= 30 and exercise_days <= 1:
        return 'Walking'
    # High BMI or poor diet
    if bmi >= 27 or diet < 40:
        return 'Cardio'
    # Already healthy & active              → build strength
    if bmi < 24 and exercise_days >= 4 and diet >= 70:
        return 'Strength Training'
    # Good diet & moderate BMI              → HIIT
    if 24 <= bmi < 27 and diet >= 50:
        return 'HIIT'
    # Default fallback
    return 'Yoga'

df['Workout_Type'] = df.apply(recommend_workout, axis=1)

# ────────────────────────────────────────────────────────────────
# 3. ENCODE TARGET
# ────────────────────────────────────────────────────────────────
le_target = LabelEncoder()
df['Workout_Type_enc'] = le_target.fit_transform(df['Workout_Type'])

# ────────────────────────────────────────────────────────────────
# 4. SELECT FEATURES & OPTIONAL SCALING
#    We keep every feature numeric; Exercise_Frequency already 0-7
# ────────────────────────────────────────────────────────────────
feature_cols = ['Age', 'BMI', 'Exercise_Frequency', 'Diet_Quality']
X = df[feature_cols].values
y = df['Workout_Type_enc'].values

# Scale numerical features (tree models don’t need it, but it never hurts)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ────────────────────────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

# ────────────────────────────────────────────────────────────────
# 6. TRAIN MODEL
# ────────────────────────────────────────────────────────────────
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42
)
clf.fit(X_train, y_train)

# ────────────────────────────────────────────────────────────────
# 7. EVALUATE
# ────────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
print("\n🌟  Classification Report")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ────────────────────────────────────────────────────────────────
# 8. SAVE MODEL + SCALER + LABEL ENCODER
# ────────────────────────────────────────────────────────────────
with open("workout_recommender_model.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("workout_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("workout_label_encoder.pkl", "wb") as f:
    pickle.dump(le_target, f)

print("\n✅  Model, scaler and label encoder saved!")

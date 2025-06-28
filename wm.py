import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('workout.csv')
# Drop irrelevant columns

# df.drop(columns=['Sleep_Hours',
#     'Smoking_Status', 'Alcohol_Consumption', 'Health_Score'], inplace=True)
def recommend_workout(row):
    bmi, ex, diet = row['BMI'], row['Exercise_Frequency'], row['Diet_Quality']

    p = np.random.rand()               # adds randomness

    if bmi >= 30 and ex <= 1:
        return 'Walking' if p < 0.95 else 'Yoga'
    if bmi >= 27 or diet < 40:
        return 'Cardio'  if p < 0.95 else 'HIIT'
    if bmi < 24 and ex >= 4 and diet >= 70:
        return 'Strength Training' if p < 0.95 else 'HIIT'
    if 24 <= bmi < 27 and diet >= 50:
        return 'HIIT' if p < 0.95 else 'Cardio'
    return 'Yoga' if p < 0.95 else 'Walking'

df['Workout_Type'] = df.apply(recommend_workout, axis=1)
 
# print(df.columns)
target = LabelEncoder()
df['WorkoutEncoded'] = target.fit_transform(df['Workout_Type'])

feature_cols = [
    'Age', 'BMI', 'Exercise_Frequency', 'Diet_Quality',
    'Sleep_Hours', 'Smoking_Status', 'Alcohol_Consumption'
]
X = df[feature_cols].values
y = df['WorkoutEncoded'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# fit ONLY on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42
)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
print("\n🌟  Classification Report")
print(classification_report(y_test, y_pred, target_names= target.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


with open("wm_model.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("wm_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("wml_encoder.pkl", "wb") as f:
    pickle.dump(target, f)

print("\n✅  Model, scaler and label encoder saved!")




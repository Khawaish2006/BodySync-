import pandas as pd
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from imblearn.ensemble import RandomForestClassifier  # Changed import
from sklearn.metrics import classification_report, roc_auc_score
import pickle

# ───────────────────────────────────────────────────────────────
# 1. Load the four NHANES modules we need
# ───────────────────────────────────────────────────────────────
dem  = pd.read_csv('demographic.csv',
                   usecols=['SEQN', 'RIDAGEYR', 'RIAGENDR'])
exam = pd.read_csv('examination.csv',
                   usecols=['SEQN', 'BMXHT', 'BMXWT', 'BPXSY1', 'BPXDI1'])
labs = pd.read_csv('labs.csv',
                   usecols=['SEQN', 'LBXTC', 'LBXSGL', 'LBXGH'])
ques = pd.read_csv('questionnaire.csv',
                   usecols=['SEQN',
                            'SMQ020', 'ALQ101', 'PAQ650',
                            'DIQ010', 'BPQ020', 'MCQ160E',
                            'MCQ160F', 'MCQ160G'])

# ───────────────────────────────────────────────────────────────
# 2. Rename columns for clarity
# ───────────────────────────────────────────────────────────────
dem.rename(columns={'RIDAGEYR': 'Age',
                    'RIAGENDR': 'Gender'}, inplace=True)

exam.rename(columns={'BMXHT': 'Height_cm',
                     'BMXWT': 'Weight_kg',
                     'BPXSY1': 'Systolic_BP',
                     'BPXDI1': 'Diastolic_BP'}, inplace=True)

labs.rename(columns={'LBXSGL': 'Glucose',
                     'LBXGH':  'HbA1c',
                     'LBXTC':  'Cholesterol'}, inplace=True)

ques.rename(columns={'SMQ020': 'Smoker',
                     'ALQ101': 'Alcohol',
                     'PAQ650': 'Physical_Activity'}, inplace=True)

# ───────────────────────────────────────────────────────────────
# 3. Merge on SEQN
# ───────────────────────────────────────────────────────────────
df = reduce(lambda l, r: pd.merge(l, r, on='SEQN', how='inner'),
            [dem, exam, labs, ques])
disease_cols = ['DIQ010', 'BPQ020', 'MCQ160E', 'MCQ160F', 'MCQ160G']
valid_vals = {1: 1, 2: 0}            # desired mapping
for col in disease_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')   # force numeric
    df = df[df[col].isin([1, 2])]                       # drop 3, 7, 9, NaN
    df[col] = df[col].map(valid_vals) 

# ───────────────────────────────────────────────────────────────
# 4. Feature engineering
# ───────────────────────────────────────────────────────────────
df['BMI']             = df['Weight_kg'] / (df['Height_cm'] / 100) ** 2
df['Pulse_Pressure']  = df['Systolic_BP'] - df['Diastolic_BP']

# ───────────────────────────────────────────────────────────────
# 5. Clean binary columns
#    · Gender: 1 = Male, 2 = Female  → 0/1
#    · Smoker / Alcohol / Activity: 1 = Yes, 2 = No  → 1/0
# ───────────────────────────────────────────────────────────────
df['Gender']             = df['Gender'].replace({1: 0, 2: 1})
df['Smoker']             = df['Smoker'].replace({1: 1, 2: 0})
df['Alcohol']            = df['Alcohol'].replace({1: 1, 2: 0})
df['Physical_Activity']  = df['Physical_Activity'].replace({1: 1, 2: 0})

# ───────────────────────────────────────────────────────────────
# 6. Target columns: drop invalid codes & map to 0/1
#    1 = "Yes", 2 = "No"; 3/7/9 = missing/unknown → drop row
# ───────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────
# 7. Define X and y
# ───────────────────────────────────────────────────────────────
feature_cols = [
    'Age', 'Gender', 'BMI',
    'Systolic_BP', 'Diastolic_BP', 'Pulse_Pressure',
    'Cholesterol', 'Glucose', 'HbA1c',
    'Smoker', 'Alcohol', 'Physical_Activity'
]

X = df[feature_cols]
y = df[disease_cols]

# ───────────────────────────────────────────────────────────────
# 8. Train / test split
# ───────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y['DIQ010']
)

# ───────────────────────────────────────────────────────────────
# 9. Pipeline: scale numeric, keep binaries, Balanced-Random-Forest multi-output
# ───────────────────────────────────────────────────────────────


rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        class_weight='balanced_subsample',
        random_state=42
)
model = MultiOutputClassifier(rf)

model.fit(X_train, y_train) 

model.fit(X_train, y_train)
# ----- 2.  Calibrate each disease estimator  ------------------
from sklearn.calibration import CalibratedClassifierCV

cal_models = []
for est, col in zip(model.estimators_, y_train.columns):
    cal = CalibratedClassifierCV(est, method='isotonic', cv=3)
    cal.fit(X_train, y_train[col])
    cal_models.append(cal)

def predict_proba_cal(X):
    """Returns list-of-arrays like original predict_proba"""
    return [cal.predict_proba(X) for cal in cal_models]

# ───────────────────────────────────────────────────────────────
# 10. Evaluation
# ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_probas = [est.predict_proba(model['prep'].transform(X_test))
            for est in model['multi'].estimators_]

print("\nPer-disease classification report + AUC")
for i, col in enumerate(disease_cols):
    print(f"\n— {col} —")
    print(classification_report(y_test[col], y_pred[:, i], zero_division=0))
    auc = roc_auc_score(y_test[col], y_probas[i][:, 1])
    print("AUC:", round(auc, 3))

from sklearn.metrics import precision_recall_curve
import numpy as np

optimal_thresholds = {}
for i, col in enumerate(disease_cols):
    probas = model['multi'].estimators_[i].predict_proba(model['prep'].transform(X_test))[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test[col], probas)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_thresholds[col] = thresholds[np.argmax(f1)]
    
print("\nOptimized thresholds for each disease:")
for col, threshold in optimal_thresholds.items():
    print(f"{col}: {threshold:.3f}")

# ───────────────────────────────────────────────────────────────
# 11. Probability → Low/Med/High helper
# ───────────────────────────────────────────────────────────────
def bucket(p, threshold=None):
    if threshold:  # For rare diseases
        return 'High' if p > threshold else 'Low'
    return 'Low' if p < 0.33 else 'Medium' if p < 0.66 else 'High'

def predict_risk(user_series):
    """user_series is a pandas Series with the 12 feature fields"""
    proba = model.predict_proba(pd.DataFrame([user_series])[feature_cols])
    result = {}
    for i, col in enumerate(disease_cols):
        p_yes = proba[i][0][1]
        # ===== MODIFIED CODE ===== #
        result[col] = {
            'prob': round(p_yes, 2),
            'bucket': bucket(p_yes, 
                           threshold=optimal_thresholds[col] if col in ['MCQ160E','MCQ160F','MCQ160G'] else None)
        }
        # ===== END MODIFICATION ===== #
    return result
# Quick demo:
print("\nExample user risk buckets:")
print(predict_risk(X_test.iloc[0]))

# ───────────────────────────────────────────────────────────────
# 12. Save artefacts
# ───────────────────────────────────────────────────────────────
with open('health_multi_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('disease_labels.pkl', 'wb') as f:
    pickle.dump(disease_cols, f)

print("\n✅ Model & labels saved!")
from flask import Flask, request , render_template
import numpy as np
import pickle
import pandas as pd
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
# ---------- load at startup ----------
c_model  = pickle.load(open("c_model.pkl",  "rb"))
c_scaler = pickle.load(open("c_scaler.pkl", "rb"))
# -------------------------------------

@app.route("/calories", methods=["GET", "POST"])
def calories():
    if request.method == "POST":
        age     = float(request.form["age"])
        weight  = float(request.form["weight"])
        height  = float(request.form["height"])
        hr      = float(request.form["heart_rate"])
        dur     = float(request.form["duration"])

        features = [age, weight, height, hr, dur]
        cal_burn = predict_calories_app(features)
        return render_template("calories.html", result=round(cal_burn, 1))

    return render_template("calories.html")

def predict_calories_app(features):
    data = np.array(features).reshape(1, -1)
    data = c_scaler.transform(data)
    return c_model.predict(data)[0]

# workout recommendation
# ---------- LOAD WORKOUT ARTEFACTS AT STARTUP ----------
workout_model  = pickle.load(open("wm_model.pkl", "rb"))
workout_scaler = pickle.load(open("wm_scaler.pkl",        "rb"))
workout_le     = pickle.load(open("wml_encoder.pkl", "rb"))
# -------------------------------------------------------

@app.route("/workout", methods=["GET", "POST"])
def workout():
    if request.method == "POST":
        age      = float(request.form["age"])
        bmi      = float(request.form["bmi"])
        days     = int(request.form["exercise_days"])
        diet     = float(request.form["diet_quality"])
        sleep    = float(request.form["sleep_hours"])
        smoking  = int(request.form["smoking_status"])  
        alcohol  = int(request.form["alcohol_consumption"])

        features = [[age, bmi, days, diet , sleep, smoking, alcohol]]
        x_scaled = workout_scaler.transform(features)
        pred_enc = workout_model.predict(x_scaled)[0]
        workout  = workout_le.inverse_transform([pred_enc])[0]

        return render_template("workout.html",
                               result=workout.title(),   # capitalize nicely
                               form_data=request.form)   # echo back inputs

    return render_template("workout.html")

# -----------------------------------------------------------------
#  LOAD artefacts once, near the top of app.py (if not already):
# -----------------------------------------------------------------
health_model   = pickle.load(open("health_multi_model.pkl", "rb"))
health_labels  = pickle.load(open("disease_labels.pkl",   "rb"))
# Probability cut-offs you printed earlier
health_thresh  = {
    'DIQ010':0.682, 'BPQ020':0.385, 'MCQ160E':0.497,
    'MCQ160F':0.623, 'MCQ160G':0.593
}

feature_cols = [      # must be EXACT order used in training
    'Age','Gender','BMI','Systolic_BP','Diastolic_BP','Pulse_Pressure',
    'Cholesterol','Glucose','HbA1c','Smoker','Alcohol','Physical_Activity'
]

def bucket(p):                         # same rule as training notebook
    return "Low" if p < .33 else "Medium" if p < .66 else "High"

# -----------------------------------------------------------------
#  RISK ROUTE
# -----------------------------------------------------------------
@app.route("/risk", methods=["GET", "POST"])
def risk():
    if request.method == "POST":

        # ---------- collect inputs ----------
        age    = float(request.form["age"])
        gender = int(request.form["gender"])                 # 0=male 1=female
        bmi    = float(request.form["bmi"])
        sysBP  = float(request.form["sys"])
        diaBP  = float(request.form["dia"])
        chol   = float(request.form["chol"])
        glu    = float(request.form["glu"])
        hba1c  = float(request.form["hba1c"])

        # coerce to 0/1 just in case
        smoker = 1 if int(request.form["smoker"])  == 1 else 0
        alco   = 1 if int(request.form["alcohol"]) == 1 else 0
        active = 1 if int(request.form["active"])  == 1 else 0

        pulse  = sysBP - diaBP

        user_df = pd.DataFrame([[  # order must match feature_cols
            age, gender, bmi, sysBP, diaBP, pulse,
            chol, glu, hba1c, smoker, alco, active
        ]], columns=feature_cols)

        # ---------- model inference ----------
        probas = health_model.predict_proba(user_df)
        results = []
        for i, code in enumerate(health_labels):
            p = probas[i][0][1]                           # probability of disease
            results.append({
                'code'  : code,
                'name'  : {'DIQ010':'Diabetes','BPQ020':'Hypertension',
                           'MCQ160E':'Heart Attack','MCQ160F':'Stroke',
                           'MCQ160G':'COPD'}[code],
                'prob'  : p,                              # keep numeric
                'bucket': bucket(p),                      # Low/Med/High
                'flag'  : int(p >= health_thresh[code])
            })

        # ---------- HEALTH score (higher = healthier) ----------
        weights = [0.30,0.25,0.20,0.15,0.10]              # same order as results
        risk_score   = sum(r['prob'] * w for r, w in zip(results, weights))  # 0–1
        health_score = (risk_score) * 100                                 # 0–100

        health_bucket = ('High'   if health_score >= 67 else
                         'Medium' if health_score >= 34 else
                         'Low')

        # convert probs back to strings for template
        for r in results:
            r['prob'] = f"{r['prob']:.2f}"

        return render_template("risk.html",
                               results=results,
                               health=round(health_score, 1),
                               health_bucket=health_bucket,
                               form_data=request.form)

    return render_template("risk.html")


if __name__ == '__main__':
    app.run(debug=True, port=5000)
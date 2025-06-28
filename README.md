# BodySync+  :muscle: :heartbeat:
**Personal Fitness & Preventive‑Health Suite powered by Machine Learning**

> “Tune your body, sync your life.”

---

## 🔑 What Is BodySync+?

BodySync+ is a Flask web application that blends **data‑driven fitness** with **preventive health screening**.  
From a single dashboard a user can:

1. **Estimate calories burned** for any workout session.  
2. **Receive a personalised workout recommendation** based on lifestyle.  
3. **Screen for five chronic diseases**—Diabetes, Hypertension, Heart‑Attack, Stroke and COPD—using calibrated risk probabilities.

All models are trained on **open datasets** (Kaggle & NHANES), persisted via Pickle, served in real time, and wrapped in a Bootstrap UI for a smooth green aesthetic.

---

## 🖼️ Live Screenshot

![landing page](static/docs/landing.png)

*(replace with your own image)*

---

## ⚙️ Tech Stack

| Layer | Libraries / Tools |
|-------|-------------------|
| **Back‑end** | Flask 2.2 · Jinja2 · Gunicorn |
| **ML** | scikit‑learn (RandomForestRegressor / Classifier, MultiOutputClassifier, Isotonic Calibration) |
| **Data** | Kaggle *Calories & Exercise* · NHANES 2015‑16 |
| **Front‑end** | Bootstrap 5.3 · Vanilla JS |
| **Ops** | Python 3.11 · virtualenv · (optional) Docker & GitHub Actions |

---

## 📊 ML Models at a Glance

| Page | Dataset & Size | Features | Algorithm & Tricks | Key Metric |
|------|----------------|----------|--------------------|------------|
| **Calorie Burn** | *exercise.csv* + *calories.csv* (~1 k rows) | Age, Weight, Height, Heart‑Rate, Duration | `RandomForestRegressor`, 100 trees | MAE ≈ **12 kcal** |
| **Workout Recommender** | Synthetic lifestyle dataset (2 k rows) | Age, BMI, Exercise Days/Wk, Diet Score, Sleep, Smoker, Alcohol | `RandomForestClassifier` with `class_weight='balanced_subsample'` | F1‑weighted **0.94** |
| **Health‑Risk Screener** | NHANES (≈ 9 k participants) | 12 vitals & lifestyle fields | Multi‑output `RandomForest`, isotonic probability calibration, custom thresholds | **AUC 0.94** (Diabetes); recall 0.45‑0.80 across labels |

### Overall Health Score  
We combine weighted disease probabilities into a 0‑100 **Health Score** (higher = healthier) so users instantly grasp their status.

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/<yourname>/bodysync-plus.git
cd bodysync-plus

# 2. Environment
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Get pre‑trained models (small zip, ~10 MB)
wget https://github.com/<yourname>/bodysync-plus/releases/download/v1.0/models.zip
unzip models.zip -d models/

# 4. Run
python app.py
# open http://127.0.0.1:5000

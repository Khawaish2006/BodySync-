# BodySync+  :muscle: :heartbeat:
**Personal Fitness & Preventive‑Health Suite powered by Machine Learning**

<img src="static/docs/hero.png" width="100%">

BodySync+ brings together calorie estimation, workout recommendation, chronic‑disease risk screening in one sleek Flask web app.

---

## ✨ Key Features

| Page | ML Problem | Algorithm | Metrics |
|------|------------|-----------|---------|
| **Calories** | Regression | `RandomForestRegressor` | MAE ≈ 12 kcal |
| **Workout**  | Classification | `RandomForestClassifier` + class weighting | F1 ≈ 0.94 |
| **Health Risk** | Multi‑label classification (5 diseases) | `RandomForest` **+ isotonic calibration** | Diabetes AUC = 0.94, recall ↑ 0.80 |

---



# 🏦 Home Credit Default Risk Predictor

> A production-style ML pipeline that predicts loan repayment difficulty for underbanked clients — combining ensemble learning, class-imbalance handling, and rigorous evaluation to build a deployable credit scorecard.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![Version](https://img.shields.io/badge/version-1.0.0-orange)

---

## 📋 Table of Contents
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Architecture](#-project-architecture)
- [Getting Started](#-getting-started)
- [Usage Example](#-usage-example)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- **End-to-end ML pipeline** — ingestion, cleaning, feature engineering, modelling, and evaluation in a single reproducible notebook
- **Smart missing-value handling** — automated 30% threshold dropping, median imputation for numerics, sentinel fill for categoricals
- **Domain-aware feature engineering** — ratio features (`credit_income_percent`, `annuity_income_percent`, `credit_term`, `days_employed_percent`) grounded in credit-risk domain knowledge
- **Multi-strategy imbalance correction** — SMOTE, Random Over-Sampling, and Random Under-Sampling with visual comparison
- **Model zoo with ensemble** — Logistic Regression, Random Forest, XGBoost, LightGBM, and a Stacking meta-classifier evaluated head-to-head
- **Advanced evaluation suite** — ROC-AUC, Precision-Recall curves, calibration curves, normalized confusion matrices, and threshold optimization
- **Exploratory richness** — KDE plots by target class, correlation heatmaps, default-rate bar charts per category, outlier analysis with 99th-percentile capping

---

## 🛠 Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.9+ |
| Data wrangling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| ML - Classic | Scikit-learn (LogReg, RandomForest, SelectKBest) |
| ML - Boosting | XGBoost, LightGBM |
| ML - Ensemble | Scikit-learn StackingClassifier |
| Imbalance | imbalanced-learn (SMOTE, ROS, RUS) |
| Serialization | Pickle |
| Environment | Jupyter Notebook |

---

## 🏗 Project Architecture

```
home-credit-default-risk/
│
├── Home_credit_default_risk.ipynb   # Main pipeline notebook
├── train.csv                        # Training dataset (Kaggle source)
├── validazione.csv                  # Holdout validation set
│
├── outputs/
│   ├── roc_pr_curves.png            # ROC & Precision-Recall curves
│   ├── calibration_curve.png        # Reliability diagrams
│   ├── confusion_matrices.png       # Normalized confusion matrices
│   ├── threshold_optimization.png   # XGBoost threshold sweep
│   └── resampling_compare.html      # Interactive resampling comparison
│
└── README.md
```

**Pipeline flow:**

```
Raw CSV → EDA → Missing Value Analysis → Drop/Impute/Encode
       → Feature Engineering → Train/Test Split
       → Resampling (SMOTE / ROS / RUS)
       → Model Training (LR / RF / XGB / LGBM / Stack)
       → Evaluation (AUC / PR / Calibration / Threshold)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/home-credit-default-risk.git
cd home-credit-default-risk

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
imbalanced-learn>=0.10
xgboost>=1.7
lightgbm>=3.3
matplotlib>=3.6
seaborn>=0.12
plotly>=5.10
jupyter
```

### Data Setup

Download `train.csv` from the [Kaggle competition](https://www.kaggle.com/c/home-credit-default-risk) and place it in the project root.

### Running the Project

```bash
jupyter notebook Home_credit_default_risk.ipynb
```

Run cells sequentially from top to bottom. Section headers (`1.0 Data Preparation`, `2.0 EDA`, `3.0 Data Prep`, `4.0 Modelling`) guide the flow.

---

## 💡 Usage Example

```python
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load your pre-trained model
with open("xgb_model.pkl", "rb") as f:
    model = load(f)

# Prepare a new applicant record
applicant = pd.DataFrame([{
    "amt_income_total": 135000,
    "amt_credit": 450000,
    "amt_annuity": 22500,
    "days_birth": -14600,      # ~40 years old (stored as negative days)
    "days_employed": -1825,    # ~5 years employed
    "ext_source_2": 0.62,
    "ext_source_3": 0.55,
}])

# Add engineered features
applicant["credit_income_percent"] = applicant["amt_credit"] / applicant["amt_income_total"]
applicant["annuity_income_percent"] = applicant["amt_annuity"] / applicant["amt_income_total"]
applicant["credit_term"] = applicant["amt_annuity"] / applicant["amt_credit"]
applicant["days_employed_percent"] = applicant["days_employed"] / applicant["days_birth"]

# Predict default probability
prob = model.predict_proba(applicant)[:, 1][0]
print(f"Default probability: {prob:.2%}")
# → Default probability: 7.43%
```

---

## 🗺 Roadmap

- [ ] **Hyperparameter tuning** — Add `Optuna` or `BayesSearchCV` for principled XGBoost/LightGBM tuning instead of hand-picked values
- [ ] **Pipeline refactor** — Wrap preprocessing into a `sklearn.Pipeline` with `ColumnTransformer` to eliminate the current data-leakage risk from fitting imputers on the full dataset before the split
- [ ] **SHAP explainability** — Integrate `shap` for global feature importance and per-prediction local explanations (critical for regulatory compliance in credit scoring)
- [ ] **MLflow experiment tracking** — Log model versions, parameters, and metrics to enable proper experiment comparison and model registry

---

## 🤝 Contributing

Contributions are welcome! Here's how to get involved:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please make sure your code follows PEP 8 and that any new modelling decisions are documented with their reasoning.

---

*Dataset sourced from the [Home Credit Default Risk Kaggle Competition](https://www.kaggle.com/c/home-credit-default-risk). For educational and portfolio purposes only.*

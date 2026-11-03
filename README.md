<div align="center">

# Diabetes Prediction Using Machine Learning

### An educational Streamlit dashboard for exploring diabetes classification models

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit--learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

</div>

This project trains and compares four machine-learning classifiers using the
Pima Indians Diabetes dataset. Its interactive dashboard lets you enter
measurements to see a model prediction and explore the saved evaluation
metrics, confusion matrices, and tree-based feature importance.

> **Medical disclaimer:** This is a learning and demonstration project, not a
> medical device. Its predictions are not a diagnosis or medical advice. Do
> not use them to make health decisions; consult a qualified healthcare
> professional.

## Contents

- [What the project does](#what-the-project-does)
- [Technology stack](#technology-stack)
- [How it works](#how-it-works)
- [Models and saved evaluation](#models-and-saved-evaluation)
- [Dataset and features](#dataset-and-features)
- [Project structure](#project-structure)
- [Setup and run](#setup-and-run)
- [Limitations](#limitations)
- [Ideas for future work](#ideas-for-future-work)

## What the project does

- Trains Logistic Regression, Decision Tree, Random Forest, and Support
  Vector Machine (SVM) classifiers.
- Compares the models using accuracy, precision, recall, and F1 score.
- Accepts eight measurements in the dashboard and displays a prediction from
  the selected model.
- Presents model metrics, confusion matrices, and feature importance in
  separate dashboard views.
- Saves trained models and evaluation artifacts under `models/`, so the
  dashboard can load them without retraining.

## Technology stack

| Area | Technologies | Role |
|---|---|---|
| Language | Python | Training, data preparation, inference, and app logic |
| Frontend / UI | Streamlit | Renders the interactive, multipage web dashboard |
| Backend / ML | scikit-learn | Splitting data, scaling features, training classifiers, and calculating metrics |
| Data handling | Pandas, NumPy | Reading and preparing tabular data and feature values |
| Charts | Matplotlib, Seaborn | Model comparison chart and confusion-matrix visualizations |
| Model persistence | Joblib | Saving and loading models, scaler, and evaluation artifacts |

There is no separate JavaScript frontend, REST API, or database in this
project. Streamlit serves the UI and invokes the Python prediction and
visualization code directly.

## How it works

### Training pipeline

Running `train_models.py` performs these steps:

1. Loads `diabetes.csv` and checks that the eight input columns and `Outcome`
   target column are present.
2. Removes duplicate rows and rows with missing values.
3. Splits the features and target into stratified training and test sets
   (80/20, with `random_state=42`).
4. Treats zero readings in Glucose, BloodPressure, SkinThickness, Insulin,
   and BMI as missing measurements. It replaces these with medians computed
   from the training set only, then applies the same medians to the test set.
5. Fits a `StandardScaler` on the training features and applies it to both
   sets.
6. Trains and evaluates all four classifiers on the same scaled data.
7. Saves the trained models, scaler, metrics, confusion matrices, selected
   best model, and available tree-based feature importances in `models/`.

### Dashboard

`app.py` loads the saved artifacts and offers these views:

| View | What it shows |
|---|---|
| Home | Project overview and the best model in the saved results |
| Prediction | Patient measurement inputs, model selection, predicted class, and probability when supported |
| Model Comparison | Evaluation metrics and a model comparison chart |
| Confusion Matrices | Correct and incorrect classifications for each model |
| Feature Importance | Random Forest and Decision Tree feature importances, when available |
| About Project | Dataset, feature, and limitation details |

The dashboard expects the artifacts created by the training script. This
repository includes a `models/` directory with trained artifacts, so you can
try the app without retraining, provided the installed library versions can
load them.

## Models and saved evaluation

The checked-in `models/metrics.csv` records the following scores from the
project's stratified 80/20 test split:

| Model | Accuracy | Precision | Recall | F1 score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7078 | 0.5882 | 0.5556 | 0.5714 |
| Decision Tree | 0.8117 | 0.7451 | 0.7037 | 0.7238 |
| Random Forest | 0.8636 | 0.8113 | 0.7963 | 0.8037 |
| SVM | 0.8377 | 0.7636 | 0.7778 | 0.7706 |

In this saved run, Random Forest has the highest F1 score. These values
describe one train/test split, not a clinical performance guarantee; results
may vary if the dataset or training procedure changes.

## Dataset and features

The project includes `diabetes.csv`, based on the Pima Indians Diabetes
dataset. Each row contains measurements and an `Outcome` label:

- `0` — no diabetes recorded
- `1` — diabetes recorded

| Input feature | Meaning |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (mg/dL) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin-fold thickness (mm) |
| `Insulin` | 2-hour serum insulin (mu U/mL) |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function score |
| `Age` | Age in years |

The training script interprets zero values in Glucose, BloodPressure,
SkinThickness, Insulin, and BMI as unavailable readings and imputes them
using training-set medians. A zero in `Pregnancies` is retained.

## Project structure

```text
.
├── app.py                         # Streamlit dashboard and prediction UI
├── train_models.py                # Data preparation, training, and evaluation
├── diabetes.csv                   # Dataset
├── requirements.txt               # Python package dependencies
├── models/
│   ├── logistic_regression.pkl     # Trained Logistic Regression model
│   ├── decision_tree.pkl          # Trained Decision Tree model
│   ├── random_forest.pkl          # Trained Random Forest model
│   ├── svm.pkl                    # Trained SVM model
│   ├── scaler.pkl                 # Fitted StandardScaler
│   ├── feature_names.pkl          # Input feature order
│   ├── metrics.csv                # Evaluation metrics
│   ├── confusion_matrices.pkl     # Confusion matrices
│   ├── feature_importance.csv     # Tree-based feature importances
│   └── best_model.json            # Model with highest F1 score
└── README.md
```

## Setup and run

Use Python with `pip`. From the project root, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### Start the dashboard with the included models

```bash
streamlit run app.py
```

Open the local address printed by Streamlit (typically
`http://localhost:8501`).

### Retrain the models

The dataset is already included. To regenerate the model and evaluation
artifacts, run:

```bash
python train_models.py
```

Then start or refresh the dashboard:

```bash
streamlit run app.py
```

`train_models.py` expects the dataset at `diabetes.csv` in the project root
and requires the columns listed in [Dataset and features](#dataset-and-features).

## Limitations

- The dataset represents a specific population of Pima Indian women; it does
  not establish that the models generalize to other populations.
- The dataset is limited in size and scope. A single holdout split is not a
  substitute for cross-validation or external validation.
- The displayed probabilities are model outputs, not calibrated clinical
  risk estimates.
- Feature importance describes how a tree model used features; it does not
  show medical causation.
- No clinical validation, privacy controls, authentication, or deployment
  hardening is provided.

## Ideas for future work

- Add cross-validation, hyperparameter tuning, and calibration analysis.
- Evaluate on a larger, more representative dataset and document its
  provenance and license.
- Improve input validation and explain how unavailable measurements are
  handled at prediction time.
- Add automated tests for preprocessing, training outputs, and dashboard
  prediction behavior.
- If deploying publicly, add appropriate privacy, security, and clinical
  review rather than presenting the app as a diagnostic service.

# 🎗️ Breast Cancer Classification System

<p align="center">
  <strong>A Machine Learning Desktop Application for Breast Tumor Classification</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/GUI-Tkinter-orange" />
  <img src="https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Database-SQLite-003B57?logo=sqlite&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## 📌 Overview

The **Breast Cancer Classification System** is a Python-based desktop application that uses machine learning to classify breast tumor measurements as **Benign** or **Malignant**.

Built with Tkinter and scikit-learn, the application provides an interactive interface for entering tumor measurements, generating predictions, tracking classification history, and exploring analytical visualizations.

The system uses the **Breast Cancer Wisconsin (Diagnostic) Dataset**, containing 569 samples and 30 numerical features extracted from digitized fine-needle-aspirate (FNA) images.

> ⚠️ **Medical Disclaimer**
>
> This project is strictly an educational machine-learning project and is **not a medical device**.
>
> It classifies numerical measurements computed from digitized FNA images. It does **not** analyze raw medical images, confirm cancer, or replace a biopsy, pathology examination, clinical evaluation, or professional medical diagnosis.
>
> A Benign prediction does not rule out cancer, and a Malignant prediction is not a confirmed diagnosis. Always consult a qualified healthcare professional.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📊 Dashboard | Total, benign, and malignant prediction counts, latest prediction, model evaluation summary, distribution chart, and recent history |
| 🧬 Tumor Classification | Interactive form containing all 30 measurements, organized into Mean, Standard Error, and Worst sections |
| ✅ Live Validation | Validates input values and supports pasting all 30 measurements at once |
| 🧪 Example Loaders | Load example measurements for testing the application |
| 🎯 Prediction Results | Displays predicted class, estimated class probability, measurement summary, date/time, and record ID |
| 🗂️ Prediction History | SQLite-powered search, filtering, record details, deletion, refresh, and CSV export |
| 📈 Analytics | Seven Matplotlib charts covering predictions, model performance, dataset distribution, and feature importance |
| 📄 Reports | CSV exports, classification summaries, per-record reports, and per-model classification reports |
| ⚙️ Settings | Application, model, dataset, database, and evaluation information |

### 📈 Analytics Dashboard

The application includes seven analytical visualizations:

1. Prediction Distribution
2. Predictions Over Time
3. Model Comparison
4. Dataset Class Distribution
5. Confusion Matrix
6. ROC Curve
7. Feature Importance

Each chart is clearly tagged as either **DATASET** or **APP PREDICTIONS** to distinguish research-dataset results from application-generated prediction history.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core programming language |
| Tkinter | Desktop graphical user interface |
| Pillow | Logo and icon generation |
| Pandas | Dataset handling and data processing |
| NumPy | Numerical computation |
| scikit-learn | Machine learning and model evaluation |
| Matplotlib | Analytics and visualizations |
| Joblib | Model serialization |
| SQLite3 | Prediction history database |

---

## 🧬 Understanding the 30 Features

The dataset contains ten measurements for each cell nucleus. Each measurement is summarized in three ways:

- **Mean:** Average measurement.
- **SE:** Standard error of the measurement.
- **Worst:** Mean of the three largest values.

| Measurement | Description |
|---|---|
| Radius | Mean distance from the centre to points on the boundary |
| Texture | Standard deviation of gray-scale values |
| Perimeter | Length of the nucleus boundary |
| Area | Area of the nucleus |
| Smoothness | Local variation in radius lengths |
| Compactness | Perimeter² / Area − 1.0 |
| Concavity | Severity of concave portions of the contour |
| Concave Points | Number of concave portions of the contour |
| Symmetry | Symmetry of the nucleus shape |
| Fractal Dimension | "Coastline approximation" − 1 |

**Measurement units:** These are image-derived measurements. The dataset does not define absolute physical units. Radius and perimeter represent lengths, area represents an area, and the remaining features are unitless or gray-level measurements.

---

## 🤖 Machine Learning Methodology

### Preprocessing Pipeline

The application uses a single scikit-learn `Pipeline` to ensure consistent preprocessing during training and prediction.

```text
Input Dataset
     │
     ▼
Median Imputation
     │
     ▼
Variance Filtering
     │
     ▼
Standard Scaling
     │
     ▼
Machine Learning Classifier
     │
     ▼
Prediction
```

The pipeline is fitted using training data only to prevent data leakage. Training and GUI predictions use the same feature names, feature order, and preprocessing pipeline.

### Data Cleaning

The training pipeline handles:

- Invalid target labels
- Non-numeric values
- Infinite and negative values
- Missing values through imputation
- Sparse rows
- Exact duplicate records

Invalid numerical values are treated as missing and imputed. All cleaning operations are reported. No cleaning was required for the supplied dataset.

### Training Configuration

| Parameter | Configuration |
|---|---|
| Train/Test Split | Stratified 80/20 |
| Random Seed | 42 |
| Cross-Validation | Stratified 5-Fold |
| Hyperparameter Tuning | Not performed |
| Preprocessing | scikit-learn Pipeline |
| Positive Class | Malignant |

### Models Compared

Seven machine learning algorithms are evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors
5. Support Vector Machine (probability-calibrated)
6. Gradient Boosting
7. Extra Trees

All algorithms use sensible fixed hyperparameter defaults to keep the experiments reproducible.

---

## 🏆 Model Selection Strategy

The best model is not selected using accuracy alone.

Models are evaluated using stratified 5-fold cross-validation on the **training split only**.

The selection score is:

```text
CV Score =
    0.40 × Sensitivity
  + 0.20 × Specificity
  + 0.20 × F1 (Malignant)
  + 0.20 × ROC-AUC
```

**Eligibility requirement:** Models with cross-validation sensitivity below `0.90` are ineligible.

Sensitivity receives the highest weight because failing to identify a malignant case is a particularly costly classification error. Specificity is also reported to show the false-alarm trade-off.

The held-out test split is never used for model selection.

---

## 📊 Model Evaluation Results

**Held-out test split:** 114 samples  
**Positive class:** Malignant

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 | ROC-AUC | FN | FP | CV Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ⭐ **Logistic Regression** | **0.956** | 0.911 | **0.976** | 0.944 | 0.943 | 0.995 | **1** | 4 | **0.9785** |
| Extra Trees | 0.956 | 0.951 | 0.929 | 0.972 | 0.940 | 0.993 | 3 | 2 | 0.9760 |
| Support Vector Machine | 0.974 | 0.953 | 0.976 | 0.972 | 0.965 | 0.994 | 1 | 2 | 0.9739 |
| Random Forest | 0.939 | 0.907 | 0.929 | 0.944 | 0.918 | 0.994 | 3 | 4 | 0.9583 |
| K-Nearest Neighbors | 0.974 | 1.000 | 0.929 | 1.000 | 0.963 | 0.988 | 3 | 0 | 0.9542 |
| Gradient Boosting | 0.956 | 0.974 | 0.905 | 0.986 | 0.938 | 0.991 | 4 | 1 | 0.9518 |
| Decision Tree | 0.939 | 0.907 | 0.929 | 0.944 | 0.918 | 0.947 | 3 | 4 | 0.9070 |

### 🥇 Selected Model: Logistic Regression

The selection strategy chooses **Logistic Regression** based on its highest cross-validation selection score.

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 95.6% |
| Sensitivity | 97.6% |
| Specificity | 94.4% |
| ROC-AUC | 0.995 |
| False Negatives | 1 |
| False Positives | 4 |

**Interpretation:** On this held-out test split, the selected model missed one malignant case and classified four benign cases as malignant.

With only 114 test samples, differences of one or two cases between models are not statistically meaningful. These results describe performance on this research dataset, **not clinical performance**.

---

## 📁 Project Structure

```text
Breast-Cancer-Classification-System/
│
├── dataset/
│   ├── breast_cancer.csv
│   └── dataset_info.json
│
├── models/
│   ├── breast_cancer_model.pkl
│   ├── model_metadata.json
│   └── model_comparison.csv
│
├── database/
│   └── breast_cancer_history.db
│
├── assets/
│   ├── logo.png
│   └── icons/
│
├── modules/
│   ├── config.py
│   ├── preprocessing.py
│   ├── predictor.py
│   ├── database.py
│   ├── analytics.py
│   └── report_generator.py
│
├── ui/
│   └── Tkinter pages and widgets
│
├── tests/
│   └── test_backend.py
│
├── reports/
│
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

### Main Modules

| File | Responsibility |
|---|---|
| `config.py` | Paths, feature definitions, labels, and colors |
| `preprocessing.py` | Dataset download, inspection, cleaning, and shared pipeline |
| `predictor.py` | Model loading, input validation, and prediction |
| `database.py` | SQLite prediction history |
| `analytics.py` | Matplotlib visualizations |
| `report_generator.py` | CSV and text report exports |
| `train_model.py` | Dataset preparation, model training, and evaluation |
| `app.py` | Main desktop application entry point |

---

## 🗃️ Prediction History & Reports

The application uses SQLite to maintain prediction history.

Available operations include:

- Search and filter previous predictions
- View individual prediction details
- Delete records with confirmation
- Refresh prediction history
- Export history to CSV
- Generate classification summary reports
- Generate individual record reports
- Export model comparison results
- Generate per-model classification reports

The system uses generated record IDs and does not store personal data.

---

## ⚠️ Limitations

- Predictions are based on numerical features from the WDBC research dataset, not raw medical images.
- The system only works with the 30 defined input measurements.
- Model evaluation results are specific to the supplied research dataset and test split.
- A high estimated class probability does not establish a medical diagnosis.
- The dataset is relatively small, so model performance may vary across different samples.
- Differences between models involving only one or two test cases should not be overinterpreted.
- The application is intended for educational and research purposes only.

---

## 🔮 Future Improvements

- [ ] Add more breast tumor datasets for broader evaluation.
- [ ] Improve model evaluation with additional validation datasets.
- [ ] Add more interactive analytics and visual reports.
- [ ] Improve prediction explanations and feature interpretation.
- [ ] Add model export and deployment options.
- [ ] Improve the user interface and accessibility.
- [ ] Explore explainability methods such as SHAP.

---

<p align="center">
  <strong>🎗️ Breast Cancer Classification System</strong><br/>
  Built with Python, Tkinter, and Machine Learning.
</p>

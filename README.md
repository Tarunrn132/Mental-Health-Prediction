# Mental Health Signal — Student Wellness Analytics

A sophisticated, full-stack predictive wellness analytics web application inspired by and cleanly recreating the concept of **"Mental Health Signal — Student Wellness Analytics"**.

The application evaluates lifestyle patterns, academic rhythm, social media habits, and perceived stress to generate an empirical, model-based mental health index score (0–10). It is built with an **editorial analytics aesthetic**, **custom SVG semicircular gauge visualization**, **real-time form validation**, and an **end-to-end scikit-learn machine learning pipeline** served through a high-performance **FastAPI** backend.

> **Important Wellness & Educational Disclaimer:**
> This application is strictly an educational and informational tool modeling statistical associations between daily habits and student wellness. It is **NOT** a diagnostic instrument, medical assessment, or substitute for professional psychiatric, psychological, or medical healthcare.

---

## Table of Contents

1. [Project Overview & Visual Language](#project-overview--visual-language)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Dataset & Empirical Schema](#dataset--empirical-schema)
5. [Machine Learning Methodology](#machine-learning-methodology)
6. [Actual Model Evaluation Metrics](#actual-model-evaluation-metrics)
7. [Directory Structure](#directory-structure)
8. [Installation & Setup](#installation--setup)
9. [Training the Model](#training-the-model)
10. [Running Locally](#running-locally)
11. [API Reference](#api-reference)
12. [Testing Suite](#testing-suite)
13. [Deployment Options](#deployment-options)
14. [Privacy & Ethical Standards](#privacy--ethical-standards)

---

## Project Overview & Visual Language

Unlike generic dashboard templates or default Bootstrap interfaces, **Mental Health Signal** utilizes an editorial/analytics design system:

- **Color Philosophy**: Warm off-white paper canvas (`#EEF2EE`, `#F4F8F3`), deep forest pine accents (`#21594A`, `#163D33`), charcoal typography (`#182420`, `#4A5850`), and muted score indicators (`#D9534F` coral, `#E3B341` amber, `#4C9A78` sage).
- **Typography Hierarchy**:
  - *Editorial Display Serif*: **Fraunces** for headings and signal classifications.
  - *Clean Sans-Serif*: **Inter** for form controls, labels, and readable body copy.
  - *Technical Monospace*: **JetBrains Mono** for numerical readouts, metric badges, and unit labels.
- **Two-Column Layout**:
  - **Left (60%)**: 3-stage organized form panel (Profile, Academic & Digital Habits, Lifestyle & Stress).
  - **Right (40%)**: Sticky predictive result panel featuring a responsive SVG gauge, dynamic needle hub, count-up numeric animation, and wellness categorization.
- **Micro-Interactions**:
  - Segmented stress controls with keyboard navigation.
  - Subtle noise texture overlay for an tactile paper feel.
  - Semicircular gauge with smooth `cubic-bezier` arc progression.
  - Dual pulse loading animation.

---

## Key Features

- **End-to-End Pipeline Serialization**: Preprocessing (log transformation, scaling, ordinal encoding, one-hot encoding) and regression are unified into a single persistent scikit-learn `Pipeline` (`mental_health_pipeline.pkl`).
- **Cardinality Reduction**: Automatically handles high-cardinality country values by identifying the top 10 historical countries and mapping all remaining inputs to `"Other"` without breaking inference.
- **Custom Client-Side & Server-Side Validation**: Immediate, inline feedback for all 12 inputs; server-side Pydantic models reject invalid inputs with structured HTTP 422 messages.
- **Live Model Performance Inspector**: Clickable header badge displaying live test metrics computed directly from held-out test records.
- **Zero-Storage Privacy Architecture**: User submissions are evaluated in memory and discarded immediately; no personally identifiable data or health records are persisted.

---

## System Architecture

```
┌────────────────────────────────────────────────────────┐
│                   Web Client / Frontend                │
│  - Vanilla HTML5 / CSS3 / ES6 Javascript               │
│  - Segmented Controls, Dynamic SVG Semicircular Gauge   │
│  - Live Form Validation & State Transitions            │
└───────────────────────────┬────────────────────────────┘
                            │ POST /predict (JSON)
                            ▼
┌────────────────────────────────────────────────────────┐
│                   FastAPI Backend                      │
│  - Pydantic Schema Validation (12 Fields)              │
│  - Country Grouping Normalization                      │
│  - CORS Middleware & Static Asset Serving              │
└───────────────────────────┬────────────────────────────┘
                            │ Raw Input DataFrame
                            ▼
┌────────────────────────────────────────────────────────┐
│             Fitted Scikit-Learn Pipeline               │
│  ┌──────────────────────────────────────────────────┐  │
│  │ ColumnTransformer Preprocessor                   │  │
│  │  - Study_Hours: log1p + StandardScaler           │  │
│  │  - Numeric Features: StandardScaler              │  │
│  │  - Stress_Level: OrdinalEncoder (Explicit Order) │  │
│  │  - Categorical: OneHotEncoder (ignore unknowns)  │  │
│  └──────────────────────────┬───────────────────────┘  │
│                             │ Preprocessed Array       │
│                             ▼                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Tuned RandomForestRegressor (n_estimators=300)    │  │
│  └──────────────────────────────────────────────────┘  │
└───────────────────────────┬────────────────────────────┘
                            │ Predicted Score [0.0 - 10.0]
                            ▼
┌────────────────────────────────────────────────────────┐
│           JSON Response & UI Score Display             │
│  - Score Count-up & Semicircular Gauge Arc Transition  │
│  - Band & Context: Strained / Balanced / Strong        │
└────────────────────────────────────────────────────────┘
```

---

## Dataset & Empirical Schema

The model is trained on the dataset: **`Student Social Media And Mental Health Impact.csv`** (5,000 student survey records).

### Target Variable
- **`Mental_Health_Score`**: Continuous numerical score ranging from `3.60` to `9.40` (mean: `6.23`, standard deviation: `1.28`).

### Input Features
| Category | Feature Name | Input Type | Allowed / Expected Range | Transformation |
| :--- | :--- | :--- | :--- | :--- |
| **Profile** | `Age` | Integer | 10 – 100 years | `StandardScaler` |
| | `Gender` | Categorical | `Male`, `Female` | `OneHotEncoder` |
| | `Country` | String | Open text / Top 10 + `Other` | Grouped + `OneHotEncoder` |
| **Habits** | `Academic_Level` | Categorical | `High School`, `Undergraduate`, `Graduate` | `OneHotEncoder` |
| | `Most_Used_Platform` | Categorical | 12 Platforms (`Instagram`, `TikTok`, `YouTube`, etc.) | `OneHotEncoder` |
| | `Purpose_Of_Use` | Categorical | `Entertainment`, `Education`, `Networking`, `News` | `OneHotEncoder` |
| | `Avg_Daily_Usage_Hours` | Float | 0.0 – 24.0 hours | `StandardScaler` |
| | `Daily_Unlocks` | Integer | $\ge 0$ unlocks | `StandardScaler` |
| **Stress** | `Study_Hours` | Float | 0.0 – 24.0 hours | `log1p` + `StandardScaler` |
| | `Physical_Activity_Hours` | Float | 0.0 – 24.0 hours | `StandardScaler` |
| | `Sleep_Hours_Per_Night` | Float | 0.0 – 24.0 hours | `StandardScaler` |
| | `Stress_Level` | Ordinal | `Low` < `Medium` < `High` < `Very High` | `OrdinalEncoder` |

---

## Machine Learning Methodology

1. **Country Grouping**: The dataset contains 111 unique countries. The top 10 most frequent countries (`Other`, `India`, `USA`, `Canada`, `Australia`, `UK`, `Germany`, `Mexico`, `Turkey`, `France`) are preserved, while all other countries are categorized into `"Other"`, preventing high-cardinality overfitting.
2. **Train/Test Split**: Clean 70/30 split (3,500 training rows, 1,500 test rows) using a fixed random seed (`42`).
3. **Preprocessing Pipeline**: Fitted exclusively on training data using `ColumnTransformer` to avoid data leakage.
4. **Model Selection**: Evaluated Linear Regression baseline, default Random Forest, and a hyperparameter-tuned Random Forest via 5-fold cross-validated `RandomizedSearchCV`.

---

## Actual Model Evaluation Metrics

> **Integrity Notice**: The metrics below are computed directly on the untouched 30% test set (1,500 rows). No numbers are fabricated or artificially inflated.

| Model | Training R² | Test R² | Test MAE | Test RMSE | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression (Baseline)** | 0.7181 | 0.7530 | 0.5132 | 0.6534 | Baseline |
| **Random Forest (Default)** | 0.9816 | 0.8820 | 0.3391 | 0.4517 | Candidate |
| **Random Forest (Tuned)** | **0.9823** | **0.8826** | **0.3380** | **0.4505** | **Active Pipeline** |

- **Best Parameters**: `n_estimators: 300, min_samples_split: 2, min_samples_leaf: 1, max_depth: None`.
- **Mean Absolute Error (MAE)**: 0.338 points on a 10-point scale.
- **Root Mean Squared Error (RMSE)**: 0.4505 points.
- **Coefficient of Determination (R²)**: 0.8826 (explains 88.26% of variance on unseen test data).

---

## Directory Structure

```
Mental-Health-Prediction/
├── backend/
│   ├── app.py                     # FastAPI application & prediction router
│   ├── train.py                   # Data ingestion, pipeline training & evaluation
│   ├── test_app.py                # Unit test suite for API endpoints
│   ├── test_integration.py        # End-to-end integration test suite
│   ├── requirements.txt           # Backend python dependencies
│   ├── data/
│   │   └── dataset.csv            # 5,000-row student wellness dataset
│   └── model/
│       ├── mental_health_pipeline.pkl  # Trained scikit-learn pipeline
│       └── metrics.json           # Genuine computed evaluation metrics
│
├── frontend/
│   ├── index.html                 # Semantic HTML5 layout & accessible form
│   ├── style.css                  # Editorial aesthetic & SVG gauge styling
│   └── script.js                  # Dynamic API resolution, gauge math & validation
│
├── Dockerfile                     # Container deployment specification
├── Procfile                       # PaaS process file (Render / Railway / Heroku)
├── requirements.txt               # Root python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Complete documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.11 and 3.13)
- `pip` package manager

### 1. Clone & Navigate
```bash
git clone <repository-url>
cd Mental-Health-Prediction
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Training the Model

The pre-trained pipeline and metrics are pre-packaged in `backend/model/`. If you wish to re-train the pipeline from scratch on the dataset:

```bash
python3 backend/train.py
```

This will:
- Load `backend/data/dataset.csv`.
- Execute the train/test split.
- Train Linear Regression, Random Forest, and perform 5-fold CV hyperparameter search.
- Save the best pipeline to `backend/model/mental_health_pipeline.pkl`.
- Output genuine metrics to `backend/model/metrics.json`.

---

## Running Locally

### Option A: Via npm (Recommended)
```bash
npm run dev
```

### Option B: Unified Server via Python / Uvicorn
FastAPI automatically serves the frontend at the root URL:

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
```
http://127.0.0.1:8000/
```

### Option C: Separate Frontend Static Server
If you prefer running the frontend independently:

1. Start the backend:
   ```bash
   uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
   ```
2. In a second terminal, serve the frontend:
   ```bash
   python3 -m http.server 3000 --directory frontend
   ```
3. Open `http://localhost:3000/`. The frontend's `script.js` automatically connects to `http://127.0.0.1:8000/predict`.

---

## API Reference

### Health Check
`GET /api/health`

**Response (`200 OK`):**
```json
{
  "status": "healthy",
  "model_ready": true,
  "metrics_summary": {
    "MAE": 0.338,
    "RMSE": 0.4505,
    "R2": 0.8826
  }
}
```

---

### Prediction Endpoint
`POST /predict`

**Request Body (`application/json`):**
```json
{
  "age": 21,
  "gender": "Male",
  "country": "India",
  "academic_level": "Undergraduate",
  "most_used_platform": "Instagram",
  "purpose_of_use": "Entertainment",
  "avg_daily_usage_hours": 5.5,
  "daily_unlocks": 60,
  "study_hours": 4.0,
  "physical_activity_hours": 1.0,
  "sleep_hours_per_night": 7.0,
  "stress_level": "Medium"
}
```

**Success Response (`200 OK`):**
```json
{
  "success": true,
  "score": 6.0,
  "predicted_mental_health_score": 6.0,
  "band": "Signal: balanced",
  "context": "Your rhythm looks fairly steady, with some room to recover and reset.",
  "scale_min": 0.0,
  "scale_max": 10.0,
  "disclaimer": "This score is a machine-learning estimate based on habit patterns. It is not a clinical assessment or medical diagnosis."
}
```

**Validation Error Response (`422 Unprocessable Entity`):**
```json
{
  "success": false,
  "error": "Input validation failed. Please check the submitted fields.",
  "detail": [...],
  "field_errors": [
    {
      "field": "age",
      "message": "Input should be less than or equal to 100",
      "type": "less_than_equal"
    }
  ]
}
```

---

## Testing Suite

Run the automated test suite with pytest:

```bash
python3 -m pytest backend/test_app.py backend/test_integration.py -v
```

### Test Coverage Highlights:
- Root and static file delivery (`index.html`, `style.css`, `script.js`).
- Health and model metrics endpoints.
- Valid prediction requests and score clamping ($0.0 \le \text{score} \le 10.0$).
- All 12 supported social platforms.
- All 4 stress levels (`Low`, `Medium`, `High`, `Very High`).
- Handling of unlisted countries via `"Other"` bucket.
- Rejection of invalid categorical choices and out-of-bounds numbers.

---

## Deployment Options

### Docker Deployment
```bash
# Build the container image
docker build -t mental-health-signal .

# Run container on port 8000
docker run -p 8000:8000 mental-health-signal
```

### Render / Railway Deployment
1. Connect this repository to Render or Railway.
2. Select **Web Service**.
3. Set build command: `pip install -r requirements.txt`.
4. Set start command: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`.

---

## Privacy & Ethical Standards

1. **No Data Storage**: The application does not store, log, or track user submissions in any database or file.
2. **Sanitized Logging**: Server logs record only system status and prediction latency without printing personal responses or sensitive wellness data.
3. **No Medical Claims**: Mental health is multifaceted and deeply personal. Statistical correlations between digital usage, study duration, and self-reported stress should never replace qualified clinical guidance.

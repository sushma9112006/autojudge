# Problem Difficulty Predictor

## Overview

The **Problem Difficulty Predictor** is a machine learning system that estimates the difficulty of competitive programming problems using **only their textual descriptions**.

The system predicts:

* **Difficulty Class**: Easy, Medium, or Hard
* **Difficulty Score**: A numerical value between **1 and 10**

A **Streamlit-based web application** allows users to paste a problem statement and receive predictions.

---

## Demo Link

[https://drive.google.com/file/d/1icXKXm__qya4ig9LF2SR0-4O47w9ToeC/view?usp=drivesdk](https://drive.google.com/file/d/1icXKXm__qya4ig9LF2SR0-4O47w9ToeC/view?usp=drivesdk)

### Application Interface

![App UI](https://github.com/user-attachments/assets/535ddc71-bedd-4631-8f33-1669dd1a6817)

---

## Live Application

[https://sushma9112006-autojudge-app-f60sdv.streamlit.app/](https://sushma9112006-autojudge-app-f60sdv.streamlit.app/)

---

## Dataset

**Source**:
[https://github.com/AREEG94FAHAD/TaskComplexityEval-24](https://github.com/AREEG94FAHAD/TaskComplexityEval-24)

**Format**: JSON Lines (JSONL)

**Available Labels**:

* `problem_class`: Easy / Medium / Hard
* `problem_score`: Numeric difficulty score

---

## Data Preprocessing

The following preprocessing steps are applied to ensure clean and consistent input:

* All textual fields are merged into a single `full_text` field
* Unicode normalization and symbol standardization
* Sentence-level deduplication within each problem
* Removal of duplicate problems based on cleaned text
* Verification that no missing values exist

These steps help reduce noise and improve model robustness.

---

## Feature Engineering

### Text-Based Features

* TF-IDF representation using **unigrams and bigrams**
* Sublinear term frequency scaling
* Vocabulary pruning using document frequency thresholds

### Numeric and Heuristic Features

* Length of the problem description
* Count of mathematical and logical symbols
* Binary indicators for common algorithmic terms
  (e.g., DP, BFS, DFS, graphs, greedy, etc.)

All features are combined using a **ColumnTransformer**.

---

## Model Design

The system follows a **two-stage classification pipeline**, followed by **class-based score calibration**.

```
Input Text
   ↓
TF-IDF + Numeric Features
   ↓
Stage 1: Hard vs Not-Hard Classifier
   ├── Hard
   └── Not-Hard
          ↓
     Stage 2: Easy vs Medium Classifier
          ↓
    Final Difficulty Class
          ↓
   Global Score Regressor
          ↓
   Class-Based Score Calibration
```

---

## Classification Models

### Stage 1: Hard vs Not-Hard

* **Model**: Random Forest Classifier
* **Output**: Probability of the problem being Hard
* **Threshold Selection**: 5-fold cross-validation
* **Final Threshold**: `0.5384`

This stage prioritizes reliable detection of Hard problems.

---

### Stage 2: Easy vs Medium

* Applied only when Stage 1 predicts *Not-Hard*
* **Model**: Linear Support Vector Classifier (Linear SVC)
* Uses class weighting to handle imbalance

Final output class ∈ **Easy, Medium, Hard**

---

## Difficulty Score Prediction

The difficulty score is predicted using a **single global regression model**.

| Component       | Model            |
| --------------- | ---------------- |
| Score Regressor | Ridge Regression |

The regressor is trained on all samples using the same feature pipeline as classification.

### Score Calibration

To maintain semantic consistency between predicted class and score, the raw regression outputs are clipped based on the final predicted class:

* **Easy**: ≤ 2.8
* **Medium**: 2.9 – 5.5
* **Hard**: 5.6 – 10.0

This calibration compensates for dataset imbalance and overlapping score distributions.

---

## Model Evaluation

### Classification Results

* **Overall Accuracy**: 51.8%

**Confusion Matrix**
(Rows = true labels, Columns = predicted labels)
Order: Easy, Hard, Medium

```
[[ 80  20  53]
 [ 39 210 140]
 [ 41 104 136]]
```

---

### Regression Results (Test Set Only)

* **MAE**: 1.65
* **RMSE**: 2.07

No regression model is trained on test data.

---

## Saved Models

The application uses the following pre-trained components:

* Stage 1 Hard classifier
* Frozen hard-decision threshold
* Stage 2 Easy/Medium classifier
* Ridge-based score regression model

---

## Running the Project Locally (Windows & Linux)

### 1. Clone the Repository

```bash
git clone https://github.com/sushma9112006/autojudge
cd autojudge
```

---

### 2. Create a Virtual Environment (Recommended)

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell / CMD)**

```powershell
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the Streamlit Application

**Linux / macOS**

```bash
streamlit run app.py
```

**Windows (recommended)**

```powershell
python -m streamlit run app.py
```

The application will open at:

```
http://localhost:8501
```

---

## Optional: AI-Based Input Validation

An optional validation layer uses **Google Gemini** to check whether the input resembles a programming problem.

* Runs **before** the ML pipeline
* Filters clearly invalid inputs (e.g., greetings, random text)
* Does **not** influence difficulty classification or scoring
*You can proceed even when the AI predicts the problem  as invalid
If no API key is provided, the system runs fully offline.

### Getting a Gemini API Key

1. Visit [https://aistudio.google.com/](https://aistudio.google.com/)
2. Sign in with a Google account
3. Generate an API key
4. Set it as an environment variable

**Windows (PowerShell)**

```powershell
setx GEMINI_API_KEY "your_api_key_here"
```

**Linux / macOS**

```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Install Required Package

```bash
pip install google-generativeai
```

---
* **Name**: Charita Sai Sushma J
* **Project**: Autojudge
---

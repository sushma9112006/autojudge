# Problem Difficulty Predictor

## Project Overview
This project predicts the difficulty of competitive programming problems using only their textual descriptions.

The system predicts:
- **Difficulty Class**: Easy, Medium, or Hard  
- **Difficulty Score**: A numeric score between 1 and 10  

A Streamlit web application is provided where users can paste a problem statement and receive instant predictions.

---

## Demo Screenshots

**App Interface**

![App UI](https://github.com/user-attachments/assets/535ddc71-bedd-4631-8f33-1669dd1a6817)

---

## Live Deployed Application
You can access the deployed application here:

https://sushma9112006-autojudge-app-f60sdv.streamlit.app/

---

## Dataset Used
**Source**: https://github.com/AREEG94FAHAD/TaskComplexityEval-24  
**Format**: JSONL  

**Labels**:
- `problem_class` (Easy, Medium, Hard)
- `problem_score` (numeric difficulty score)

**Preprocessing Steps**:
- All textual fields were combined into a single text field.
- Duplicate problem statements were removed.
- No duplicate entries with conflicting labels were found.
- No missing (null) values were present in the dataset.

---

## Feature Extraction

### Text Features
- TF-IDF with unigrams and bigrams
- Sublinear term frequency scaling
- Vocabulary filtered using document frequency thresholds

### Numeric Features
- Length of the problem text
- Count of mathematical symbols
- Binary presence of common algorithmic keywords such as DP, graphs, BFS, DFS, greedy, etc.

---

## Model Architecture

Classification is performed using a two-stage approach.

### Stage 1: Hard vs Not-Hard
- **Model**: Random Forest Classifier  
- **Output**: Probability of being Hard  
- **Decision threshold**: Fixed using 5-fold cross-validation  
- **Final threshold value**: 0.5343  

### Stage 2: Easy vs Medium
- Applied only if the problem is not classified as Hard  
- **Model**: Linear Support Vector Classifier with class weights  

The final output is one of: **Easy**, **Medium**, or **Hard**.

---

## Regression Model
- **Model**: Histogram-based Gradient Boosting Regressor  
- Predicts a continuous difficulty score  
- Output score is clipped between **1 and 10**

---

## Evaluation Metrics

### Classification Performance
- **Overall accuracy**: 51%

**Confusion Matrix**  
(rows = true labels, columns = predicted labels)  
Order: Easy,Hard,Medium

```

[[ 81  20  52 ]
[ 41 211 137 ]
[ 40 113 128 ]]

````

### Regression Performance
- **Mean Absolute Error (MAE)**: 1.66  
- **Root Mean Squared Error (RMSE)**: 1.99  

---

## Saved Trained Models
The following trained model files are included:
- `stage1_hard_classifier.pkl`
- `hard_threshold.pkl`
- `stage2_easy_medium.pkl`
- `score_regressor.pkl`

---

## Web Application

The Streamlit web application allows users to:
- Paste a full coding problem description
- Optionally include input and output formats
- Receive the predicted difficulty class and difficulty score

---

## Steps to Run the Project Locally

### 1. Clone the repository
```bash
git clone https://github.com/sushma9112006/autojudge
cd autojudge
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Ensure the trained model files are present in the project root

```
stage1_hard_classifier.pkl
hard_threshold.pkl
stage2_easy_medium.pkl
score_regressor.pkl
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

---

## Optional: AI-Based Input Validation

This project optionally supports an AI-based validation step using **Google Gemini**.
This feature only checks whether the input resembles a valid coding problem and does **not** affect difficulty prediction.

### How to Enable

1. Install the optional dependency:

```bash
pip install google-generativeai
```

2. Visit [https://aistudio.google.com/](https://aistudio.google.com/)
3. Sign in with a Google account.
4. Generate a new API key.
5. Paste the key into the **“Gemini API Key (Optional)”** field in the application sidebar.

If no API key is provided or the library is not installed, the application runs entirely using local machine learning models.

---

## Demo Video


---

## Author Details

**Name**: Charita Sai Sushma J
**Project**: Problem Difficulty Predictor
**Technologies**: Python, scikit-learn, Streamlit

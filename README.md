# AutoJudge – Programming Problem Difficulty Predictor

AutoJudge is a machine learning–based system that predicts the difficulty of programming problems based on their textual description.  
The project outputs:
- A difficulty class: Easy / Medium / Hard
- A numerical difficulty score on a scale of 1–10

The system is deployed as an interactive Streamlit web application and runs completely locally.

---

Project Overview
Difficulty estimation of programming problems is a subjective and non-trivial task. AutoJudge aims to automate this process by analyzing problem statements using natural language processing and machine learning techniques.

The project combines text-based features (TF-IDF) with handcrafted numerical features to estimate difficulty levels in a consistent and explainable manner.

---

Dataset Used
- Source: Custom dataset (`dataset.csv`)
- Contents:
  - Problem Title
  - Problem Description
  - Input Description
  - Output Description
  - Difficulty Class (Easy / Medium / Hard)
  - Difficulty Score (1–10)

The dataset is preprocessed and used to train both classification and regression models.

---

Approach and Models Used:

Data Preprocessing
- Missing values handled using empty string filling
- All text fields combined into a single textual input

Feature Extraction
1. TF-IDF Vectorization  
   - Converts textual problem statements into numerical vectors
2. Handcrafted Features  
   - Length of problem text  
   - Count of mathematical symbols  
   - Presence of algorithmic keywords (DP, graph, tree, BFS, DFS, etc.)
3. Numeric features are scaled before model training

Models Used
- Classification Model:  
  Logistic Regression  
  Predicts difficulty class (Easy / Medium / Hard)

- Regression Model:  
  Gradient Boosting Regressor  
  Predicts difficulty score (1–10)

---

## Evaluation Metrics

### Classification
- Accuracy (computed during training)-0.48

### Regression
- Mean Absolute Error (MAE)-1.76
- Root Mean Squared Error (RMSE)-2.05

These metrics are used to evaluate how close the predicted difficulty score is to the ground-truth score.

---

Steps to Run the Project Locally

1. Clone the Repository
git clone https://github.com/AshiAg13/AutoJudge.git
cd AutoJudge

2. Install Required Libraries
pip install streamlit scikit-learn pandas numpy scipy joblib

3. Run the Streamlit App
streamlit run autojudge_app.py

---

Web Interface Explanation
The project includes a Streamlit-based web UI that allows users to:
1. Enter:
   - Problem Title
   - Problem Description
   - Input Description
   - Output Description
2. Click Predict Difficulty
3. View:
   - Predicted Difficulty Class
   - Predicted Difficulty Score (1–10)

The web app loads the trained models and performs real-time inference locally.

---

Link to a 2–3 minute demo video:  <>


---

ASHI AGRAWAL
24118012
METALLURGICAL AND MATERIALS ENGINEERING


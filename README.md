# XGBoostShotClassifier

Project Overview

This project develops an XGBoost-based model to predict shot outcomes in the NBA using shot-level data from the 2015–2016 season. The model estimates the probability of making each shot based on player identity, team, shot type, and court zone. These probabilities are weighted by shot value (2 or 3 points) to generate expected points tables for players across all shot types and court areas.

The primary goal is to assess the reliability of XGBoost for predicting shot success and to derive actionable insights for coaches and analysts on high- and low-value shooting decisions.

Data

Source: NBA shot-level data, 2015–2016 season

Features Used:

Player identity

Team

Shot type (layup, jump shot, etc.)

Court zone (restricted area, mid-range, three-point)

Target Variable: SHOT_MADE_FLAG (1 = made, 0 = missed)

Preprocessing:

Categorical variables are one-hot encoded

Backcourt shots removed for relevance

Only shots with sufficient frequency considered in final tables

Methods

Model: XGBoost Classifier

Gradient boosting framework: each tree corrects errors of previous trees

Probability output for each shot

Pipeline:

One-hot encoding for categorical variables

Randomized hyperparameter search with 3-fold cross-validation

Metrics Evaluated:

Classification: Accuracy, Precision, Recall, F1 Score

Probability-based: Log Loss, Brier Score, AUC-ROC

Calibration curves to assess probability reliability

Output:

Expected points per player per shot type and court zone

Comparison with true shooting percentages

Usage

Clone the repository:

git clone <repository_url>


Install dependencies (Python 3.10+ recommended):

pip install -r requirements.txt


Run the model training and evaluation notebook:

jupyter notebook NBA_XGBoost_Shot_Recommendation.ipynb


Explore generated tables and plots for league-wide and team-level analyses.

Key Findings

League Level:

Predicted shot success closely matches actual success (45.46% vs. 45.36%)

Accuracy: 65%, F1: 0.557, Precision: 66.5%, Recall: 48%

Probability metrics show improvements over baseline: Log Loss 0.6213, Brier Score 0.2167, AUC-ROC 0.693

Team Level (Washington Wizards):

Predicted success: 47.15%, actual success: 46.18%

Accuracy: 73.8%, F1: 0.687, Precision: 76.5%, Recall: 62.4%

Probability metrics show significant improvement: Log Loss 0.5593, Brier Score 0.1878, AUC-ROC 0.778

Insights:

Model well-calibrated and more effective for team-specific predictions

Identifies high-value and low-value shooting zones for strategic decision-making

File Structure
NBA_XGBoost_Shot_Recommendation/
│
├─ data/                  # Raw and processed shot-level data
├─ notebooks/             # Jupyter notebooks with analysis and model training
├─ src/                   # Python scripts for model pipeline and evaluation
├─ figures/               # Plots and heatmaps of shot metrics
├─ requirements.txt       # Python dependencies
└─ README.md

Dependencies

pandas

numpy

scikit-learn

xgboost

seaborn

matplotlib

References

NBA shot-level data, 2015–2016 season

XGBoost Documentation: https://xgboost.readthedocs.io

Basketball analytics literature on expected points and shot success modeling


<img width="1337" height="1116" alt="Sheet 1 (8)" src="https://github.com/user-attachments/assets/8fc98f06-9ac0-4925-be55-07466ba2de11" />

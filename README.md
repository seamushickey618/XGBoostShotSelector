# XGBoostShotClassifier

**Project Overview**

This project develops an XGBoost-based model to predict shot outcomes in the NBA using shot-level data from the 2015–2016 season. The model estimates the probability of making each shot based on player identity, team, shot type, and court zone. These probabilities are weighted by shot value (2 or 3 points) to generate expected points tables for players across all shot types and court areas.

The primary goal is to assess the reliability of XGBoost for predicting shot success and to derive insights for coaches and analysts on high- and low-value shot selection

**Data**

**Source**: NBA shot-level data, 2015–2016 season


**Features Used:**

Player 
Team
Shot type 
Court zone 

Target Variable: SHOT_MADE_FLAG 


**Preprocessing:**

Categorical variables are one-hot encoded
Backcourt shots removed for relevance
Only shots with sufficient frequency considered in final table


**Model: XGBoost Classifier**

Gradient boosting framework: each tree corrects errors of previous trees
Probability output for each shot

Stores in XGModel.py module, for functions including: tuning hyper-parameters and running the model (train_xgb_tuned), running the model with fixed hyperparameters (league_xgb_tuned) and ranking teams based on model accuracy score (make_team_ranking_report)

**Pipeline:**

One-hot encoding for categorical variables
Randomized hyperparameter search with 3-fold cross-validation


Metrics Evaluated: Accuracy, Confusion Matrices


**Output:**

Expected points per player per shot type and court zone
Comparison with true shooting percentages


Usage

Clone the repository:
git clone https://github.com/seamushickey618/XGBoostShotSelector/tree/main


Install dependencies:
pip install -r requirements.txt


Run the model training and evaluation notebook:
jupyter notebook NBA_XGBoost_Shot_Recommendation.ipynb


Dependencies

pandas
numpy
scikit-learn
xgboost
seaborn
matplotlib


End Report:

<img width="1337" height="1116" alt="Sheet 1 (8)" src="https://github.com/user-attachments/assets/8fc98f06-9ac0-4925-be55-07466ba2de11" />

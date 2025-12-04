# XGBoostShotClassifier

**Project Overview**

This project develops an XGBoost-based model to predict shot outcomes in the NBA using shot-level data from the 2015–2016 season. The model estimates the probability of making each shot based on player identity, team, shot type, and court zone. These probabilities are weighted by shot value (2 or 3 points) to generate expected points tables for players across all shot types and court areas.

The primary goal is to assess the reliability of XGBoost for predicting shot success and to derive insights for coaches and analysts on high- and low-value shot selection

**Data**

**Source**: https://github.com/wyatt-ai/nba-movement-data/tree/master/data/shots : `shots.csv`

NBA shot-level data, 2015–2016 season

**Features Used:**

- Player 
- Team
- Shot type 
- Court zone 


**Target Variable:** Shot made / missed


**Preprocessing:**

- Categorical variables are one-hot encoded
- Backcourt shots removed for relevance
- Only shots with sufficient frequency considered in final table


**Model: XGBoost Classifier & Module**

- Gradient boosting framework: each tree corrects errors of previous trees
Probability output for each shot

- *`XGModel.py` module*, stores functions including: tuning hyperparameters and running the model `train_xgb_tuned`, running the model with fixed hyperparameters `league_xgb_tuned` and ranking teams based on model accuracy score `make_team_ranking_report`

**Pipeline:**

- One-hot encoding for categorical variables
- Randomized hyperparameter search with 3-fold cross-validation


**Metrics Evaluated:** Accuracy, Confusion Matrices


**Output:**

- Expected points per player per shot type and court zone
- Comparison with true shooting percentages
- `wiz_full_analysis.csv` contains results from all Wizards shots taken over 25 times

**Other softwares used in project:** 
- Excel: used to isolate the results for the starting 5 players after full analysis stage
- Tableau: used for the visualization of the final report

**Use of ChatGPT:** 
- Guidance on model structure and navigating libraries `scikit-learn` and `xgboost`
- Troubleshooting syntax 
- Visualizing confusion matrices in `seaborn`

**Usage**

Clone the repository:
`git clone https://github.com/seamushickey618/XGBoostShotSelector/tree/main`


Install dependencies:
`pip install -r requirements.txt`


Run the model training and evaluation notebook:
`jupyter notebook Main.ipynb`


Dependencies

`pandas
numpy
scikit-learn
xgboost
seaborn
matplotlib`


Final Report Visualization:

<img width="1274" height="1024" alt="Sheet 4" src="https://github.com/user-attachments/assets/f115d9b7-142e-4192-8976-a7794aea01b3" />


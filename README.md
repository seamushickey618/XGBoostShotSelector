# XGBoostShotClassifier

**Project Overview**

This project develops an XGBoost-based model to predict shot outcomes in the NBA using shot-level data from the 2015–2016 season. The model estimates the probability of making each shot based on player , team, shot type, and court zone. These probabilities are weighted by shot value (2 or 3 points) to generate expected points tables for players across all shot types and court areas.

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


**`XGModel.py`** module stores functions including: tuning hyperparameters and running the model `train_xgb_tuned`, running the model with fixed hyperparameters `league_xgb_tuned` and ranking teams based on model accuracy score `make_team_ranking_report`


**Metrics Evaluated:** Accuracy, Confusion Matrices


**Output:**

- Expected points per player per shot type and court zone
- Comparison with true shooting percentages
- `wiz_full_analysis.csv` contains results from running the `Main.ipynb` notebook

**Other softwares used in project:** 
- Excel: used to isolate the results for the starting 5 players from `wiz_full_analysis.csv`
- Tableau: used for the visualization of the final report

**Use of ChatGPT:** 
- Guidance on model structure 
- Assistance navigating libraries `scikit-learn` and `xgboost`
- Troubleshooting syntax 
- Visualizing confusion matrices in `seaborn`
  
**Usage**

Clone the repository:
`git clone https://github.com/seamushickey618/XGBoostShotSelector/tree/main`


Install dependencies:
`pip install -r Requirements.txt`


Run the notebook:
`jupyter notebook Main.ipynb`


Dependencies

`pandas
numpy
scikit-learn
xgboost
seaborn
matplotlib
jupyter`


Final Report Visualization made in Tableau using the `wiz_full_analysis.csv` filtered to include only the starting 5 players

<img width="1274" height="1024" alt="Sheet 4" src="https://github.com/user-attachments/assets/f115d9b7-142e-4192-8976-a7794aea01b3" />


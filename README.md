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

<img width="1337" height="1116" alt="Sheet 1 (8)" src="https://github.com/user-attachments/assets/8fc98f06-9ac0-4925-be55-07466ba2de11" />

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss, brier_score_loss, roc_auc_score, make_scorer
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*use_label_encoder.*")
import pandas as pd


# --------------------------
# TRAIN + TUNE XGBoost Model
# --------------------------
def train_xgb_tuned(shots, features, target, param_grid):


    # --------------------------
    # Prepare data
    # --------------------------
    X = shots[features]
    y = shots[target]

    # Identify and one-hot encode categorical features
    cat_features = [c for c in X.columns if X[c].dtype == "object"]
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
        remainder="passthrough"
    )

    # --------------------------
    # Pipeline
    # --------------------------
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("xgb", XGBClassifier(eval_metric="logloss", n_jobs=1, random_state=42))
    ])

    # --------------------------
    # Train/test split
    # --------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --------------------------
    # Randomized hyperparameter search
    # --------------------------
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=20,
        scoring=make_scorer(f1_score),
        cv=3,
        verbose=1,
        n_jobs=1,
        random_state=42
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print("\nBest Hyperparameters:")
    print(search.best_params_)

    # --------------------------
    # Predictions
    # --------------------------
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    # --------------------------
    # Standard metrics
    # --------------------------
    accuracy = best_model.score(X_test, y_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # --------------------------
    # Probability-based metrics
    # --------------------------
    logloss = log_loss(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    # --------------------------
    # Calibration Curve
    # --------------------------
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy='uniform')

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='XGBoost')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --------------------------
    # Overall predicted vs actual shot success
    # --------------------------
    predicted_success_rate = np.mean(y_proba)
    actual_success_rate = np.mean(y_test)


    # --------------------------
    # Print metrics with interpretation
    # --------------------------
    print(f"\nPredicted Overall Shot Success: {predicted_success_rate:.10f}")
    print(f"Actual Overall Shot Success: {actual_success_rate:.10f}")
    print(f"\nAccuracy: {accuracy:.3f} | F1 Score: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")
    print(f"Log Loss: {logloss:.4f} ")
    print(f"Brier Score: {brier:.4f} ")
    print(f"AUC-ROC: {auc:.3f} ")
    print("Calibration curve indicates probability reliability.\n"
          "Points on diagonal = well-calibrated; above = underestimates; below = overestimates.")

    return best_model, X_test, y_test, y_proba, accuracy, f1, precision, recall, logloss, brier, auc, predicted_success_rate, actual_success_rate


# --------------------------
# Evaluate Model Performance on Teams
# --------------------------

def make_team_ranking_report(shots, features, model):
    teams = shots["TEAM_NAME"].unique()
    metrics = []

    for team in teams:
        team_df = shots[shots["TEAM_NAME"] == team]
        X_team = team_df[features]
        y_true = team_df["SHOT_MADE_FLAG"]
        y_pred_proba = model.predict_proba(X_team)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_proba)
        ll = log_loss(y_true, y_pred_proba)
        br = brier_score_loss(y_true, y_pred_proba)

        metrics.append([team, acc, f1, auc, ll, br])

    df = pd.DataFrame(metrics, columns=["TEAM_NAME", "Accuracy", "F1", "AUC", "LogLoss", "Brier"])


    # -------------------------
    # Arrange metrics and ranks together
    # -------------------------
    cols_order = [
        "TEAM_NAME",
        "Accuracy", 
        "F1", 
        "AUC", 
        "LogLoss", 
        "Brier", 
    ]
    
    # Format metrics to 4 decimals
    df[["Accuracy","F1","AUC","LogLoss","Brier"]] = df[["Accuracy","F1","AUC","LogLoss","Brier"]].round(4)
    
    print("\n================ TEAM PERFORMANCE TABLE ================")
    print(df[cols_order].sort_values("Accuracy", ascending = False))


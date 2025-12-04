from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message=".*use_label_encoder.*")


# --------------------------
# TRAIN + TUNE XGBoost Model
# --------------------------
def train_xgb_tuned(shots, features, target, param_grid):

    # --------------------------
    # Prepare data
    # --------------------------
    X = shots[features]
    y = shots[target]

    # --------------------------
    # Identify and one-hot encode categorical features
    # --------------------------
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
        scoring=make_scorer(accuracy_score),
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

    # --------------------------
    # Overall predicted vs actual shot success
    # --------------------------
    predicted_success_rate = np.mean(y_proba)
    actual_success_rate = np.mean(y_test)

    # --------------------------
    # Confusion matrix
    # --------------------------
    cm = confusion_matrix(y_test, y_pred)

    # Normalize to percentages (overall)
    cm_norm = cm / cm.sum()

    # Extract raw counts
    tn, fp, fn, tp = cm.ravel()

    # Compute conditional frequencies
    tn_freq = tn / (tn + fp)
    tp_freq = tp / (tp + fn)
    fn_freq = fn / (fn + tp)
    fp_freq = fp / (fp + tn)

    # Create labeled matrix for annotation (showing proportion + frequency)
    labels = np.array([
        [f"TN\n{cm_norm[0,0]*100:.1f}%\n({tn_freq*100:.1f}%)",
         f"FP\n{cm_norm[0,1]*100:.1f}%\n({fp_freq*100:.1f}%)"],
        [f"FN\n{cm_norm[1,0]*100:.1f}%\n({fn_freq*100:.1f}%)",
         f"TP\n{cm_norm[1,1]*100:.1f}%\n({tp_freq*100:.1f}%)"]
    ])

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar_kws={"label": "Proportion"}
    )

    plt.title("Normalized Confusion Matrix with Conditional Frequencies")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0.5, 1.5], ["Miss (0)", "Make (1)"])
    plt.yticks([0.5, 1.5], ["Miss (0)", "Make (1)"], rotation=0)

    plt.tight_layout()
    plt.show()

    # --------------------------
    # Print metrics with interpretation
    # --------------------------
    print(f"\nPredicted Overall Shot Success: {predicted_success_rate:.10f}")
    print(f"Actual Overall Shot Success: {actual_success_rate:.10f}")
    print(f"\nAccuracy: {accuracy:.3f}")

    return best_model, X_test, y_test, y_proba, accuracy, predicted_success_rate, actual_success_rate

def league_xgb_tuned(shots, features, target):

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
        ("xgb", XGBClassifier(
            eval_metric="logloss",
            n_jobs=1,
            random_state=42,
            
            # --------------------------
            # FIXED HYPERPARAMETERS
            # --------------------------
            subsample=0.75,
            n_estimators=750,
            max_depth=12,
            learning_rate=0.01,
            colsample_bytree=0.5
        ))
    ])

    # --------------------------
    # Train/test split
    # --------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --------------------------
    # Train model (no hyperparameter search)
    # --------------------------
    pipeline.fit(X_train, y_train)
    best_model = pipeline  # naming consistency

    # --------------------------
    # Predictions
    # --------------------------
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    # --------------------------
    # Standard metrics
    # --------------------------
    accuracy = best_model.score(X_test, y_test)

    # --------------------------
    # Overall predicted vs actual shot success
    # --------------------------
    predicted_success_rate = np.mean(y_proba)
    actual_success_rate = np.mean(y_test)

    # --------------------------
    # Confusion matrix
    # --------------------------
    cm = confusion_matrix(y_test, y_pred)

    # Normalize to percentages (overall)
    cm_norm = cm / cm.sum()

    # Extract raw counts
    tn, fp, fn, tp = cm.ravel()

    # Compute conditional frequencies
    tn_freq = tn / (tn + fp)
    tp_freq = tp / (tp + fn)
    fn_freq = fn / (fn + tp)
    fp_freq = fp / (fp + tn)

    # Create labeled matrix for annotation (showing proportion + frequency)
    labels = np.array([
        [f"TN\n{cm_norm[0,0]*100:.1f}%\n({tn_freq*100:.1f}%)",
         f"FP\n{cm_norm[0,1]*100:.1f}%\n({fp_freq*100:.1f}%)"],
        [f"FN\n{cm_norm[1,0]*100:.1f}%\n({fn_freq*100:.1f}%)",
         f"TP\n{cm_norm[1,1]*100:.1f}%\n({tp_freq*100:.1f}%)"]
    ])

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar_kws={"label": "Proportion"}
    )

    plt.title("Normalized Confusion Matrix with Conditional Frequencies")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0.5, 1.5], ["Miss (0)", "Make (1)"])
    plt.yticks([0.5, 1.5], ["Miss (0)", "Make (1)"], rotation=0)

    plt.tight_layout()
    plt.show()


    # --------------------------
    # Print metrics
    # --------------------------
    print(f"\nPredicted Overall Shot Success: {predicted_success_rate:.10f}")
    print(f"Actual Overall Shot Success: {actual_success_rate:.10f}")
    print(f"Accuracy: {accuracy:.3f}")

    return best_model, X_test, y_test, y_proba, accuracy, predicted_success_rate, actual_success_rate

# --------------------------
# Evaluate Model Performance on Teams
# --------------------------

def make_team_ranking_report(shots, features, model):
    teams = shots["TEAM_NAME"].unique()
    metrics = []

    for team in teams:
        team_df = shots[shots["TEAM_NAME"] == team]

        # Raw features 
        X_team = team_df[features]
        y_true = team_df["SHOT_MADE_FLAG"]

        # Model predictions
        y_pred_proba = model.predict_proba(X_team)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Accuracy
        acc = accuracy_score(y_true, y_pred)

        metrics.append([team, acc])

    # Convert to DataFrame
    df = pd.DataFrame(metrics, columns=["TEAM_NAME", "Accuracy"])

    # Round
    df["Accuracy"] = df["Accuracy"].round(4)

    # Sort
    df_sorted = df.sort_values("Accuracy", ascending=False)

    print("\n TEAM ACCURACCY TABLE ")
    print(df_sorted.to_string(index=False))

    return df_sorted

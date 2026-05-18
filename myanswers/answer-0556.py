import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


def comparar_clasificadores_estratificado(X, y, n_folds=5):

    logistic_model = LogisticRegression(
        max_iter=500,
        random_state=42
    )

    tree_model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=42
    )

    logistic_scores = []
    tree_scores = []

    for train_idx, test_idx in skf.split(X, y):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Logistic Regression
        logistic_model.fit(X_train, y_train)
        y_pred_log = logistic_model.predict(X_test)

        f1_log = f1_score(
            y_test,
            y_pred_log,
            average='weighted',
            zero_division=0
        )

        logistic_scores.append(f1_log)

        # Decision Tree
        tree_model.fit(X_train, y_train)
        y_pred_tree = tree_model.predict(X_test)

        f1_tree = f1_score(
            y_test,
            y_pred_tree,
            average='weighted',
            zero_division=0
        )

        tree_scores.append(f1_tree)

    logistic_mean = round(float(np.mean(logistic_scores)), 4)
    logistic_std = round(float(np.std(logistic_scores)), 4)

    tree_mean = round(float(np.mean(tree_scores)), 4)
    tree_std = round(float(np.std(tree_scores)), 4)

    if logistic_std <= tree_std:
        modelo_mas_estable = "LogisticRegression"
    else:
        modelo_mas_estable = "DecisionTreeClassifier"

    return {
        "logistic_f1_mean": logistic_mean,
        "logistic_f1_std": logistic_std,
        "tree_f1_mean": tree_mean,
        "tree_f1_std": tree_std,
        "modelo_mas_estable": modelo_mas_estable
    }

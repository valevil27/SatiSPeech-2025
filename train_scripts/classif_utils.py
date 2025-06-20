import time
from typing import Any, Callable
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def get_classifiers(
    random_state: int, cv: int = 5, verbose: int = 1
) -> dict[str, GridSearchCV]:
    """
    Creates a dictionary of scikit-learn GridSearchCV objects for different classifiers with predefined hyperparameter grids.

    Parameters:
            random_state (int): Seed for random number generators to ensure reproducibility.
            cv (int, optional): Number of cross-validation folds to use in GridSearchCV. Default is 1.
            verbose (int, optional): Controls the verbosity of the GridSearchCV output. Default is 1.

    Returns:
            dict: A dictionary where keys are classifier names ("SVM", "LogisticRegression", "RandomForest") and values are corresponding GridSearchCV objects configured with their respective hyperparameter grids.

    Notes:
            - The classifiers included are SVM (with probability estimates), Logistic Regression, and Random Forest.
            - The scoring metric used for hyperparameter tuning is "f1_macro".
            - All GridSearchCV objects use parallel processing with n_jobs=-1.
    """
    param_grids = {
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        },
        "LogisticRegression": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
            "max_iter": [100, 200],
        },
        "RandomForest": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
    }
    return {
        "SVM": GridSearchCV(
            SVC(probability=True, random_state=random_state),
            param_grid=param_grids["SVM"],
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=verbose,
        ),
        "LogisticRegression": GridSearchCV(
            LogisticRegression(random_state=random_state),
            param_grid=param_grids["LogisticRegression"],
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=verbose,
        ),
        "RandomForest": GridSearchCV(
            RandomForestClassifier(random_state=random_state),
            param_grid=param_grids["RandomForest"],
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=verbose,
        ),
    }


def timeit(label: str, results: dict[str, Any]):
    def decorator(func: Callable):
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            rets = func(*args, **kwargs)
            duration = time.perf_counter() - start
            print(f"[TIMER] {label} took {duration:.4} seconds to run.")
            results[label]["duration"] = duration
            return rets

        return wrapper

    return decorator

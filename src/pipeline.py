from .preprocess import preprocess_data
from .models import get_models
from .tuner import tune_model
from .evaluator import evaluate_model


def run_automl(df, target_column):
    preprocessor, X_train, X_test, y_train, y_test = preprocess_data(df, target_column)

    models = get_models()

    # Parameter grids for all models
    param_grids = {
        "Logistic Regression": {"classifier__C": [0.1, 1, 10]},
        "Decision Tree": {"classifier__max_depth": [None, 5, 10]},
        "Random Forest": {"classifier__n_estimators": [50, 100], "classifier__max_depth": [None, 10]},
        "Gradient Boosting": {"classifier__n_estimators": [50, 100], "classifier__learning_rate": [0.05, 0.1]},
        "AdaBoost": {"classifier__n_estimators": [50, 100], "classifier__learning_rate": [0.5, 1]},
        "SVM": {"classifier__C": [0.1, 1], "classifier__kernel": ["rbf", "linear"]},
        "KNN": {"classifier__n_neighbors": [3, 5, 7]},
        "Extra Trees": {"classifier__n_estimators": [50, 100], "classifier__max_depth": [None, 10]}
    }

    results = []

    for name, model in models.items():
        grid = param_grids.get(name, {})  # Safety: avoid KeyError

        best_model, best_params, cv_score = tune_model(
            name, model, grid, preprocessor, X_train, y_train
        )

        accuracy, f1 = evaluate_model(best_model, X_test, y_test)

        results.append({
            "model": name,
            "best_model": best_model,
            "accuracy": accuracy,
            "f1_score": f1,
            "params": best_params
        })

    # Select best model based on F1 score
    best_result = max(results, key=lambda x: x["f1_score"])

    return best_result, results

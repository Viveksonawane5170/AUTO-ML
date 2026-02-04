from .preprocess import preprocess_data
from .models import get_models
from .tuner import tune_model
from .evaluator import evaluate_model


def run_automl(df, target_column):
    preprocessor, X_train, X_test, y_train, y_test, problem_type = preprocess_data(df, target_column)

    models = get_models(problem_type)

    param_grids = {}  # Keep empty for now or add later

    results = []

    for name, model in models.items():
        grid = param_grids.get(name, {})

        best_model, best_params, cv_score = tune_model(
            name, model, grid, preprocessor, X_train, y_train, problem_type
        )

        metrics = evaluate_model(best_model, X_test, y_test, problem_type)

        result = {
            "model": name,
            "best_model": best_model,
            "params": best_params
        }
        result.update(metrics)
        results.append(result)

    if problem_type == "classification":
        best_result = max(results, key=lambda x: x["f1_score"])
    else:
        best_result = max(results, key=lambda x: x["r2_score"])

    return best_result, results, problem_type



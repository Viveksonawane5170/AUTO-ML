from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def tune_model(name, model, param_grid, preprocessor, X_train, y_train, problem_type="classification"):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scoring = "f1_weighted" if problem_type == "classification" else "r2"

    if not param_grid:
        pipeline.fit(X_train, y_train)
        return pipeline, {}, None

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring=scoring,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

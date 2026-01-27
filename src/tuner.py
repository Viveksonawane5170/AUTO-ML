from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from collections import Counter


def tune_model(name, model, param_grid, preprocessor, X_train, y_train):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    class_counts = Counter(y_train)
    min_class_samples = min(class_counts.values())

    # If too few samples for CV, skip GridSearch
    if min_class_samples < 3:
        print(f"⚠ Skipping hyperparameter tuning for {name} (not enough samples per class)")
        pipeline.fit(X_train, y_train)
        return pipeline, {}, None

    cv_folds = min(3, min_class_samples)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_folds,
        scoring="f1_weighted",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

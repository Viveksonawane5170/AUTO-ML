from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def evaluate_model(model, X_test, y_test, problem_type="classification"):
    y_pred = model.predict(X_test)

    if problem_type == "classification":
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        return {"accuracy": accuracy, "f1_score": f1}
    else:
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        return {"mae": mae, "rmse": rmse, "r2_score": r2}

import shap
import matplotlib.pyplot as plt


def explain_model(model, X_train):
    explainer = shap.Explainer(model.named_steps["classifier"])
    shap_values = explainer(model.named_steps["preprocessor"].transform(X_train))

    return shap_values

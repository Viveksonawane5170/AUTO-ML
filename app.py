import streamlit as st
import pandas as pd
from src.pipeline import run_automl

st.set_page_config(page_title="AutoML System", layout="wide")
st.title("🔮 Intelligent AutoML System")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### 📄 Dataset Preview")
    st.dataframe(df.head())

    st.write("### 📊 Dataset Shape:", df.shape)

    target_column = st.selectbox("🎯 Select Target Column", df.columns)

    if st.button("🚀 Run AutoML"):
        if df[target_column].nunique() < 2:
            st.error("❌ Target column must have at least 2 unique values.")
        else:
            with st.spinner("Training models... Please wait ⏳"):
                try:
                    best_result, all_results, problem_type = run_automl(df, target_column)

                    st.success("✅ AutoML Completed!")

                    st.subheader("🏆 Best Model")
                    st.write(f"**Model:** {best_result['model']}")
                    st.write("**Best Parameters:**", best_result["params"])

                    st.subheader("📊 Performance Metrics")

                    if problem_type == "classification":
                        st.write(f"**Accuracy:** {best_result['accuracy']:.4f}")
                        st.write(f"**F1 Score:** {best_result['f1_score']:.4f}")
                        results_df = pd.DataFrame(all_results)[["model", "accuracy", "f1_score"]]

                    else:  # Regression
                        st.write(f"**MAE:** {best_result['mae']:.4f}")
                        st.write(f"**RMSE:** {best_result['rmse']:.4f}")
                        st.write(f"**R² Score:** {best_result['r2_score']:.4f}")
                        results_df = pd.DataFrame(all_results)[["model", "mae", "rmse", "r2_score"]]

                    st.subheader("📈 All Model Results")
                    st.dataframe(results_df)

                except Exception as e:
                    st.error(f"⚠️ Error during training: {str(e)}")








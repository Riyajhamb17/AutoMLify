import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime


def display_dataset_summary(X, y):
    st.subheader("📄 Dataset Summary")
    st.write(f"Number of rows: {X.shape[0]}")
    st.write(f"Number of features: {X.shape[1]}")
    st.write("Target column distribution:")
    st.bar_chart(y.value_counts() if y.dtype == 'object' or y.nunique() < 20 else y.describe())


def display_preprocessing_summary():
    st.subheader("🧼 Preprocessing Summary")
    st.markdown("- Missing values handled via mean/median/mode or dropped based on threshold")
    st.markdown("- Categorical encoding: LabelEncoder or One-Hot Encoding")
    st.markdown("- Numerical scaling: StandardScaler or MinMaxScaler based on skew")
    st.markdown("- Class imbalance: SMOTE or RandomOverSampler")


def display_feature_importance(importance_df):
    st.subheader("📊 Feature Importance")
    st.dataframe(importance_df)
    fig, ax = plt.subplots()
    sns.barplot(data=importance_df.head(10), x="Importance", y="Feature", ax=ax)
    st.pyplot(fig)


def evaluate_model(model, X_test, y_test, task_type):
    st.subheader("📈 Model Evaluation")
    preds = model.predict(X_test)

    if task_type == "classification":
        st.text("Classification Report:")
        report = classification_report(y_test, preds, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

    else:
        mse = mean_squared_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        st.write(f"✅ RMSE: {rmse:.4f}")
        st.write(f"✅ MAE: {mae:.4f}")
        st.write(f"✅ R² Score: {r2:.4f}")


def generate_report_dict(model_name, model_params, metrics_dict, task_type):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = {
        "Project": "AutoML Model Evaluation Report",
        "Timestamp": timestamp,
        "Model": model_name,
        "Hyperparameters": model_params,
        "Task": task_type,
        "Metrics": metrics_dict
    }
    return report

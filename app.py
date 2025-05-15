import streamlit as st
import pandas as pd
from preprocessing.preprocess_pipeline import handle_missing_values, balance_dataset, encode_and_scale_features
from eda.eda_pipeline import run_eda
from feature_engineering.feature_pipeline import (
    construct_new_features, remove_low_variance_features,
    remove_highly_correlated_features, select_important_features, scale_features
)
from sklearn.preprocessing import LabelEncoder

from model_training.training_pipeline import auto_mode, manual_mode, detect_task_type
from evaluation.evaluation_pipeline import (
    display_dataset_summary,
    display_preprocessing_summary,
    display_feature_importance,
    evaluate_model,
    generate_report_dict
)

st.title("🧠 AutoMLify Dashboard")

page = st.sidebar.selectbox("📂 Select Section", ["Upload Dataset", "Preprocessing", "EDA","Feature Engineering","Model Training and Evaluation","Evaluation"])

if page == "Upload Dataset":
    st.header("📤 Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.write("📄 Data Preview:")
        st.dataframe(df.head())

elif page == "Preprocessing":
    if "df" in st.session_state:
        df = st.session_state.df.copy()
        df_cleaned = handle_missing_values(df)
        st.session_state.df_cleaned = df_cleaned

        st.write("✅ Cleaned Dataset:")
        st.dataframe(df_cleaned.head())

        target_col = st.selectbox("🎯 Select Target Column (for balancing)", df_cleaned.columns)
        if target_col:
            X = df_cleaned.drop(columns=[target_col])
            y = df_cleaned[target_col]

            # Encode categorical target variable
            if y.dtype == 'object':
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)

            X_balanced, y_balanced = balance_dataset(X, y)
            X_encoded_scaled = encode_and_scale_features(X_balanced)

            st.session_state.X_processed = X_encoded_scaled
            st.session_state.y_processed = y_balanced

            st.success("✅ Preprocessing complete. Proceed to EDA or Modeling.")
    else:
        st.warning("Please upload a dataset first.")

elif page == "EDA":
    if "df_cleaned" in st.session_state:
        df = st.session_state.df_cleaned
        run_eda(df)
    else:
        st.warning("Please run preprocessing first.")

# elif page == "Feature Engineering":
#     if "X_processed" in st.session_state and "y_processed" in st.session_state:
#         X = st.session_state.X_processed.copy()
#         y = st.session_state.y_processed

#         X = construct_new_features(X)
#         X = remove_low_variance_features(X)
#         X = remove_highly_correlated_features(X)

#         method = st.selectbox("🎯 Feature selection method", ["mutual_info", "f_classif"])
#         k = st.slider("🔢 Number of top features to select", 5, min(20, X.shape[1]), 10)
#         X = select_important_features(X, y, method, k)

#         X_scaled = scale_features(X)

#         st.session_state.X_final = X_scaled
#         st.session_state.y_final = y

#         st.success("✅ Feature engineering complete. Ready for model training!")
#     else:
#         st.warning("Please complete preprocessing first.")
elif page == "Feature Engineering":
    if "X_processed" in st.session_state and "y_processed" in st.session_state:
        X = st.session_state.X_processed.copy()
        y = st.session_state.y_processed

        # Removed complex feature generation to avoid errors and simplify logic
        X = remove_low_variance_features(X)
        X = remove_highly_correlated_features(X)

        if X.shape[0] > 0:
            method = st.selectbox("🎯 Feature selection method", ["f_classif"])
            k = st.slider("🔢 Number of top features to select", 1, min(20, X.shape[1]), 5)
            X = select_important_features(X, y, method, k)
        else:
            st.warning("No data samples available for feature selection. Please check your preprocessing.")

        X_scaled = scale_features(X)

        st.session_state.X_final = X_scaled
        st.session_state.y_final = y

        st.success("✅ Feature engineering complete. Ready for model training!")
    else:
        st.warning("Please complete preprocessing first.")

elif page == "Model Training and Evaluation":
    if "X_final" in st.session_state and "y_final" in st.session_state:
        X = st.session_state.X_final
        y = st.session_state.y_final

        task_type = detect_task_type(y)
        mode = st.radio("Choose Training Mode", ["Auto", "Manual"])

        if mode == "Auto":
            auto_mode(X, y, task_type)
        else:
            manual_mode(X, y, task_type)
    else:
        st.warning("Please complete feature engineering before training.")
elif page == "Evaluation":
    if all(k in st.session_state for k in ["model", "X_test", "y_test", "task_type"]):
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        task_type = st.session_state.task_type

        report = generate_report_dict(model, X_test, y_test, task_type)

        st.subheader("📊 Evaluation Report")
        for section, content in report.items():
            st.markdown(f"### {section}")
            st.write(content)

        if st.button("📥 Download Report as CSV"):
            report_df = pd.DataFrame(list(report.items()), columns=["Section", "Details"])
            csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name='evaluation_report.csv',
                mime='text/csv'
            )
    else:
        st.warning("Please train a model first before accessing evaluation.")
          

import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
# from ydata_profiling import ProfileReport
from streamlit.components.v1 import html

def run_eda(df):
    st.subheader("📊 Dataset Summary")
    st.write(df.describe(include='all'))

    st.subheader("📈 Data Type Breakdown")
    st.write(df.dtypes.value_counts())

    # Correlation Heatmap (numerical only)
    st.subheader("🧊 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric features available for correlation.")

    # Target Distribution (if user picks one)
    st.subheader("🎯 Target Column Visualization")
    target = st.selectbox("Select target column to visualize", df.columns)
    if df[target].nunique() < 20:
        fig, ax = plt.subplots()
        df[target].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Too many unique values to visualize as bar chart.")

    # Show data types
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    st.write(f"🔡 Categorical Columns: {categorical}")
    st.write(f"🔢 Numerical Columns: {numeric}")

    # Optional: Display ProfileReport
    # st.subheader("🧠 Auto EDA Report with Narration")
    # if st.button("Generate Pandas Profiling Report"):
    #     profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    #     profile_html = profile.to_html()
    #     html(profile_html, height=800, scrolling=True)

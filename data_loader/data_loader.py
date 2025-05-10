import pandas as pd
import streamlit as st

def load_dataset(uploaded_file):
    """
    Load dataset from uploaded file.
    Supports CSV, Excel, and JSON.
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, or JSON.")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

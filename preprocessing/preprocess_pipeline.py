import pandas as pd
import numpy as np
from scipy.stats import skew
import streamlit as st
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler,LabelEncoder


def handle_missing_values(df, threshold_ratio=0.01):
    total_rows = df.shape[0]
    null_counts = df.isnull().sum()
    null_ratio = null_counts / total_rows

    st.subheader("🔍 Missing Values Summary")
    st.write(null_counts[null_counts > 0])

    if null_counts.sum() == 0:
        st.success("No missing values found!")
        return df

    if all(null_ratio < threshold_ratio):
        st.info(f"Null values < {threshold_ratio*100}% of rows — dropping affected rows.")
        df_cleaned = df.dropna()
        return df_cleaned

    st.info("Performing intelligent imputation...")

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                col_skew = skew(df[col].dropna())
                if abs(col_skew) < 0.5:
                    fill_value = df[col].mean()
                    df[col].fillna(fill_value, inplace=True)
                    st.write(f"➡️ Filled missing in **{col}** with **mean**")
                else:
                    fill_value = df[col].median()
                    df[col].fillna(fill_value, inplace=True)
                    st.write(f"➡️ Filled missing in **{col}** with **median**")
            else:
                fill_value = df[col].mode()[0]
                df[col].fillna(fill_value, inplace=True)
                st.write(f"➡️ Filled missing in **{col}** with **mode**")
    return df


def balance_dataset(X, y):
    st.subheader("⚖️ Class Balance Analysis")

    class_counts = y.value_counts()
    st.write("Original Class Distribution:")
    st.bar_chart(class_counts)

    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]

    st.info(f"Imbalance Ratio: {imbalance_ratio:.2f}")

    if imbalance_ratio < 1.5:
        st.success("✅ Dataset is already balanced. No action needed.")
        return X, y

    try:
        if len(y.value_counts()) == 2:
            st.info("Using SMOTE (for binary classification)...")
            smote = SMOTE()
            X_res, y_res = smote.fit_resample(X, y)
        else:
            st.info("Using RandomOverSampler (multi-class)...")
            ros = RandomOverSampler()
            X_res, y_res = ros.fit_resample(X, y)

        st.success("✅ Dataset successfully balanced.")
        st.write("Balanced Class Distribution:")
        st.bar_chart(pd.Series(y_res).value_counts())
        return X_res, y_res
    except Exception as e:
        st.error(f"⚠️ Error during balancing: {e}")
        return X, y
    
def encode_and_scale_features(X):
    """
    Encodes categorical features and scales numerical features.
    - Binary categorical: Label Encoding
    - Multi-class categorical: One-Hot Encoding
    - Scaling: StandardScaler or MinMaxScaler based on skew
    """
    st.subheader("🧬 Encoding & Scaling")

    X_encoded = X.copy()
    categorical_cols = X_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X_encoded.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Encode categorical
    for col in categorical_cols:
        if X_encoded[col].nunique() == 2:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col])
            st.write(f"🔁 Binary encoding on **{col}**")
        else:
            dummies = pd.get_dummies(X_encoded[col], prefix=col)
            X_encoded.drop(col, axis=1, inplace=True)
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
            st.write(f"🔁 One-hot encoding on **{col}**")

    # Scale numerical
    for col in numeric_cols:
        col_skew = skew(X_encoded[col])
        if abs(col_skew) < 0.5:
            scaler = StandardScaler()
            X_encoded[[col]] = scaler.fit_transform(X_encoded[[col]])
            st.write(f"📏 Standard scaled **{col}** (normal)")
        else:
            scaler = MinMaxScaler()
            X_encoded[[col]] = scaler.fit_transform(X_encoded[[col]])
            st.write(f"📏 MinMax scaled **{col}** (skewed)")

    return X_encoded    



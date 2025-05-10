import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def construct_new_features(X):
    st.subheader("🧱 Constructing New Features")

    df = X.copy()
    new_features = []

    # Add polynomial interaction terms (only for numerical)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) >= 2:
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_feats = poly.fit_transform(df[num_cols])
        poly_df = pd.DataFrame(poly_feats, columns=poly.get_feature_names_out(num_cols))
        poly_df = poly_df.drop(columns=num_cols, errors='ignore')  # Keep only interaction terms
        df = pd.concat([df, poly_df], axis=1)
        new_features.extend(poly_df.columns.tolist())
        st.write(f"➕ Added interaction terms: {new_features}")
    else:
        st.info("Not enough numeric columns for feature interactions.")

    # Example: extract features from datetime if exists
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            df[col + "_day"] = df[col].dt.day
            df[col + "_month"] = df[col].dt.month
            df[col + "_weekday"] = df[col].dt.weekday
            new_features.extend([col + "_day", col + "_month", col + "_weekday"])
            st.write(f"🕒 Extracted datetime parts from {col}")

    return df

def remove_low_variance_features(X, threshold=0.01):
    st.subheader("🔍 Removing Low Variance Features")
    selector = VarianceThreshold(threshold)
    X_reduced = selector.fit_transform(X)
    kept_features = X.columns[selector.get_support()]
    removed = list(set(X.columns) - set(kept_features))
    st.write(f"🗑 Removed low variance features: {removed}")
    return pd.DataFrame(X_reduced, columns=kept_features)

def remove_highly_correlated_features(X, threshold=0.9):
    st.subheader("🧯 Removing Highly Correlated Features")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X_filtered = X.drop(columns=to_drop)
    st.write(f"🧹 Dropped highly correlated features: {to_drop}")
    return X_filtered

def select_important_features(X, y, method="mutual_info", k=10):
    st.subheader("💡 Selecting Top Important Features")

    if method == "f_classif":
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    else:
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))

    selector.fit(X, y)
    mask = selector.get_support()
    selected = X.columns[mask]
    st.write(f"⭐ Top {k} features using `{method}`: {selected.tolist()}")
    return X[selected]

def scale_features(X):
    st.subheader("⚖️ Final Feature Scaling")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.success("📏 Applied StandardScaler on all features.")
    return pd.DataFrame(X_scaled, columns=X.columns)

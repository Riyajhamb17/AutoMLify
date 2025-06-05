import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# def detect_task_type(y):
#     if y.dtype == 'object' or y.nunique() <= 15:
#         return "classification"
#     return "regression"
def detect_task_type(y):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    try:
        if y.dtype == 'object' or y.nunique() <= 15:
            return "classification"
        else:
            return "regression"
    except Exception as e:
        st.error(f"Error detecting task type: {e}")
        return "classification"  # default fallback


def auto_mode(X, y, task_type):
    st.subheader("🚀 Auto Mode: Training Multiple Models")

    models = {
        "classification": {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(),
            "Naive Bayes": GaussianNB(),
            "KNN": KNeighborsClassifier()
        },
        "regression": {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Decision Tree": DecisionTreeRegressor(),
            "SVR": SVR()
        }
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = []

    for name, model in models[task_type].items():
        try:
            scores = cross_val_score(model, X_train, y_train, cv=5, error_score='raise')
            avg_score = scores.mean()
            model.fit(X_train, y_train)
            st.session_state['trained_model'] = model
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['task_type'] = task_type
            params = model.get_params()
            results.append((name, avg_score, params))
        except Exception as e:
            st.warning(f"⚠️ Model {name} failed: {e}")

    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        best_model, best_score, best_params = results[0]
        st.success(f"✅ Best Model: {best_model} with score: {round(best_score, 4)}")
        st.write("🔧 Best Model Hyperparameters:")
        st.json(best_params)

        st.write("📊 All Model Results:")
        for name, score, _ in results:
            st.write(f"• {name}: {round(score, 4)}")
    else:
        st.error("❌ No models could be successfully trained.")

def manual_mode(X, y, task_type):
    st.subheader("🛠 Manual Mode: Choose Your Algorithm")

    if task_type == "classification":
        algo = st.selectbox("Choose Algorithm", ["Logistic Regression", "Random Forest", "SVM", "Naive Bayes", "KNN"])
    else:
        algo = st.selectbox("Choose Algorithm", ["Linear Regression", "Random Forest", "Decision Tree", "SVR"])

    params = {}

    if algo == "Random Forest":
        n_estimators = st.slider("Number of Trees", 10, 200, 100)
        params["n_estimators"] = n_estimators
        model = RandomForestClassifier(n_estimators=n_estimators) if task_type == "classification" else RandomForestRegressor(n_estimators=n_estimators)

    elif algo == "Logistic Regression":
        c = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
        params["C"] = c
        model = LogisticRegression(C=c, max_iter=500)

    elif algo == "Linear Regression":
        model = LinearRegression()

    elif algo == "Decision Tree":
        max_depth = st.slider("Max Depth", 1, 20, 5)
        params["max_depth"] = max_depth
        model = DecisionTreeRegressor(max_depth=max_depth)

    elif algo == "SVM":
        c = st.slider("C", 0.01, 10.0, 1.0)
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
        model = SVC(C=c, kernel=kernel) if task_type == "classification" else SVR(C=c, kernel=kernel)

    elif algo == "Naive Bayes":
        model = GaussianNB()

    elif algo == "KNN":
        k = st.slider("K", 1, 20, 5)
        model = KNeighborsClassifier(n_neighbors=k)

    # Train
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
       
        st.session_state['trained_model'] = model
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['task_type'] = task_type       
        preds = model.predict(X_test)

        # Metrics
        if task_type == "classification":
            acc = accuracy_score(y_test, preds)
            st.success(f"✅ Accuracy: {round(acc, 4)}")
        else:
            mse = mean_squared_error(y_test, preds)
            st.success(f"✅ MSE: {round(mse, 4)}")

        st.write("🔧 Final Model Hyperparameters:")
        st.json(model.get_params())

    except Exception as e:
        st.error(f"❌ Model training failed: {e}")

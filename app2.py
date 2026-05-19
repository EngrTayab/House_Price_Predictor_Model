import streamlit as st
import numpy as np
import pandas as pd
import joblib
import gdown

# =========================
# LOAD TRAINED PIPELINE MODEL
# =========================
# =========================
# DOWNLOAD MODEL FROM DRIVE
# =========================
 

file_id = "1lT6bwTTJn1NcNQXuBkn4xFGuvBjq8cy "
 
url = f"https://drive.google.com/uc?id={file_id}"
output = "model.pkl"
 
gdown.download(url, output, quiet=False)
 
# Load pipeline model

model = joblib.load("model.pkl")

# =========================
# LOAD DATA (for UI only)
# =========================
df = pd.read_csv("enhanced_house_price_dataset.csv")

# Features (WITHOUT target column)
X_raw = df.drop("Price", axis=1)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="House Price Predictor", layout="wide")

# =========================
# SIDEBAR MENU
# =========================
st.sidebar.title("🏡 ML Project Menu")
menu = st.sidebar.radio("Select Option", ["Home", "Predict", "About Model"])

# =========================
# HOME PAGE
# =========================
if menu == "Home":
    st.title("🏡 House Price Prediction System")

    st.write(
        "This AI system predicts house prices using a Machine Learning model "
        "trained with preprocessing + regression pipeline."
    )

    st.success("✔ Built with Python, Scikit-learn, Streamlit")

    st.markdown("### 📊 Model Features")
    st.markdown("""
    - Automatic Categorical Encoding
    - Feature Scaling inside Pipeline
    - No manual preprocessing required
    - ML Model: Regression (e.g., XGBoost / RandomForest)
    """)

# =========================
# PREDICT PAGE
# =========================
elif menu == "Predict":
    st.title("🔮 Predict House Price")
    st.write("Enter house details below:")

    user_inputs = {}

    col1, col2 = st.columns(2)

    # Dynamic input form
    for i, col_name in enumerate(X_raw.columns):
        target_col = col1 if i % 2 == 0 else col2

        # categorical column
           # categorical column
        if pd.api.types.is_object_dtype(X_raw[col_name]) or pd.api.types.is_string_dtype(X_raw[col_name]):
            options = list(X_raw[col_name].unique())
            user_inputs[col_name] = target_col.selectbox(f"{col_name}", options)

    # numeric column
        elif pd.api.types.is_numeric_dtype(X_raw[col_name]):
            user_inputs[col_name] = target_col.number_input(
            f"{col_name}",
            value=float(X_raw[col_name].median())
        )

    # fallback for unsupported column types
    else:
        user_inputs[col_name] = None
      

    # =========================
    # PREDICTION
    # =========================
    if st.button("🚀 Predict Price"):
        input_df = pd.DataFrame([user_inputs])

        # Direct prediction (NO scaler, NO get_dummies, NO reindex)
        prediction = model.predict(input_df)[0]

        st.success(f"💰 Estimated House Price: RS{prediction:,.2f}")
        st.info("Prediction generated using trained ML pipeline model.")

# =========================
# ABOUT PAGE
# =========================
elif menu == "About Model":
    st.title("📘 About This Model")

    st.write("""
    This project predicts house prices using Machine Learning.

    The model uses:
    - Automatic preprocessing (encoding + scaling)
    - Regression algorithm (best selected model)
    - Streamlit web interface
    """)

    st.metric("Framework", "Streamlit + Scikit-learn Pipeline")
    st.metric("Model Type", "Regression (XGBoost / RF / Linear)")

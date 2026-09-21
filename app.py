import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")


# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient details below to predict the result.")


# =====================================================
# LOAD DATASET
# =====================================================

data = pd.read_csv("heart.csv")


# =====================================================
# DATA PREPROCESSING
# =====================================================

X = data.drop("target", axis=1)
y = data["target"]


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================================
# MODEL TRAINING
# =====================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)


# =====================================================
# USER INPUT
# =====================================================

st.header("Enter Patient Details")


col1, col2 = st.columns(2)


# ---------------- LEFT COLUMN ----------------

with col1:

    age = st.number_input(
        "Age",
        min_value=int(data["age"].min()),
        max_value=int(data["age"].max()),
        value=50
    )

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x:
        "Female" if x == 0 else "Male"
    )

    cp = st.selectbox(
        "Chest Pain Type (cp)",
        sorted(data["cp"].unique())
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=int(data["trestbps"].min()),
        max_value=int(data["trestbps"].max()),
        value=int(data["trestbps"].median())
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=int(data["chol"].min()),
        max_value=int(data["chol"].max()),
        value=int(data["chol"].median())
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar (fbs)",
        sorted(data["fbs"].unique())
    )

    restecg = st.selectbox(
        "Resting ECG (restecg)",
        sorted(data["restecg"].unique())
    )


# ---------------- RIGHT COLUMN ----------------

with col2:

    thalach = st.number_input(
        "Maximum Heart Rate (thalach)",
        min_value=int(data["thalach"].min()),
        max_value=int(data["thalach"].max()),
        value=int(data["thalach"].median())
    )

    exang = st.selectbox(
        "Exercise Induced Angina (exang)",
        sorted(data["exang"].unique())
    )

    oldpeak = st.number_input(
        "ST Depression (oldpeak)",
        min_value=float(data["oldpeak"].min()),
        max_value=float(data["oldpeak"].max()),
        value=float(data["oldpeak"].median()),
        step=0.1
    )

    slope = st.selectbox(
        "Slope",
        sorted(data["slope"].unique())
    )

    ca = st.selectbox(
        "Number of Major Vessels (ca)",
        sorted(data["ca"].unique())
    )

    thal = st.selectbox(
        "Thalassemia (thal)",
        sorted(data["thal"].unique())
    )


# =====================================================
# PREDICTION BUTTON
# =====================================================

st.write("")

predict_button = st.button(
    "❤️ Predict Heart Disease",
    use_container_width=True
)


# =====================================================
# PREDICTION
# =====================================================

if predict_button:

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    st.header("🔍 Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ Model Prediction: Heart Disease Detected"
        )

        st.write(
            f"Prediction probability: "
            f"{probability[1] * 100:.2f}%"
        )

    else:

        st.success(
            "✅ Model Prediction: No Heart Disease Detected"
        )

        st.write(
            f"Prediction probability: "
            f"{probability[0] * 100:.2f}%"
        )

    st.info(
        "This is a machine-learning prediction for an academic "
        "project and is not a medical diagnosis."
    )
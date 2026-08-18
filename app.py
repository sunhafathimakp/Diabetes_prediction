import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "diabetes_model.pkl"

try:
    model = joblib.load(MODEL_PATH)

except FileNotFoundError:
    st.error("❌ Model file not found!")
    st.write(
        "Please make sure the file is located at:"
    )
    st.code("model/diabetes_model.pkl")
    st.stop()

except Exception as e:
    st.error("❌ Error loading the model")
    st.exception(e)
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🩺 Diabetes Prediction System")

st.markdown(
    """
    Enter the patient's medical information below to predict
    whether the patient is likely to have diabetes.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🩺 Diabetes Predictor")

st.sidebar.info(
    """
    This application uses a Machine Learning
    Logistic Regression model to predict
    diabetes based on patient health parameters.
    """
)


# ============================================================
# PATIENT INPUT
# ============================================================

st.subheader("👤 Patient Information")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# COLUMN 1
# ------------------------------------------------------------

with col1:

    pregnancies = st.number_input(
        "🤰 Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "🩸 Glucose",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0
    )

    blood_pressure = st.number_input(
        "💓 Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    skin_thickness = st.number_input(
        "📏 Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )


# ------------------------------------------------------------
# COLUMN 2
# ------------------------------------------------------------

with col2:

    insulin = st.number_input(
        "💉 Insulin",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        step=1.0
    )

    bmi = st.number_input(
        "⚖️ BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "🧬 Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01
    )

    age = st.number_input(
        "🎂 Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Diabetes",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]


    # --------------------------------------------------------
    # PREDICTION PROBABILITY
    # --------------------------------------------------------

    probability = model.predict_proba(input_data)[0]

    non_diabetic_probability = probability[0] * 100
    diabetic_probability = probability[1] * 100


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader("📊 Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ **The model predicts: Diabetic**"
        )

        st.metric(
            "Diabetes Probability",
            f"{diabetic_probability:.2f}%"
        )

    else:

        st.success(
            "✅ **The model predicts: Non-Diabetic**"
        )

        st.metric(
            "Non-Diabetes Probability",
            f"{non_diabetic_probability:.2f}%"
        )


    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    st.divider()

    st.subheader("📌 Patient Summary")

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric(
            "Glucose",
            f"{glucose:.0f}"
        )

    with metric2:
        st.metric(
            "Blood Pressure",
            f"{blood_pressure:.0f}"
        )

    with metric3:
        st.metric(
            "BMI",
            f"{bmi:.1f}"
        )

    with metric4:
        st.metric(
            "Age",
            f"{age}"
        )


    # ========================================================
    # CHART 1 - HEALTH METRICS
    # ========================================================

    st.divider()

    st.subheader("📊 Health Metrics")

    health_data = pd.DataFrame({
        "Metric": [
            "Glucose",
            "Blood Pressure",
            "BMI",
            "Insulin"
        ],
        "Value": [
            glucose,
            blood_pressure,
            bmi,
            insulin
        ]
    })

    fig_health = px.bar(
        health_data,
        x="Metric",
        y="Value",
        title="Patient Health Measurements",
        text="Value"
    )

    fig_health.update_traces(
        textposition="outside"
    )

    fig_health.update_layout(
        xaxis_title="Health Parameter",
        yaxis_title="Value",
        height=450
    )

    st.plotly_chart(
        fig_health,
        use_container_width=True
    )


    # ========================================================
    # CHART 2 - PREDICTION PROBABILITY
    # ========================================================

    st.subheader("📈 Prediction Probability")

    probability_data = pd.DataFrame({
        "Outcome": [
            "Non-Diabetic",
            "Diabetic"
        ],
        "Probability": [
            non_diabetic_probability,
            diabetic_probability
        ]
    })

    fig_probability = px.bar(
        probability_data,
        x="Outcome",
        y="Probability",
        title="Diabetes Prediction Probability",
        text="Probability",
        range_y=[0, 100]
    )

    fig_probability.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_probability.update_layout(
        xaxis_title="Prediction",
        yaxis_title="Probability (%)",
        height=450
    )

    st.plotly_chart(
        fig_probability,
        use_container_width=True
    )


    # ========================================================
    # CHART 3 - ALL PATIENT MEASUREMENTS
    # ========================================================

    st.subheader("📋 Patient Measurements")

    patient_data = pd.DataFrame({
        "Measurement": [
            "Pregnancies",
            "Glucose",
            "Blood Pressure",
            "Skin Thickness",
            "Insulin",
            "BMI",
            "Diabetes Pedigree",
            "Age"
        ],
        "Value": [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]
    })

    fig_patient = px.bar(
        patient_data,
        x="Measurement",
        y="Value",
        title="Complete Patient Health Profile",
        text="Value"
    )

    fig_patient.update_traces(
        textposition="outside"
    )

    fig_patient.update_layout(
        xaxis_title="Measurement",
        yaxis_title="Value",
        height=500
    )

    st.plotly_chart(
        fig_patient,
        use_container_width=True
    )


    # ========================================================
    # INPUT DATA TABLE
    # ========================================================

    st.divider()

    st.subheader("📄 Patient Input Details")

    st.dataframe(
        input_data,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # PREDICTION DETAILS
    # ========================================================

    st.subheader("🎯 Prediction Details")

    result_data = pd.DataFrame({
        "Prediction": [
            "Diabetic" if prediction == 1 else "Non-Diabetic"
        ],
        "Non-Diabetic Probability": [
            f"{non_diabetic_probability:.2f}%"
        ],
        "Diabetic Probability": [
            f"{diabetic_probability:.2f}%"
        ]
    })

    st.dataframe(
        result_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🩺 Diabetes Prediction System | "
    "Machine Learning Project"
)
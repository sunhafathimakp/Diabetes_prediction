import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model/diabetes_model.pkl")

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("data/diabetes.csv")

# -----------------------------
# Title
# -----------------------------
st.title("🩺 Diabetes Prediction System")

st.markdown("---")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Enter Patient Details")

preg = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.number_input("Glucose", 0, 300, 120)
bp = st.sidebar.number_input("Blood Pressure", 0, 200, 70)
skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.number_input("Age", 1, 120, 30)

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Predict"):

    input_data = [[
        preg,
        glucose,
        bp,
        skin,
        insulin,
        bmi,
        dpf,
        age
    ]]

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Patient is likely Diabetic")
    else:
        st.success("✅ Patient is likely Non-Diabetic")

    st.subheader("Prediction Probability")

    st.write(f"Non-Diabetic : **{probability[0][0]*100:.2f}%**")
    st.write(f"Diabetic : **{probability[0][1]*100:.2f}%**")

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(data.head(), use_container_width=True)

st.markdown("---")

# -----------------------------
# Dataset Information
# -----------------------------
st.subheader("Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Total Records", len(data))

with col2:
    st.metric("Total Features", len(data.columns) - 1)

st.markdown("---")

st.subheader("Dataset Summary")
st.write(data.describe())
st.markdown("---")

# -----------------------------
# Diabetes Distribution
# -----------------------------
st.subheader("Diabetes Distribution")

fig, ax = plt.subplots(figsize=(5, 4))
data["Outcome"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    ax=ax
)
ax.set_ylabel("")
st.pyplot(fig)
st.markdown("---")

# -----------------------------
# Glucose Distribution
# -----------------------------
st.subheader("Glucose Distribution")

fig, ax = plt.subplots(figsize=(5, 4))
ax.hist(data["Glucose"], bins=20)
ax.set_xlabel("Glucose")
ax.set_ylabel("Count")
st.pyplot(fig)
st.markdown("---")

# -----------------------------
# BMI Distribution
# -----------------------------
st.subheader("BMI Distribution")

fig, ax = plt.subplots(figsize=(5, 4))
ax.hist(data["BMI"], bins=20)
ax.set_xlabel("BMI")
ax.set_ylabel("Count")
st.pyplot(fig)
st.markdown("---")

# -----------------------------
# Model Accuracy Comparison
# -----------------------------
st.subheader("📈 Model Accuracy Comparison")

accuracy = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Decision Tree"
        ],
        "Accuracy (%)": [
            74.68,
            72.08,
            74.68
        ]
    }
)

st.dataframe(accuracy, use_container_width=True)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(accuracy["Model"], accuracy["Accuracy (%)"], color=["#4C78A8", "#F58518", "#54A24B"])
ax.set_xlabel("Model")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
plt.xticks(rotation=15)
st.pyplot(fig)

st.markdown("---")

# -----------------------------
# Correlation Table
# -----------------------------
st.subheader("📊 Correlation Table")
st.dataframe(data.corr().round(2), use_container_width=True)
st.markdown("---")

# -----------------------------
# Diabetes Cases
# -----------------------------
st.subheader("📋 Diabetes Cases")

positive = len(data[data["Outcome"] == 1])
negative = len(data[data["Outcome"] == 0])

col1, col2 = st.columns(2)

with col1:
    st.metric("Diabetic Patients", positive)

with col2:
    st.metric("Non-Diabetic Patients", negative)

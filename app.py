"""
Heart Disease Prediction System - Streamlit web app
----------------------------------------------------
Loads the trained Support Vector Classifier (svc_trained_model.pkl) made by
the Heart_Disease_Prediction notebook and lets a user enter 4 patient
attributes to get a prediction.

LabelEncoder gives codes by the sorted order of the category list, so fitting
the encoders on the same fixed lists as the notebook gives the same codes.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

SEX = ["Male", "Female"]
CHEST_PAIN = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
EXERCISE_ANGINA = ["Yes", "No"]
ST_SLOPE = ["Upsloping", "Flat", "Downsloping"]


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    return {
        "Sex": LabelEncoder().fit(SEX),
        "ChestPain": LabelEncoder().fit(CHEST_PAIN),
        "ExerciseAngina": LabelEncoder().fit(EXERCISE_ANGINA),
        "STSlope": LabelEncoder().fit(ST_SLOPE),
    }


model = load_model()
encoders = build_label_encoders()

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    Enter the four patient attributes below. The trained Support Vector
    Classifier (SVC) will predict whether heart disease is likely.

    *Trained on the UCI Heart Disease (Cleveland) dataset, 303 patients.*
    """
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", SEX)
    chest_pain = st.selectbox("Chest Pain Type", CHEST_PAIN, index=3)
with col2:
    exercise_angina = st.selectbox("Exercise Induced Angina", EXERCISE_ANGINA)
    st_slope = st.selectbox("ST Slope (peak exercise)", ST_SLOPE, index=1)

if st.button("Predict", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "Sex": [sex],
        "ChestPain": [chest_pain],
        "ExerciseAngina": [exercise_angina],
        "STSlope": [st_slope],
    })

    encoded = pd.DataFrame({
        "Sex": encoders["Sex"].transform(user_input["Sex"]),
        "ChestPain": encoders["ChestPain"].transform(user_input["ChestPain"]),
        "ExerciseAngina": encoders["ExerciseAngina"].transform(user_input["ExerciseAngina"]),
        "STSlope": encoders["STSlope"].transform(user_input["STSlope"]),
    })

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE LIKELY")
    else:
        st.success("### ✅ NO HEART DISEASE LIKELY")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "Built for learning (university ML assignment). Uses only 4 attributes, "
    "so it is NOT a medical tool and must not be used for real diagnosis."
)

import streamlit as st
import pandas as pd
import joblib
import base64
model = joblib.load("models/titanic_best_model.pkl")
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)
def set_background(image_path):
    with open(image_path, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.35);
        z-index: -1;
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)

set_background("images/titanic.jpg")
st.title("🚢 Titanic Survival Prediction")

st.write(
    "Enter passenger details to predict whether the passenger "
    "would have survived the Titanic disaster."
)

st.subheader("Passenger Information")
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)
sex = st.selectbox(
    "Gender",
    ["male", "female"]
)
age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=30.0,
    step=1.0
)
sibsp = st.number_input(
    "Number of Siblings/Spouses",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)
parch = st.number_input(
    "Number of Parents/Children",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)
fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=32.0,
    step=1.0
)
embarked = st.selectbox(
    "Port of Embarkation",
    ["S", "C", "Q"]
)
if st.button("Predict Survival"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    non_survival_probability = probabilities[0]
    survival_probability = probabilities[1]
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(
            "✅ Passenger is predicted to SURVIVE."
        )
    else:

        st.error(
            "❌ Passenger is predicted NOT TO SURVIVE."
        )
    st.info(
        f"Survival Probability: "
        f"{survival_probability:.2%}"
    )


    st.warning(
        f"Non-Survival Probability: "
        f"{non_survival_probability:.2%}"
    )
    st.write("Survival Probability")
    st.progress(
        float(survival_probability)
    )
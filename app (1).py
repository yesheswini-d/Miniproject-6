import streamlit as st
import pandas as pd
import pickle

# Page settings
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():

    with open("loan_model.pkl", "rb") as file:
        model = pickle.load(file)

    return model


model = load_model()

# Title
st.title("🏦 Loan Approval Prediction")

st.write(
    "Enter the applicant details below to predict "
    "whether the loan is likely to be approved."
)

st.divider()

# Two columns
col1, col2 = st.columns(2)

# First column
with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["No", "Yes"]
    )

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000,
        step=500
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0,
        value=0,
        step=500
    )


# Second column
with col2:

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=1,
        value=150,
        step=10
    )

    loan_term = st.selectbox(
        "Loan Term (Months)",
        [120, 180, 240, 300, 360, 480],
        index=4
    )

    credit_history = st.selectbox(
        "Credit History",
        [1, 0],
        format_func=lambda x:
        "Good (1)" if x == 1
        else "Bad/Unavailable (0)"
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )


st.divider()

# Prediction
if st.button("🔮 Predict Loan Approval"):

    input_data = pd.DataFrame([{

        "Gender": gender,

        "Married": married,

        "Education": education,

        "Self_Employed": self_employed,

        "ApplicantIncome": applicant_income,

        "CoapplicantIncome": coapplicant_income,

        "LoanAmount": loan_amount,

        "Loan_Amount_Term": loan_term,

        "Credit_History": credit_history,

        "Property_Area": property_area

    }])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probability = (
        model.predict_proba(input_data)[0][int(prediction)]
        * 100
    )

    # Display result
    if prediction == 1:

        st.success("✅ LOAN APPROVED")

    else:

        st.error("❌ LOAN NOT APPROVED")

    st.info(
        f"Prediction Confidence: {probability:.2f}%"
    )


st.caption(
    "This application is developed for educational "
    "and mini-project purposes."
)

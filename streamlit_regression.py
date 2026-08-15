import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle


# Load trained model
model = tf.keras.models.load_model("salary_regression_model.h5")


# Load encoder and scaler files
with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geography = pickle.load(f)

with open("scaler_regression.pkl", "rb") as f:
    scaler = pickle.load(f)


# App title
st.title("Bank Customer Salary Prediction")


# Take input from user
geography = st.selectbox(
    "Select Geography",
    ["France", "Spain", "Germany"]
)

gender = st.selectbox(
    "Select Gender",
    ["Male", "Female"]
)

age = st.slider(
    "Select Age",
    18,
    100,
    30
)

balance = st.number_input(
    "Enter Balance",
    value=0.0
)

credit_score = st.number_input(
    "Enter Credit Score",
    value=600
)

tenure = st.slider(
    "Select Tenure",
    0,
    10,
    5
)

num_products = st.slider(
    "Select Number of Products",
    1,
    4,
    2
)

has_cr_card = st.checkbox(
    "Does the customer have a credit card?"
)

is_active_member = st.checkbox(
    "Is the customer an active member?"
)


# Create dataframe from user input
input_data = pd.DataFrame([{
    "Geography": geography,
    "Gender": label_encoder_gender.transform([gender])[0],
    "Age": age,
    "Balance": balance,
    "CreditScore": credit_score,
    "Tenure": tenure,
    "NumOfProducts": num_products,
    "HasCrCard": int(has_cr_card),
    "IsActiveMember": int(is_active_member)
}])


# Encode Geography
geo_encoded = onehot_encoder_geography.transform(
    input_data[["Geography"]]
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geography.get_feature_names_out(
        ["Geography"]
    )
)


# Add encoded columns
input_data = pd.concat(
    [
        input_data.drop("Geography", axis=1),
        geo_encoded_df
    ],
    axis=1
)


# Arrange features in the same order as training
input_data = input_data[scaler.feature_names_in_]


# Scale input data
input_data_scaled = scaler.transform(input_data)


# Predict salary
prediction = model.predict(input_data_scaled)

predicted_salary = prediction[0][0]


# Show result
st.success(
    f"Predicted Estimated Salary: ₹{predicted_salary:,.2f}"
)
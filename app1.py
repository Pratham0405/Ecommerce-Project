import streamlit as st
import pandas as pd
import numpy as np
import pickle5 as pickle
import joblib

# Load the pre-trained model
model = joblib.load("pipe4.pkl")

# Define the function to make predictions
def predict(length_of_membership, avg_session_length, time_on_app, time_on_website):
    input_data = pd.DataFrame({
        'LOM': [length_of_membership],
        'ASL': [avg_session_length],
        'TOA': [time_on_app],
        'TOW': [time_on_website]
    })
    prediction = model.predict(input_data)
    return prediction[0]

# Create the Streamlit app
def main():
    st.title("Ecommerce Customer Revenue Prediction")
    st.write("This app predicts the yearly amount spent by a customer based on their input.")

    # Add input fields for user data
    length_of_membership = st.slider("Years of Membership", 0, 10, 1)
    avg_session_length = st.slider("Avg. Session Length (minutes)", 0.0, 60.0, 30.0)
    time_on_app = st.slider("Time on App (minutes)", 0.0, 120.0, 60.0)
    time_on_website = st.slider("Time on Website (minutes)", 0.0, 120.0, 60.0)

    # Add a button to make predictions
    if st.button("Predict"):
        prediction = predict(length_of_membership, avg_session_length, time_on_app, time_on_website)
        st.success(f"The predicted yearly amount spent is: ${prediction:.2f}")

# Run the app
if __name__ == "__main__":
    main()



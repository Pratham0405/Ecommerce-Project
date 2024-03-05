import streamlit as st
import pandas as pd
import numpy as np
import pickle5 as pickle

# Import necessary libraries

import joblib


# Load your trained model
model = joblib.load('pipe2.pkl')

# Define the function to make predictions
def predict(years_of_membership, avg_session_length, time_on_app, time_on_website):
    # Prepare the input data as a dictionary
    input_data = {
        'Length of Membership': years_of_membership,
        'Avg. Session Length': avg_session_length,
        'Time on App': time_on_app,
        'Time on Website': time_on_website
    }
    
    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make predictions
    prediction = model.predict(input_df)
    
    return prediction[0]

# Define the Streamlit app
def main():
    # Set the title of the app
    st.title('Ecommerce Customer Income Predictor')
    
    # Add input fields for user input
    st.sidebar.header('User Input')
    years_of_membership = st.sidebar.number_input('Years of Membership', min_value=0, max_value=10, step=1)
    avg_session_length = st.sidebar.number_input('Average Session Length (minutes)', min_value=0.0, max_value=60.0, step=0.1)
    time_on_app = st.sidebar.number_input('Time on App (minutes)', min_value=0.0, max_value=24.0, step=0.1)
    time_on_website = st.sidebar.number_input('Time on Website (minutes)', min_value=0.0, max_value=24.0, step=0.1)
    
    # Add a button to trigger predictions
    if st.sidebar.button('Predict'):
        # Call the predict function with user inputs
        prediction = predict(years_of_membership, avg_session_length, time_on_app, time_on_website)
        
        # Display the prediction
        st.success(f'The predicted annual income is ${prediction:.2f}')

# Run the app
if __name__ == '__main__':
    main()




import streamlit as st
import pandas as pd
import numpy as np
import pickle5 as pickle

# Import necessary libraries

import joblib


# Load your trained model
model = joblib.load('pipe4.pkl')

# Define the function to make predictions
def predict(YAS, ASL, TOA, TOW):
    # Prepare the input data as a dictionary
    input_data = {
        'Length of Membership': YAS,
        'Avg. Session Length': ASL,
        'Time on App': TOA,
        'Time on Website': TOW
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
    YAS = st.sidebar.number_input('Years of Membership', min_value=0, max_value=10, step=1)
    ASL = st.sidebar.number_input('Average Session Length (minutes)', min_value=0.0, max_value=60.0, step=0.1)
    TOA = st.sidebar.number_input('Time on App (minutes)', min_value=0.0, max_value=24.0, step=0.1)
    TOW = st.sidebar.number_input('Time on Website (minutes)', min_value=0.0, max_value=24.0, step=0.1)
    
    # Add a button to trigger predictions
    if st.sidebar.button('Predict'):
        # Call the predict function with user inputs
        prediction = predict(YAS, ASL, TOA, TOW)
        
        # Display the prediction
        st.success(f'The predicted annual income is ${prediction:.2f}')

# Run the app
if __name__ == '__main__':
    main()




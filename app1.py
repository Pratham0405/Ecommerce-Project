import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle5 as pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
@st.cache
def load_data():
    data = pd.read_csv("Ecommerce_Customers.csv")
    return data

data = load_data()

# Sidebar with buttons
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Data Summary", "Data Visualization", "Prediction"])

# Data Summary
if selection == "Data Summary":
    st.title("Data Summary")

    # Display basic information about the dataset
    st.write("Basic Information:")
    st.write(f"Number of rows: {data.shape[0]}")
    st.write(f"Number of columns: {data.shape[1]}")
    st.write(f"Column names: {', '.join(data.columns)}")
    st.write(f"Data types: {', '.join(data.dtypes.astype(str))}")  # Convert data types to strings

    # Display the first few rows of the dataset
    st.write("First Few Rows:")
    st.write(data.head())

    # Display summary statistics of numerical columns
    st.write("Summary Statistics:")
    st.write(data.describe())

    # Additional points about the dataset
    st.write("Additional Points:")
    st.write("- This dataset contains information about customers.")
    st.write("- It includes numerical features such as 'Avg. Session Length', 'Time on App', 'Time on Website', and 'Length of Membership'.")
    st.write("- The target variable is 'Yearly Amount Spent', representing the amount spent by each customer annually.")


# Data Visualization
# Data Visualization
elif selection == "Data Visualization":
    st.title("Data Visualization")
    # Filter out numerical columns
    numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    selected_column = st.selectbox("Select a column for visualization", numerical_columns)
    
    # If there are numerical columns available
    if numerical_columns:
        # Histogram
        st.subheader("Histogram")
        plt.hist(data[selected_column])
        st.pyplot()

        # Boxplot
        st.subheader("Boxplot")
        sns.boxplot(data[selected_column])
        st.pyplot()
    else:
        st.write("No numerical columns found in the dataset.")
        
        
        st.set_option('deprecation.showPyplotGlobalUse', False)



# Modeling
# Modeling
elif selection == "Prediction":
    st.title("Prediction")
    st.write("This app predicts the yearly amount spent by a customer based on their input.")

    # Add input fields for user data
    avg_session_length = st.slider("Avg. Session Length (minutes)", 0.0, 60.0, 30.0)
    time_on_app = st.slider("Time on App (minutes)", 0.0, 120.0, 60.0)
    time_on_website = st.slider("Time on Website (minutes)", 0.0, 120.0, 60.0)
    length_of_membership = st.slider("Years of Membership", 0, 10, 1)

    # Load the pre-trained model
    model = joblib.load("pipe4.pkl")

    # Define the function to make predictions
    def predict(avg_session_length, time_on_app, time_on_website, length_of_membership):
        input_data = pd.DataFrame({
            'ASL': [avg_session_length],
            'TOA': [time_on_app],
            'TOW': [time_on_website],
            'LOM': [length_of_membership],
        })
        prediction = model.predict(input_data)
        return prediction[0]

    # Add a button to make predictions
    if st.button("Predict"):
        prediction = predict(avg_session_length, time_on_app, time_on_website, length_of_membership)
        st.success(f"Yearly Amount spent by customer is: ${prediction:.2f}")


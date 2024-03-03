import streamlit as st
import pandas as pd
import numpy as np
import pickle5 as pickle

file = open('pipe2.pkl', 'rb')
rf = pickle.load(file)
file.close()


data = pd.read_csv("traineddata.csv")

#data['IPS'].unique()

st.title("Yearly Amount Spent")
 
# Average Session Length

ASL = st.number_input('Average Session Length')



# Time on App

TOA = st.number_input('Time on App')

# Time on Website

TOW = st.number_input('Time on Website')

# Length of Membership

LOM = st.number_input('Length of Membership (in months)')




query = np.array([ASL, TOA, TOW, LOM])

query = query.reshape(1, 4)

prediction = int(model.predict(query)[0])

st.title("Predicted Yearly Amount spent by customer is" +
             str(prediction-50)+"₹" + " to " + str(prediction+50)+"₹")

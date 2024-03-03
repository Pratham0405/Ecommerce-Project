import streamlit as st
import pandas as pd
import numpy as np
import pickle5 as pickle

file1 = open('ecommercewebsite.pkl', 'rb')
model = pickle.load(file1)
file1.close()



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

prediction = int(np.exp(model.predict(query)[0]))

st.title("Predicted Yearly Amount spent by customer is" +
             str(prediction-50)+"₹" + " to " + str(prediction+50)+"₹")

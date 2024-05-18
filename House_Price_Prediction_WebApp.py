# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:57:51 2024

@author: prachet
"""


import numpy as np
import pickle
import streamlit as st

#loading. the saved model
loaded_model = pickle.load(open('C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-3-House_Price_Prediction/house_price_prediction_model.sav','rb'))

#creating a function for prediction

def house_price_prediction(input_data):

    #changing the input data to numpy
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting on 1 instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    #print(prediction)

    return prediction[0]
  
    
  
def main():
    
    #giving a title
    st.title('House Price Prediction Web App')
    
    col1 , col2 , col3 = st.columns(3)
    #getting input data from user
    with col1:
        SquareMeters = st.number_input("Size of house in square meters")
    with col2:
        NumberOfRooms = st.number_input("Number Of Rooms")
    with col3:
        option1 = st.selectbox('Has Yard',('Yes', 'No')) 
        HasYard = 1 if option1 == 'Yes' else 0
    with col1:
        option2 = st.selectbox('Has Pool',('Yes', 'No')) 
        HasPool = 1 if option2 == 'Yes' else 0
    with col2:
        Floors = st.number_input("Number of floors")
    with col3:
        CityCode = st.number_input("City Code")
    with col1:
        CityPartRange = st.number_input("City Part Range (range - 0 - cheapest, 10 - the most expensive)")
    with col2:
        NumPrevOwners = st.number_input("Number Prev Owners")
    with col3:
        Made = st.number_input('Made in Year')
    with col1:
        option3 = st.selectbox('Is New Built',('Yes', 'No'))
        IsNewBuilt = 1 if option3 == 'Yes' else 0
    with col2:
        option4 = st.selectbox('Has Storm Protector',('Yes', 'No'))
        HasStormProtector = 1 if option4 == 'Yes' else 0
    with col3:
        Basement = st.number_input('Basement in square meters')
    with col1:
        Attic = st.number_input('Attic in square meteres')
    with col2:
        Garage = st.number_input('Garage Size in square meteres')
    with col3:
        option5 = st.selectbox('Has Storage Room',('Yes', 'No'))
        HasStorageRoom = 1 if option5 == 'Yes' else 0
    with col1:
        HasGuestRoom = st.number_input('Number of guest rooms')	
    
    
    # code for prediction
    price = ''
    
    #creating a button for Prediction
    if st.button('Predict House Price'):
        price = house_price_prediction((SquareMeters,NumberOfRooms,HasYard,HasPool,Floors,CityCode,CityPartRange,NumPrevOwners,Made,IsNewBuilt,HasStormProtector,Basement,Attic,Garage,HasStorageRoom,HasGuestRoom))
        
    st.success('The Predicted Price: '+ str(price)+'$')
    
    
    
if __name__ == '__main__':
    main()
    
    
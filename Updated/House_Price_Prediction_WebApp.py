# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:57:51 2024

@author: prachet
"""

import json
import pickle
import streamlit as st
import pandas as pd

#loading. the saved model
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-3-House_Price_Prediction/Updated/columns.pkl", 'rb') as f:
    all_features = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-3-House_Price_Prediction/Updated/scaler.pkl", 'rb') as f:
    scalers = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-3-House_Price_Prediction/Updated/best_features_lr.json", 'r') as file:
    best_features_lr = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-3-House_Price_Prediction/Updated/best_features_rfr.json", 'r') as file:
    best_features_rfr = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-3-House_Price_Prediction/Updated/best_features_xgb.json", 'r') as file:
    best_features_xgb = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-3-House_Price_Prediction/Updated/house_price_prediction_trained_lr_model.sav", 'rb') as f:
    loaded_model_lr = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-3-House_Price_Prediction/Updated/house_price_prediction_trained_rfr_model.sav", 'rb') as f:
    loaded_model_rfr = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-3-House_Price_Prediction/Updated/house_price_prediction_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb = pickle.load(f)


# converting the category columns to string data type
#cat_cols = ['hasYard', 'hasPool', 'cityPartRange', 'numPrevOwners', 'made', 'isNewBuilt','hasStormProtector', 'hasStorageRoom','hasGuestRoom']

#creating a function for prediction

def house_price_prediction(input_data):

    df = pd.DataFrame([input_data], columns=all_features)
  #  df[cat_cols] = df[cat_cols].astype('str')

    df[all_features] = scalers.transform(df[all_features])
    
    df_best_features_lr = df[best_features_lr]
    df_best_features_rfr = df[best_features_rfr]
    df_best_features_xgb = df[best_features_xgb]
    
    prediction1 = loaded_model_lr.predict(df_best_features_lr)
    prediction2 = loaded_model_rfr.predict(df_best_features_rfr)
    prediction3 = loaded_model_xgb.predict(df_best_features_xgb)
    
    return prediction1 , prediction2, prediction3

    
  
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
        option1 = st.selectbox('Has Yard',('No', 'Yes')) 
        HasYard = 1 if option1 == 'Yes' else 0
    with col1:
        option2 = st.selectbox('Has Pool',('No', 'Yes')) 
        HasPool = 1 if option2 == 'Yes' else 0
    with col2:
        Floors = st.number_input("Number of floors")
    with col3:
        CityCode = st.number_input("City Code")
    with col1:
        CityPartRange = st.selectbox('City Part Range(cheapest to expensive)',('1','2','3','4','5','6','7','8','9','10')) 
    with col2:
        NumPrevOwners = st.selectbox('Number Prev Owners',('1','2','3','4','5','6','7','8','9','10'))
    with col3:
        Made = st.number_input("Made in Year")
    with col1:
        option3 = st.selectbox('Is New Built',('No', 'Yes'))
        IsNewBuilt = 1 if option3 == 'Yes' else 0
    with col2:
        option4 = st.selectbox('Has Storm Protector',('No', 'Yes'))
        HasStormProtector = 1 if option4 == 'Yes' else 0
    with col3:
        Basement = st.number_input('Basement in square meters')
    with col1:
        Attic = st.number_input('Attic in square meteres')
    with col2:
        Garage = st.number_input('Garage Size in square meteres')
    with col3:
        option5 = st.selectbox('Has Storage Room',('No', 'Yes'))
        HasStorageRoom = 1 if option5 == 'Yes' else 0
    with col1:
        HasGuestRoom = st.number_input('Number of guest rooms')	
    
    
    # code for prediction
    house_price_prediction_lr = ''
    house_price_prediction_rfr = ''
    house_price_prediction_xgb = ''
    

    house_price_prediction_lr,house_price_prediction_rfr,house_price_prediction_xgb = house_price_prediction([SquareMeters,NumberOfRooms,HasYard,HasPool,Floors,CityCode,CityPartRange,NumPrevOwners,Made,IsNewBuilt,HasStormProtector,Basement,Attic,Garage,HasStorageRoom,HasGuestRoom])
        
    #creating a button for Prediction
    if st.button("Predict House Price"):
        prediction = house_price_prediction_lr[0]
        prediction = "{:.2f}".format(prediction)
        st.write(f"The Predicted Price: {prediction} $")
    
    if st.checkbox("Show Advanced Options"):
        if st.button("Predict House Price with Linear Regression Model"):
            prediction = house_price_prediction_lr[0]
            prediction = "{:.2f}".format(prediction)
            st.write(f"The Predicted Price: {prediction} $")
        if st.button("Predict House Price with Random Forest Regressor Model"):
            prediction = house_price_prediction_rfr[0]
            prediction = "{:.2f}".format(prediction)
            st.write(f"The Predicted Price: {prediction} $")
        if st.button("Predict House Price with XG Boost Regressor"):
            prediction = house_price_prediction_xgb[0]
            prediction = "{:.2f}".format(prediction)
            st.write(f"The Predicted Price: {prediction} $")   
    
    
    
if __name__ == '__main__':
    main()
    
    
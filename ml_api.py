# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
import pickle 
import json 
import numpy as np
import jsonify

app = FastAPI()


class dia_model(BaseModel):
    Pregnancies:int 
    Glucose:int
    BloodPressure:int
    SkinThickness:int
    Insulin:int
    BMI:int
    DiabetesPedigreeFunction:int
    Age:int


class model_input(BaseModel):
    age:int
    sex:int
    cp:int
    trestbps:int
    chol:int
    fbs:int
    restecg:int
    thalach:int
    exang:int
    oldpeak:float
    slope:int
    ca:int
    thal:int
    
heart_model = pickle.load(open('heart_disease_model.sav','rb'))

diabetes_model = pickle.load(open('diabetes_model.sav','rb'))

@app.post('/heartprediction')
def read_root(input_parameters : model_input):
    input_data = input_parameters.json() 
    input_dictionary = json.loads(input_data) 
    age = input_dictionary['age']
    sex = input_dictionary['sex']
    cp = input_dictionary['cp']
    trestbps = input_dictionary['trestbps']
    chol = input_dictionary['chol']
    fbs = input_dictionary['fbs']
    restecg = input_dictionary['restecg']
    thalach = input_dictionary['thalach']
    exang = input_dictionary['exang']
    oldpeak = input_dictionary['oldpeak']
    slope = input_dictionary['slope']
    ca = input_dictionary['ca']
    thal = input_dictionary['thal']
    
    

    input_d = (age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
    
    print(input_d)
    
    input_data_as_numpy_array= np.asarray(input_d)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    print(input_data_reshaped)
    
    
    
    
    
        
    #input_list = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
    
    result = heart_model.predict(input_data_reshaped)

    if(result[0] == 0):
        return "The Person has no Heart Disease"
    return "The Person has Heart Disease"




@app.post('/diabetesprediction')
def read_root(input_parameters : dia_model):
    input_data = input_parameters.json() 
    input_dictionary = json.loads(input_data) 
    Preg = input_dictionary['Pregnancies']
    glu = input_dictionary['Glucose']
    bp = input_dictionary['BloodPressure']
    skt = input_dictionary['SkinThickness']
    ins = input_dictionary['Insulin']
    bmi = input_dictionary['BMI']
    dpf = input_dictionary['DiabetesPedigreeFunction']
    age = input_dictionary['Age']
    
    input_d = (Preg,glu,bp,skt,ins,bmi,dpf,age)

    input_data_as_numpy_array= np.asarray(input_d)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    
    result = diabetes_model.predict(input_data_reshaped)
    

    if(result[0] == 0):
        return "The Person has no Diabetes"
    return "The Person has Diabetes"
    
    
    
    
    
    

    
    
    
    
   
    
    
    
    
        
    
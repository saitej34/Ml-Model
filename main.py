from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
import pickle 
import json 
import numpy as np
from time import time
from fastapi import FastAPI, __version__



app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



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



@app.get("/")
async def root():
    return "Hello World"



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
def read_root(input_parameters: dict):
    try:
        Preg = input_parameters['Pregnancies']
        glu = input_parameters['Glucose']
        bp = input_parameters['BloodPressure']
        skt = input_parameters['SkinThickness']
        ins = input_parameters['Insulin']
        bmi = input_parameters['BMI']
        dpf = input_parameters['DiabetesPedigreeFunction']
        age = input_parameters['Age']
        
        input_d = (Preg, glu, bp, skt, ins, bmi, dpf, age)

        input_data_as_numpy_array = np.asarray(input_d)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Perform prediction using diabetes_model
        result = diabetes_model.predict(input_data_reshaped)
        print(result)
        if result[0] == 0:
            return "The Person has no Diabetes"
        return "The Person has Diabetes"
    except Exception as e:
        return f"Error: {str(e)}"

# @app.post('/dietrec')
# def read_root(input_parameters : diet_model):
#     input_data = input_parameters.json() 
#     input_dictionary = json.loads(input_data) 
#     bmi = input_dictionary['Pregnancies']
#     input_nutrition = [250, 15, 2, 80, 150, 30,5,3,20]
#     if bmi < 18.5:
#         input_ingredients = ['rice', 'eggs', 'beans', 'bananas', 'apples', 'carrots']
#     elif bmi < 25:
#         input_ingredients = ['oats', 'chicken', 'yogurt', 'spinach', 'berries', 'nuts']
#     elif bmi < 30:
#         input_ingredients = ['brown rice', 'salmon', 'sweet potato', 'quinoa', 'tomatoes', 'bell peppers']
#     else:
#         input_ingredients = ['kale', 'tofu', 'broccoli', 'mushrooms', 'cauliflower','garlic']
    
#     recommendation_dataframe = diet_mod.recommend(input_nutrition,input_ingredients)
#     output = diet_mod.output_recommended_recipes(recommendation_dataframe)

#     return output

    
    
    
    
    
    

    
    
    
    
   
    
    
    
    
        
    
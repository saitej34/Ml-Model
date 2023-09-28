import numpy as np
import re
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import joblib

class DietModel:
    def _init_(self):
        self.dataset = pd.read_csv('C:\\Users\\saite\\OneDrive\\Desktop\\dataset (1).csv', compression='gzip')
        self.params = {'n_neighbors': 3, 'return_distance': False}
        self.pipeline = None

    def scaling(self, dataframe):
        scaler = StandardScaler()
        prep_data = scaler.fit_transform(dataframe.iloc[:, 6:15].to_numpy())
        return prep_data, scaler

    def nn_predictor(self, prep_data):
        neigh = NearestNeighbors(metric='cosine', algorithm='brute')
        neigh.fit(prep_data)
        return neigh

    def build_pipeline(self, neigh, scaler, params):
        transformer = FunctionTransformer(neigh.kneighbors, kw_args=params)
        pipeline = Pipeline([('std_scaler', scaler), ('NN', transformer)])
        return pipeline
    def extract_data(self, dataframe, ingredients):
        extracted_data = dataframe.copy()
        extracted_data = self.extract_ingredient_filtered_data(extracted_data, ingredients)
        return extracted_data

    def extract_ingredient_filtered_data(self, dataframe, ingredients):
        extracted_data = dataframe.copy()
        regex_string = ''.join(map(lambda x: f'(?=.*{x})', ingredients))
        extracted_data = extracted_data[extracted_data['RecipeIngredientParts'].str.contains(regex_string, regex=True, flags=re.IGNORECASE)]
        return extracted_data

    def apply_pipeline(self, _input, extracted_data):
        _input = np.array(_input).reshape(1, -1)
        return extracted_data.iloc[self.pipeline.transform(_input)[0]]

    def recommend(self, _input, ingredients=[]):
        extracted_data = self.extract_data(self.dataset, ingredients)
        if extracted_data.shape[0] >= self.params['n_neighbors']:
            prep_data, scaler = self.scaling(extracted_data)
            neigh = self.nn_predictor(prep_data)
            self.pipeline = self.build_pipeline(neigh, scaler, self.params)
            return self.apply_pipeline(_input, extracted_data)
        else:
            return None

    def extract_quoted_strings(self, s):
        # Find all the strings inside double quotes
        strings = re.findall(r'"([^"]*)"', s)
        # Join the strings with 'and'
        return strings
    def output_recommended_recipes(self, dataframe):
        if dataframe is not None:
            output = dataframe.copy()
            output = output.to_dict("records")
            for recipe in output:
                recipe['RecipeIngredientParts'] = self.extract_quoted_strings(recipe['RecipeIngredientParts'])
                recipe['RecipeInstructions'] = self.extract_quoted_strings(recipe['RecipeInstructions'])
        else:
            output = None
        return output

    def save_model(self, filename):
        filename = 'dietrecs_model.sav'
        pickle.dump(self, open(filename, 'wb'))

    @staticmethod
    def load_model(filename):
        return pickle.load(filename)




model = DietModel.load_model('diet_model.sav')

# Get user input
height_cm = float(input("Enter your height in centimeters: "))
weight_kg = float(input("Enter your weight in kilograms: "))
age = int(input("Enter your age: "))
gender = input("Enter your gender (M or F): ")
activity_level = float(input("Enter your activity level (1.2 for sedentary, 1.375 for lightly active, 1.55 for moderately active, 1.725 for very active, and 1.9 for extremely active): "))

# Calculate BMI
bmi = weight_kg / (height_cm / 100) ** 2
print(f"Your BMI is {bmi:.2f}")

# Calculate calorie intake
if gender == "M":
    bmr = 88.36 + (13.4 * weight_kg) + (4.8 * height_cm) - (5.7 * age)
else:
    bmr = 447.6 + (9.2 * weight_kg) + (3.1 * height_cm) - (4.3 * age)

calorie_intake = bmr * activity_level
print(f"Your daily calorie intake should be {calorie_intake:.2f} calories")

# Add suitable ingredients to input_ingredients array based on BMI
input_ingredients = []

if bmi < 18.5:
    input_ingredients = ['rice', 'eggs', 'beans', 'bananas', 'apples', 'carrots']
elif bmi < 25:
    input_ingredients = ['oats', 'chicken', 'yogurt', 'spinach', 'berries', 'nuts']
elif bmi < 30:
    input_ingredients = ['brown rice', 'salmon', 'sweet potato', 'quinoa', 'tomatoes', 'bell peppers']
else:
    input_ingredients = ['kale', 'tofu', 'broccoli', 'mushrooms', 'cauliflower', 'garlic']

print("Your recommended ingredients are:")
print(input_ingredients)

import random

# randomly choose 2 ingredients
input_ingredients = random.sample(input_ingredients, k=2)

print("Chosen ingredients:", input_ingredients)
input_nutrition = [250, 15, 2, 80, 150, 30, 5, 3, 20]
# Call recommend function to get recommended recipes
recommendation_dataframe = model.recommend(input_nutrition,input_ingredients)
output = model.output_recommended_recipes(recommendation_dataframe)

# Print output
if output is None:
    print("No recipes found.")
else:
    for recipe in output:
        print(f"Name: {recipe['Name']}")
        print(f"Ingredients: {recipe['RecipeIngredientParts']}")
        print(f"Instructions: {recipe['RecipeInstructions']}")
        print("--------")

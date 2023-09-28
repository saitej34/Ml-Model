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

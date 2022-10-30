#Import required libraries
from random import Random
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 

import lightgbm as lgb


model_fname = "model.save"
MODEL_NAME = "reg_base_lightgbm_lime"

class Regressor(): 
    
    def __init__(self, boosting_type="gbdt", n_estimators = 250, num_leaves=31, 
                 learning_rate=1e-1, **kwargs) -> None:
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators        
        self.model = self.build_model(**kwargs)
        
        
        
    def build_model(self, **kwargs): 
        model = lgb.LGBMRegressor(
            boosting_type = self.boosting_type,
            num_leaves = self.num_leaves,
            learning_rate = self.learning_rate,
            n_estimators = self.n_estimators,
            num_iterations = 500,
            **kwargs, 
            random_state=42
        )
        return model
    
    
    def fit(self, train_X, train_y):        
                 
    
        self.model.fit(
                X = train_X,
                y = train_y
            )
    
    
    def predict(self, X): 
        preds = self.model.predict(X)
        return preds 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)        

    
    def save(self, model_path): 
        joblib.dump(self, os.path.join(model_path, model_fname))

    @classmethod
    def load(cls, model_path): 
        regressor = joblib.load(os.path.join(model_path, model_fname))
        return regressor


def save_model(model, model_path):    
    model.save(model_path) 
    

def load_model(model_path):     
    model = joblib.load(os.path.join(model_path, model_fname))   
    return model


def get_data_based_model_params(data): 
    return { }


#!/usr/bin/env python

import os, warnings, sys
warnings.filterwarnings('ignore') 

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split


import algorithm.preprocessing.pipeline as pp_pipe
import algorithm.preprocessing.preprocess_utils as pp_utils
import algorithm.utils as utils

#import algorithm.scoring as scoring
from algorithm.model.regressor import Regressor, get_data_based_model_params
from algorithm.utils import get_model_config


# get model configuration parameters 
model_cfg = get_model_config()


def get_trained_model(train_data, data_schema, hyper_params):  
    
    # set random seeds
    utils.set_seeds()
        
    # preprocess data
    print("Pre-processing data...")
    train_X, train_y, inputs_pipeline, _, _ = preprocess_data(train_data, None, data_schema)  
              
    # Create and train model     
    print('Fitting model ...')  
    model= train_model(train_X, train_y, hyper_params)    
    
    return inputs_pipeline, model


def train_model(train_X, train_y, hyper_params):      
    # get model hyper-paameters parameters 
    data_based_params = get_data_based_model_params(train_X)
    #model_params = { **data_based_params, **hyper_params }
    model_params = { **hyper_params, **data_based_params }
    
    # Create and train model   
    model = Regressor(  **model_params )  
    # model.summary()  
    model.fit(  train_X=train_X, train_y=train_y )  
    return model


def preprocess_data(train_data, valid_data, data_schema):
    # print('Preprocessing train_data of shape...', train_data.shape)
    pp_params = pp_utils.get_preprocess_params(train_data, data_schema, model_cfg)   
    
    inputs_pipeline = pp_pipe.get_inputs_pipeline(pp_params, model_cfg)
    inputs = train_data.loc[:, train_data.columns != pp_params["target_attr_name"]]
    train_X = inputs_pipeline.fit_transform(inputs)
    
    # we are not doing any transformation on the targets, but we could have (e.g. standard scaling)
    train_y = train_data[[pp_params["target_attr_name"]]].values
    print("Processed train X/y data shape", train_X.shape, train_y.shape)
    
    if valid_data is not None: 
        valid_X = inputs_pipeline.transform(valid_data.loc[:, train_data.columns != pp_params["target_attr_name"]])
        valid_y = valid_data[[pp_params["target_attr_name"]]]
    else: 
        valid_X, valid_y = None, None
          
    return train_X, train_y, inputs_pipeline, valid_X, valid_y



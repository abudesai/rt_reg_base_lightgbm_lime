from feature_engine.encoding import RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sys, os
import joblib
import pandas as pd 

import algorithm.preprocessing.preprocessors as preprocessors



preprocessor_fname = "preprocessor.save"


def get_inputs_pipeline(pp_params, model_cfg):    
    
    pp_step_names = model_cfg["pp_params"]["pp_step_names"]
    pp_cat_params = model_cfg["pp_params"]["cat_params"] 
    
    pipe_steps = []
        
    # ===== KEEP ONLY COLUMNS WE USE   =====
    pipe_steps.append(
        (
            "column_selector",
            preprocessors.ColumnSelector(
                columns=pp_params['cat_vars']+pp_params['num_vars']
                ),
        )
    )
    
    # ===============================================================
    # ===== CATEGORICAL VARIABLES =====
    
    pipe_steps.append(
        # ===== CAST CAT VARS TO STRING =====
        (
            "string_type_caster",
            preprocessors.StringTypeCaster(
                cat_vars=pp_params['cat_vars']
                ),
        )
    )        
    
    # impute categorical na with string 'missing'           
    if len(pp_params['cat_na_impute_with_str_missing']):
        pipe_steps.append(
            (
                "cat_imputer_missing_tag",
                CategoricalImputer(
                    imputation_method="missing",
                    variables=pp_params["cat_na_impute_with_str_missing"],
                ),
            )
        )
        
    # impute categorical na with most frequent category
    if len(pp_params['cat_na_impute_with_freq']):
        pipe_steps.append(
            (
                "cat_imputer_most_frequent",
                CategoricalImputer(
                    imputation_method="frequent",
                    variables=pp_params["cat_na_impute_with_freq"],
                ),
            )
        )
        
    if len(pp_params['cat_vars']):
        # rare-label encoder
        pipe_steps.append(
            (
                "cat_rare_label_encoder",
                RareLabelEncoder(
                    tol=pp_cat_params["rare_perc_threshold"], 
                    n_categories=1, 
                    variables=pp_params["cat_vars"]
                ),
            )
        )
        
        # one-hot encoder cat vars
        pipe_steps.append(
            (
                'cat_one_hot_encoder',
                preprocessors.OneHotEncoderMultipleCols(                    
                    ohe_columns=pp_params["cat_vars"],
                ),
            )
        )
        
        # == DROP UNWANTED FEATURES 
        pipe_steps.append(
            (
                "feature_dropper",
                preprocessors.ColumnSelector(
                    columns=pp_params["cat_vars"],
                    selector_type='drop')
            )
        )
        
    # ===============================================================
    # ===== NUMERICAL VARIABLES =====
    if len(pp_params['num_vars']):  
        pipe_steps.append(
            # ===== CAST CAT VARS TO STRING =====
                (
                    "float_type_caster",
                    preprocessors.FloatTypeCaster(
                        num_vars=pp_params['num_vars']
                        ),
                )
            )
    
    
    if len(pp_params['num_na']):
        # add missing indicator to nas in numerical features 
        pipe_steps.append(
            (
                "numerical_missing_indicator",
                AddMissingIndicator(variables=pp_params["num_na"]),
            )
        )
        # impute numerical na with the mean
        pipe_steps.append(
            (
                "numerical_missing_mean_imputer",
                MeanMedianImputer(
                    imputation_method="mean",
                    variables=pp_params["num_na"],
                )
            )
        )    
    
    # Transform numerical variables - standard scale
    if len(pp_params['num_vars']):   
        
        # Standard Scale num vars
        pipe_steps.append(
            (
                "numerical_standard_scaler", 
                SklearnTransformerWrapper(                    
                    StandardScaler(),
                    variables=pp_params["num_vars"] 
                ),    
            )
        )     
        
        # Clip values to +/- 4 std devs to remove outliers
        pipe_steps.append(
            (
                "value_clipper", 
                preprocessors.ValueClipper(
                    fields_to_clip=pp_params["num_vars"],
                    min_val=-4.0,   # - 4 std dev
                    max_val=4.0,    # + 4 std dev    
                ),    
            )
        )      
    
    # ===============================================================          
    input_pipeline = Pipeline( pipe_steps )    
    return input_pipeline

        

def save_preprocessor(inputs_pipeline, file_path):
    joblib.dump(inputs_pipeline, os.path.join(file_path, preprocessor_fname), protocol=3)    
    

def load_preprocessor(file_path):
    inputs_pipeline = joblib.load(os.path.join(file_path, preprocessor_fname))       
    return inputs_pipeline
    
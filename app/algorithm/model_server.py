import numpy as np, pandas as pd
import json
import os, sys
import pprint
from interpret.blackbox import LimeTabular

import algorithm.utils as utils
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.regressor as regressor


# get model configuration parameters
model_cfg = utils.get_model_config()


class ModelServer:
    def __init__(self, model_path, data_schema):
        self.model_path = model_path
        self.preprocessor = None
        self.model = None
        self.data_schema = data_schema
        self.id_field_name = self.data_schema["inputDatasets"][
            "regressionBaseMainInput"
        ]["idField"]
        self.has_local_explanations = True
        self.MAX_LOCAL_EXPLANATIONS = 3

    def _get_preprocessor(self):
        if self.preprocessor is None:
            self.preprocessor = pipeline.load_preprocessor(self.model_path)
        return self.preprocessor

    def _get_model(self):
        if self.model is None:
            self.model = regressor.load_model(self.model_path)
        return self.model

    def predict(self, data):

        preprocessor = self._get_preprocessor()
        model = self._get_model()

        if preprocessor is None:
            raise Exception("No preprocessor found. Did you train first?")
        if model is None:
            raise Exception("No model found. Did you train first?")

        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        pred_X = preprocessor.transform(data)
        # make predictions
        preds = model.predict(pred_X)
        # return te prediction df with the id and prediction fields
        preds_df = data[[self.id_field_name]].copy()
        preds_df["prediction"] = np.round(preds, 4)
        return preds_df

    def _get_preds_array(self, X):
        model = self._get_model()
        preds = model.predict(X)
        return preds

    def explain_local(self, data):

        if data.shape[0] > self.MAX_LOCAL_EXPLANATIONS:
            msg = f"""Warning!
            Maximum {self.MAX_LOCAL_EXPLANATIONS} explanation(s) allowed at a time. 
            Given {data.shape[0]} samples. 
            Selecting top {self.MAX_LOCAL_EXPLANATIONS} sample(s) for explanations."""
            print(msg)

        model = self._get_model()
        preprocessor = self._get_preprocessor()

        data2 = data.head(self.MAX_LOCAL_EXPLANATIONS)
        # transform data - returns a dict of X (transformed input features) and Y(targets, if any, else None)
        pred_X = preprocessor.transform(data2)

        print(f"Generating local explanations for {pred_X.shape[0]} sample(s).")
        lime = LimeTabular(
            predict_fn=self._get_preds_array, data=model.train_X, random_state=1
        )

        # Get local explanations
        lime_local = lime.explain_local(
            X=pred_X, y=None, name=f"{regressor.MODEL_NAME} local explanations"
        )

        ids = list(data2[self.id_field_name])
        explanations = []
        for i, sample_exp in enumerate(lime_local._internal_obj["specific"]):
            sample_expl_dict = {}
            # intercept
            sample_expl_dict["baseline"] = np.round(sample_exp["extra"]["scores"][0], 5)

            sample_expl_dict["feature_scores"] = {
                f: np.round(s, 5)
                for f, s in zip(sample_exp["names"], sample_exp["scores"])
            }

            explanations.append(
                {
                    self.id_field_name: ids[i],
                    "prediction": np.round(sample_exp["perf"]["predicted"], 5),
                    "explanations": sample_expl_dict,
                }
            )
        explanations = {"predictions": explanations}
        explanations = json.dumps(explanations, cls=utils.NpEncoder, indent=2)
        return explanations

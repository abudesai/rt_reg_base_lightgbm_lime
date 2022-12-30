LightGBM Regressor using Scikit-Learn and LIME for Regression

- lightgbm
- lime
- black box
- bagging
- ensemble
- local explanations
- xai
- python
- feature engine
- scikit optimize
- flask
- nginx
- gunicorn
- docker

This is an explainable version of LightGBM regressor, with LIME for model interpretability.

Model explainability is provided using LIME. Local explanations are provided here. Explanations at each instance can be understood using LIME. These explanations can be viewed by means of various plots.

The data preprocessing step includes:

- for categorical variables
  - Handle missing values in categorical:
    - When missing values are frequent, then impute with 'missing' label
    - When missing values are rare, then impute with most frequent
- Group rare labels to reduce number of categories
- One hot encode categorical variables

- for numerical variables

  - Add binary column to represent 'missing' flag for missing values
  - Impute missing values with mean of non-missing
  - Standard scale data after yeo-johnson
  - Clip values to +/- 4.0 (to remove outliers)

HPT based on Bayesian optimization is included for tuning Random Forest hyper-parameters.The tunable hyperparameters are: boosting_type, n_estimators, num_leaves, and learning_rate.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as abalone, ailerons, auto_prices, computer_activity, diamond, energy, heart_disease, house_prices, medical_costs, and white_wine.

The main programming language is Python. Other tools include Scikit-Learn for main algorithm, interpret package for model explainability, feature-engine for preprocessing, Scikit-Optimize for HPT, Flask + Nginx + gunicorn for web service. The web service provides three endpoints- /ping for health check, /infer for predictions in real time and /explain to generate local explanations.

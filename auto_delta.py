#!/usr/bin/env python3

# set params
data_path = "ocv_features_normalized.csv"
target_column_name = "expt_ocv"

primary_column_name = "md_ocv"
using_primary_as_training = True

drop_col_list = ["name", "Li_num", "Unnamed: 0", "cif_idx", "original_md_ocv"]

# can be changed
self_define_train_test = True
test_proportion = 0.2

from pprint import pprint
#import sklearn.datasets
#import sklearn.metrics
from sklearn.model_selection import train_test_split

import autosklearn.regression
#import matplotlib.pyplot as plt
#import numpy as np
#import utils
#from utils import evaluation, plot_scatter

import pandas as pd

df = pd.read_csv(data_path)
#train, test = train_test_split(df["name"].unique(), test_size=0.2, random_state=279)
train, test = train_test_split(df["name"].unique(), test_size=0.2, random_state=461)
train_data = df[df["name"].isin(train)]
test_data = df[df["name"].isin(test)]
X_train = train_data.drop(["name", "Li_num", "Unnamed: 0", "cif_idx", "expt_ocv", "original_md_ocv"], axis=1).values.astype(float)
y_train = train_data["expt_ocv"].values.astype(float)
X_test = test_data.drop(["name", "Li_num", "Unnamed: 0", "cif_idx", "expt_ocv", "original_md_ocv"], axis=1).values.astype(float)
y_test = test_data["expt_ocv"].values.astype(float)

automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=30,
#    per_run_time_limit=30,
)
automl.fit(X_train, y_train)
print(automl.leaderboard())
pprint(automl.show_models(), indent=5)

for weight, model in automl.get_models_with_weights():
    # Obtain the step of the underlying scikit-learn pipeline
    print(weight)

#y_train_pred = automl.predict(X_train)
#y_test_pred = automl.predict(X_test)

import joblib
joblib.dump(automl, 'model.joblib')

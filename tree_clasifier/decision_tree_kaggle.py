import numpy as np
import pandas as pd
import os as os

from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif
from preprocess import *
from random_forest import RandomForest
from bagging import Bagging
from simple_tree import SimpleTree
from boosting import AdaBoost

from copy import deepcopy
import math
import ast

# params = {'imputer_type': 'SI', 'strategy': 'median', 'scaling': False, 'resampling': False, 'select_features': False, 'model': 'BAGGING', 'n_trees': 900, 'depth': 5, 'criterion': 'log_loss', 'splitter': 'random', 'balance_weights': 'balanced'}
params_path = "submission_kaggle/params.txt"

with open(params_path) as f: 
    data = f.read() 

params = ast.literal_eval(data) 

print(params)

np.random.seed(42)

# Cargamos csv con los datos de train
df_train = pd.read_csv(
    "../data_raw/training_data.csv", sep=",", header=0, na_values=["?", "", "NA"]
)
# Cargamos csv con los datos de test
df_test = pd.read_csv(
    "../data_raw/test_data.csv", sep=",", header=0, na_values=["?", "", "NA"]
)

df_train, df_test = encoder(df_train, df_test)

if params["removing_outlier"]:
    df_train = outliers_remove(df_train.dropna())

if params["imputer_type"] == "KNN":
    imputer_func = KNNImputer(n_neighbors=params["n_neighbors"])

elif params["imputer_type"] == "SI":
    imputer_func = SimpleImputer(missing_values=np.nan, strategy=params["strategy"])

df_train, df_test = imputation_test(
    df_train, df_test, imputer_func
)

if params["scaling"]:
    if params["scaling_method"] == "STANDARD":
        scaler = StandardScaler()
    elif params["scaling_method"] == "MINMAX":
        scaler = MinMaxScaler()

    df_train, df_test = scale_test(df_train, df_test, scaler)

if params["resampling"]:
    df_train = train_resampling(df_train)

max_features = len(df_train.columns) - 2
n_selected_features = max_features

if params["PCA_decomposition"]:
    df_train, df_test, n_selected_features = apply_PCA_test(df_train, df_test, params["n_components"])

if params["select_features"]:

    if params["feature_selector_type"] == "KBEST":
        feature_selector = SelectKBest(f_classif, k=params["k"])

    elif params["feature_selector_type"] == "KPERCENTILE":
        feature_selector = SelectPercentile(f_classif, percentile=params["percentile"])

    df_train, df_test, n_selected_features = feature_selection_test(
        df_train, df_test, feature_selector
    )

df_train = df_train.drop(["ID"], axis=1)
df_train_label = df_train["RATE"]
df_train = df_train.drop(["RATE"], axis=1)

df_test_id = df_test["ID"]
df_test = df_test.drop(["ID"], axis=1)

if params["model"] == "BOOSTING":
    boosting_model = AdaBoost()

    boosting_model.fit(
        df_train,
        df_train_label,
        n_trees=params["n_trees"],
    )
    pred = boosting_model.predict(df_test)

    submission = {"ID": df_test_id.astype(int).to_list(), "RATE": pred}
    submission = pd.DataFrame(submission)

if params["model"] == "RF":
    n_features_min = int(math.sqrt(n_selected_features)) - 2
    if n_features_min < 1:
        n_features_min = 1
    n_features_max = int(math.sqrt(n_selected_features)) + 2

    random_forest_model = RandomForest()

    random_forest_model.fit(
        df_train,
        df_train_label,
        depth=params["depth"],
        n_trees=params["n_trees"],
        n_features=params["n_features"],
        criterion=params["criterion"],
        splitter=params["splitter"],
        balance_weights=params["balance_weights"],
    )
    pred = random_forest_model.predict(df_test_id)

    submission = {"ID": df_test_id.astype(int).to_list(), "RATE": pred}
    submission = pd.DataFrame(submission)

elif params["model"] == "BAGGING":

    bagging_model = Bagging()

    bagging_model.fit(
        df_train,
        df_train_label,
        depth=params["depth"],
        n_trees=params["n_trees"],
        criterion=params["criterion"],
        splitter=params["splitter"],
        balance_weights=params["balance_weights"],
    )
    pred = bagging_model.predict(df_test)

    submission = {"ID": df_test_id.astype(int).to_list(), "RATE": pred}
    submission = pd.DataFrame(submission)

elif params["model"] == "SIMPLE":

    simple_model = SimpleTree()

    simple_model.fit(
        df_train,
        df_train_label,
        depth=params["depth"],
        criterion=params["criterion"],
        splitter=params["splitter"],
        balance_weights=params["balance_weights"],
    )
    pred = simple_model.predict(df_test)

    submission = {"ID": df_test_id.astype(int).to_list(), "RATE": pred}
    submission = pd.DataFrame(submission)


os.makedirs("submission_kaggle", exist_ok=True)

df_train.to_csv(
    os.path.join(
        "submission_kaggle",
        "train_preprocess.csv"
    ),
    index=False,
)

df_test.to_csv(
    os.path.join(
        "submission_kaggle",
        "test_preprocess.csv"
    ),
    index=False,
)

submission.to_csv(
    os.path.join(
        "submission_kaggle",
        "submission.csv"
    ),
    index=False,
)

print(submission)

print(params)


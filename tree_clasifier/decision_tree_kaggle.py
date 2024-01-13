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

from imblearn.over_sampling import SMOTEN

import optuna

from copy import deepcopy
import logging
import math

def encoder(df_train, df_test):
    orden_x24 = ["VLOW", "LOW", "MED", "HIGH", "VHIGH"]

    ordinal_encoder_x24 = OrdinalEncoder(categories=[orden_x24], dtype=int)

    df_train["X24"] = ordinal_encoder_x24.fit_transform(df_train[["X24"]])
    df_test["X24"] = ordinal_encoder_x24.transform(df_test[["X24"]])

    orden_x25 = ["NO", "YES"]

    ordinal_encoder_x25 = OrdinalEncoder(categories=[orden_x25], dtype=int)

    df_train["X25"] = ordinal_encoder_x25.fit_transform(df_train[["X25"]])
    df_test["X25"] = ordinal_encoder_x25.transform(df_test[["X25"]])

    df_train_encoded = df_train.copy()
    df_test_encoded = df_test.copy()

    df_train_encoded.loc[df_train["X30"] == "VTKGN", "X30"] = 1
    df_train_encoded.loc[df_train["X30"] != "VTKGN", "X30"] = 0

    df_test_encoded.loc[df_test["X30"] == "VTKGN", "X30"] = 1
    df_test_encoded.loc[df_test["X30"] != "VTKGN", "X30"] = 0

    df_train_encoded["X30"] = pd.to_numeric(df_train_encoded["X30"])
    df_test_encoded["X30"] = pd.to_numeric(df_train_encoded["X30"])

    return df_train_encoded, df_test_encoded


def random_forest(
    df_train,
    df_test,
    depth=4,
    n_trees=100,
    n_features=7,
    criterion="gini",
    splitter="best",
    balance_weights=None,
):
    clf = tree.DecisionTreeClassifier(
        max_depth=depth,
        criterion=criterion,
        splitter=splitter,
        class_weight=balance_weights,
    )
    df_train = df_train.drop(["ID"], axis=1)
    df_train_label = df_train["RATE"]
    df_train = df_train.drop(["RATE"], axis=1)

    df_test_id = df_test["ID"]
    df_test = df_test.drop(["ID"], axis=1)

    filter_list = []
    for i in range(n_trees):
        filter_list.append(
            [
                col
                for col in sorted(np.random.choice(df_train.columns, n_features, False))
            ]
        )

    fitted_clf = []
    test_list = []
    oob_errors = []
    for filter in filter_list:
        X_train, X_test, y_train, y_test = train_test_split(
            df_train, df_train_label, test_size=1 / 3
        )
        fitted_clf.append(deepcopy(clf.fit(X_train[filter], y_train)))

        pred = clf.predict(X_test[filter])

        oob_errors.append(accuracy_score(pred, y_test))

    logging.info("RANDOM FOREST OOB: {}".format(np.mean(np.array(oob_errors))))

    pred_list = []
    for clf, filter in zip(fitted_clf, filter_list):
        pred = clf.predict(df_test[filter])
        pred_list.append(pred)

    pred_submission = []

    for pred in np.array(pred_list).T:
        pred_submission.append(pd.Series(pred).value_counts().idxmax())

    submission = {"ID": df_test_id.astype(int).to_list(), "RATE": pred_submission}
    submission = pd.DataFrame(submission)

    return submission


def bagging(
    df_train,
    df_test,
    depth=4,
    n_trees=100,
    criterion="gini",
    splitter="best",
    balance_weights=None,
):
    clf = tree.DecisionTreeClassifier(
        max_depth=depth,
        criterion=criterion,
        splitter=splitter,
        class_weight=balance_weights,
    )
    df_train = df_train.drop(["ID"], axis=1)
    df_train_label = df_train["RATE"]
    df_train = df_train.drop(["RATE"], axis=1)

    df_test_id = df_test["ID"]
    df_test = df_test.drop(["ID"], axis=1)

    fitted_clf = []
    test_list = []
    oob_errors = []
    for i in range(n_trees):
        X_train, X_test, y_train, y_test = train_test_split(
            df_train, df_train_label, test_size=1 / 3
        )

        fitted_clf.append(deepcopy(clf.fit(X_train, y_train)))

        pred = clf.predict(X_test)

        oob_errors.append(accuracy_score(pred, y_test))

    logging.info("BAGGING OOB: {}".format(np.mean(np.array(oob_errors))))

    pred_list = []
    for clf in fitted_clf:
        pred = clf.predict(df_test)
        pred_list.append(pred)

    pred_submission = []

    for pred in np.array(pred_list).T:
        pred_submission.append(pd.Series(pred).value_counts().idxmax())

    submission = {"ID": df_test_id.astype(int).to_list(), "RATE": pred_submission}
    submission = pd.DataFrame(submission)

    return submission


def prediction(
    df_train, df_test, depth=4, criterion="gini", splitter="best", balance_weights=None
):
    df_train = df_train.drop(["ID"], axis=1)
    df_train_label = df_train["RATE"]
    df_train = df_train.drop(["RATE"], axis=1)

    df_test_id = df_test["ID"]
    df_test = df_test.drop(["ID"], axis=1)

    clf = tree.DecisionTreeClassifier(
        max_depth=depth,
        criterion=criterion,
        splitter=splitter,
        class_weight=balance_weights,
    )
    clf.fit(df_train, df_train_label)

    pred = clf.predict(df_test)

    submission = {"ID": df_test_id.astype(int).to_list(), "RATE": pred}
    submission = pd.DataFrame(submission)

    return submission


def train_resampling(df_train):
    train_id = df_train["ID"].to_frame()
    train_labels = df_train["RATE"].to_frame()
    train_data = df_train.drop(["RATE", "ID"], axis=1, inplace=False)

    sm = SMOTEN(random_state=42)

    X_res, y_res = sm.fit_resample(train_data, train_labels)

    train_id_dict = train_id.to_dict()
    key = len(train_id)
    for i in np.arange(9000, 9000 + (len(X_res) - len(train_data))):
        train_id_dict["ID"][key] = i
        key += 1

    train_id = pd.DataFrame.from_dict(train_id_dict)

    df_train = train_id.join(other=[X_res, y_res])
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_train


def scale(df_train, df_test, scaler):
    train_id = df_train["ID"].to_frame()
    train_labels = df_train["RATE"].to_frame()
    train_data = df_train.drop(["RATE", "ID"], axis=1, inplace=False)

    scaler.fit(train_data)
    imputed_X_train = pd.DataFrame(
        scaler.transform(train_data), columns=train_data.columns
    )

    df_train = train_id.join(other=[imputed_X_train, train_labels])

    test_id = df_test["ID"].to_frame()
    test_data = df_test.drop(["ID"], axis=1, inplace=False)

    imputed_X_test = pd.DataFrame(
        scaler.transform(test_data), columns=test_data.columns
    )

    df_test = test_id.join(other=imputed_X_test)

    return df_train, df_test


def imputation(df_train, df_test, imputer_func):
    train_id = df_train["ID"].to_frame()
    train_labels = df_train["RATE"].to_frame()
    train_data = df_train.drop(["RATE", "ID"], axis=1, inplace=False)

    imp_train = imputer_func.fit(train_data)
    imputed_X_train = pd.DataFrame(
        imp_train.transform(train_data), columns=train_data.columns
    )

    df_train = train_id.join(other=[imputed_X_train, train_labels])

    test_id = df_test["ID"].to_frame()
    test_data = df_test.drop(["ID"], axis=1, inplace=False)

    imputed_X_test = pd.DataFrame(
        imp_train.transform(test_data), columns=test_data.columns
    )

    df_test = test_id.join(other=imputed_X_test)

    return df_train, df_test


def feature_selection(df_train, df_test, feature_selector):
    train_id = df_train["ID"].to_frame()
    train_labels = df_train["RATE"].to_frame()
    train_data = df_train.drop(["RATE", "ID"], axis=1, inplace=False)

    feature_train = feature_selector.fit(train_data, train_labels.to_numpy().flatten())

    selection_train = pd.DataFrame(
        feature_train.transform(train_data),
        columns=feature_train.get_feature_names_out(train_data.columns),
    )

    n_features = len(feature_train.get_feature_names_out(train_data.columns))

    df_train = train_id.join(other=[selection_train, train_labels])

    test_id = df_test["ID"].to_frame()
    test_data = df_test.drop(["ID"], axis=1, inplace=False)

    selection_test = pd.DataFrame(
        feature_train.transform(test_data),
        columns=feature_train.get_feature_names_out(test_data.columns),
    )

    df_test = test_id.join(other=selection_test)

    return df_train, df_test, n_features

params = {'imputer_type': 'SI', 'strategy': 'median', 'scaling': False, 'resampling': False, 'select_features': False, 'model': 'BAGGING', 'n_trees': 900, 'depth': 5, 'criterion': 'log_loss', 'splitter': 'random', 'balance_weights': 'balanced'}

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


if params["imputer_type"] == "KNN":
    imputer_func = KNNImputer(n_neighbors=params["n_neighbors"])

elif params["imputer_type"] == "SI":
    imputer_func = SimpleImputer(missing_values=np.nan, strategy=params["strategy"])

df_train, df_test = imputation(
    df_train, df_test, imputer_func
)

if params["scaling"]:
    if params["scaling_method"] == "STANDARD":
        scaler = StandardScaler()
    elif params["scaling_method"] == "MINMAX":
        scaler = MinMaxScaler()

    df_train, df_validation, df_test = scale(df_train, df_validation, df_test, scaler)

if params["resampling"]:
    df_train = train_resampling(df_train)

if params["select_features"]:

    if params["feature_selector_type"] == "KBEST":
        feature_selector = SelectKBest(f_classif, k=params["k"])

    elif params["feature_selector_type"] == "KPERCENTILE":
        feature_selector = SelectPercentile(f_classif, percentile=params["percentile"])

    df_train, df_test, n_selected_features = feature_selection(
        df_train, df_test, feature_selector
    )

else:
    n_selected_features = len(df_train.columns) - 2

if params["model"] == "RF":
    n_features_min = int(math.sqrt(n_selected_features)) - 2
    if n_features_min < 1:
        n_features_min = 1
    n_features_max = int(math.sqrt(n_selected_features)) + 2
    submission = random_forest(
        df_train,
        df_test,
        depth=params["depth"],
        n_trees=params["n_trees"],
        n_features=params["n_features"],
        criterion=params["criterion"],
        splitter=params["splitter"],
        balance_weights=params["balance_weights"],
    )

elif params["model"] == "BAGGING":
    submission = bagging(
        df_train,
        df_test,
        depth=params["depth"],
        n_trees=params["n_trees"],
        criterion=params["criterion"],
        splitter=params["splitter"],
        balance_weights=params["balance_weights"]
    )

elif params["model"] == "SIMPLE":
    submission = prediction(
        df_train,
        df_test,
        depth=params["depth"],
        criterion=params["criterion"],
        splitter=params["splitter"],
        balance_weights=params["balance_weights"]
    )

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

print(params)


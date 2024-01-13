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

from random_forest import RandomForest


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


def train_validation_split(df_train):
    df_train_label = df_train["RATE"].to_frame()
    df_train = df_train.drop(["RATE"], axis=1, inplace=False)

    (
        df_train,
        df_validation,
        df_train_label,
        df_validation_label,
    ) = train_test_split(
        df_train,
        df_train_label,
        test_size=int(len(df_train) * 0.3),
        random_state=42,
        stratify=df_train_label,
    )

    df_train = df_train.join(other=df_train_label).sort_index().reset_index(drop=True)
    df_validation = (
        df_validation.join(other=df_validation_label)
        .sort_index()
        .reset_index(drop=True)
    )

    return df_train, df_validation


# def random_forest(
#     df_train,
#     df_test,
#     depth=4,
#     n_trees=100,
#     n_features=7,
#     criterion="gini",
#     splitter="best",
#     balance_weights=None,
# ):
#     clf = tree.DecisionTreeClassifier(
#         max_depth=depth,
#         criterion=criterion,
#         splitter=splitter,
#         class_weight=balance_weights,
#     )
#     df_train = df_train.drop(["ID"], axis=1)
#     df_train_label = df_train["RATE"]
#     df_train = df_train.drop(["RATE"], axis=1)

#     df_test_id = df_test["ID"]
#     df_test = df_test.drop(["ID"], axis=1)

#     filter_list = []
#     for i in range(n_trees):
#         filter_list.append(
#             [
#                 col
#                 for col in sorted(np.random.choice(df_train.columns, n_features, False))
#             ]
#         )

#     fitted_clf = []
#     test_list = []
#     oob_errors = []
#     for filter in filter_list:
#         X_train, X_test, y_train, y_test = train_test_split(
#             df_train, df_train_label, test_size=1 / 3
#         )
#         fitted_clf.append(deepcopy(clf.fit(X_train[filter], y_train)))

#         pred = clf.predict(X_test[filter])

#         oob_errors.append(accuracy_score(pred, y_test))

#     logging.info("RANDOM FOREST OOB: {}".format(np.mean(np.array(oob_errors))))

#     pred_list = []
#     for clf, filter in zip(fitted_clf, filter_list):
#         pred = clf.predict(df_test[filter])
#         pred_list.append(pred)

#     pred_submission = []

#     for pred in np.array(pred_list).T:
#         pred_submission.append(pd.Series(pred).value_counts().idxmax())

#     submission = {"ID": df_test_id.astype(int).to_list(), "RATE": pred_submission}
#     submission = pd.DataFrame(submission)

#     return submission


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


def scale(df_train, df_validation, df_test, scaler):
    train_id = df_train["ID"].to_frame()
    train_labels = df_train["RATE"].to_frame()
    train_data = df_train.drop(["RATE", "ID"], axis=1, inplace=False)

    scaler.fit(train_data)
    imputed_X_train = pd.DataFrame(
        scaler.transform(train_data), columns=train_data.columns
    )

    df_train = train_id.join(other=[imputed_X_train, train_labels])

    validation_id = df_validation["ID"].to_frame()
    validation_labels = df_validation["RATE"].to_frame()
    validation_data = df_validation.drop(["RATE", "ID"], axis=1, inplace=False)

    imputed_X_validation = pd.DataFrame(
        scaler.transform(validation_data), columns=train_data.columns
    )

    df_validation = validation_id.join(other=[imputed_X_validation, validation_labels])

    test_id = df_test["ID"].to_frame()
    test_data = df_test.drop(["ID"], axis=1, inplace=False)

    imputed_X_test = pd.DataFrame(
        scaler.transform(test_data), columns=test_data.columns
    )

    df_test = test_id.join(other=imputed_X_test)

    return df_train, df_validation, df_test


def imputation(df_train, df_validation, df_test, imputer_func):
    train_id = df_train["ID"].to_frame()
    train_labels = df_train["RATE"].to_frame()
    train_data = df_train.drop(["RATE", "ID"], axis=1, inplace=False)

    imp_train = imputer_func.fit(train_data)
    imputed_X_train = pd.DataFrame(
        imp_train.transform(train_data), columns=train_data.columns
    )

    df_train = train_id.join(other=[imputed_X_train, train_labels])

    validation_id = df_validation["ID"].to_frame()
    validation_labels = df_validation["RATE"].to_frame()
    validation_data = df_validation.drop(["RATE", "ID"], axis=1, inplace=False)

    imputed_X_validation = pd.DataFrame(
        imp_train.transform(validation_data), columns=validation_data.columns
    )

    df_validation = validation_id.join(other=[imputed_X_validation, validation_labels])

    test_id = df_test["ID"].to_frame()
    test_data = df_test.drop(["ID"], axis=1, inplace=False)

    imputed_X_test = pd.DataFrame(
        imp_train.transform(test_data), columns=test_data.columns
    )

    df_test = test_id.join(other=imputed_X_test)

    return df_train, df_validation, df_test


def feature_selection(df_train, df_validation, df_test, feature_selector):
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

    validation_id = df_validation["ID"].to_frame()
    validation_labels = df_validation["RATE"].to_frame()
    validation_data = df_validation.drop(["RATE", "ID"], axis=1, inplace=False)

    selection_validation = pd.DataFrame(
        feature_train.transform(validation_data),
        columns=feature_train.get_feature_names_out(validation_data.columns),
    )

    df_validation = validation_id.join(other=[selection_validation, validation_labels])

    test_id = df_test["ID"].to_frame()
    test_data = df_test.drop(["ID"], axis=1, inplace=False)

    selection_test = pd.DataFrame(
        feature_train.transform(test_data),
        columns=feature_train.get_feature_names_out(test_data.columns),
    )

    df_test = test_id.join(other=selection_test)

    return df_train, df_validation, df_test, n_features


def objective(trial):
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

    df_train, df_validation = train_validation_split(df_train)

    imputer_type = trial.suggest_categorical("imputer_type", ["KNN", "SI"])

    if imputer_type == "KNN":
        n_neighbors = trial.suggest_int("n_neighbors", 2, 10)
        imputer_func = KNNImputer(n_neighbors=n_neighbors)

    elif imputer_type == "SI":
        strategy = trial.suggest_categorical(
            "strategy", ["mean", "median", "most_frequent"]
        )
        imputer_func = SimpleImputer(missing_values=np.nan, strategy=strategy)

    df_train, df_validation, df_test = imputation(
        df_train, df_validation, df_test, imputer_func
    )

    scaling = trial.suggest_categorical("scaling", [True, False])

    if scaling:
        scaling_method = trial.suggest_categorical("scaling_method", ["STANDARD", "MINMAX"])
        if scaling_method == "STANDARD":
            scaler = StandardScaler()
        elif scaling_method == "MINMAX":
            scaler = MinMaxScaler()

        df_train, df_validation, df_test = scale(df_train, df_validation, df_test, scaler)

    resampling = trial.suggest_categorical("resampling", [True, False])

    if resampling:
        df_train = train_resampling(df_train)

    select_features = trial.suggest_categorical("select_features", [True, False])

    if select_features:
        feature_selector_type = trial.suggest_categorical(
            "feature_selector_type", ["KBEST", "KPERCENTILE"]
        )

        if feature_selector_type == "KBEST":
            k = trial.suggest_int("k", 2, 39)
            feature_selector = SelectKBest(f_classif, k=k)

        elif feature_selector_type == "KPERCENTILE":
            percentile = trial.suggest_int("percentile", low=10, high=90, step=10)
            feature_selector = SelectPercentile(f_classif, percentile=percentile)

        df_train, df_validation, df_test, n_selected_features = feature_selection(
            df_train, df_validation, df_test, feature_selector
        )

    else:
        n_selected_features = len(df_train.columns) - 2

    model = trial.suggest_categorical("model", ["RF"])#, "BAGGING", "SIMPLE"])
    n_trees = trial.suggest_int("n_trees", low=100, high=1000, step=100)
    depth = trial.suggest_int("depth", 2, 6)
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    splitter = trial.suggest_categorical("splitter", ["best", "random"])
    balance_weights = trial.suggest_categorical("balance_weights", ["balanced", None])

    df_train = df_train.drop(["ID"], axis=1)
    df_train_label = df_train["RATE"]
    df_train = df_train.drop(["RATE"], axis=1)

    df_validation_id = df_validation["ID"]
    df_validation = df_validation.drop(["ID"], axis=1)
    df_validation_nolabel = df_validation.drop(["RATE"], axis=1, inplace=False)
    
    if model == "RF":
        n_features_min = int(math.sqrt(n_selected_features)) - 2
        if n_features_min < 1:
            n_features_min = 1
        n_features_max = int(math.sqrt(n_selected_features)) + 2
        n_features = trial.suggest_int("n_features", n_features_min, n_features_max)


        random_forest_model = RandomForest()

        random_forest_model.fit(
            df_train,
            df_train_label,
            depth=depth,
            n_trees=n_trees,
            n_features=n_features,
            criterion=criterion,
            splitter=splitter,
            balance_weights=balance_weights,
        )
        pred = random_forest_model.predict(df_validation_nolabel)

        submission = {"ID": df_validation_id.astype(int).to_list(), "RATE": pred}
        submission = pd.DataFrame(submission)
        accuracy = accuracy_score(submission["RATE"], df_validation["RATE"])

    elif model == "BAGGING":

        df_validation_nolabel = df_validation.drop(["RATE"], axis=1, inplace=False)

        validation_pred_bagging = bagging(
            df_train,
            df_validation_nolabel,
            depth=depth,
            n_trees=n_trees,
            criterion=criterion,
            splitter=splitter,
            balance_weights=balance_weights,
        )
        accuracy = accuracy_score(
            validation_pred_bagging["RATE"], df_validation["RATE"]
        )
        submission = bagging(
            df_train,
            df_test,
            depth=depth,
            n_trees=n_trees,
            criterion=criterion,
            splitter=splitter,
            balance_weights=balance_weights,
        )

    elif model == "SIMPLE":

        df_validation_nolabel = df_validation.drop(["RATE"], axis=1, inplace=False)

        validation_pred = prediction(
            df_train,
            df_validation_nolabel,
            depth=depth,
            criterion=criterion,
            splitter=splitter,
            balance_weights=balance_weights,
        )
        accuracy = accuracy_score(validation_pred["RATE"], df_validation["RATE"])
        submission = prediction(
            df_train,
            df_test,
            depth=depth,
            criterion=criterion,
            splitter=splitter,
            balance_weights=balance_weights,
        )

    n_trial = trial.number

    os.makedirs(
        os.path.join("submission_optuna", "trial_{}".format(n_trial)), exist_ok=True
    )

    df_train.to_csv(
        os.path.join(
            "submission_optuna",
            "trial_{}".format(n_trial),
            "train_preprocess_{}.csv".format(n_trial),
        ),
        index=False,
    )
    
    df_validation.to_csv(
        os.path.join(
            "submission_optuna",
            "trial_{}".format(n_trial),
            "validation_preprocess_{}.csv".format(n_trial),
        ),
        index=False,
    )
    
    # df_test.to_csv(
    #     os.path.join(
    #         "submission_optuna",
    #         "trial_{}".format(n_trial),
    #         "test_preprocess_{}.csv".format(n_trial),
    #     ),
    #     index=False,
    # )

    submission.to_csv(
        os.path.join(
            "submission_optuna",
            "trial_{}".format(n_trial),
            "submission_{}.csv".format(n_trial),
        ),
        index=False,
    )

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)

    print(study.best_trial)
    print()
    print(study.best_params)
    print()
    print(study.best_trial.value)
    print()
    print(
        os.path.join(
            "submission_optuna",
            "trial_{}".format(study.best_trial.number),
            "submission_{}.csv".format(study.best_trial.number),
        )
    )

    df = study.trials_dataframe()
    df.to_csv("trials_history.csv", sep=";", index=False)

    print(df)

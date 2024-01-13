import numpy as np
import pandas as pd
import os as os
import logging
import math
import optuna

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from preprocess import *
from random_forest import RandomForest
from bagging import Bagging
from simple_tree import SimpleTree
from boosting import AdaBoost

import argparse


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

    removing_outlier = trial.suggest_categorical("removing_outlier", [True, False])

    if removing_outlier:
        df_train = outliers_remove(df_train.dropna())
        # print(df_train)

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

    max_features = len(df_train.columns) - 2
    n_selected_features = max_features

    PCA_decomposition = trial.suggest_categorical("PCA_decomposition", [True, False])

    if PCA_decomposition:
        n_components = trial.suggest_int("n_components", 2, n_selected_features)
        df_train, df_validation, df_test, n_selected_features = apply_PCA(df_train, df_validation, df_test, n_components)

    select_features = trial.suggest_categorical("select_features", [True, False])

    if select_features:
        feature_selector_type = trial.suggest_categorical(
            "feature_selector_type", ["KBEST", "KPERCENTILE"]
        )

        if feature_selector_type == "KBEST":
            k = trial.suggest_int("k", 2, n_selected_features)
            feature_selector = SelectKBest(f_classif, k=k)

        elif feature_selector_type == "KPERCENTILE":
            percentile = trial.suggest_int("percentile", low=10, high=90, step=10)
            feature_selector = SelectPercentile(f_classif, percentile=percentile)

        df_train, df_validation, df_test, n_selected_features = feature_selection(
            df_train, df_validation, df_test, feature_selector
        )

    model = trial.suggest_categorical("model", ["BOOSTING", "RF", "BAGGING", "SIMPLE"])
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

    if model == "BOOSTING":
        boosting_model = AdaBoost()

        boosting_model.fit(
            df_train,
            df_train_label,
            n_trees=n_trees,
        )
        pred = boosting_model.predict(df_validation_nolabel)

        submission = {"ID": df_validation_id.astype(int).to_list(), "RATE": pred}
        submission = pd.DataFrame(submission)
        accuracy = accuracy_score(submission["RATE"], df_validation["RATE"])

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

        bagging_model = Bagging()

        bagging_model.fit(
            df_train,
            df_train_label,
            depth=depth,
            n_trees=n_trees,
            criterion=criterion,
            splitter=splitter,
            balance_weights=balance_weights,
        )
        pred = bagging_model.predict(df_validation_nolabel)

        submission = {"ID": df_validation_id.astype(int).to_list(), "RATE": pred}
        submission = pd.DataFrame(submission)
        accuracy = accuracy_score(submission["RATE"], df_validation["RATE"])

    elif model == "SIMPLE":

        simple_model = SimpleTree()

        simple_model.fit(
            df_train,
            df_train_label,
            depth=depth,
            criterion=criterion,
            splitter=splitter,
            balance_weights=balance_weights,
        )
        pred = simple_model.predict(df_validation_nolabel)

        submission = {"ID": df_validation_id.astype(int).to_list(), "RATE": pred}
        submission = pd.DataFrame(submission)
        accuracy = accuracy_score(submission["RATE"], df_validation["RATE"])

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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t", "--n_trials", 
        required=False,
        default=200,
        type=int
    )

    args = parser.parse_args()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

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

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report
from copy import deepcopy
import argparse
import logging

def random_forest(df_train, df_test, depth = 4, n_trees = 100, n_features = 7):
    clf = tree.DecisionTreeClassifier(max_depth = depth)#, class_weight=class_weight)
    df_train = df_train.drop(["ID"], axis=1)
    df_train_label =  df_train["RATE"]
    df_train = df_train.drop(["RATE"], axis=1)

    df_test_id = df_test["ID"]
    df_test = df_test.drop(["ID"], axis=1)

    filter_list = []
    for i in range(n_trees):
        filter_list.append(
            [ "X{}".format(col) for col in sorted(np.random.choice(np.arange(1,40), n_features, False))]
        )

    fitted_clf = []
    test_list = []
    oob_errors = []
    for filter in filter_list:
        X_train, X_test, y_train, y_test = train_test_split(df_train, df_train_label, test_size=1/3)
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

def bagging(df_train, df_test, depth = 4, n_trees = 100):
    clf = tree.DecisionTreeClassifier(max_depth = depth)
    df_train = df_train.drop(["ID"], axis=1)
    df_train_label =  df_train["RATE"]
    df_train = df_train.drop(["RATE"], axis=1)

    df_test_id = df_test["ID"]
    df_test = df_test.drop(["ID"], axis=1)

    fitted_clf = []
    test_list = []
    oob_errors = []
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(df_train, df_train_label, test_size=1/3)
        
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

def prediction(df_train, df_test, depth = 4):
    df_train = df_train.drop(["ID"], axis=1)
    df_train_label =  df_train["RATE"]
    df_train = df_train.drop(["RATE"], axis=1)

    df_test_id = df_test["ID"]
    df_test= df_test.drop(["ID"], axis=1)

    clf = tree.DecisionTreeClassifier(max_depth = depth)
    clf.fit(df_train, df_train_label)

    pred = clf.predict(df_test)

    submission = {"ID": df_test_id.astype(int).to_list(), "RATE": pred}
    submission = pd.DataFrame(submission)
    
    return submission

def main(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    np.random.seed(42)
    submission_RF = random_forest(df_train, df_test)
    submission_RF.to_csv("submission_RF.csv", index=False)

    np.random.seed(42)
    submission_bagging = bagging(df_train, df_test)
    submission_bagging.to_csv("submission_bagging.csv", index=False)

    np.random.seed(42)
    submission = prediction(df_train, df_test)
    submission.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "--test",
        type=str,
        help="",
        required=True
    )

    parser.add_argument(
        "-v", 
        "--verbose", 
        type=int, 
        required=False, 
        default=0
    )

    args = parser.parse_args()

    log_level = logging.WARNING
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose == 2:
        log_level = logging.DEBUG
    else:
        logging.warning('Log level not recognised. Using WARNING as default')

    logging.getLogger().setLevel(log_level)

    logging.warning("Verbose level set to {}".format(logging.root.level))

    main(args.train, args.test)
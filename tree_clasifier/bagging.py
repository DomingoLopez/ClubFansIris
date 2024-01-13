import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from copy import deepcopy
import logging


class Bagging:
    def __init__(self):
        self.depth = 0
        self.n_trees = None
        self.criterion="gini"
        self.splitter="best"
        self.balance_weights=None
        self.fitted_clf = []
        self.oob_errors = []

    def fit(self, X, y, 
        depth=4,
        n_trees=100,
        criterion="gini",
        splitter="best",
        balance_weights=None,
    ):
        self.depth = depth
        self.n_trees = n_trees
        self.criterion=criterion
        self.splitter=splitter
        self.balance_weights=balance_weights

        clf = tree.DecisionTreeClassifier(
            max_depth=depth,
            criterion=criterion,
            splitter=splitter,
            class_weight=balance_weights,
        )

        for i in range(n_trees):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1 / 3
            )

            self.fitted_clf.append(deepcopy(clf.fit(X_train, y_train)))

            pred = clf.predict(X_test)

            self.oob_errors.append(accuracy_score(pred, y_test))

        logging.info("BAGGING OOB: {}".format(np.mean(np.array(self.oob_errors))))

    def predict(self, X):
        pred_list = []
        for clf in self.fitted_clf:
            pred = clf.predict(X)
            pred_list.append(pred)

        pred_submission = []

        for pred in np.array(pred_list).T:
            pred_submission.append(pd.Series(pred).value_counts().idxmax())

        return pred_submission
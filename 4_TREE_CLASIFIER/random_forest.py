import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from copy import deepcopy
import logging


class RandomForest:
    def __init__(self):
        self.depth = 0
        self.n_trees = None
        self.n_features = 0
        self.criterion="gini"
        self.splitter="best"
        self.balance_weights=None
        self.fitted_clf = []
        self.filter_list = []
        self.oob_errors = []

    def fit(self, X, y, 
        depth=4, 
        n_trees=100,
        n_features=7,
        criterion="gini",
        splitter="best",
        balance_weights=None
    ):
        self.depth = depth
        self.n_trees = n_trees
        self.n_features = n_features
        self.criterion=criterion
        self.splitter=splitter
        self.balance_weights=balance_weights

        clf = tree.DecisionTreeClassifier(
            max_depth=depth,
            criterion=criterion,
            splitter=splitter,
            class_weight=balance_weights,
        )

        
        for i in range(self.n_trees):
            self.filter_list.append(
                [
                    col
                    for col in sorted(np.random.choice(X.columns, self.n_features, False))
                ]
            )

        for filter in self.filter_list:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=1 / 3
            )
            self.fitted_clf.append(deepcopy(clf.fit(X_train[filter], y_train)))

            pred = clf.predict(X_test[filter])

            self.oob_errors.append(accuracy_score(pred, y_test))

        logging.info("RANDOM FOREST OOB: {}".format(np.mean(np.array(self.oob_errors))))

    def predict(self, X):
        pred_list = []
        for clf, filter in zip(self.fitted_clf, self.filter_list):
            pred = clf.predict(X[filter])
            pred_list.append(pred)

        pred_submission = []

        for pred in np.array(pred_list).T:
            pred_submission.append(pd.Series(pred).value_counts().idxmax())

        return pred_submission
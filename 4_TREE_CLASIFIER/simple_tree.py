import numpy as np
import pandas as pd
from sklearn import tree

class SimpleTree:

    def __init__(self):
        self.depth = 0
        self.criterion="gini"
        self.splitter="best"
        self.balance_weights=None
        self.clf = None

    def fit(self, X, y, 
        depth=4,
        criterion="gini",
        splitter="best",
        balance_weights=None,
    ):
        self.clf = tree.DecisionTreeClassifier(
            max_depth=depth,
            criterion=criterion,
            splitter=splitter,
            class_weight=balance_weights,
        )
        self.clf.fit(X, y)
        
    def predict(self, X):
        pred = self.clf.predict(X)

        return pred
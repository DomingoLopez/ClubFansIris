import numpy as np
import pandas as pd
from sklearn import tree
import time


class AdaBoost:
    def __init__(self):
        self.alphas = []
        self.fitted_clf = []
        self.depth = 0
        self.n_trees = None
        self.training_errors = []
        self.prediction_errors = []
        self.class_econder = {"A":1, "B":2, "C":3, "D":4}

    def compute_error(self, y, y_pred, w_i):
        return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)

    def compute_alpha(self, error):
        return np.log((1 - error) / error)

    def update_weights(self, w_i, alpha, y, y_pred):
        return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))

    def fit(self, X, y, depth=4, n_trees = 100):
        self.alphas = [] 
        self.training_errors = []
        self.depth = depth
        self.n_trees = n_trees

        for m in range(n_trees):
            
            # Set weights for current boosting iteration
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)  # At m = 0, weights are all the same and equal to 1 / N
            else:
                # Update w_i
                w_i = self.update_weights(w_i, alpha_m, y, y_pred)
            
            # Fit weak classifier and predict labels
            clf = tree.DecisionTreeClassifier(max_depth = self.depth)
            clf.fit(X, y, sample_weight = w_i)
            y_pred = clf.predict(X)
            
            self.fitted_clf.append(clf) # Save to list of weak classifiers

            # Compute error
            error_m = self.compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)

            # Compute alpha
            alpha_m = self.compute_alpha(error_m)
            self.alphas.append(alpha_m)

        assert len(self.fitted_clf) == len(self.alphas)

    def predict(self, X):
        weak_preds = pd.DataFrame(index = range(len(X)), columns = range(self.n_trees)) 

        y_preds = []
        pred_list = []
        alpha_list = []
        for m in range(self.n_trees):
            pred_list.append(self.fitted_clf[m].predict(X))
            alpha_list.append(self.alphas[m])

        alpha_list = np.array(alpha_list)

        for pred in np.array(pred_list).T:
            assert len(pred) == len(alpha_list)
            # print(pred)
            # print(alpha_list)
            weighted_sum = []
            for value in np.unique(pred):
                index_value = np.argwhere(pred == value).flatten()
                weighted_sum.append(np.sum(alpha_list[index_value]))
            
            # print(np.unique(pred), weighted_sum)
            y_pred = np.unique(pred)[np.argmax(weighted_sum)]
            # print(y_pred)
            y_preds.append(y_pred)

        return y_preds
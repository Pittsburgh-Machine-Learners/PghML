"""CART on the Bank Note dataset"""
from __future__ import print_function
from collections import defaultdict
# import json
from random import seed  # randrange, sample
import sys
from time import time

import numpy as np
import pandas as pd
from sklearn_utils import Bunch
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

def load_banknote_authentication():
    """Load a CSV file, converting strings to floats and returns the data"""
    df = pd.read_csv("banknote_authentication.csv")
    return Bunch(dataframe=df,
                 data=df[df.columns[:-1]].as_matrix(),
                 target=df[df.columns[-1]].as_matrix(),
                 feature_names=df.columns[:-1].tolist())


def display_tree(tree, classes, indent=''):
    """prints out decision tree, called recursively"""
    if not isinstance(tree, Node):
        # is leaf node
        print(str(tree))
    else:
        print("if feat[%d] <= %0.3f: (gini: %.3f samples: %d %s)" %
              (tree.index, tree.value, tree.gini, tree.samples,
               tree.get_dist(classes)))
        # Print the branches
        print(indent + 'T->', end=" ")
        display_tree(tree.left, classes, indent + '  ')
        print(indent + 'F->', end=" ")
        display_tree(tree.right, classes, indent + '  ')


class Node(object):
    """represents a node in decision tree"""
    def __init__(self, index, value, groups, gini):
        self.index = index
        self.value = value
        self.groups = groups
        self.gini = gini
        self.samples = 0
        self.dist = defaultdict(int)
        for group in groups:
            targets = group['y']
            self.samples += len(targets)
            for tgt in targets:
                self.dist[tgt] += 1
        self.left = None
        self.right = None

    def get_dist(self, targets):
        """returns an array of counts for each target value"""
        ret_val = []
        for target in targets:
            ret_val.append(self.dist.get(target, 0))
        return ret_val


class CartDecisionTreeClassifier(object):
    """CART algorithm based Decision Tree learner"""
    def __init__(self, max_depth=5, min_samples_split=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None
        self.classes_ = None
        self.X_ = None
        self.y_ = None

    def __repr__(self):
        return "CartDecisionTreeClassifier(max_depth=%d, min_samples_split=%d)" % \
            (self.max_depth, self.min_samples_split)

    def get_params(self, deep=True):
        """returns params passed to __init__ as dict"""
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split
        }

    def set_params(self, **params):
        """allows supported params (accessable by get_params() to be modified
        by passed them as in as kwargs

        Necessary for supporting parameter optimizers such as GridSearchCV
        """
        for k, v in params.items():
            setattr(self, k, v)
        return self
        
    def fit(self, X, y):
        """trains a decision tree"""
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.classes_ = unique_labels(y)
        # get_split() determines which feature is the best decision point
        # for the root of the tree
        root = self._get_split(self.X_, self.y_)
        # flesh out the rest of the tree, constrained by params max_depth and
        # min_size, '1' here is the current depth (root)
        # this function recursively calls itself
        self._split(root, 1)
        self.tree_ = root
        return self

    def predict(self, X):
        """outputs predicted value(s) for data X"""
        X = check_array(X)
        return [self._predict(self.tree_, row) for row in X]

    def score(self, X, y):
        """Calculate accuracy percentage"""
        actual = y
        predicted = self.predict(X)
        num_actual = len(actual)
        num_correct = sum(1 for i in range(num_actual)
                          if actual[i] == predicted[i])
        return num_correct / float(num_actual) * 100.0

    def _predict(self, node, row):
        """Make a prediction with a decision tree"""
        if row[node.index] < node.value:
            if isinstance(node.left, Node):
                return self._predict(node.left, row)
            else:
                return node.left
        else:
            if isinstance(node.right, Node):
                return self._predict(node.right, row)
            else:
                return node.right

    def _get_split(self, X, y):
        """
        returns the best split point for a dataset (what should be root of
        subtree)
        """
        # get distinct set of class values
        class_values = np.unique(y)
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for feat_index in range(len(X[0])):
            for row in X:
                feat_val = row[feat_index]
                # 'groups' represents the dataset split into (left, right) by
                # how each row's value of the current 'feat_index' compares to
                # 'feat_val' of our current row
                candiate_feat_split = \
                    self._split_on_feature(feat_index, feat_val, X, y)
                # the gini index is a value between 0.0 and 1.0.  A 'group'
                # with a lower gini index of predicting the class
                gini = self._gini_index(candiate_feat_split, class_values)
                if gini < b_score:
                    # this iteration is an improvement, store the values
                    b_index, b_value, b_score, b_groups = \
                        feat_index, feat_val, gini, candiate_feat_split
        return Node(b_index, b_value, b_groups, gini)

    def _split_on_feature(self, feat_index, feat_value, X, y):
        """
        return dataset split into two groups (left, right)
        based on how each of the datasets rows value for the feature identified
        by feat_idx compares to feat_value. if the current row's value
        is < feat_value place it in the left group, else right
        """
        feat_vals = X[:, feat_index]
        left_idx = np.where(feat_vals < feat_value)
        right_idx = np.where(feat_vals >= feat_value)
        left_X, left_y = X[left_idx], y[left_idx]
        right_X, right_y = X[right_idx], y[right_idx]
        return [{'X': left_X, 'y': left_y}, {'X': right_X, 'y': right_y}]

    def _gini_index(self, groups, class_values):
        """
        return the calculated Gini index for a split dataset

        The name doesn't meaning anything - it was named after economist
        Corrando Gini, who first used it for comparing incomes in populations.

        The index represents the equality of values in a set of data
        0.0 represents perfect equality whereas 1.0 represents totally
        inequality
        """
        gini = 0.0
        for class_value in class_values:
            for group in groups:
                group_targets = group['y']
                group_size = len(group_targets)
                # simple prevention of divide by zero
                if group_size == 0:
                    continue
                # count how many times class_value occurs in group's target
                # values
                grp_class_val_cnt = np.sum(group_targets == class_value)
                # proportion is how many times the current class_value is
                # present in the current group divided by the size of the
                # group
                proportion = grp_class_val_cnt / float(group_size)
                # add the normalized proportion value to the gini index
                gini += (proportion * (1.0 - proportion))
        return gini

    def _to_terminal(self, grp):
        """return a terminal node - majority class in 'y'"""
        # targets = grp['y']
        return np.argmax(np.bincount(grp['y']))

    def _split(self, node, depth):
        """Create child splits for a node or make terminal"""
        left_gr, right_gr = node.groups
        node.groups = None
        # check for a no split - both left and right will be same terminal node
        # and this node should be pruned later (how can I pre-prune?)
        if not len(left_gr['y']) or not len(right_gr['y']):
            term = None
            if len(left_gr['y']):
                term = self._to_terminal(left_gr)
            elif len(right_gr['y']):
                term = self._to_terminal(right_gr)
            else:
                print ("ERROR: both left and right groups are empty")
                sys.exit(1)
            node.left, node.right = term, term
            return
        # if we've hit max depth force both left and right to be terminal nodes
        if depth >= self.max_depth:
            node.left = self._to_terminal(left_gr)
            node.right = self._to_terminal(right_gr)
            return
        # process left child
        if len(left_gr['y']) <= self.min_samples_split:
            node.left = self._to_terminal(left_gr)
        else:
            node.left = self._get_split(left_gr['X'], left_gr['y'])
            self._split(node.left, depth+1)
        # process right child
        if len(right_gr['y']) <= self.min_samples_split:
            node.right = self._to_terminal(right_gr)
        else:
            node.right = self._get_split(right_gr['X'], right_gr['y'])
            self._split(node.right, depth+1)

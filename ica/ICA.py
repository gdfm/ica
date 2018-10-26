#ICA.py

import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import random
import heapq

# Utility functions

def recall_at_k(y_true, y_score, k=5000):
    pivot = heapq.nlargest(k, y_score)[-1]
    y_pred = [0 if x < pivot else 1 for x in y_score]
    return sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred)


## ICA

class Classifier(object):
    def __init__(self, scikit_classifier_name, **classifier_args):
        classifer_class = get_class(scikit_classifier_name)
        self.clf = classifer_class(**classifier_args)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    md = __import__(module)
    for comp in parts[1:]:
        md = getattr(md, comp)
    return md

class Aggregator(object):
    def __init__(self, domain_labels):
        self.domain_labels = domain_labels  # list of labels in the domain

    def aggregate(self, G, node, conditional_node_to_label_map):
        raise NotImplementedError

class Count(Aggregator):
    def aggregate(self, G, node, conditional_node_to_label_map):
        neighbor_undirected = []
        for x in self.domain_labels:
            neighbor_undirected.append(0.0)
        for i in G.adj[node].keys():
            if i in conditional_node_to_label_map.keys():
                index = self.domain_labels.searchsorted(conditional_node_to_label_map[i])
                neighbor_undirected[index] += 1.0
        return neighbor_undirected
    
def create_map(Y, train_indices):
    return Y.loc[train_indices].to_dict()

class RelationalClassifier(Classifier):
    def __init__(self, scikit_classifier_name, aggregator, **classifier_args):
        super(RelationalClassifier, self).__init__(scikit_classifier_name, **classifier_args)
        self.aggregator = aggregator

    def fit(self, G, X, Y, train_indices, local_classifier, bootstrap):
        conditional_map = {}
        # X and G need to be sorted in the same way consistently
        if bootstrap:
            predictclf = local_classifier.predict(X)
            conditional_map = self.cond_mp_upd(G, conditional_map, predictclf, sorted(G.nodes()))
        for i in train_indices:
            conditional_map[i] = Y.loc[i]
        aggregates = [np.matrix(self.aggregator.aggregate(G, i, conditional_map), dtype=np.float64) for i in train_indices]
        features = X.loc[train_indices].values
        labels = Y.loc[train_indices].values      
        aggregates = np.vstack(aggregates)
        features = np.hstack([features, aggregates])
        self.clf.fit(features, labels)            

    def predict(self, G, X, test_indices, conditional_map=None):
        aggregates = [np.matrix(self.aggregator.aggregate(G, i, conditional_map), dtype=np.float64) for i in test_indices]
        features = X.loc[test_indices].values
        aggregates = np.vstack(aggregates)
        features = np.hstack([features, aggregates])
        return self.clf.predict(features)

    def cond_mp_upd(self, G, conditional_map, pred, indices):
        for x in range(len(pred)):
            conditional_map[indices[x]] = pred[x]
        return conditional_map

class ICA(Classifier):
    def __init__(self, local_classifier, relational_classifier, bootstrap, max_iteration=10):
        self.local_classifier = local_classifier
        self.relational_classifier = relational_classifier
        self.bootstrap = bootstrap
        self.max_iteration = max_iteration

    def fit(self, G, X, Y, train_indices):
        self.local_classifier.fit(X.loc[train_indices], Y.loc[train_indices])
        self.relational_classifier.fit(G, X, Y, train_indices, self.local_classifier, self.bootstrap)

    def predict(self, G, X, eval_indices, test_indices, conditional_node_to_label_map=None):
        predictclf = self.local_classifier.predict(X.loc[eval_indices])
        conditional_node_to_label_map = self.cond_mp_upd(G, conditional_node_to_label_map, predictclf, eval_indices)

        relation_predict = []
        for iter in range(self.max_iteration):
            for ei in random.sample(eval_indices, len(eval_indices)): # shuffle order
                rltn_pred = list(self.relational_classifier.predict(G, X, [ei], conditional_node_to_label_map))
                conditional_node_to_label_map = self.cond_mp_upd(G, conditional_node_to_label_map, rltn_pred, [ei])
        for ti in test_indices:
            relation_predict.append(conditional_node_to_label_map[ti])
        return relation_predict

    def cond_mp_upd(self, G, conditional_map, pred, indices):
        for x in range(len(pred)):
            conditional_map[indices[x]] = pred[x]
        return conditional_map


## Probabilistic version

class InMean(Aggregator):
    def aggregate(self, G, node, conditional_node_to_label_map):
        s = np.zeros(len(self.domain_labels))
        n = 0
        for u in G.pred[node].keys():
            if u in conditional_node_to_label_map.keys():
                s += conditional_node_to_label_map[u]
                n += 1
        return s/n if n > 0 else s
    
class RelationalProbaClassifier(Classifier):
    def __init__(self, scikit_classifier_name, aggregator, **classifier_args):
        super(RelationalProbaClassifier, self).__init__(scikit_classifier_name, **classifier_args)
        self.aggregator = aggregator

    def fit(self, G, X, Y, train_indices, local_classifier, bootstrap):
        conditional_map = {}
        # X and G need to be sorted in the same way consistently
        if bootstrap:
            predictclf = local_classifier.predict_proba(X)[:,1] # predict class 1 probabilities
            conditional_map = self.cond_mp_upd(G, conditional_map, predictclf, sorted(G.nodes()))
#        for i in train_indices:
#            conditional_map[i] = Y.loc[i]
        aggregates = [np.matrix(self.aggregator.aggregate(G, i, conditional_map), dtype=np.float64) for i in train_indices]
        features = X.loc[train_indices].values
        labels = Y.loc[train_indices].values      
        aggregates = np.vstack(aggregates)
        features = np.hstack([features, aggregates])
        self.clf.fit(features, labels)

    def predict(self, G, X, test_indices, conditional_map=None):
        aggregates = [np.matrix(self.aggregator.aggregate(G, i, conditional_map), dtype=np.float64) for i in test_indices]
        features = X.loc[test_indices].values
        aggregates = np.vstack(aggregates)
        features = np.hstack([features, aggregates])
        return self.clf.predict_proba(features)[:,1] # predict class 1 probabilities

    def cond_mp_upd(self, G, conditional_map, pred, indices):
        for x in range(len(pred)):
            conditional_map[indices[x]] = pred[x]
        return conditional_map

    
class ICAProba(Classifier):
    def __init__(self, local_classifier, relational_classifier, bootstrap, max_iteration=10):
        self.local_classifier = local_classifier
        self.relational_classifier = relational_classifier
        self.bootstrap = bootstrap
        self.max_iteration = max_iteration

    def fit(self, G, X, Y, train_indices):
        self.local_classifier.fit(X.loc[train_indices], Y.loc[train_indices])
        self.relational_classifier.fit(G, X, Y, train_indices, self.local_classifier, self.bootstrap)

    def predict(self, G, X, eval_indices, test_indices, conditional_node_to_label_map=None):
        predictclf = self.local_classifier.predict_proba(X.loc[eval_indices])[:,1] # predict class 1 probabilities
        conditional_node_to_label_map = self.cond_mp_upd(G, conditional_node_to_label_map, predictclf, eval_indices)

        relation_predict = []
        for iter in range(self.max_iteration):
            for ei in random.sample(eval_indices, len(eval_indices)): # shuffle order
                rltn_pred = list(self.relational_classifier.predict(G, X, [ei], conditional_node_to_label_map))
                conditional_node_to_label_map = self.cond_mp_upd(G, conditional_node_to_label_map, rltn_pred, [ei])
        for ti in test_indices:
            relation_predict.append(conditional_node_to_label_map[ti])
        return relation_predict

    def cond_mp_upd(self, G, conditional_map, pred, indices):
        for x in range(len(pred)):
            conditional_map[indices[x]] = pred[x]
        return conditional_map



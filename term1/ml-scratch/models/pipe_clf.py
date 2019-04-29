#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 08:55:18 2019

@author: arimoto
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def log_reg(X_train, X_test, y_train, y_test):
    estimators = [('sc', StandardScaler()),
                  ('lr', LogisticRegression())]
    parameters = {"lr__penalty" : ["l1","l2"],
                  "lr__C" : np.logspace(-3, 3, 7).tolist(), 
                  "lr__solver" : ["liblinear"]}
    pl = Pipeline(estimators)
    clf = GridSearchCV(pl, parameters, n_jobs=-1, cv=5)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    print("------Logistic Regression------")
    print("訓練データの正解率: ", accuracy_score(y_train, y_train_pred))
    print("テストデータの正解率: ", accuracy_score(y_test, y_test_pred))
    print("テストデータの適合率: ", precision_score(y_test, y_test_pred))
    print("テストデータの再現率: ", recall_score(y_test, y_test_pred))
    print("テストデータのf1スコア: ", f1_score(y_test, y_test_pred))
    
def svc(X_train, X_test, y_train, y_test):
    estimators = [('sc', StandardScaler()),
                  ('svc', SVC())]
    parameters = {"svc__C" : np.logspace(-3, 3, 7).tolist()}
    pl = Pipeline(estimators)
    clf = GridSearchCV(pl, parameters, n_jobs=-1, cv=5)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    print("------SVC------")
    print("訓練データの正解率: ", accuracy_score(y_train, y_train_pred))
    print("テストデータの正解率: ", accuracy_score(y_test, y_test_pred))
    print("テストデータの適合率: ", precision_score(y_test, y_test_pred))
    print("テストデータの再現率: ", recall_score(y_test, y_test_pred))
    print("テストデータのf1スコア: ", f1_score(y_test, y_test_pred))

def dec_tree(X_train, X_test, y_train, y_test):
    estimators = [('sc', StandardScaler()),
                  ('dtc', DecisionTreeClassifier())]
    parameters = {"dtc__max_depth" : np.arange(1,11).tolist()}
    pl = Pipeline(estimators)
    clf = GridSearchCV(pl, parameters, n_jobs=-1, cv=5)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    print("------Decision Tree Classifier------")
    print("訓練データの正解率: ", accuracy_score(y_train, y_train_pred))
    print("テストデータの正解率: ", accuracy_score(y_test, y_test_pred))
    print("テストデータの適合率: ", precision_score(y_test, y_test_pred))
    print("テストデータの再現率: ", recall_score(y_test, y_test_pred))
    print("テストデータのf1スコア: ", f1_score(y_test, y_test_pred))

def random_forest(X_train, X_test, y_train, y_test):
    estimators = [('sc', StandardScaler()),
                  ('rfc', RandomForestClassifier())]
    parameters = {"rfc__max_depth" : np.arange(1,11).tolist(), 
                  "rfc__n_estimators" : np.arange(1,21).tolist()}
    pl = Pipeline(estimators)
    clf = GridSearchCV(pl, parameters, n_jobs=-1, cv=5)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    print("------Random Forest Classifier------")
    print("訓練データの正解率: ", accuracy_score(y_train, y_train_pred))
    print("テストデータの正解率: ", accuracy_score(y_test, y_test_pred))
    print("テストデータの適合率: ", precision_score(y_test, y_test_pred))
    print("テストデータの再現率: ", recall_score(y_test, y_test_pred))
    print("テストデータのf1スコア: ", f1_score(y_test, y_test_pred))

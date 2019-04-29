#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:25:37 2019

@author: arimoto
"""

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def lin_reg(X_train, X_test, y_train, y_test):
    estimators = [('sc', StandardScaler()),
                  ('lr', LinearRegression())]
    parameters = {"lr__fit_intercept" : [False, True]}
    pl = Pipeline(estimators)
    reg = GridSearchCV(pl, parameters, n_jobs=-1, cv=5)
    reg.fit(X_train, y_train)
    y_test_pred = reg.predict(X_test)
    print("------Linear Regression------")
    print("テストデータのMSE: ", mean_squared_error(y_test, y_test_pred))
    print("テストデータのR2_score: ", r2_score(y_test, y_test_pred))
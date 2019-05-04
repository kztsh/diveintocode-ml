#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:25:37 2019

@author: arimoto
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def lin_reg(X, y, X_val, y_val, n_iter=50, lr=0.1, bias=True, verbose=True, random_state=1):
    """
    以下を順に実行するパイプライン実装
    
    1. X, y（いずれも訓練データ）によるfitスケーリング
    2. X, y, X_val, y_valのtransformスケーリング
    3. ScratchLinearRegressionにて、X, y, X_val, y_valによるfitting
    4. ScratchLinearRegressionにて、X, X_valによるpredict
    5. y_pred, y_val_predと各真値との残差プロット
    
    """
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    # Xスケーリング
    sc_X.fit(X)
    X = sc_X.transform(X)
    X_val = sc_X.transform(X_val)
    # yスケーリング
    # sklearnのscalerは2次元配列を想定しているので、2次元にしてfit, transformした後に1次元に戻している
    sc_y.fit(y[:, np.newaxis])
    y = sc_y.transform(y[:, np.newaxis]).ravel()
    y_val = sc_y.transform(y_val[:, np.newaxis]).ravel()
    
    # fit
    reg = ScratchLinearRegression(n_iter=n_iter, lr=lr, bias=bias, verbose=verbose, random_state=random_state)
    reg.fit(X, y, X_val, y_val)
    # predict
    y_pred = reg.predict(X)
    y_val_pred = reg.predict(X_val)
    print("------Scratch Linear Regression------")
    print("検証データのMSE: ", mean_squared_error(y_val, y_val_pred))
    print("検証データのR2_score: ", r2_score(y_val, y_val_pred))
    
    # 残差プロット
    plt.scatter(y_pred, (y_pred - y), c="steelblue", edgecolor="white", marker="o", label="Training data")
    plt.scatter(y_val_pred, (y_val_pred - y_val), c="orangered", edgecolor="white", marker="s", label="Test data")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc="best")
    y_pred_min = min(y_pred)
    y_pred_max = max(y_pred)
    y_val_pred_min = min(y_val_pred)
    y_val_pred_max = max(y_val_pred)
    plt.hlines(y=0, xmin=min(y_pred_min, y_val_pred_min)-0.1, xmax=max(y_pred_max, y_val_pred_max)+0.1, color="black", lw=2)
    plt.tight_layout()
    plt.show()
    
    # 後にバイアス項にアクセスできるように、学習したインスタンスを返しておく
    return reg

class ScratchLinearRegression():
    """
    最急降下法による線形回帰のスクラッチ実装

    Parameters
    ----------
    n_iter : int
      イテレーション数
    lr : float
      学習率
    bias : bool
      バイアス項を入れない場合はTrue
    verbose : bool
      学習過程を出力する場合はTrue
    random_state : int
      重みを初期化するための乱数シード

    Attributes
    ----------
    self.w_ : 次の形のndarray, shape (n_features,)
      パラメータ
    self.loss : 次の形のndarray, shape (self.iter,)
      学習用データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証用データに対する損失の記録

    """
    
    def __init__(self, n_iter=50, lr=0.1, bias=True, verbose=True, random_state=1):
        # ハイパーパラメータを属性として記録
        self.n_iter = n_iter
        self.lr = lr
        self.bias = bias
        self.verbose = verbose
        self.random_state = random_state
        # 損失を記録する配列を用意
        self.loss = np.zeros(self.n_iter)
        self.val_loss = np.zeros(self.n_iter)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        線形回帰を学習する。検証用データが入力された場合はそれに対する損失と精度もイテレーションごとに計算する。

        Parameters
        ----------
        X : 次の形のndarray, shape (n_samples, n_features)
            学習用データの特徴量
        y : 次の形のndarray, shape (n_samples, )
            学習用データの正解値
        X_val : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量
        y_val : 次の形のndarray, shape (n_samples, )
            検証用データの正解値
        """
        rgen = np.random.RandomState(self.random_state)
        # バイアス項なしの場合
        if self.bias == False:
            self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
            for i in range(self.n_iter):
                # インプット計算
                calc = np.dot(X, self.w_)
                # 係数の更新
                errors = calc - y
                self.w_ -= self.lr * X.T.dot(errors)/X.shape[0]
                # 最小二乗誤差（損失）の計算と格納
                cost = (errors**2).sum() / (2.0 * X.shape[0])
                self.loss[i] = cost
                # 検証用データがあれば、その損失も記録
                if X_val is not None:
                    val_calc = np.dot(X_val, self.w_)
                    val_errors = val_calc - y_val
                    val_cost = (val_errors**2).sum() / (2.0 * X_val.shape[0])
                    self.val_loss[i] = val_cost
                
        # バイアス項ありの場合
        elif self.bias == True:
            self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
            for i in range(self.n_iter):
                # インプット計算
                calc = np.dot(X, self.w_[1:]) + self.w_[0]
                # 係数の更新
                errors = calc - y
                self.w_[1:] -= self.lr * X.T.dot(errors)/X.shape[0]
                self.w_[0] -= self.lr * errors.sum()/X.shape[0]
                # 最小二乗誤差（損失）の計算と格納
                cost = (errors**2).sum() / (2.0 * X.shape[0])
                self.loss[i] = cost
                # 検証用データがあれば、その損失も記録
                if X_val is not None:
                    val_calc = np.dot(X_val, self.w_[1:]) + self.w_[0]
                    val_errors = val_calc - y_val
                    val_cost = (val_errors**2).sum() / (2.0 * X_val.shape[0])
                    self.val_loss[i] = val_cost
        
        # これ以外に、バイアス項に異常値が入った場合
        else:
            raise ValueError("Set boolean value to 'bias' parameter !")
        
        #verboseをTrueにした際は損失の推移リストと推移プロットを出力
        if self.verbose:
            print("学習データにおける損失の推移(変数リスト：self.loss)：\n", self.loss)
            plt.plot(np.arange(len(self.loss)), self.loss, color="steelblue", label="learning loss curve")
            # 検証データがあればその損失（精度）リストも出力
            if X_val is not None:
                print("検証データにおける損失（精度）の推移(変数リスト：self.val_loss)：\n", self.val_loss)
                plt.plot(np.arange(len(self.val_loss)), self.val_loss, color="orangered", label="validating loss curve")
            plt.xlabel("n_iter")
            plt.ylabel("Mean Squared Error (MSE)")
            plt.legend(loc="best")
            plt.show()
            
    
    def predict(self, X_val):
        """
        線形回帰を使い推定する。

        Parameters
        ----------
        X_val : 次の形のndarray, shape (n_samples, n_features)
            サンプル

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            線形回帰による推定結果
        """
        if self.bias == False:
            return np.dot(X_val, self.w_)
        elif self.bias == True:
            return np.dot(X_val, self.w_[1:]) + self.w_[0]
        else:
            raise ValueError("Set boolean value to 'bias' parameter !")
    
    def intercept_(self):
        if self.bias==True:
            return self.w_[0]
        else:
            return None
    
    def coef_(self):
        if self.bias==True:
            return self.w_[1:]
        else:
            return self.w_
        
    

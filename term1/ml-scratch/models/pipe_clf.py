#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 08:55:18 2019

@author: arimoto
"""

import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython.display import display
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def log_reg(X_train, X_test, y_train, y_test, feature_names, target_names, val_size, test_size, lr=0.1, two_features=False):
    """
    最急降下法によるScratchLogistic回帰のパイプライン。
    入力データを標準化、学習、評価し、各種スコア値を算出する。
    決定領域をプロットする場合は、入力特徴量のうち2種類の全組合わせでそれぞれ学習・評価し、可視化する。

    Parameters
    ----------------
    X_train : ndarray, shape(n_samples, n_features)
        学習用データの特徴量
    X_test : ndarray, shape(n_samples, n_features)
        テスト用データの特徴量
    y_train : ndarray, shape(m_samples,)
        学習用データの正解値
    y_test : ndarray, shape(m_samples,)
        テスト用データの正解値
    feature_names : list of str
        使用する全特徴量名のリスト
    target_names= : list of str
        使用する目的変数（ラベル）名のリスト
    val_size : float
      訓練データのうちfittingに使用する検証用データの比率
    test_size : float
      元データからX_trainとX_testに分割した際のX_testの比率
    lr : float （初期値： 0.1）
      ScratchLogisticRegressionの学習率
    two_features : boolean （初期値： False）
      多次元特徴量で学習し、可視化しない場合：False
      多次元特徴量から2種類の全組合わせで学習し、それぞれで可視化する場合：True
    """
    # X_train, y_trainを更に訓練データと検証データに分割
    X, X_val, y, y_val = train_test_split(X_train, y_train, test_size=val_size, stratify=y_train, random_state=0)
    
    # XとX_valをスケーリング
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)
    X_val = sc.transform(X_val)
    
    # X_trainとX_testをスケーリング
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    
    # X, yでfitting（X_val, y_valも検証用として引数に入れる）
    # 学習、評価し、5種スコアの集計表と混同行列・決定領域の可視化をする関数scoring_funcを出力する
    model_names = []
    models = []
    test_idx=range(np.floor((1-test_size)*X_train.shape[0]).astype("int"), X_train.shape[0])
    
    # 例外処理(TrueでもFalseでもない場合)
    if not (two_features==True or two_features==False):
        raise ValueError("Set boolean value to 'two_features' parameter !")
        
    # 多次元特徴量から2種類の全組み合わせのそれぞれで学習・評価し、それぞれで決定領域を可視化する場合
    if two_features==True or X.shape[1]==2:
        # 2種特徴量の組み合わせの列indexのタプルのリストを生成
        combs = list(itertools.combinations(np.arange(X.shape[1]), 2))
        for i, comb in enumerate(combs):
            print("------Taken features: {}, {}------".format(feature_names[comb[0]], feature_names[comb[1]]))
            clf = ScratchLogisticRegression(lr=lr)
            clf.fit(X[:, comb], y, X_val[:, comb], y_val)
            model_names.append("Scratch Logistic Regression")
            models.append(clf)
        # 別途定義のスコアリング関数にパラメータ移譲
        scoring_func(X_train, X_test, y_train, y_test, 
                 model_names, models, feature_names, target_names, test_idx, two_features)
        
    # 多次元特徴量で学習・評価し、決定領域は可視化しない（できない）場合
    else:
        clf = ScratchLogisticRegression(lr=lr)
        clf.fit(X, y, X_val, y_val)
        model_names.append("Scratch Logistic Regression")
        models.append(clf)
        # 別途定義のスコアリング関数にパラメータ移譲
        scoring_func(X_train, X_test, y_train, y_test, 
                 model_names, models, feature_names, target_names, test_idx)


class ScratchLogisticRegression:
    """
    最急降下法によるLogistic回帰のスクラッチ実装

    Parameters
    ----------
    n_iter : int （初期値： 50）
      イテレーション数
    lr : float （初期値： 0.1）
      学習率
    bias : bool （初期値： True）
      バイアス項を入れる場合はTrue
    verbose : bool （初期値： True）
      学習過程を出力する場合はTrue
    lam : float （初期値： 0.1）
      正則化項パラメータの変数ラムダ
    tol : float （初期値： 1e-3）
      毎回の学習によるコスト関数値の変化をストップするための閾値
    random_state : int （初期値： 1）
      重みを初期化するための乱数シード

    Attributes
    ----------
    self.w_ : 次の形のndarray, shape (n_features,)
      重みパラメータ
    self.loss : 次の形のndarray, shape (self.iter,)
      学習用データに対する損失の記録
    self.val_loss : 次の形のndarray, shape (self.iter,)
      検証用データに対する損失の記録
    """
    
    def __init__(self, n_iter=50, lr=0.1, bias=True, verbose=True, lam=0.1, tol=1e-3, random_state=1):
        self.n_iter = n_iter
        self.lr = lr
        self.bias = bias
        self.verbose = verbose
        self.lam = lam
        self.tol = tol
        self.random_state = random_state
        # w_をまずNoneに設定
        self.w_ = None
        # コスト関数値を格納するリストを用意（便宜上、初期値はリストに入れておく）
        self.loss = [1, ]
        self.val_loss = [1, ]
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        線形回帰を学習する。損失もイテレーション毎に記録する。

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
        diff = np.inf
        iteration = 0
        
        # バイアス項なしの場合
        if self.bias == False:
            # w_をランダムに初期化
            self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
            # コスト関数値の変化が変数tol以下または更新回数が変数n_iter以上となったらストップ
            while diff > self.tol and iteration < self.n_iter:
                w_prev = self.w_
                # yの推測値をXとX_valで計算
                y_pred = self.sigmoid(np.dot(X, self.w_))
                if X_val is not None and y_val is not None:
                    y_val_pred = self.sigmoid(np.dot(X_val, self.w_))
                else:
                    y_val_pred = None
                # コスト関数値の計算、リスト格納、変化の値の更新
                self._cost_func(X, y, y_pred, X_val, y_val, y_val_pred)
                # 学習率 * 勾配を計算し、パラメータ更新
                self.w_ -= self.lr * (np.dot(X.T, (y_pred - y)) / X.shape[0] + (self.lam / X.shape[0]) * self.w_)
                iteration += 1
        
        # バイアス項ありの場合
        elif self.bias == True:
            # w_をランダムに初期化
            self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
            # X, X_valにx_0=1の列を追加
            X = np.hstack([np.ones(X.shape[0]).reshape(-1,1), X])
            if X_val is not None:
                X_val = np.hstack([np.ones(X_val.shape[0]).reshape(-1,1), X_val])
            # コスト関数値の変化が変数tol以下または更新回数が変数n_iter以上となったらストップ
            while diff > self.tol and iteration < self.n_iter:
                w_prev = self.w_
                # yの推測値をXとX_valで計算
                y_pred = self.sigmoid(np.dot(X, self.w_))
                if X_val is not None and y_val is not None:
                    y_val_pred = self.sigmoid(np.dot(X_val, self.w_))
                else:
                    y_val_pred = None
                # コスト関数値の計算、リスト格納、変化の値の更新
                self._cost_func(X, y, y_pred, X_val, y_val, y_val_pred)
                 # 学習率 * 勾配を計算し、パラメータ更新
                self.w_[0] -= self.lr * (np.dot(X[:, 0], (y_pred - y)) / X.shape[0])
                self.w_[1:] -= self.lr * (np.dot(X[:, 1:].T, (y_pred - y)) / X.shape[0] + (self.lam / X.shape[0]) * self.w_[1:])
                iteration += 1
        
        # バイアス項エラーの場合
        else:
            raise ValueError("Set boolean value to 'bias' parameter !")
        
        # 学習過程をプロットする場合
        if self.verbose:
            self._cost_plot_func(self.loss, self.val_loss)
    
    def predict(self, X_test):
        """
        fittingした回帰器を使い、ラベル推定結果を返す。

        Parameters
        ----------
        X_test : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量

        Returns
        -------
            次の形のndarray, shape (n_samples, 1)
            Logistic回帰によるラベル推定結果
        """
        if self.bias == True:
            X_test = np.hstack([np.ones(X_test.shape[0]).reshape(-1,1), X_test])
        y_pred = self.sigmoid(np.dot(X_test, self.w_))
        return np.where(y_pred > 0.5, 1, 0)
    
    def predict_proba(self, X_test):
        """
        fittingした回帰器を使い、ラベル推定の確率を返す。

        Parameters
        ----------
        X_test : 次の形のndarray, shape (n_samples, n_features)
            検証用データの特徴量

        Returns
        -------
            次の形のndarray, shape (n_samples, n_classes)
            Logistic回帰によるラベル推定確率
        """
        if self.bias == True:
            X_test = np.hstack([np.ones(X_test.shape[0]).reshape(-1,1), X_test])
        y_pred = self.sigmoid(np.dot(X_test, self.w_)).reshape(-1,1)
        return np.hstack((1 - y_pred, y_pred))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _cost_func(self, X, y, y_pred, X_val=None, y_val=None, y_val_pred=None):
        # コスト関数値を計算し、リストに格納
        cost = (-np.dot(y, np.log(y_pred)) - np.dot(
            1 - y, np.log(1 - y_pred)))/X.shape[0] + self.lam / (2.0 * X.shape[0]) * np.sum(self.w_**2)
        self.loss.append(cost)
        # 検証用データがあれば同様に処理
        if X_val is not None and y_val is not None and y_val_pred is not None:
            val_cost = (-np.dot(y_val, np.log(y_val_pred)) - np.dot(
                1 - y_val, np.log(1 - y_val_pred)))/X_val.shape[0] + self.lam / (2.0 * X_val.shape[0]) * np.sum(self.w_**2)
            self.val_loss.append(val_cost)
        # コスト関数値の変化の値を更新
        diff = self.loss[-2] - self.loss[-1]
    
    def _cost_plot_func(self, loss, val_loss):
        print("------コスト関数値の推移（学習用データ）------\n", loss)
        plt.plot(np.arange(len(loss)), loss, color="steelblue", linestyle="-", label="Train")
        if len(val_loss) > 1:
            print("------コスト関数値の推移（検証用データ）------\n", val_loss)
            plt.plot(np.arange(len(val_loss)), val_loss, color="orangered", linestyle="--", label="Validation")
        plt.xlabel("n_iter")
        plt.ylabel("cost func with L2")
        plt.title("Learning curve")
        plt.legend(loc="best")
        plt.show()


class ScratchSVMClassifier:
    """
    ラグランジュ未定乗数法の制約条件に従ったイテレーションによるSVMClassifierのスクラッチ実装（最急降下法ではない）

    Parameters
    ----------
    n_iter : int （初期値： 50）
      イテレーション数
    C : float （初期値： 1e+10）
      スラック変数を導入したソフトマージンにおけるパラメータC
      （初期値のように無限大に近い値ではハードマージンに相当する）

    Attributes
    ----------
    self.evaluation_ : list of float, shape (self.iter,)
      訓練データによるイテレーション毎の評価関数値の記録
    self.lam_ : 次の形のndarray, shape (n_samples,)
      学習後の、各サンプル点のラグランジュ未定乗数の値
    self.w_ : 次の形のndarray, shape (n_features,)
      重みパラメータ
    self.w0_ : float
      バイアス項
    """
    def __init__(self, n_iter=50, C=1e+10):
        self.n_iter = n_iter
        self.C = C
        self.evaluation_ = [0,]
    
    def fit(self, X, y):
        # 初期値設定
        lam = np.zeros(X.shape[0]) # ラグランジュ未定乗数ラムダの初期値（要素数：サンプル数）
        lamy = 0 # lamとyの内積の初期値
        lamyx = np.zeros(X.shape[1]) # lamとyとXの内積の初期値（要素数：特徴量数）
        
        # 固定値算出（最初から作れる）
        yx = y.reshape(-1,1) * X # X^T * y
        # データ点識別用の番号生成
        indices = np.arange(X.shape[0])
        
        # 勾配学習
        for g in range(self.n_iter):
            # データ番号列（0列目）とy * 勾配の列（1列目）の行列index_y_gradを生成
            gradient = 1 - np.dot(yx, lamyx.T) # 勾配計算（サンプル数,）
            y_grad = y * gradient # y * 勾配（サンプル数,）
            index_y_grad = np.c_[indices, y_grad] # indexを0列目に追加（サンプル数, 2）
            
            # i（ラベル-1の点のうちy * 勾配が最小のデータ番号）を決定
            # (y=+1かつlam>0)または（y=-1かつlam<self.C）の行を抽出
            row_minus = index_y_grad[((y>0) & (lam>0)) | ((y<0) & (lam<self.C)), :]
            min_row_minus = np.argmin(row_minus[:, 1]) # その中でy * 勾配が最小の行インデックスを決定
            i = int(row_minus[min_row_minus, 0]) # その行インデックスに該当するデータ番号を格納
            
            # j（ラベル+1の点のうちy * 勾配が最大のデータ番号）を決定
            # (y=-1かつlam>0)または（y=+1かつlam<self.C）の行を抽出
            row_plus = index_y_grad[((y<0) & (lam>0)) | ((y>0) & (lam<self.C)), :] # y = +1またはlam>0の行を抽出
            max_row_plus = np.argmax(row_plus[:, 1]) # その中でy * 勾配が最大の行インデックスを決定
            j = int(row_plus[max_row_plus, 0]) # その行インデックスに該当するデータ番号を格納
            
            # KKT条件から導出する最適解の条件式を満たしているか確認
            if y_grad[i] >= y_grad[j]:
                break
            
            # 満たしていなければパラメータ更新フローに入る
            lamy2 = lamy - lam[i] * y[i] - lam[j] * y[j] # lamyの現在の仮変数を計算
            lamyx2 = lamyx - lam[i] * y[i] * X[i, :] - lam[j] * y[j] * X[j, :] # lamyxの現在の仮変数を計算
            
            # lamy2とlamyx2を使い、制約条件を無視したlam_iをまず作成
            lam_i = (1 - y[i] * y[j] + y[i] * np.dot(X[i, :] - X[j, :], X[j, :] * lamy2 - lamyx2)) / (((X[i] - X[j])**2).sum())
            # 制約条件：lamの値は0以上self.C以下
            if lam_i < 0:
                lam_i = 0
            elif lam_i > self.C:
                lam_i = self.C
            # lam_iが決まればlam_jが決まる
            lam_j = y[j] * (-lam_i * y[i] - lamy2)
            # 制約条件：lamの値は0以上self.C以下
            if lam_j < 0:
                lam_j = 0
                lam_i = y[i] * (-lam_j * y[j] - lamy2) # lam_jが変わる場合は改めて計算
            elif lam_j > self.C:
                lam_j = self.C
                lam_i = y[i] * (-lam_j * y[j] - lamy2) # lam_jが変わる場合は改めて計算
            
            # lamyとlamyxの変化分で更新
            lamy += (lam_i - lam[i]) * y[i] + (lam_j - lam[j]) * y[j]
            lamyx += (lam_i - lam[i]) * y[i] * X[i, :] + (lam_j - lam[j]) * y[j] * X[j, :]
            
            # 計算されたlam_iに変化がなければ終了、あれば更新
            if lam_i == lam[i]:
                break
            lam[i] = lam_i
            lam[j] = lam_j
            
            # 評価関数L（最大化）の値をリストに格納
            evaluation = lam.sum() - (1/2) * lamyx.sum()
            self.evaluation_.append(evaluation)
        
        # 学習が終わったらラグランジュ乗数と学習係数をインスタンス変数に格納
        self.lam_ = lam
        ind = lam != 0. # ラグランジュ乗数が更新されたサンプルのインデックス
        self.w_ = ((lam[ind] * y[ind]).reshape(-1,1) * X[ind, :]).sum(axis=0) # lam*y*Xのサンプル方向の和（要素数：特徴量数）
        # lam!=0の全サンプルでy * f(x)=1となることを利用して算出したw_0値の平均をとる
        self.w0_ = (y[ind] - np.dot(X[ind, :], self.w_)).sum() / ind.sum()
        
        # 評価関数Lの値の推移を可視化
        print("評価関数Lの値の推移（最大化）：\n", self.evaluation_)
        plt.plot(np.arange(len(self.evaluation_)), self.evaluation_, label="evaluation func")
        plt.xlabel("n_iter")
        plt.ylabel("evaluation func value")
        plt.legend(loc="best")
        plt.show()
        
    def predict(self, X):
        return np.sign(self.w0_ + np.dot(X, self.w_))


class ScratchDecisionTreeClassifier:
    """
    DecisionTreeClassifierのスクラッチ実装

    Parameters
    ----------
    max_depth : int （初期値： None）
      決定木の階層の深さの制限値

    Attributes
    ----------
    self.tree : class Node
      決定木のルートノード（別途定義クラスのインスタンス）
    self.feature_importances_ : ndarray, shape (n_features,)
      学習後の、各特徴量の重要度
      
    """
    def __init__(self, max_depth=None):
        self.tree = None
        self.max_depth = max_depth
    
    def fit(self, X_train, y_train):
        # fitting
        self.n_samples = X_train.shape[0]
        total_labels = np.unique(y_train)
        self.tree = Node(self.max_depth)
        self.tree.divide(X_train, y_train, 0, total_labels) # データ2種、根の深さ0、訓練データのラベル種が引数
        
        # feature importances計算
        self.feature_importances_ = np.zeros(X_train.shape[1])
        
        # self.tree（ルートノード）から関数実行開始（関数は本クラス内に別途定義）
        self._feature_importances_calc(self.tree) # 各ノードにおいて再帰的に計算する関数
        
        # 正規化（ゼロ除算の場合は何もしない）
        try:
            self.feature_importances_ /= self.feature_importances_.sum()
        except:
            return
    
    def predict(self, X_test):
        pred = []
        # 各サンプル行毎にNodeクラスのpredictメソッドを実行
        for row in X_test:
            pred.append(self.tree.predict(row))
        return np.array(pred)
    
    def accuracy_score(self, X_test, y_test):
        return sum(self.predict(X_test)==y_test) / float(len(y_test))
    
    def _feature_importances_calc(self, node):
        # リーフに達したら計算終了
        if node.info_gain == 0.0 or node.depth == node.max_depth:
            return
        
        # 親ノードのfeature_importancesを計算し、上記ndarrayに格納
        self.feature_importances_[node.feature] += node.info_gain * node.n_samples
        
        # 左子ノードと右子ノードで同様の処理に進む
        self._feature_importances_calc(node.left)
        self._feature_importances_calc(node.right)
    
class Node:
    """
    決定木分類器内で呼ばれるノードクラス
    
    """
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.depth = None
        self.label = None
        self.left = None
        self.right = None
        self.info_gain = 0.0
        self.feature = None
        self.threshold = None
    
    def divide(self, X, y, depth, total_labels):
        # ノードに渡されたデータのサンプル数、ラベル種、深さをこの時点で格納
        self.n_samples = X.shape[0]
        self.labels = np.unique(y)
        self.depth = depth
        # ラベルが1種類だけ（ピュア）の状態になったらリーフとして終了
        if len(self.labels) == 1:
            self.label = y[0]
            return
        
        # ラベルが2種類以上あったら最多ラベルをこのノードのラベルとする
        unique, count = np.unique(y, return_counts=True)
        y_count_dict = dict(zip(unique, count))
        self.label = max(y_count_dict, key=y_count_dict.get)
        
        # 情報利得を計算
        for column in range(X.shape[1]):
            for threshold in X[:, column]:
                y_left = y[X[:, column] <= threshold]
                y_right = y[X[:, column] > threshold]
                info_gain_calculated = self._info_gain_func(y, y_left, y_right) # スクラッチ関数
                # 情報利得が大きければ都度更新
                if self.info_gain < info_gain_calculated:
                    self.info_gain = info_gain_calculated
                    self.feature = column
                    self.threshold = threshold        
        # for文を回しても情報利得がゼロのままだったらリーフとして終了
        if self.info_gain == 0.0:
            return
        
        # max_depthまで達したらリーフとして終了
        if self.depth == self.max_depth:
            return
        
        # 左の子ノード用に、データ割付と新たなノード生成を行い、関数再帰させる
        X_left = X[X[:, self.feature] <= self.threshold]
        y_left = y[X[:, self.feature] <= self.threshold]
        self.left = Node(self.max_depth)
        # 割付したデータ、+1した深さ、訓練データのラベル種を渡す
        self.left.divide(X_left, y_left, self.depth+1, total_labels)
        
        # 右の子ノード用に、データ割付と新たなノード生成を行い、関数再帰させる
        X_right = X[X[:, self.feature] > self.threshold]
        y_right = y[X[:, self.feature] > self.threshold]
        self.right = Node(self.max_depth)
        # 割付したデータ、+1した深さ、訓練データのラベル種を渡す
        self.right.divide(X_right, y_right, self.depth+1, total_labels)
    
    def predict(self, row):
        # リーフまで達したら、既に学習された結果としてそのノードに格納されているラベルを返す
        # max_depth==Noneのtreeであれば、info_gain=0がリーフとなっている
        if self.info_gain == 0.0 or self.depth == self.max_depth:
            return self.label
        
        # まだ途中なら更に下流のリーフに向かう
        else:
            # 着目する特徴量が閾値以下なら左の子ノードのpredictへ
            if row[self.feature] <= self.threshold:
                return self.left.predict(row)
            # それ以外は右の子ノードのpredictへ
            else:
                return self.right.predict(row)
    
    def _info_gain_func(self, y_parent, y_left, y_right):
        return self._gini_func(y_parent) - (
            y_left.shape[0] / y_parent.shape[0]) * self._gini_func(y_left) - (
            y_right.shape[0] / y_parent.shape[0]) * self._gini_func(y_right)
    
    def _gini_func(self, y):
        gini = 1
        y_species = np.unique(y)
        for y_i in y_species:
            gini -= (sum(y==y_i) / y.shape[0])**2
        return gini


def scoring_func(X_train, X_test, y_train, y_test, 
                 model_names, models, feature_names, target_names, test_idx, two_features=False, sv_show=False):
    """
    多値分類（3種まで）を多次元特徴量において複数の学習済モデルで予測・評価し、以下1.、2.、3.を出力する。
    
    2次元特徴量データを評価・可視化する場合、
    または多次元特徴量データから2種類取り出す全組合せにてそれぞれ評価・可視化する場合は、
    4.も追加で出力する。
    
    1. 訓練データの正解率、テストデータの正解率、適合率、再現率、f1スコアの5種の集計表
    2. 1.の集計表の棒グラフ
    3. 混同行列をモデル別に行ごとにまとめた図
    4. 決定領域をモデル別に行ごとにまとめた図
    
    決定領域の出力には別途作成したdecision_region関数を利用する。

    Parameters
    ----------------
    X_train : ndarray, shape(n_samples, n_features)
        学習用データの特徴量
    X_test : ndarray, shape(n_samples, n_features)
        テスト用データの特徴量
    y_train : ndarray, shape(m_samples,)
        学習用データの正解値
    y_test : ndarray, shape(m_samples,)
        テスト用データの正解値
    model_names : list of str
        グラフタイトルに組み込むmodel名のリスト
    models : list of instances
        学習するモデルのインスンタスのリスト
    feature_names : list of str
        入力する特徴量名のリスト
    target_names= : list of str
        目的ラベル名のリスト
    test_idx : range of int
        テストデータのindex(eg. range(80,100))
    two_features : boolean （初期値：False）
        多次元特徴量から2種類の特徴量を取り出す全組合せでそれぞれ評価、可視化する場合はTrueに設定
    sv_show : boolean
        Trueでsuppor vectorを表示する
    """
    score = []
    scoring_names = ["train_accuracy", "test_accuracy", "test_precision", "test_recall", "test_f1_score"]
    # 予測後、評価リストを作成し、model毎に混同行列と決定境界を可視化
    for i, model_name, model in zip(np.arange(len(models)), model_names, models):
        # 予測
        if two_features==True:
            combs = list(itertools.combinations(np.arange(X_train.shape[1]), 2))
            y_train_pred = model.predict(X_train[:, combs[i]])
            y_test_pred = model.predict(X_test[:, combs[i]])
        elif two_features==False:
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
        # 例外処理(TrueでもFalseでもない場合)
        else:
            raise ValueError("Set boolean value to 'two_features' parameter !")
            
        # 評価
        score.append(accuracy_score(y_train, y_train_pred)) # 訓練データの正解率を算出
        score.append(accuracy_score(y_test, y_test_pred)) # テストデータの正解率を算出
        score.append(precision_score(y_test, y_test_pred)) # テストデータの適合率を算出
        score.append(recall_score(y_test, y_test_pred)) # テストデータの再現率を算出
        score.append(f1_score(y_test, y_test_pred)) # テストデータのf1スコアを算出
        # 混同行列を可視化
        plt.rcParams["font.size"] = 16
        plt.figure(figsize=(20,5))
        plt.subplot(1,2,1)
        conf_mat = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(conf_mat, annot=True)
        plt.title(model_name)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        # 2次元特徴量であれば決定境界を可視化
        if X_train.shape[1]==2 or two_features==True:
            plt.subplot(1,2,2)
            if X_train.shape[1]==2:
                X_combined = np.vstack((X_train, X_test))
                y_combined = np.hstack((y_train, y_test))
                # 別途定義の決定領域出力関数にパラメータ移譲
                decision_region(X=X_combined, y=y_combined, model=model, step=0.01, test_idx=test_idx, 
                                feature_names=feature_names, 
                                target_names=target_names, model_name=model_name, sv_show=sv_show)
                plt.show()
            else:
                X_combined = np.vstack((X_train[:, combs[i]], X_test[:, combs[i]]))
                y_combined = np.hstack((y_train, y_test))
                # 別途定義の決定領域出力関数にパラメータ移譲
                decision_region(X=X_combined, y=y_combined, model=model, step=0.01, test_idx=test_idx, 
                                feature_names=[feature_names[combs[i][0]], feature_names[combs[i][1]]], 
                                target_names=target_names, model_name=model_name, sv_show=sv_show)
                plt.show()
    # 評価したスコアを集計して表示
    score = np.array(score).reshape(-1,5).round(2) # 各列を5種のスコアに変形
    score = pd.DataFrame(score, index=model_names, columns=scoring_names)
    display(score) # スコアのDataFrameを表示
    score.T.plot(kind="bar") # スコア毎に棒グラフを表示
    plt.legend(bbox_to_anchor=(1,1))
    plt.ylim(0.6,1.05)
    plt.show()


def decision_region(X, y, model, step=0.01, test_idx=None, sv_show=False, feature_names=["f0","f1"], 
                    target_names=None, model_name="unknown model"):
    """
    多値分類を2次元の特徴量で学習したモデルの決定領域を描く。
    背景の色が学習したモデルによる推定値から描画される。
    散布図の点は学習用・テスト用データである。

    Parameters
    ----------------
    X : ndarray, shape(n_samples, 2)
        学習用・テスト用データの特徴量
    y : ndarray, shape(n_samples,)
        学習用・テスト用データの正解値
    model : object
        学習したモデルのインスンタス
    step : float, (default : 0.01)
        推定値を計算する間隔
    test_idx : range of int
        テストデータのindex
    sv_show : boolean
        Trueでsuppor vectorを表示する
    feature_names : list of str
        軸ラベルの一覧
    target_names= : list of str
        凡例の一覧
    model_name : str
        グラフタイトルに組み込むmodel名
    """
    # setting
    n_classes = np.unique(y).shape[0]
    colors = [cm.jet(float(i) / n_classes) for i in range(n_classes)]
    if target_names is None:
        target_names=[i for i in range(len(np.unique(y)))] # target_namesに指定がなければ[0,1,2,...]をラベルとする

    # pred
    x1_min, x1_max = np.min(X[:, 0])-0.5, np.max(X[:, 0])+0.5
    x2_min, x2_max = np.min(X[:, 1])-0.5, np.max(X[:, 1])+0.5
    xx1, xx2  = np.meshgrid(np.arange(x1_min, x1_max, step), np.arange(x2_min, x2_max, step))
    mesh = np.c_[np.ravel(xx1), np.ravel(xx2)]
    Z = model.predict(mesh).reshape(xx1.shape)

    # train & test plot
    plt.contourf(xx1, xx2, Z, n_classes-1, cmap=ListedColormap(colors), alpha=0.2)
    plt.contour(xx1, xx2, Z, n_classes-1, colors='black', linewidths=2, alpha=0.5)
    for i, target in enumerate(np.unique(y)):
        plt.scatter(X[y==target][:, 0], X[y==target][:, 1], 
                    s=80, color=colors[i], alpha=0.3, label=target_names[i], marker='o')
    plt.title("decision region["+model_name+"]")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    
    # emphasizing test plot
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        for i, target in enumerate(np.unique(y_test)):
            plt.scatter(X_test[y_test==target][:, 0], X_test[y_test==target][:, 1], 
                        s=100, color=scatter_color[i], edgecolor="black", alpha=1.0, linewidth=1, 
                        label=target_names[i]+"[test_set]", marker="o")
    
    # emphasizing support-vector plot
    if sv_show:
        if test_idx:
            plt.scatter(X[np.arange(X.shape[0] - len(test_idx))][model.lam_!=0.][:, 0], 
                        X[np.arange(X.shape[0] - len(test_idx))][model.lam_!=0.][:, 1], 
                        s=200, color=(0,0,0,0), edgecolor="black", label="support vector", marker='o')
        else:
            plt.scatter(X[model.lam_!=0.][:, 0], X[model.lam_!=0.][:, 1], 
                        s=200, color=(0,0,0,0), edgecolor="black", label="support vector", marker='o')
    
    plt.legend(loc="best")
    
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

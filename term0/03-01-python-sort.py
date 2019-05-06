#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 20:21:37 2019

@author: arimoto
"""

import numpy as np
import csv, operator
import pandas as pd

# CSVファイル（utf-8）をカレントディレクトリに別途作成済の状況を想定


# 方法1:csvモジュールを利用
with open("03-01-python-sort.csv") as csv_file:
    # csvファイルをそのまま出力
    reader = csv.reader(csv_file) # イテレータを作成
    for row in reader:
        print(row)
with open("03-01-python-sort.csv") as csv_file:
    # csvファイルをソートして出力
    reader = csv.reader(csv_file) # イテレータを作成
    # 郵便番号の列（0列目の次の1列目）でソート
    reader_s = sorted(reader, key=operator.itemgetter(1))
    for row in reader_s:
        print(row)

# 方法2:DataFrameを利用
df = pd.read_csv("03-01-python-sort.csv")
print(df)
df_sorted = df.sort_values("郵便番号")
print(df_sorted)


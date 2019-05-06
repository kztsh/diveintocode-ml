#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:08:52 2019

@author: arimoto
"""

# 期待する出力
"""
月曜日は、3時間勉強する予定です。
1限目 数学
2限目 機械学習
3限目 深層学習
火曜日は、1時間勉強する予定です。
1限目 エンジニアプロジェクト
水曜日は、3時間勉強する予定です。
1限目 Python
2限目 数学
3限目 機械学習
木曜日は、お休みです。
金曜日は、4時間勉強する予定です。
1限目 深層学習
2限目 エンジニアプロジェクト
3限目 Python
4限目 数学
土曜日は、2時間勉強する予定です。
1限目 機械学習
2限目 深層学習
日曜日は、2時間勉強する予定です。
1限目 エンジニアプロジェクト
2限目 Python
"""

WEEK_LIST = ['月', '火', '水', '木', '金', '土', '日']
SUBJECT_LIST = ['Python', '数学', '機械学習', '深層学習','エンジニアプロジェクト']

def output_schedule(study_time_list, study_term_list, holiday):
    '''今週の勉強予定を出力します'''
    # その日の曜日、勉強時間、勉強科目を取り出し
    for weekday, study_time, term in zip(
            WEEK_LIST, study_time_list, study_term_list):
        # 休日は勉強科目を表示せず次の曜日へ
        if weekday == holiday:
            print("{}曜日は、お休みです。".format(weekday))
            continue
        else:
            print("{}曜日は、{}時間勉強する予定です。".format(
                    weekday, study_time))
            # その日の各勉強科目tを順番に取り出し
            for idx, t in enumerate(term):
                print("{}限目 {}".format(idx+1, SUBJECT_LIST[t]))


def main():
    '''勉強情報をoutput_scheduleに渡します'''
    # 1日に何時間勉強するか（1週間　月曜日〜日曜日）
    study_time_list = [3, 1, 3, 0, 4, 2, 2]
    # 今週のholidayを新たに定義
    holiday = "木" 
    # 今週のstudy_term_listを新たに定義。
    # 各曜日の学習科目がSUBJECT_LISTの各インデックスに対応。
    study_term_list = [[1,2,3],[4,],[0,1,2],[],[3,4,0,1],[2,3],[4,0]]
    output_schedule(study_time_list, study_term_list, holiday)


if __name__ == '__main__':
    main()


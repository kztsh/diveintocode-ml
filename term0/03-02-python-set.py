#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:35:56 2019

@author: arimoto
"""

course_dict = {
    'AIコース': {'Aさん', 'Cさん', 'Dさん'},
    'Railsコース': {'Bさん', 'Cさん', 'Eさん'},
    'Railsチュートリアルコース': {'Gさん', 'Fさん', 'Eさん'},
    'JS': {'Aさん', 'Gさん', 'Hさん'},
}


def find_person(want_to_find_person):
    """
    受講生がどのコースに在籍しているかを出力する。
    まずはフローチャートを書いて、どのようにアルゴリズムを解いていくか考えてみましょう。
    """
    # ここにコードを書いてみる
    for course_name in course_dict:
        result = course_dict[course_name] & want_to_find_person
        if len(result) >= 2:
            print("{}に{}は在籍しています。".format(course_name, result))
        elif len(result) == 1:
            print("{}に{}のみ在籍しています。".format(course_name, result))
        else:
            print("{}に{}は在籍していません。".format(
                    course_name, want_to_find_person))


def main():
    want_to_find_person = {'Cさん', 'Aさん'}
    print('探したい人: {}'.format(want_to_find_person))
    find_person(want_to_find_person)


if __name__ == '__main__':
    main()
    

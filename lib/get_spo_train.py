# -*- coding: utf-8 -*-
"""
This module to generate training data for training a so-labeling model
"""

import io
import os
import sys
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

def get_p(input_file):
    """
    Generate training data for so labeling model
    """
    with open(input_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            try:
                dic = json.loads(line.strip())
            except:
                continue
            spo_list = dic['spo_list']
            p_list = [item['predicate'] for item in spo_list]
            for p in p_list:
                print("\t".join([json.dumps(dic, ensure_ascii=False), p]))


if __name__ == '__main__':
    input_file = sys.argv[1]
    get_p(input_file)

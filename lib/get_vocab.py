# -*- coding: utf-8 -*-
"""
This module to generate vocabulary list
"""

import os
import sys
import json
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

def load_word_file(f_input):
    """
    Get all words in files
    :param string: input file
    """
    file_words = {}
    with open(f_input, 'r', encoding='utf-8') as fr:
        words = []
        for line in fr:
            try:
                dic = json.loads(line.strip())
                postag = dic['postag']
                words = [item["word"].strip() for item in postag]
            except:
                continue
            for word in words:
                file_words[word] = file_words.get(word, 0) + 1
    return file_words


def get_vocab(train_file, dev_file):
    """
    Get vocabulary file from the field 'postag' of files
    :param string: input train data file
    :param string: input dev data file
    """
    word_dic = load_word_file(train_file)
    if len(word_dic) == 0:
        raise ValueError('The length of train word is 0')
    dev_word_dic = load_word_file(dev_file)
    if len(dev_word_dic) == 0:
        raise ValueError('The length of dev word is 0')
    for word in dev_word_dic:
        if word in word_dic:
            word_dic[word] += dev_word_dic[word]
        else:
            word_dic[word] = dev_word_dic[word]
    print('<UNK>')
    vocab_set = set()
    value_list = sorted(word_dic.items(), key=lambda d: d[1], reverse=True)
    for word in value_list[:30000]:
        print(word[0])
        vocab_set.add(word[0])

    # add predicate in all_50_schemas
    if not os.path.exists('./data/all_50_schemas'):
        raise ValueError('./data/all_50_schemas')
    with open('./data/all_50_schemas', 'r', encoding='utf-8') as fr:
        for line in fr:
            dic = json.loads(line.strip())
            p = dic['predicate']
            if p not in vocab_set:
                vocab_set.add(p)
                print(p)

if __name__ == '__main__':
    train_file = sys.argv[1]
    dev_file = sys.argv[2]
    get_vocab(train_file, dev_file)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N Final Project
utils.py utility functions and classes for math_baseline LSTM with attention
should be the same as math_baseline LSTM
Tyler Roost <Roost@usc.edu>
Modified from CS224N Assignment 5 utils.py made by:
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""
import math, random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def pad_sents_char(sents, char_pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[int]]): list of sentences, result of `sents2charindices()`
        from `vocab.py`
    @param char_pad_token (int): index of the character-padding token
    @returns sents_padded (list[list[int]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the appropriate pad token, such that
        each sentence in the batch now has same number of chars
        Output shape: (batch_size, max_sentence_length)
    """

    sents_padded = []
    max_length = max([len(sent) for sent in sents])

    for sent in sents:
        # pad sents to be max_length
        if len(sent) < max_length:
            for _ in range(max_length - len(sent)):
                sent.append(char_pad_token)
        sents_padded.append(sent)

    return sents_padded

def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e['question']), reverse=True)
        src_sents = [e['question'] for e in examples]
        tgt_sents = [e['answer'] for e in examples]

        yield src_sents, tgt_sents

def batch_iter_new(src, tgt, modules: List[str], levels: List[str], batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """

    for level in levels:
        # data_len = 0
        # for module in modules:
        #     data_len += len(src[level][module])

        data_len = min([len(src[level][module]) for module in modules])
        batch_num = math.ceil(data_len / batch_size)
        index_array = list(range(data_len))
        module_array = np.random.randint(low = 0, high = len(modules), size = data_len)
        if shuffle:
            np.random.shuffle(index_array)

        for i in range(batch_num):
            indices = index_array[i * batch_size: (i + 1) * batch_size]
            module_indices = module_array[i * batch_size: (i+1) * batch_size]
            actual_modules = [modules[module_index] for module_index in module_indices]


            src_examples = [src[level][actual_module][idx] for actual_module, idx in zip(actual_modules, indices)]
            tgt_examples = [tgt[level][actual_module][idx] for actual_module, idx in zip(actual_modules, indices)]

            src_sents, tgt_sents = zip(*sorted(zip(src_examples, tgt_examples), key=lambda x: len(x[0]), reverse=True))
            # print(zipped[0])
            # src_sents = zipped[0]
            # tgt_sents = zipped[1]
            yield src_sents, tgt_sents


def get_size_of_model(model):
    size = 0
    for mod in list(model.modules()):
        for param in list(mod.parameters()):
            size += np.prod(np.array(param.size()))
    return size

def get_data(modules: List[str], levels: List[str], filepath, isTest: bool):
    if isTest:
        test_src = {}
        test_tgt = {}
        for level in levels:
            test_src[level] = {}
            test_tgt[level] = {}
            for module in modules:
                test_src[level][module] = []
                test_tgt[level][module] = []
                print('reading from ',level, module)
                #assume file path is D:\Data Science\Machine Learning\Deep Learning\CS224n-Stanford\cs224nFinalProject\datasets\math_dataset
                for i, line in enumerate(open(filepath + '\\' + level + '\\' + module + '.txt')):
                    if i < 2000:
                        sent = line.strip()
                        if i % 2 == 0:
                            test_src[level][module].append(sent)
                        else:
                            test_tgt[level][module].append(sent)

        return test_src, test_tgt
    else:
        src = {}
        tgt = {}
        for level in levels:
            src[level] = {}
            tgt[level] = {}
            for module in modules:
                src[level][module] = []
                tgt[level][module] = []
                print('reading from ', module)
                #assume file path is D:\Data Science\Machine Learning\Deep Learning\CS224n-Stanford\cs224nFinalProject\datasets\math_dataset
                for i, line in enumerate(open(filepath + '\\' + level + '\\' + module + '.txt')):
                    sent = line.strip()
                    if i % 2 == 0:
                        src[level][module].append(sent)
                    else:
                        tgt[level][module].append(sent)

        dev_src = {}
        dev_tgt = {}
        for level in levels:
            dev_src[level] = {}
            dev_tgt[level] ={}
            for module in modules:
                dev_src[level][module] = []
                dev_tgt[level][module] = []
                for i in range(len(src[level][module])):
                    if i % 20 == 0:
                        dev_src[level][module].append(src[level][module].pop())
                        dev_tgt[level][module].append(tgt[level][module].pop())

        return src, tgt, dev_src, dev_tgt

class MathDataset(torch.utils.data.Dataset):
    def __int__(self, dataset_path, modules, level):
        self.data_src = []
        self.data_tgt = []
        self.modules = modules
        self.levels = levels
        for level in levels:
            for module in modules:
                print('reading from ', level, module)
                #assume file path is D:\Data Science\Machine Learning\Deep Learning\CS224n-Stanford\cs224nFinalProject\datasets\math_dataset
                for i, line in enumerate(open(dataset_path + '\\' + level + '\\' + module + '.txt')):
                    sent = line.strip()
                    if i % 2 == 0:
                        self.data_src.append(sent)
                    else:
                        self.data_tgt.append(sent)

    def __len__(self):
        return len(self.data_src)

    def __getitem__(self, idx):
        return (self.data_src[idx], self.data_tgt[idx])

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
import math
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

def get_size_of_model(model):
    size = 0
    for mod in list(model.modules()):
        for param in list(mod.parameters()):
            size += np.prod(np.array(param.size()))
    return size

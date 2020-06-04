#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N Final Project
utils.py utility functions and classes for math_baseline LSTM
Tyler Roost <Roost@usc.edu>
Modified from CS224N Assignment 5 utils.py made by:
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""
import math
import copy
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import AdamW

def get_optimizers(model, lr, eps = 1e-8) -> torch.optim.Optimizer:
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        """
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(params = optimizer_grouped_parameters, lr = lr, eps = eps)
        return optimizer

def log(logs):
    print("\nloss: %f learning_rate: %f epoch %d step: %d" % (logs["loss"], logs["lr"], logs["epoch"], logs["step"]))

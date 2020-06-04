#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N Final Project
model.py model for math_baseline LSTM with attention
should be the same as math_baseline LSTM
Tyler Roost <Roost@usc.edu>

Usage:
    run.py train --vocab=<file> [options]
    run.py decode [options] MODEL_PATH OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --load                                  load model
    --vocab=<file>                          vocab file
    --dataset_path=<file>                   path to dataset [default: D:\Data Science\Machine Learning\Deep Learning\CS224n-Stanford\cs224nFinalProject\datasets\math_dataset]
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 256]
    --embed-size=<int>                      embedding size [default: 50]
    --hidden-size=<int>                     hidden size [default: 256]
    --thinking-steps=<int>                  number of thinking steps [default: 15]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 100]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.004]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --load-from=<file>                      model load path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""
import math
import sys
import pickle
import time

from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import batch_iter, get_size_of_model, batch_iter_new, get_data, MathDataset
from vocab import Vocab, VocabEntry
from knockknock import email_sender
from docopt import docopt
import torch
import numpy as np
import nlp
from vocab import Vocab, VocabEntry
from model import MathBaselineLSTMwAttention, Hypothesis

def evaluate_ppl(model, dev_data, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def evaluate_ppl_new(model, dev_src, dev_tgt, modules, levels, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter_new(dev_src, dev_tgt, modules, levels, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl

def compute_corpus_score(src_sents, references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level Accuracy.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns score: corpus-level Accuracy score
    """
    if references[0][0] == '<sos>':
        references = [ref[1:-1] for ref in references]
    score = 0
    for src, ref, hyp in zip(src_sents, references, hypotheses):

        hyp = "".join(hyp.value)

        if(ref == hyp):
            score += 1
            print(ref, hyp)
    score /= len(references)
    return score


@email_sender(recipient_emails=["tyler.roost@gmail.com"], sender_email="tyler.roost@gmail.com")
def train(args):
    train_data, dev_data = nlp.load_dataset('math_dataset', split = ['train[:95%]', 'train[95%:]'], cache_dir = "D:/.cache/nlp/")
    vocab = Vocab.load(args['--vocab'])
    embed_size = int(args['--embed-size'])
    hidden_size = int(args['--hidden-size'])
    thinking_steps = int(args['--thinking-steps'])
    dropout_rate = float(args['--dropout'])

    train_batch_size = int(args['--batch-size'])

    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']
    modules = ["algebra__linear_1d",
            "algebra__linear_1d_composed",
            "algebra__linear_2d",
            "algebra__linear_2d_composed",
            "algebra__polynomial_roots",
            "algebra__polynomial_roots_composed",
            "algebra__sequence_next_term",
            "algebra__sequence_nth_term",
            "calculus__differentiate",
            "calculus__differentiate_composed",
            "polynomials__add",
            "polynomials__coefficient_named",
            "polynomials__collect",
            "polynomials__compose",
            "polynomials__evaluate",
            "polynomials__evaluate_composed",
            "polynomials__expand",
            "polynomials__simplify_power"
            ]

    levels = ["train-easy",
            "train-medium",
            "train-hard"
            ]
    device = torch.device("cuda:0" if args['--cuda'] else "cpu")

    # print('reading in data...')
    # filepath = 'D:\Data Science\Machine Learning\Deep Learning\CS224n-Stanford\cs224nFinalProject\datasets\math_dataset'
    # train_src, train_tgt, dev_src, dev_tgt = get_data(modules, levels, filepath, False)
    # print('finished reading data')
    model = MathBaselineLSTMwAttention(
        embed_size,
        hidden_size,
        vocab,
        thinking_steps,
        dropout_rate
    )
    if args['--load']:
        model_load_path = args['--load-from']
        print('loading previously stored model from ', model_load_path)
        params = torch.load(model_load_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(params['state_dict'])
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

        print('restore parameters of the optimizers', file=sys.stderr)
        optimizer.load_state_dict(torch.load(model_load_path + '.optim'))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    size = get_size_of_model(model)

    print("model size %d parameters" % size)
    model.train()

    uniform_init = float(args['--uniform-init'])
    # if np.abs(uniform_init) > 0.:
    #     print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
    #     for p in model.parameters():
    #         p.data.uniform_(-uniform_init, uniform_init)
    # else:
    #     print('Xavier initialize parameters', file=sys.stderr)
    #     for p in model.parameters():
    #         p.data.xavier_uniform_()


    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')
    easy = ['train-easy']
    medium = ['train-easy',
              'train-medium']
    hard = ['train-easy',
            'train-medium',
            'train-hard']
    filepath = 'D:\Data Science\Machine Learning\Deep Learning\CS224n-Stanford\cs224nFinalProject\datasets\math_dataset'
    train_easy = MathDataset(filepath, easy, modules)
    train_medium = MathDataset(filepath, medium, modules)
    train_hard = MathDataset(filepath, hard, modules)
    while True:
        epoch += 1
        if epoch == 1:
            train_data = train_easy
        elif epoch <= 5:
            train_data = train_medium
        else:
            train_data = train_hard
        # for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
        # for src_sents, tgt_sents in batch_iter_new(train_src, train_tgt, modules, levels, batch_size=train_batch_size, shuffle=True):
        dataloader = torch.utils.data.Dataloader(train_data, batch_size=train_batch_size, shuffle=True)
        for batch in dataloader:
            src_sents, tgt_sents = batch[0], batch[1]
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_sents)

            example_losses = -model(src_sents, tgt_sents) # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s) for s in tgt_sents)  # omitting leading `<sos>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=train_batch_size)   # dev batch size can be a bit larger
                # dev_ppl = evaluate_ppl_new(model, dev_src, dev_tgt, modules, levels, batch_size=train_batch_size)
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            return

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

            if epoch == int(args['--max-epoch']):
                print('reached maximum number of epochs!', file=sys.stderr)
                return


def decode(args: Dict[str, str]):
    """ Performs decoding on a test set, and save the best-scoring decoding results
        and computes question accuracy
    @param args (Dict): args from cmd line
    """

    test_data = nlp.load_dataset('math_dataset', split = ['test[:1%]'], cache_dir = "D:/.cache/nlp/")
    test_data_src = test_data[0]['question']
    test_data_tgt = test_data[0]['answer']
    # modules = ["algebra__linear_1d",
    #         "algebra__linear_1d_composed",
    #         "algebra__linear_2d",
    #         "algebra__linear_2d_composed",
    #         "algebra__polynomial_roots",
    #         "algebra__polynomial_roots_composed",
    #         "algebra__sequence_next_term",
    #         "algebra__sequence_nth_term",
    #         "calculus__differentiate",
    #         "calculus__differentiate_composed",
    #         "polynomials__add",
    #         "polynomials__coefficient_named",
    #         "polynomials__collect",
    #         "polynomials__compose",
    #         "polynomials__evaluate",
    #         "polynomials__evaluate_composed",
    #         "polynomials__expand",
    #         "polynomials__simplify_power"
    #         ]
    #
    # levels = ["train-easy",
    #         "train-medium",
    #         "train-hard"
    #         ]
    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    model = MathBaselineLSTMwAttention.load(args['MODEL_PATH'])
    size = get_size_of_model(model)

    print("model size %d parameters" % size)
    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    hypotheses = beam_search(model, test_data_src, beam_size=int(args['--beam-size']),  max_decoding_time_step=int(args['--max-decoding-time-step']))
    # print(hypotheses)
    top_hypotheses = [hyps[0] for hyps in hypotheses]
    score = compute_corpus_score(test_data_src, test_data_tgt, top_hypotheses)
    # hypotheses = beam_search(model, test_data_src[level][module], beam_size=int(args['--beam-size']),  max_decoding_time_step=int(args['--max-decoding-time-step']))
    print('Corpus Accuracy: {}'.format(score * 100), file=sys.stderr)


    # filepath = 'D:\Data Science\Machine Learning\Deep Learning\CS224n-Stanford\cs224nFinalProject\datasets\math_dataset'
    # full_data_src = []
    # full_data_tgt = []
    # full_hypotheses = []
    # for level in levels:
    #     for module in modules:
    #         test_data_src, test_data_tgt = get_data([module], [level], filepath, True)
    #
    #
    #         hypotheses = beam_search(model, test_data_src[level][module], beam_size=int(args['--beam-size']),  max_decoding_time_step=int(args['--max-decoding-time-step']))
    #
    #
    #         top_hypotheses = [hyps[0] for hyps in hypotheses]
    #         # print(level, module)
    #         score = compute_corpus_score(test_data_src[level][module], test_data_tgt[level][module], top_hypotheses)
    #         print('Corpus Accuracy: {}'.format(score * 100), file=sys.stderr)
    #         for hyp in top_hypotheses:
    #             full_hypotheses.append(hyp)
    #         with open(args['OUTPUT_FILE'], 'w') as f:
    #             for src_sent, hyps in zip(test_data_src, hypotheses):
    #                 top_hyp = hyps[0]
    #                 hyp_sent = ''.join(top_hyp.value)
    #                 f.write(hyp_sent + '\n')
    #         for sent in test_data_src[level][module]:
    #             full_data_src.append(sent)
    #         for sent in test_data_tgt[level][module]:
    #             full_data_tgt.append(sent)
    #
    # full_score = compute_corpus_score(full_data_src, full_data_tgt, full_hypotheses)
    # print('Full Score: ', full_score)


def beam_search(model: MathBaselineLSTMwAttention, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in test_data_src:
            example_hyps, thinking_hidden_states, thinking_attention_scores = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses




def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)

    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()

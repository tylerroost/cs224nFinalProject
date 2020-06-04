#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N Final Project
run.py model for math_baseline Transformer
Tyler Roost <Roost@usc.edu>

Usage:
    run.py train [options]
    run.py train_without_trainer [options]
    run.py insepct [options]
    run.py decode [options]
    run.py dataset [options]

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --load                                  load model
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 42]
    --batch-size=<int>                      batch size [default: 8]
    --log-every=<int>                       log every [default: 100]
    --valid-niter=<int>                     number of steps to perform validation [default: 1000]
    --save-every=<int>                      save every [default: 1000]
    --max-epoch=<int>                       max epoch [default: 4]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.00005]
    --save-to=<file>                        model save path [default: model.bin]
    --load-from=<file>                      model load path [default: model.bin]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
    --clip-grad=<float>                     value to clip the gradient to [default: 1.0]
"""
import os
import math
import sys
import pickle
import time
import logging
import re
import shutil
from pathlib import Path

from typing import List, Tuple, Dict, Set, Union

from tqdm.auto import tqdm
from docopt import docopt
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import numpy as np
from AnswerMaskDataCollator import AnswerMaskDataCollator
from MathDataset import MathDataset
from utils import get_optimizers, log
dataset_path = 'D:\Data Science\Machine Learning\Deep Learning\CS224n-Stanford\cs224nFinalProject\datasets\math_dataset'
cache_dir = 'D:\.cache\\transformers'
data_path = './data/'
def train(args):
    logging.basicConfig(level=logging.INFO)
    tokenizer = transformers.AlbertTokenizer.from_pretrained('albert-base-v2', cache_dir = cache_dir)
    albert_for_math_config = transformers.AlbertConfig(hidden_size=768,
                                                       num_attention_heads=12,
                                                       intermediate_size=3072,
                                                       )

    if args['--load']:
        model = transformers.AlbertForMaskedLM.from_pretrained(args['--load-from'])
        training_args = transformers.TrainingArguments(
                            output_dir = args['--save-to'],
                            overwrite_output_dir = True,
                            num_train_epochs = int(args['--max-epoch']),
                            per_gpu_train_batch_size = int(args['--batch-size']),
                            per_gpu_eval_batch_size = int(args['--batch-size']),
                            logging_steps = int(args['--log-every']),
                            save_steps = int(args['--save-every']),
                            save_total_limit = 10,
                            learning_rate = float(args['--lr']),
                            seed = int(args['--seed']),
                        )

    else:
        model = transformers.AlbertForMaskedLM(albert_for_math_config)
        training_args = transformers.TrainingArguments(
                            output_dir = args['--save-to'],
                            num_train_epochs = int(args['--max-epoch']),
                            per_gpu_train_batch_size = int(args['--batch-size']),
                            per_gpu_eval_batch_size = int(args['--batch-size']),
                            logging_steps = int(args['--log-every']),
                            save_steps = int(args['--save-every']),
                            save_total_limit = 10,
                            learning_rate = float(args['--lr']),
                            seed = int(args['--seed']),
                        )

    #load datasets
    print('Loading Data...')
    train_data = torch.load('./data/train_data_train-easy_algebra__linear_1d.pt')
    dev_data = torch.load('./data/dev_data_train-easy_algebra__linear_1d.pt')
    print('Finished loading data')
    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    model.to(device)
    trainer = transformers.Trainer(
        model = model,
        args = training_args,
        data_collator = AnswerMaskDataCollator(tokenizer),
        train_dataset = train_data,
        eval_dataset = dev_data,
        prediction_loss_only = True,
    )

    if args['--load']:
        trainer.train(model_path = args['--load-from'])
    else:
        trainer.train()

def train_without_trainer(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    batch_size = int(args['--batch-size'])
    logging_steps = int(args['--log-every'])

    tokenizer = transformers.AlbertTokenizer.from_pretrained('albert-base-v2', cache_dir = cache_dir)
    albert_for_math_config = transformers.AlbertConfig(hidden_size=768,
                                                       num_attention_heads=12,
                                                       intermediate_size=3072,
                                                       )
    print('Loading Data...')
    train_data = torch.load('./data/train_data_train-easy_algebra__linear_1d.pt')
    dev_data = torch.load('./data/dev_data_train-easy_algebra__linear_1d.pt')
    print('Finished loading data')
    data_collator = AnswerMaskDataCollator(tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size = batch_size,
                                                   sampler = torch.utils.data.sampler.RandomSampler(train_data),
                                                   collate_fn = data_collator.collate_batch
                                                   )


    if args['--load']:
        model = transformers.AlbertForMaskedLM.from_pretrained(args['--load-from'])
        optimizer = get_optimizers(model, float(args['--lr']))
        optimizer.load_state_dict(torch.load(os.path.join(args['--load-from'], "optimizer.pt"), map_location=device))
        global_step = int(args['--load-from'].split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader))
        steps_trained_in_current_epoch = global_step % len(train_dataloader)
        epoch = epochs_trained
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    else:
        model = transformers.AlbertForMaskedLM(albert_for_math_config)
        optimizer = get_optimizers(model, float(args['--lr']))
        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        epoch = 0
    model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    max_epoch = int(args['--max-epoch'])
    t_total = len(train_dataloader) * max_epoch
    tr_loss = 0.0
    logging_loss = 0.0
    min_eval_loss = 1e20 # might be too high
    valid_niter = int(args['--valid-niter'])
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data))
    logger.info("  Num Epochs = %d", max_epoch)
    logger.info("  train batch size = %d", batch_size)
    logger.info("  Total optimization steps = %d", t_total)
    num_eval_samples = 4096
    checkpoint_prefix = 'checkpoint'
    while(epoch < max_epoch):

        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, inputs in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            tr_loss += train_step(model, inputs, device)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args['--clip-grad']))
            optimizer.step()
            model.zero_grad()
            global_step += 1
            if global_step % logging_steps == 0:
                logs: Dict[str, float] = {}
                logs["loss"] = (tr_loss - logging_loss) / logging_steps
                logs["lr"] = (optimizer.defaults['lr']) # possible RuntimeError
                logs["epoch"] = epoch
                logs["step"] = global_step
                logging_loss = tr_loss
                log(logs)
            if global_step % valid_niter == 0:
                eval_loss = 0.0
                description = "Evaluation"
                sampler = torch.utils.data.sampler.SequentialSampler(dev_data[:num_eval_samples])
                eval_dataloader = torch.utils.data.DataLoader(
                                    dev_data[:num_eval_samples],
                                    sampler=sampler,
                                    batch_size=batch_size,
                                    collate_fn=data_collator.collate_batch,
                                )
                logger.info("***** Running %s *****", description)
                logger.info("   Num Examples = %d", num_eval_samples)
                logger.info("   Batch size = %d", batch_size)
                for inputs in tqdm(eval_dataloader, desc=description):
                    for k, v in inputs.items():
                        inputs[k] = v.to(device)
                    model.eval()
                    with torch.no_grad():
                        outputs = model(**inputs)
                        loss = outputs[0]
                        eval_loss += loss.item()
                print("\nEvaluation loss = %f" % (eval_loss / num_eval_samples))
                if eval_loss / num_eval_samples * batch_size < min_eval_loss:
                    min_eval_loss = eval_loss / num_eval_samples * batch_size
                    # save model and optimizer

                    output_dir = os.path.join(args['--save-to'] + '/validations/', f"{checkpoint_prefix}-{global_step}")
                    os.makedirs(output_dir, exist_ok=True)
                    model.save_pretrained(output_dir)
                    output_dir = os.path.join(args['--save-to'] + '/validations/')
                    rotate_checkpoints(output_dir)
                    output_dir = os.path.join(args['--save-to'] + '/validations/', f"{checkpoint_prefix}-{global_step}")
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            if global_step % int(args['--save-every']) == 0:
                output_dir = os.path.join(args['--save-to'], f"{checkpoint_prefix}-{global_step}")
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                output_dir = output_dir = os.path.join(args['--save-to'])
                rotate_checkpoints(output_dir)
                output_dir = os.path.join(args['--save-to'], f"{checkpoint_prefix}-{global_step}")
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        epoch_iterator.close()
        epoch += 1
    logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
def sorted_checkpoints(output_dir, checkpoint_prefix):
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")]

    for path in glob_checkpoints:
        print(path)
        regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
        if regex_match and regex_match.groups():
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted

def rotate_checkpoints(output_dir, checkpoint_prefix = 'checkpoint'):
    checkpoints_sorted = sorted_checkpoints(output_dir, checkpoint_prefix)
    if len(checkpoints_sorted) > 10:
        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - 10)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            print("     Deleting older checkpoint [%s] due to there being more than 10" % (checkpoint))
            shutil.rmtree(checkpoint)
def train_step(model, inputs, device):
    model.train()
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    outputs = model(**inputs)
    loss = outputs[0]
    loss.backward()
    return loss.item()


def decode(args, modules, levels):
    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    tokenizer = transformers.AlbertTokenizer.from_pretrained('albert-base-v2', cache_dir = cache_dir)
    model = transformers.AlbertForMaskedLM.from_pretrained(args['--load-from']).to(device)
    answer_mask = '[MASK]'
    model.eval()
    for level in levels:
        for module in modules:
            count = 0
            total_correct = 0
            all_decodeds = []
            for i, line in enumerate(open(dataset_path + '\\' + level + '\\' + module + '.txt')):
                sent = line.strip()
                if level == 'interpolate':
                    if i < 200:
                        if i % 2 == 0:
                            question = sent
                        else:
                            count += 1
                            input_encodings = tokenizer.encode(question, return_tensors = 'pt').to(device)
                            # masked_lm_labels = tokenizer.encode((question, sent), return_tensors = 'pt')
                            with torch.no_grad():
                                outputs = model(input_encodings)
                            # print(outputs[0].shape)
                            prediction_scores = outputs[0]
                            predicted_index = torch.argmax(prediction_scores[0, -1, :]).item()
                            decoded = tokenizer.decode(predicted_index)
                            all_decodeds.append(decoded)
                            # print(decoded)
                            # print(sent)
                            if decoded[-1] == sent:
                                total_correct += 1

            print('Accuracy for', level, module, '=', total_correct/count)
            print(set(all_decodeds))
def inspecet(args):
    pass

def create_dataset(args, modules, levels):
    train_src = []
    dev_src = []
    test_src = []
    answer_mask = '[MASK]'
    tokenizer = transformers.AlbertTokenizer.from_pretrained('albert-base-v2', cache_dir = cache_dir)
    for level in levels:
        for module in modules:
            print('reading from ', level, module)
            #assume file path is D:\Data Science\Machine Learning\Deep Learning\CS224n-Stanford\cs224nFinalProject\datasets\math_dataset
            for i, line in enumerate(open(dataset_path + '\\' + level + '\\' + module + '.txt')):
                sent = line.strip()
                if level == 'interpolate':
                    if i < 200:
                        if i % 2 == 0:
                            question = sent
                        else:
                            test_src.append((question, answer_mask))
                else:
                    if i % 2 == 0:
                        question = sent
                    else:
                        train_src.append((question, sent))
            if level != 'interpolate':
                for i in range(len(train_src)):
                    if i % 20:
                        dev_src.append(train_src.pop())

            if level == 'interpolate':
                print('Tokenizing test_src for', level, module)
                test_src = tokenizer.batch_encode_plus(test_src, pad_to_max_length = True, max_length = 161, return_tensors = 'pt')
                print('Finished Tokenizing test_src for', level, module)
                print('Saving test_src for', level, module)
                test_src = MathDataset(test_src)
                torch.save(test_src, data_path + 'test_data_' + level + '_' + module + '.pt')
                print('Finished saving test_src for', level, module)
            else:
                print('Tokenizing train_src for', level, module)
                train_src = tokenizer.batch_encode_plus(train_src, pad_to_max_length = True, max_length = 190, return_tensors = 'pt')
                print('Finished Tokenizing train_src for', level, module)
                print('Saving train_src for', level, module)
                train_src = MathDataset(train_src)
                torch.save(train_src, data_path + 'train_data_' + level + '_' + module + '.pt')
                print('Finished saving train_src for', level, module)
                print('Tokenizing dev_src for', level, module)
                dev_src = tokenizer.batch_encode_plus(dev_src, pad_to_max_length = True, max_length = 190, return_tensors = 'pt')
                print('Finished Tokenizing dev_src for', level, module)
                print('Saving dev_src for', level, module)
                dev_src = MathDataset(dev_src)
                torch.save(dev_src, data_path + 'dev_data_' + level + '_' + module + '.pt')
                print('Finished saving dev_src for', level, module)

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
    modules = ['algebra__linear_1d']
    levels = ['interpolate']
    if args['train']:
        train(args)
    elif args['train_without_trainer']:
        train_without_trainer(args)
    elif args['decode']:
        decode(args, modules, levels)
    elif args['dataset']:
        create_dataset(args, modules, levels)
    elif args['inspect']:
        inspect(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()

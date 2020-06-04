#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N Final Project
model.py model for math_baseline LSTM with attention
Tyler Roost <Roost@usc.edu>
Modified heavily from CS224N Assignment 5 nmt_model.py made by:
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""
import sys
from collections import namedtuple

from typing import List, Tuple, Dict, Set, Union
import torch.nn as nn
import torch
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class MathBaselineLSTMwAttention(nn.Module):
    """ Simple seq2seq encoder-decoder architecture for solving simple math problems:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
    """
    def __init__(self, embed_size, hidden_size, vocab, thinking_steps, dropout_rate=0.2):
        """ Init MathBaselineLSTM Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing singular vocab of chars
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(MathBaselineLSTMwAttention, self).__init__()
        self.embed_size = embed_size
        self.encoder_hidden_size = int(hidden_size)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.thinking_steps = thinking_steps

        self.model_embeddings_source = nn.Embedding(len(self.vocab.vocab.char2id), self.embed_size, self.vocab.vocab.char2id['<pad>'])
        self.model_embeddings_target = nn.Embedding(len(self.vocab.vocab.char2id), self.embed_size, self.vocab.vocab.char2id['<pad>'])
        self.encoder = nn.LSTM(self.embed_size, self.encoder_hidden_size, bidirectional = True)
        self.h_projection = nn.Linear(2 * self.encoder_hidden_size, self.hidden_size, bias = False)
        self.c_projection = nn.Linear(2 * self.encoder_hidden_size, self.hidden_size, bias = False)
        self.decoder = nn.LSTMCell(self.embed_size + self.encoder_hidden_size, self.hidden_size)
        self.att_projection = nn.Linear(2 * self.encoder_hidden_size, self.hidden_size, bias=False)
        self.combined_output_projection = nn.Linear(2 * self.encoder_hidden_size + self.hidden_size, self.encoder_hidden_size, bias=False)
        self.target_vocab_projection = nn.Linear(self.encoder_hidden_size, len(self.vocab.vocab.char2id), bias = False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the system.

        @param source (List[str]): list of source sentences
        @param target (List[str]): list of target sentences

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        source_lengths = [len(s) for s in source]

        source_padded_chars = self.vocab.vocab.to_input_tensor_char(source, device=self.device, isTarget = False) # (source_length, b, 1)
        target_padded_chars = self.vocab.vocab.to_input_tensor_char(target, device=self.device, isTarget = True) # (target_length, b, 1)
        # encode
        enc_hiddens, dec_init_state = self.encode(source_padded_chars, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        # decode
        combined_outputs, _ = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded_chars)


        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)
        target_char_masks = (target_padded_chars != self.vocab.vocab.char2id['<pad>']).float()
        # Compute log probability of generating true target chars
        rand_int = random.randint(0, 99)
        if rand_int == 0:
            print("".join([self.vocab.vocab.id2char[target_padded_chars[i, 0].item()] for i in range(1, target_padded_chars.shape[0]) if self.vocab.vocab.id2char[target_padded_chars[i,0].item()] != '<pad>' and self.vocab.vocab.id2char[target_padded_chars[i,0].item()] != '<eos>']))
            print("".join([self.vocab.vocab.id2char[torch.argmax(P, dim= 2)[i][0].item()] for i in range(0, P.shape[0]) if self.vocab.vocab.id2char[torch.argmax(P, dim= 2)[i][0].item()] != '<pad>' and self.vocab.vocab.id2char[torch.argmax(P, dim= 2)[i][0].item()] != '<eos>']))
        target_gold_chars_log_prob = torch.gather(P, index=target_padded_chars[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_char_masks[1:]
        scores = target_gold_chars_log_prob.sum()
        return scores


    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.
        @param source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b, max_word_length), where
                                        b = batch_size, src_len = maximum source sentence length. Note that
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                                hidden state and cell.
        """

        X = self.model_embeddings_source(source_padded)
        X_packed = pack_padded_sequence(X, source_lengths)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(X_packed)
        dec_init_state = (self.h_projection(torch.cat((last_hidden[0], last_hidden[1]), 1)), self.c_projection(torch.cat((last_cell[0], last_cell[1]), 1)))
        (enc_hiddens, _) = pad_packed_sequence(enc_hiddens)
        enc_hiddens = enc_hiddens.permute(1, 0, 2)
        return enc_hiddens, dec_init_state


    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
                dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.
        @param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                                     b = batch size, src_len = maximum source sentence length.
        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b, max_word_length), where
                                       tgt_len = maximum target sentence length, b = batch size.
        @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Chop of the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = target_padded.shape[1]
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []
        enc_hiddens_proj = self.att_projection(enc_hiddens)
        Y = self.model_embeddings_target(target_padded)
        thinking_y = torch.zeros(Y.shape[0], batch_size, self.embed_size, device = self.device)
        thinking_o_prev = torch.zeros(batch_size, self.encoder_hidden_size, device=self.device)
        thinking_hiddens_stacked = []
        for i in range(self.thinking_steps):
            thinking_hiddens = []
            for thinking_y_t in torch.split(thinking_y, 1):
                thinking_y_t = torch.squeeze(thinking_y_t, 0)
                thinking_ybar_t = torch.cat((thinking_y_t, thinking_o_prev), dim = 1)
                dec_state, o_t, e_t = self.step(thinking_ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
                thinking_hiddens.append(o_t)
                thinking_o_prev = o_t
            thinking_hiddens_stacked.append(torch.stack(thinking_hiddens))

        o_prev = thinking_o_prev
        for Y_t in torch.split(Y, 1):
            Y_t = torch.squeeze(Y_t, 0)
            Ybar_t = torch.cat((Y_t, o_prev), dim=1)
            dec_state, o_t, e_t = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            combined_outputs.append(o_t)
            o_prev = o_t
        combined_outputs = torch.stack((combined_outputs))
        thinking_hiddens_stacked = torch.stack(thinking_hiddens_stacked)
        return combined_outputs.to(self.device), thinking_hiddens_stacked.to(self.device)

    def step(self, Ybar_t: torch.Tensor,
            dec_state: Tuple[torch.Tensor, torch.Tensor],
            enc_hiddens: torch.Tensor,
            enc_hiddens_proj: torch.Tensor,
            enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:

        """ Compute one forward step of the LSTM decoder, including the attention computation.
        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length.
        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """

        dec_state = self.decoder(Ybar_t, dec_state)
        # dec_hidden.shape = [b x h] [5 x 3]
        # enc_hiddens_proj.shape = [b x src_length x h] [5 x 20 x 3]
        dec_hidden, dec_cell = dec_state
        e_t = torch.squeeze(torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_hidden,2)), 2)
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks, -float('inf'))

        alpha_t = F.softmax(e_t, dim = 1)
        a_t = torch.squeeze(torch.bmm(torch.unsqueeze(alpha_t, 1), enc_hiddens), 1) # (b x 2enc_h)
        U_t = torch.cat((a_t, dec_hidden), dim = 1) # (b x (2enc_h + h))
        V_t = self.combined_output_projection(U_t) # (b x enc_h)
        O_t = self.dropout(torch.tanh(V_t))

        combined_output = O_t
        return dec_state, combined_output, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size.
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.

        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.bool)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)


    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding answers
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

        src_sents_var = self.vocab.vocab.to_input_tensor_char([src_sent], self.device, isTarget=False)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.vocab.char2id['<eos>']

        hypotheses = [['<sos>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        #Thinking steps
        thinking_y = torch.zeros(5, 1, self.embed_size, device = self.device)
        thinking_o_prev = torch.zeros(1, self.encoder_hidden_size, device=self.device)
        thinking_hiddens_stacked = []
        thinking_attention_scores_stacked = []
        for i in range(self.thinking_steps):
            thinking_hiddens = []
            thinking_attention_scores = []
            for thinking_y_t in torch.split(thinking_y, 1):
                thinking_y_t = torch.squeeze(thinking_y_t, 0)
                thinking_ybar_t = torch.cat((thinking_y_t, thinking_o_prev), dim = 1)
                h_tm1, o_t, e_t = self.step(thinking_ybar_t, h_tm1, src_encodings, src_encodings_att_linear, enc_masks = None)
                thinking_hiddens.append(o_t)
                thinking_attention_scores.append(e_t)
                thinking_o_prev = o_t
            thinking_hiddens_stacked.append(torch.stack(thinking_hiddens))
            thinking_attention_scores_stacked.append(torch.stack(thinking_attention_scores))
        thinking_hiddens_stacked = torch.stack(thinking_hiddens_stacked)
        thinking_attention_scores_stacked = torch.stack(thinking_attention_scores_stacked)
        att_tm1 = thinking_o_prev
        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = self.vocab.vocab.to_input_tensor_char(list([hyp[-1]] for hyp in hypotheses), device=self.device, isTarget=False)
            y_t_embed = self.model_embeddings_target(y_tm1)
            y_t_embed = torch.squeeze(y_t_embed, dim=0)
            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, e_t  = self.step(x, h_tm1, exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.vocab.char2id)
            hyp_char_ids = top_cand_hyp_pos % len(self.vocab.vocab.char2id)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            decoderStatesForUNKsHere = []
            for prev_hyp_id, hyp_char_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_char_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_char_id = hyp_char_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_char = self.vocab.vocab.id2char[hyp_char_id]

                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_char]
                if hyp_char == '<eos>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
        return completed_hypotheses, thinking_hiddens_stacked.to(self.device), thinking_attention_scores_stacked.to(self.device)


    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.h_projection.weight.device

    @staticmethod
    def load(model_path: str, no_char_decoder=False):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = MathBaselineLSTMwAttention(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.embed_size,
                        hidden_size=self.hidden_size,
                        thinking_steps=self.thinking_steps,
                        dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)

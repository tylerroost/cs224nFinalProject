# -*- coding: utf-8 -*-

"""
CS224N Final Project
vocab.py: Vocabulary Generation
Tyler Roost <Roost@usc.edu>
Modified from CS224N Assignment 5 vocab.py made by:
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>

Modification allows for strictly character encoding and decoding

Usage:
    vocab.py VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

from docopt import docopt
import json
import torch
from typing import List
from utils import pad_sents_char

class VocabEntry(object):
    """ Vocabulary Entry, i.e. structure containing either
    src or tgt language terms.
    """
    def __init__(self, char2id=None):
        """ Init VocabEntry Instance.
        @param word2id (dict): dictionary mapping words 2 indices
        """
        # if word2id:
        #     self.word2id = word2id
        # else:
        #     self.word2id = dict()
        #     self.word2id['<pad>'] = 0   # Pad Token
        #     self.word2id['<s>'] = 1 # Start Token
        #     self.word2id['</s>'] = 2    # End Token
        #     self.word2id['<unk>'] = 3   # Unknown Token
        # self.unk_id = self.word2id['<unk>']
        # self.id2word = {v: k for k, v in self.word2id.items()}

        ## Additions to the A4 code:
        if char2id:
            self.char2id = char2id
        else:
            self.char_list = list("""ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} """)

            self.char2id = dict() # Converts characters to integers
            self.char2id['<pad>'] = 0
            self.char2id['<unk>'] = 1
            self.char2id['<sos>'] = 2
            self.char2id['<eos>'] = 3
            for c in self.char_list:
                self.char2id[c] = len(self.char2id)
        self.char_unk = self.char2id['<unk>']
        self.start_of_word = self.char2id['<sos>']
        self.end_of_word = self.char2id['<eos>']

        self.id2char = {v: k for k, v in self.char2id.items()} # Converts integers to characters
        ## End additions to the A4 code

    def __getitem__(self, c):
        """ Retrieve char's index. Return the index for the unk
        token if the char is out of vocabulary.
        @param c (str): char to look up.
        @returns index (int): index of word
        """
        return self.char2id.get(c, self.char_unk)

    def __contains__(self, c):
        """ Check if word is captured by VocabEntry.
        @param c (str): word to look up
        @returns contains (bool): whether word is contained
        """
        return char in self.char2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the VocabEntry.
        """
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return len(self.char2id)

    def __repr__(self):
        """ Representation of VocabEntry to be used
        when printing the object.
        """
        return 'Vocabulary[size=%d]' % len(self.char2id)

    # def id2word(self, wid):
    #     """ Return mapping of index to word.
    #     @param wid (int): word index
    #     @returns word (str): word corresponding to index
    #     """
    #     return self.id2word[wid]

    def add(self, c):
        """ Add char to VocabEntry, if it is previously unseen.
        @param c (str): char to add to VocabEntry
        @return index (int): index that the word has been assigned
        """
        if c not in self.char2id:
            cid = self.char2id[c] = len(self.char2id)
            self.id2word[cid] = c
            return cid
        else:
            return self.char2id[c]

    def sents2charindices(self, sents, isTarget):
        """ Convert list of sentences of words into list of list of list of character indices.
        @param sents (list[str]): sentence(s) in words
        @return word_ids (list[list[int]]): sentence(s) in indices
        """
        sents_word_ids = []
        for sent in sents:
            if isTarget:
                char_sent_idx = [self.start_of_word] + [self.char2id[char] for char in sent] + [self.end_of_word]
            else:
                char_sent_idx = [self.char2id[char] for char in sent]
            sents_word_ids.append(char_sent_idx)
        return sents_word_ids
        

    def to_input_tensor_char(self, sents: List[List[str]], device: torch.device, isTarget: bool) -> torch.Tensor:
        """ Convert list of sentences (chars) into tensor with necessary padding for
        shorter sentences.

        @param sents (List[str]): list of sentences (chars)
        @param device: device on which to load the tensor, i.e. CPU or GPU

        @returns sents_var: tensor of (max_sentence_length, batch_size)
        """

        char_ids = self.sents2charindices(sents, isTarget)
        chars_t = pad_sents_char(char_ids, self.char2id['<pad>'])
        chars_var = torch.tensor(chars_t, dtype=torch.long, device=device) # (batch_size, max_sentence_length)
        return torch.transpose(chars_var, 0, 1)


class Vocab(object):
    """ Vocab encapsulating src and target langauges.
    """
    def __init__(self, vocab: VocabEntry):
        """ Init Vocab.
        @param vocab (VocabEntry): VocabEntry for source and target language
        """
        self.vocab = vocab

    @staticmethod
    def build() -> 'Vocab':
        """ Build Vocabulary.
        @param src_sents (list[str]): Source sentences provided by read_corpus() function
        @param tgt_sents (list[str]): Target sentences provided by read_corpus() function
        @param vocab_size (int): Size of vocabulary for both source and target languages
        @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word.
        """

        print('initialize vocabulary ..')
        vocab = VocabEntry()

        return Vocab(vocab)

    def save(self, file_path):
        """ Save Vocab to file as JSON dump.
        @param file_path (str): file path to vocab file
        """
        json.dump(dict(char2id = self.vocab.char2id), open(file_path, 'w'), indent=2)

    @staticmethod
    def load(file_path):
        """ Load vocabulary from JSON dump.
        @param file_path (str): file path to vocab file
        @returns Vocab object loaded from JSON dump
        """
        entry = json.load(open(file_path, 'r'))
        char2id = entry['char2id']

        return Vocab(VocabEntry(char2id))

    def __repr__(self):
        """ Representation of Vocab to be used
        when printing the object.
        """
        return 'Vocab(vocab %d chars)' % (len(self.vocab.char2id))


if __name__ == '__main__':
    args = docopt(__doc__)

    vocab = Vocab.build()
    print('generated vocabulary')

    vocab.save(args['VOCAB_FILE'])
    print('vocabulary saved to %s' % args['VOCAB_FILE'])

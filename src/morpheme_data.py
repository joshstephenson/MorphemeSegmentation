# -*- coding: utf-8 -*-
import pandas as pd
import os.path as pt
import os
import torch
import torch.nn as nn
import torchtext
import numpy as np
from datasets import Dataset
from functools import reduce
from typing import List
import yaml


path = pt.join(pt.dirname(pt.abspath(__file__)), 'config/config.yaml')
with open(path, 'r') as f:
    config = yaml.load(f, Loader = yaml.FullLoader)

UNKNOWN_TOKEN = config['special_tokens']['unknown_token']
PAD_TOKEN = config['special_tokens']['pad_token']
SOS_TOKEN = config['special_tokens']['sos_token']
EOS_TOKEN = config['special_tokens']['eos_token']

class MorphemeDataLoader():
    class Wrapper():
        def __init__(self, words: List[str], morph_str: List[str], morphs: List[str]):
            self.words = words #np.reshape(words, (-1,1)).tolist()
            self.morph_str = morph_str
            self.morphs = morphs
            # print(self.words[:10])
            # print(self.morphs[:10])
            # exit(0)

            self.process_vocab()
            # self.chars = set("".join(X))
            # self.chars |= set("".join(y_orig))
            # self.chars |= set(['<PAD>'])
            # self.n_chars = len(self.chars)
            # self.X, self.y = [], []
            # longest = max(max(X, key=len), max(y_orig, key=len), key = len)
            #
            # for word in X:
            #     padding = [PAD] * (len(longest) - len(word))
            #     self.X.append(list(word)+padding)
            # for word in y:
            #     padding = [PAD] * (len(longest) - len(word))
            #     self.y.append(list(word) + padding)
            # self.char_to_ix = {c: i for i, c in enumerate(self.chars)}
            # TODO: Refactor this
            # self.morphemes = set()
            # for morpheme in self.y:
            #     for morph in morpheme.split():
            #         self.morphemes.add(morph)
            # self.n_morphemes = len(self.morphemes)
            # self.morpheme_to_ix = {m: i for i, m in enumerate(self.morphemes)}
            # print(self.morpheme_to_ix)

        def process_vocab(self):
            special_tokens = [
                UNKNOWN_TOKEN,
                PAD_TOKEN,
                SOS_TOKEN,
                EOS_TOKEN
            ]
            self.word_vocab = torchtext.vocab.build_vocab_from_iterator(
                self.words,
                min_freq=config['preprocessing']['min_freq'],
                specials=special_tokens,
            )
            self.morph_vocab = torchtext.vocab.build_vocab_from_iterator(
                self.morphs,
                min_freq=config['preprocessing']['min_freq'],
                specials=special_tokens,
            )

            assert self.word_vocab[UNKNOWN_TOKEN] == self.morph_vocab[UNKNOWN_TOKEN]
            assert self.word_vocab[PAD_TOKEN] == self.morph_vocab[PAD_TOKEN]

            self.unk_index = self.word_vocab[UNKNOWN_TOKEN]
            self.pad_index = self.word_vocab[PAD_TOKEN]
            self.word_vocab.set_default_index(self.unk_index)
            self.morph_vocab.set_default_index(self.unk_index)

            self.word_len = len(self.word_vocab)
            self.morph_len = len(self.morph_vocab)

            # print(word_vocab["tanításokért"])
            # print(morph_vocab["tanít"])
            # print(morph_vocab["@@ás"])
            # print(f'word len: {len(word_vocab)}, morph len: {len(morph_vocab)}')
            # print(word_vocab.get_itos()[:100])

            # print("here")
            ids = Dataset.from_dict({"words": [self.word_vocab.lookup_indices(token) for token in self.words], "morphs": [self.morph_vocab.lookup_indices(token) for token in self.morphs]})
            ids = ids.with_format(type = 'torch', columns = ['words', 'morphs'], output_all_columns = True)

            # print(self.ids['words'][:10])

            def get_collate_fn(pad_index):
                def collate_fn(batch):
                    batch_word_ids = [example["words"] for example in batch]
                    batch_morph_ids = [example["morphs"] for example in batch]
                    batch_word_ids = nn.utils.rnn.pad_sequence(batch_word_ids, padding_value = pad_index)
                    batch_morph_ids = nn.utils.rnn.pad_sequence(batch_morph_ids, padding_value = pad_index)
                    batch = {
                        "word_ids": batch_word_ids,
                        "morph_ids": batch_morph_ids,
                    }
                    return batch

                return collate_fn

            def get_data_loader(dataset, pad_index, **kwargs):
                collate_fn = get_collate_fn(pad_index)
                data_loader = torch.utils.data.DataLoader(
                    dataset = dataset,
                    batch_size = kwargs['batch_size'],
                    collate_fn = collate_fn,
                    shuffle = kwargs['shuffle'],
                )
                return data_loader

            self.loader = get_data_loader(ids, self.pad_index, **config['preprocessing'])

        # def word_to_indices(self, chars):
        #     lookup = [self.char_to_ix[char] for char in chars]
        #     return lookup

    def __init__(self, lang = 'hun'):
        self.lang = lang
        def get_path(dset):
            return pt.join(pt.dirname(pt.abspath(__file__)), f'data/{lang}.word.{dset}.tsv')

        def get_data(path):
            if os.path.exists(path):
                # words = []
                # morph_strings = []
                # morphs = []
                # with open(path, 'r', encoding='utf') as f:
                #     for line in f.readlines():
                #         parts = line.split('\t')
                #         words.append(parts[0])
                #         morph_strings.append(parts[1])
                #         morphs = morphs + parts[1].split()
                table = pd.read_table(path, sep='\t', header=None)
                table.columns = ['words', 'morph_str', 'identifier']
                words = [[SOS_TOKEN] + list(w) + [EOS_TOKEN] for w in table['words'].to_numpy()]
                morphs = [[SOS_TOKEN] + list(m) + [EOS_TOKEN] for m in table['morph_str'].to_numpy()]
                # morphs = " ".join(table['morph_str']).split()
                return self.Wrapper(words, table['morph_str'].to_numpy(), morphs)
            else:
                logging.critical(f'No such file: {path}')
                exit(1)

        self.train = get_data(get_path('train'))
        self.test = get_data(get_path('test.gold'))
        self.validation = get_data(get_path('dev'))

# -*- coding: utf-8 -*-
import pandas as pd
import os.path as pt
import os
import torch
import torch.nn as nn
import torchtext
from datasets import Dataset
from typing import List
import yaml
import logging


path = pt.join(pt.dirname(pt.abspath(__file__)), 'config/config.yaml')
with open(path, 'r') as f:
    config = yaml.load(f, Loader = yaml.FullLoader)

UNKNOWN_TOKEN = config['special_tokens']['unknown_token']
PAD_TOKEN = config['special_tokens']['pad_token']
SOS_TOKEN = config['special_tokens']['sos_token']
EOS_TOKEN = config['special_tokens']['eos_token']

class MorphemeDataLoader():
    class Wrapper():
        def __init__(self, words: List[str], morphs: List[str], word_classes: List[int]):
            self.words = words
            self.morphs = morphs
            self.word_classes = word_classes
            self.process_vocab()

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

            ids = Dataset.from_dict({"words": [self.word_vocab.lookup_indices(token) for token in self.words], "morphs": [self.morph_vocab.lookup_indices(token) for token in self.morphs]})
            ids = ids.with_format(type = 'torch', columns = ['words', 'morphs'], output_all_columns = True)

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

    def __init__(self, lang = 'hun'):
        self.lang = lang
        def get_path(dset):
            return pt.join(pt.dirname(pt.abspath(__file__)), f'data/{lang}.word.{dset}.tsv')

        def get_data(path):
            if os.path.exists(path):
                table = pd.read_table(path, sep='\t', header=None)
                table.columns = ['words', 'morph_str', 'identifier']

                # Replace ' @@' morpheme marker with '@' which will make things easier
                table['morph_str'] = table['morph_str'].str.replace(' @@', '@')
                words = [[SOS_TOKEN] + list(w) + [EOS_TOKEN] for w in table['words'].to_numpy()]
                morphs = [[SOS_TOKEN] + list(m) + [EOS_TOKEN] for m in table['morph_str'].to_numpy()]
                word_classes = table['identifier'].to_numpy()
                return self.Wrapper(words, morphs, word_classes)
            else:
                logging.critical(f'No such file: {path}')

        self.train = get_data(get_path('train'))
        self.test = get_data(get_path('test.gold'))
        self.validation = get_data(get_path('dev'))

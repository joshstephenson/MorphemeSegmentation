# -*- coding: utf-8 -*-
import pandas as pd
import torch
import torch.nn as nn
import os
import os.path as pt
import torchtext
from datasets import Dataset
import logging

MORPH_SEPARATOR = ' |'

class MorphemeDataLoader:
    def __init__(self, config):
        SOS_TOKEN = config['special_tokens']['sos_token']
        EOS_TOKEN = config['special_tokens']['eos_token']

        self.lang = config['language']

        def get_path(dset):
            data_dir = config['data_dir']
            return pt.join(pt.dirname(pt.abspath(__file__)), data_dir, f'{self.lang}.word.{dset}.tsv')

        def get_data(path):
            if os.path.exists(path):
                table = pd.read_table(path, sep='\t', header=None)
                table.columns = ['words', 'morph_str', 'identifier']

                # print("Trimming to first 10000 lines")
                # table = table[:1000]

                # Replace ' @@' morpheme marker with '@' which will make things easier
                # table['morph_str'] = table['morph_str'].str.replace(' @@', MORPH_SEPARATOR)
                words = [[SOS_TOKEN] + list(w) + [EOS_TOKEN] for w in table['words'].to_numpy()]
                morphs = [[SOS_TOKEN] + list(m) + [EOS_TOKEN] for m in table['morph_str'].to_numpy()]
                # word_classes = table['identifier'].to_numpy()
                return self.Wrapper(words, morphs, config)
            else:
                logging.critical(f'No such file: {path}')

        self.train = get_data(get_path('train'))
        self.test = get_data(get_path('test.gold'))
        self.validation = get_data(get_path('dev'))

    class Wrapper:
        def __init__(self, words, morphs, config):
            self.words = words
            self.morphs = morphs
            # self.word_classes = word_classes

            special_tokens = config['special_tokens'].values()
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

            UNKNOWN_TOKEN = config['special_tokens']['unknown_token']
            PAD_TOKEN = config['special_tokens']['pad_token']

            assert self.word_vocab[UNKNOWN_TOKEN] == self.morph_vocab[UNKNOWN_TOKEN]
            assert self.word_vocab[PAD_TOKEN] == self.morph_vocab[PAD_TOKEN]

            self.unk_index = self.word_vocab[UNKNOWN_TOKEN]
            self.pad_index = self.word_vocab[PAD_TOKEN]
            self.word_vocab.set_default_index(self.unk_index)
            self.morph_vocab.set_default_index(self.unk_index)

            self.word_len = len(self.word_vocab)
            self.morph_len = len(self.morph_vocab)

            ids = Dataset.from_dict({"word_ids": [self.word_vocab.lookup_indices(word) for word in self.words],
                                     "morph_ids": [self.morph_vocab.lookup_indices(morph_set) for morph_set in
                                                   self.morphs]})
            ids = ids.with_format(type='torch', columns=['word_ids', 'morph_ids'], output_all_columns=True)

            def get_collate_fn(pad_index):
                def collate_fn(batch):
                    batch_word_ids = [example["word_ids"] for example in batch]
                    batch_morph_ids = [example["morph_ids"] for example in batch]
                    batch_word_ids = nn.utils.rnn.pad_sequence(batch_word_ids, padding_value=pad_index)
                    batch_morph_ids = nn.utils.rnn.pad_sequence(batch_morph_ids, padding_value=pad_index)
                    batch = {
                        "word_ids": batch_word_ids,
                        "morph_ids": batch_morph_ids,
                    }
                    return batch

                return collate_fn

            def get_data_loader(dataset, pad_index, **kwargs):
                collate_fn = get_collate_fn(pad_index)
                data_loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=kwargs['batch_size'],
                    collate_fn=collate_fn,
                    shuffle=kwargs['shuffle'],
                )
                return data_loader

            self.loader = get_data_loader(ids, self.pad_index, **config['preprocessing'])

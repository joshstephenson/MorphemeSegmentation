#!/usr/bin/env python
from transformers import BertTokenizer
import argparse
import sys

MODELS = {'def': 'bert-base-multilingual-cased', 'eng': 'bert-base-uncased', 'mon': 'bert-base-uncased'}
def get_tokenizer(file):
    """
    Use a different model for English and Mongolian languages
    :param file: a full path to a file
    :return: a BertTokenizer object from pretrained model
    """
    file = file
    lang = file.split('/')[-1].split('.')[0]
    if lang in MODELS.keys():
        model = MODELS[lang]
    else:
        model = MODELS['def']
    return BertTokenizer.from_pretrained(model)

def main(args):
    tokenizer = get_tokenizer(args.file)
    with open(args.file, 'r') as infile:
        for line_in in infile.readlines():
            columns = line_in.split('\t')
            word = columns[0].replace('\n', '')
            guess = " ".join(tokenizer.tokenize(word)).replace('##', '@@')
            line_out = f'{word}\t{guess}'
            sys.stdout.write(line_out + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenize Sigmorphon files with Bert Tokenizer')
    parser.add_argument("file", type=str)
    opt = parser.parse_args()
    main(opt)
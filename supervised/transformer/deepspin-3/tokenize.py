#!/usr/bin/env python

"""
This is the first part of the shared task pipeline. The input is raw shared
task data stored as tsv files. The output is {train,dev,test}.{src,tgt} files
containing that data, but with tokens separated by whitespace. The supported
tokenization techniques are

1) char: each character is a separate token. Spaces in the raw text are
         replaced by underscores.
2) spm: either learn a new sentencepiece model on the task data, or apply an
        external one.

Regardless of tokenization strategy, this script replaces " @@" with "|" in the
target data.
"""

import argparse
from collections import Counter
from functools import partial
from itertools import chain
import os
import shutil

import sentencepiece as spm


def read_tsv(path, category=False):
    # tsv without header
    col_names = ["surface", "segment"]
    if category:
        col_names.append("category")
    data = {name: [] for name in col_names}
    with open(path, encoding='utf-8') as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            for name, field in zip(col_names, fields):
                if name == "segment":
                    # replace @@ with | as morpheme boundary character
                    # (without destroying whitespace, because it might be needed
                    # for training an spm model)
                    field = field.replace('@@', '|')
                    # field = field.replace(' ', '|')
                data[name].append(field)
    return data


# ok, that's part of it but not the whole thing
def character_tokenize(string):
    # remove spaces before morpheme boundaries. turn other spaces into
    # underscores.
    string = string.replace(' |', '|')
    string = string.replace(" ", "_")  # maybe only on word level
    return list(string.strip())


def write_tokenized_corpus(path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write(" ".join(line) + "\n")


def build_spm_tokenizer(new_prefix, train_iter, vocab_size):
    spm.SentencePieceTrainer.train(
        sentence_iterator=train_iter,
        model_prefix=new_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0
    )
    # spm_model_path = new_prefix + ".model"
    # processor = spm.SentencePieceProcessor(model_file=spm_model_path)

    # return processor


def copy_spm_model(existing_path, new_path):
    # but what about the vocab?
    for suffix in [".model", ".vocab"]:
        try:
            shutil.copy(existing_path + suffix, new_path + suffix)
        except shutil.SameFileError:
            pass


def write_character_vocab(path, char_seqs):
    vocab = Counter()
    for seq in char_seqs:
        vocab.update(seq)
    with open(path, 'w') as f:
        for sym in ["<unk>", "<s>", "</s>"]:
            f.write("\t".join([sym, '1']) + "\n")
        for char, count in vocab.most_common():
            f.write("\t".join([char, str(count)]) + "\n")


def prepare_spm(existing_spm_path, spm_path, train_iter, vocab_size, model_type="unigram"):
    """
    If existing_spm_path is not None, train_iter and vocab_size are ignored

    This function either copies an existing sentencepiece model to the spm_path
    location (inside the tokenized data directory) or it makes a new
    sentencepiece model and puts it there.

    It returns an spm.SentencePieceProcessor instance, whose encode() method
    can be used as a tokenizer
    """
    # spm path should not have the suffix (.model or .vocab)

    # src_spm_path = os.path.join(args.out_dir, args.new_spm_prefix + ".src")
    if existing_spm_path is not None:
        # preexisting model; copy it to the right directory
        copy_spm_model(existing_spm_path, spm_path)
    else:
        # spm model needs to be trained
        spm.SentencePieceTrainer.train(
            sentence_iterator=train_iter,
            model_prefix=spm_path,
            vocab_size=vocab_size,
            character_coverage=1.0,
            model_type=model_type
        )

    # now: the spm model has been built. Use it to make a processor
    processor = spm.SentencePieceProcessor(model_file=spm_path + ".model")
    # todo: character coverage, alpha hyperparameter
    return processor


def main(args):
    # make the output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # read data
    data = read_tsv(args.corpus)
    src = data["surface"]
    tgt = data["segment"]

    # another thing to think about: you should not create an spm vocab if
    # split != train

    if args.src_tok_type == "spm":
        assert args.split == "train" or args.existing_src_spm is not None
        src_spm_path = os.path.join(args.out_dir, "src")
        src_processor = prepare_spm(
            args.existing_src_spm,
            src_spm_path,
            chain(src, tgt) if args.shared_data else iter(src),
            args.vocab_size,
            model_type="bpe" if args.bpe else "unigram"
        )
        # todo: character coverage, alpha hyperparameter
        src_tokenizer = partial(
            src_processor.encode,
            out_type=str,
            enable_sampling=args.sample,
            alpha=args.alpha
        )
    else:
        src_tokenizer = character_tokenize

    if args.tgt_tok_type == "spm":
        assert args.split == "train" or args.existing_tgt_spm is not None
        tgt_spm_path = os.path.join(args.out_dir, "tgt")
        tgt_processor = prepare_spm(
            args.existing_tgt_spm,
            tgt_spm_path,
            chain(src, tgt) if args.shared_data else iter(tgt),
            args.vocab_size,
            model_type="bpe" if args.bpe else "unigram"
        )
        # todo: character coverage, alpha hyperparameter
        tgt_tokenizer = partial(
            tgt_processor.encode,
            out_type=str,
            enable_sampling=args.sample,
            alpha=args.alpha
        )
    else:
        tgt_tokenizer = character_tokenize

    # write to the correct directory
    # I would like to simplify this by separating the tokenization from the
    # writing.
    src_toks = [src_tokenizer(s) for s in src * args.n_samples]
    write_tokenized_corpus(
        os.path.join(args.out_dir, args.split + ".src"), src_toks
    )
    tgt_toks = [tgt_tokenizer(t) for t in tgt * args.n_samples]
    write_tokenized_corpus(
        os.path.join(args.out_dir, args.split + ".tgt"), tgt_toks,
    )

    if args.split == "train":
        if args.src_tok_type == "char":
            write_character_vocab(
                os.path.join(args.out_dir, "src.vocab"),
                chain(src_toks, tgt_toks) if args.shared_data else src_toks
            )
        if args.tgt_tok_type == "char":
            write_character_vocab(
                os.path.join(args.out_dir, "tgt.vocab"),
                chain(src_toks, tgt_toks) if args.shared_data else tgt_toks
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', help="tsv file from which to build tokenized data")
    parser.add_argument("--src-tok-type", "-s", default="char", choices=["char", "spm"])
    parser.add_argument("--tgt-tok-type", "-t", default="char", choices=["char", "spm"])
    parser.add_argument('--existing-src-spm', default=None,
                        help="Path to existing sentencepiece model")
    parser.add_argument('--existing-tgt-spm', default=None,
                        help="Path to existing sentencepiece model")
    parser.add_argument("--vocab-size", "-v", default=1000, type=int,
                        help="Vocab size if training a new sentencepiece model")
    parser.add_argument("--out-dir", "-o", required=True)
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"])
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--shared-data", action="store_true")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Number of times to repeat the corpus (useful for subword regularization)")
    parser.add_argument("--bpe", action="store_true", help="use BPE instead of ULM")
    opt = parser.parse_args()
    main(opt)

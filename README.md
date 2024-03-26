# Morpheme Segmentation with LSTM RNN
Morpheme segmentation is the process of separating words into their fundamental units of meaning. For example:

- **foundationalism** &rarr; **found**+**ation**+**al**+**ism**

This project is an attempt at reproduction of [DeepSPIN-2](https://aclanthology.org/2022.sigmorphon-1.14/), a recurrent neural network (LSTM) model for morpheme segmentation which was the 2nd place winner in the [2022 Sigmorphon competition](https://github.com/sigmorphon/2022SegmentationST/), designed by the same team that won 1st place. To be clear: this is not DeepSPIN-2, but an attempt at recreating it based on the information shared in the paper.

### The Data
Here is a sample of the training data for Hungarian:
```
tanításokért	tanít @@ás @@ok @@ért	110
Algériánál	Algéria @@nál	100
metélőhagymába	metélő @@hagyma @@ba	101
fülésztől	fül @@ész @@től	110
```
After training, the model is expected to be able to receive just the first column (the untokenized word) and be able to separate it into morphemes, with the ` @@` morpheme separator. The final column, which can be used for training has 3 bits that represent the types of morphology (inflection, derivation, & compounding).

### Setup
- Make sure to clone this with `--recuse-submodules` to ensure you get the data from the competition.
- After creating a virtual environment with `python -m venv <name>`, you can install necessary python libraries with `pip install -r requirements.txt` from the root directory of this repository.

### Organization
This repository is organized as such:
```
baseline/
baseline/bert
baseline/morfessor
supervised/
supervised/lstm/
supervised/lstm/deepspin-2
supervised/lstm/mine
supervised/transformer
supervised/transformer/deepspin-3
```

1. The `baseline` directory has 2 scripts for generating baseline segmentations. One uses a pretrained BertTokenizer (`baseline/bert`) and the other uses Morfessor 2.0 (`baseline/morfessor`), an unsupervised utility that is not pretrained.
2. The `supervised` directory has two subdirectories: one for an `LSTM` implementation and one for a `Transformer` based implementation. Within `supervised/lstm/deepspin-2` you can find a reproduction of DeepSpin-2, written with fairseq by Ben Peters, as well as a more or less from scratch implementation written by me, with the hopes of recreating the results of DeepSpin-2 without fairseq. Within `supervised/transformer/deepspin-3` is another fairseq implementation written by Ben Peters that uses (you guessed it) a Transformer architecture.

### Configuration
All configuration variables including the target language and hyperparameters are set in `config/config.yaml`. This project only trains one language at a time, based on the 'language' variable in config.yaml.

Language options include:
| Language | Language code |
|----------|---------------|
|English|eng|
|French|fra|
|Hungarian|hun|
|Italian|ita|
|Latin|lat|
|Mongolian|mon|
|Russian|rus|
|Spanish|spa|

### Training
Please refer to README's within each subdirectory for training and evaluation.




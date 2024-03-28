# Morpheme Segmentation with LSTM and Transformers
Morpheme segmentation is the process of separating words into their fundamental units of meaning. For example:

- **foundationalism** &rarr; **found**+**ation**+**al**+**ism**

This project is an attempt at reproducing the 2nd and 1st place systems from the [2022 Sigmorphon competition](https://github.com/sigmorphon/2022SegmentationST/) in word segmentation. These systems, from Ben Peters and Andre Martins are [DeepSPIN-2](https://aclanthology.org/2022.sigmorphon-1.14/), a recurrent neural network (LSTM) model and [DeepSPIN-3](https://aclanthology.org/2022.sigmorphon-1.14/) a transformer based model.

### Organization
This repository is organized as such:
```
baseline/
baseline/bert       # simple BertTokenizer generator and evaluator
baseline/morfessor  # simple Morfessor 2.0 trainer, generator, and evaluator
deepspin-2          # streamlined implementation of DeepSpin-2, written by (Ben Peters)[https://github.com/bpopeters], using fairseq and an LSTM architecture
deepspin-3          # streamlined implementation of DeepSpin-3, written by (Ben Peters)[https://github.com/bpopeters], using fairseq and a transformer architecture
lstm                # an LSTM architecture not built on fairseq (for academic purposes)
```

1. The `baseline` directory has 2 scripts for generating baseline segmentations. One uses a pretrained BertTokenizer (`baseline/bert`) and the other uses Morfessor 2.0 (`baseline/morfessor`), an unsupervised utility that is not pretrained.
2. The `supervised` directory has two subdirectories: one for an `LSTM` implementation and one for a `Transformer` based implementation. Within `supervised/lstm/deepspin-2` you can find a reproduction of DeepSpin-2, as well as a more or less from scratch implementation written by me, as an academic exercise. Within `deepspin-3` is another fairseq implementation that uses a transformer model.

In the case of DeepSPIN-2 and DeepSPIN-3, the original implementations were written by (Ben Peters)[https://github.com/bpopeters], but the scripts in this repository streamline their usage within a single shell script.

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
Please refer to README's within each subdirectory for training and evaluation of each individual project.




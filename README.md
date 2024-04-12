# Morpheme Segmentation with LSTM and Transformers
Morpheme segmentation is the process of separating words into their fundamental units of meaning. For example:

- **foundationalism** &rarr; **found**+**ation**+**al**+**ism**

This project is a reproduction of the 2nd and 1st place systems from the [2022 Sigmorphon competition](https://github.com/sigmorphon/2022SegmentationST/) in word segmentation. These systems, from [Ben Peters](https://github.com/bpopeters) and [Andre Martins](https://github.com/andre-martins), are [DeepSPIN-2](https://aclanthology.org/2022.sigmorphon-1.14/), a recurrent neural network (LSTM) model with character-level tokenization, and [DeepSPIN-3](https://aclanthology.org/2022.sigmorphon-1.14/) a transformer based model that uses an [entmax](https://github.com/deep-spin/entmax) loss function and [ULM subword tokenization](https://github.com/google/sentencepiece).

### Organization
This repository is organized as such:
```
baseline/
baseline/bert       # simple BertTokenizer generator and evaluator
baseline/morfessor  # simple Morfessor 2.0 trainer, generator, and evaluator
deepspin            # flexible implementation of DeepSpin-2 and DeepSpin-3 with fairseq and an LSTM architecture as outlined in the paper above.
yoyodyne            # A basic implementation using yoyodyne - https://github.com/CUNY-CL/yoyodyne
lstm                # an LSTM work-in-progress architecture built with basic PyTorch (for academic purposes).
```

1. The `baseline` directory has 2 scripts for generating baseline segmentations. One uses a pretrained BertTokenizer (`baseline/bert`) and the other uses Morfessor 2.0 (`baseline/morfessor`), an unsupervised utility that is not pretrained.

In the case of DeepSPIN-2 and DeepSPIN-3, the original implementations were written by [Ben Peters](https://github.com/bpopeters), but the scripts in this repository streamline their usage and decouple tokenization from each. This enables exploring of a transformer architecture with character-level encoding *or* an LSTM architecture with subword tokenization. This is helpful to really determine whether subword tokenization is a crucial ingredient in the high performance of DeepSPIN-3. Spoiler alert: it only accounts for 0.2% of the F-score.

### The Data
Here is a sample of the training data for Hungarian:
```
tanításokért	tanít @@ás @@ok @@ért	110
Algériánál	Algéria @@nál	100
metélőhagymába	metélő @@hagyma @@ba	101
fülésztől	fül @@ész @@től	110
```
After training, the model is expected to be able to receive just the first column (the untokenized word) and be able to separate it into morphemes, with the ` @@` morpheme separator. The final column, which can be used for training has 3 bits that represent the types of morphology (inflection, derivation, & compounding). This is currently unused.

### Setup
- Make sure to clone this with `--recurse-submodules` to ensure you get the data from the competition.
- After creating a virtual environment with `python -m venv <name>` or `conda create...`, you can install necessary python libraries with `pip install -r requirements.txt` from the root directory of this repository.

### Training
Please refer to README's within each subdirectory for training and evaluation for each individual architecture.




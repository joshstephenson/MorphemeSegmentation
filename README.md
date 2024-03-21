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
- Use `python train.py` to train the model. It will use the corresponding language training data found in `2022SegmentationST/data`. The trained model will be placed in the `output/` directory and the filename will contain all the hyperparameters found in config.yaml. For example:
```
hun-embeddings_256-hidden_1024-n_layers_2-dropout_0.2-tfr_0.5-lr_0.001-clip_1.0-by_char.pt
```

### Evaluation
- Use `python generate.py` in the `src` directory of this repository to write the predictions to a TRV file in the `output/` directory. You can also use `python predict.py` to print predictions to STDOUT for debugging.
- The generate script will also call the evaluation script from `2022SegmentationST` and the output will look something like this:

Output:
```
category: all
distance	2.50
f_measure	54.13
precision	53.94
recall	54.32
```




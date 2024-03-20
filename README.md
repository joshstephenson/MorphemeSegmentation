# Purpose
This project is an attempt at reproduction of [DeepSPIN-2](https://aclanthology.org/2022.sigmorphon-1.14/), a recurrent neural network (LSTM) model for morpheme segmentation.

This was the 2nd place winner in the [2022 Sigmorphon competition](https://github.com/sigmorphon/2022SegmentationST/).

To be clear: this is not DeepSPIN-2, but an attempt at recreating it based on the information shared in the paper.

##### A note about cloning
Make sure to clone this with `--recuse-submodules` to ensure you get the data from the competition.

### Configuration
All configuration variables including the target language and hyperparameters are set in `config/config.yaml`. This project only trains one language at a time, based on the 'language' variable in config.yaml.

Language options include:
| Language | Language code |
|----------|---------------|
| English | eng |
|French|fra|
|Hungarian|hun|
|Italian|ita|
|Latin|lat|
|Mongolian|mon|
|Russian|rus|
|Spanish|spa|

### Training
Use `python train.py` to train the model. It will use the corresponding language training data found in `2022SegmentationST/data`. The trained model will be placed in the `model/` directory.

### Evaluation
First, use `python generate.py` in the `src` directory of this repository to write the predictions to a TRV file in the `model/` directory.

You can also use `python predict.py` to print predictions to STDOUT, for debugging.

Next, use the `evaluation/evaluate.py` script from the competition's repository (`2022Segmentation/evaluation/`) to calculate F1 and other stats for the model. The following example is for Hungarian with 512 hidden dimensions (again, based on the configuration variables in `config/config.yaml`).

```
python 2022SegmentationST/evaluation/evaluate.py --gold 2022SegmentationST/data/hun.word.test.gold.tsv --guess src/model/hun-512-by_char-test.tsv
```

Output:
```
category: all
distance	2.83
f_measure	48.04
precision	48.35
recall	47.74
```




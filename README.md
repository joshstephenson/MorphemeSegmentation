# Purpose
This project is an attempt at reproduction of [DeepSPIN-2](https://aclanthology.org/2022.sigmorphon-1.14/), a recurrent neural network (LSTM) model for morpheme segmentation.

This was the 2nd place winner in the [2022 Sigmorphon competition](https://github.com/sigmorphon/2022SegmentationST/).

To be clear: this is not DeepSPIN-2, but an attempt at recreating it based on the information shared in the paper.

##### A note about cloning
Make sure to clone this with `--recuse-submodules` to ensure you get the data from the competition.

### Setup
After creating a virtual environment with `python -m venv <name>`, you can install necessary python libraries with `pip install -r requirements.txt` from the root directory of this repository.

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
Use `python train.py` to train the model. It will use the corresponding language training data found in `2022SegmentationST/data`. The trained model will be placed in the `output/` directory and the filename will contain all the hyperparameters found in config.yaml. For example:
```
hun-embeddings_256-hidden_1024-n_layers_2-dropout_0.2-tfr_0.5-lr_0.001-clip_1.0-by_char.pt
```

### Evaluation
First, use `python generate.py` in the `src` directory of this repository to write the predictions to a TRV file in the `output/` directory.

You can also use `python predict.py` to print predictions to STDOUT for debugging.

Next, use the `evaluation/evaluate.py` script from the competition's repository (`2022Segmentation/evaluation/`) to calculate F1 and other stats for the model. The following example is for Hungarian.

```
python 2022SegmentationST/evaluation/evaluate.py --gold 2022SegmentationST/data/hun.word.test.gold.tsv --guess hun-embeddings_256-hidden_1024-n_layers_2-dropout_0.2-tfr_0.5-lr_0.001-clip_1.0-by_char.pt
```

Output:
```
category: all
distance	2.50
f_measure	54.13
precision	53.94
recall	54.32
```




import logging

from predict import segment_word, segment_word2
from helpers import postprocess
from morpheme_data import MorphemeDataLoader
from models import *
from helpers import *
from tqdm import tqdm
import subprocess

config = Config()
logger = get_logger()
logger.setLevel(logging.WARN)

SOS_TOKEN = config['special_tokens']['sos_token']
EOS_TOKEN = config['special_tokens']['eos_token']

def main():
    config = Config()
    data = MorphemeDataLoader(config)
    device = config.device()  # Look for Metal GPU device (for Silicon Macs) and default to CUDA (for hosted GPU service)
    model = Seq2Seq(data.train.word_len, data.train.morph_len, device).to(device)
    model.load_from_file(config.model_file)
    logger.info(model)

    dataset = data.test

    def write_predictions(dataset, use_heuristic=False):
        pred_file = project_file(config, config['predictions_ext'],
                                 config['model_suffix'] + '-test' + '-heuristic' if use_heuristic else '')
        assert pred_file is not None
        description = 'Writing predictions with heuristic...' if use_heuristic else 'Writing predictions...'
        with open(pred_file, 'w') as f:
            for word, morphs in tqdm(dataset[:], total=len(data.test.words), desc=description):
                try:
                    pred = segment_word2(word, model, dataset, device) if use_heuristic else segment_word(word, model,
                                                                                                      dataset, device)
                    logger.warn(f"Unable to segment word: {word}")
                except Exception as _:
                    pred = "".join(word)
                pred = postprocess(pred)
                line = "".join(word[1:-1]) + '\t' + pred
                # logger.info(line + " (" + "".join(morphs[1:-1]) + ")")
                f.write(line + "\n")
        logger.info(f"Predictions written to: {pred_file}")

    def test_predictions(use_heuristic=False):
        pred_file = project_file(config, config['predictions_ext'],
                                 config['model_suffix'] + '-test' + '-heuristic' if use_heuristic else '')
        lang = config['language']
        command = ["python", f"../2022SegmentationST/evaluation/evaluate.py", "--gold", f"../2022SegmentationST/data/{lang}.word.test.gold.tsv", "--guess", pred_file]
        print(pred_file)
        output = subprocess.run(command)
        print(output)

    write_predictions(dataset)
    test_predictions()
    # write_predictions(dataset, use_heuristic=True)
    # test_predictions(use_heuristic=True)


if __name__ == "__main__":
    main()

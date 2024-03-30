import logging

from predict import segment_word
from morpheme_data import MorphemeDataLoader
from models import *
from helpers import *
from tqdm import tqdm
import subprocess
import argparse

config = Config()
logger = get_logger()
logger.setLevel(logging.WARN)

SOS_TOKEN = config['special_tokens']['sos_token']
EOS_TOKEN = config['special_tokens']['eos_token']

def main(file, cpu = False):
    config = Config()
    data = MorphemeDataLoader(config)
    device = torch.device('cpu') if cpu else config.device()
    model = Seq2Seq(data.train.word_len, data.train.morph_len, device).to(device)

    model.load_from_file(file)
    logger.info(model)

    dataset = data.test

    def write_predictions(dataset, pred_file):
        assert pred_file is not None
        description = 'Writing predictions...'
        with open(pred_file, 'w') as f:
            for word, morphs in tqdm(dataset[:], total=len(data.test.words), desc=description):
                pred = segment_word(word, model, dataset, device)
                pred = postprocess(pred)
                line = "".join(word[1:-1]) + '\t' + pred
                f.write(line + "\n")

    def test_predictions(pref_file, results_file):
        lang = config['language']
        command = ["python", f"../2022SegmentationST/evaluation/evaluate.py", "--gold", f"../2022SegmentationST/data/{lang}.word.test.gold.tsv", "--guess", pred_file]
        print(results_file)
        output = subprocess.run(command, capture_output=True).stdout.decode("utf-8")
        print(output)

        with open(results_file, 'w') as f:
            f.write(output)

    pred_file = file.replace(config['model_ext'], config['predictions_ext'])
    results_file = file.replace(config['model_ext'], config['results_ext'])
    print(f'pred_file: {pred_file}, results_file: {results_file}')
    write_predictions(dataset, file)
    logger.info(f"Predictions written to: {pred_file}")
    test_predictions(file, results_file)
    # write_predictions(dataset, use_heuristic=True)
    # test_predictions(use_heuristic=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generator')
    parser.add_argument("file", type=str, help = 'path to model file')
    parser.add_argument("--use-cpu", type=bool, default=False)
    opt = parser.parse_args()
    main(opt.file, opt.use_cpu)

import logging

from predict import segment_word, segment_word2
from helpers import postprocess
from morpheme_data import MorphemeDataLoader
from models import *
from helpers import *
from tqdm import tqdm

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

    pred_file = project_file(config, config['predictions_ext'], config['model_suffix'] + '-test-improved')
    assert pred_file is not None

    dataset = data.test
    # logger.setLevel(logging.WARN)
    with open(pred_file, 'w') as f:
        for word, morphs in tqdm(dataset[:], total = len(data.test.words)):
            pred = segment_word2(word, model, dataset, device)
            pred = postprocess(pred)
            line = "".join(word[1:-1]) + '\t' + pred
            # logger.info(line + " (" + "".join(morphs[1:-1]) + ")")
            f.write(line + "\n")
    logger.info(f"Predictions written to: {pred_file}")

if __name__ == "__main__":
    main()

import logging

from predict import segment_word
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
    device = config.device()
    model_file = get_most_recent_model_file(config)
    model = Seq2Seq(data.train.word_len, data.train.morph_len, device, config['encoder_decoder']['dropout']).to(device)
    print(f'model_file: {model_file}')
    model.load_from_file(model_file)

    dataset = data.test

    def write_predictions(dataset, pred_file):
        assert pred_file is not None
        logger.info(f'Writing predictions to {pred_file}')
        description = 'Writing predictions...'
        with open(pred_file, 'w') as f:
            for word, morphs in tqdm(dataset[:], total=len(data.test.words), desc=description):
                pred = segment_word(word, model, dataset, device)
                pred = postprocess(pred)
                line = "".join(word[1:-1]) + '\t' + pred
                f.write(line + "\n")

    pred_file = model.model_file.replace(config['model_ext'], config['predictions_ext'])
    write_predictions(dataset, pred_file)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Generator')
    # parser.add_argument("file", type=str, help = 'path to model file')
    # parser.add_argument("--use-cpu", type=bool, default=False)
    # opt = parser.parse_args()
    # main(opt.file, opt.use_cpu)
    main()
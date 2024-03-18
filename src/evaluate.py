import torch
from config import Config
from predict import segment_word
from helpers import project_file
from morpheme_data import MorphemeDataLoader
from models import *
from helpers import *

config = Config()
logger = get_logger()

SOS_TOKEN = config['special_tokens']['sos_token']
EOS_TOKEN = config['special_tokens']['eos_token']

def main():
    data = MorphemeDataLoader(config)

    # Look for Metal GPU device (for Silicon Macs) and default to CUDA (for hosted GPU service)
    device = config.device()

    model = Seq2Seq(data.test, device, config).to(device)
    model.load_from_file(config.model_file)
    print(model)

    lang = config['language']
    pred_file = project_file(config, config['predictions_ext'] + '-train', config['model_suffix'])
    assert pred_file is not None

    with open(pred_file, 'w') as f:
        for word in data.test.words:
            pred = segment_word(word, model, data.test, device)
            f.write("".join(word[1:-1]) + '\t' + pred + '\n')
    print("Predictions written to: {pred_file")





if __name__ == "__main__":
    main()

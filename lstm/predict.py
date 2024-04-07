import argparse

import torch
from config import Config
from morpheme_data import MorphemeDataLoader
from tqdm import tqdm
from models import Seq2Seq
from helpers import get_logger, postprocess, get_most_recent_model_file

config = Config()
logger = get_logger()
SOS_TOKEN = config['special_tokens']['sos_token']
EOS_TOKEN = config['special_tokens']['eos_token']


def segment_word(word, model, dataset, device, max_output_length=config['predictions']['max_output']):
    word_vocab = dataset.word_vocab
    morph_vocab = dataset.morph_vocab
    model.eval()
    with torch.no_grad():
        tokens = word
        ids = word_vocab.lookup_indices(tokens)
        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
        hidden, cell = model.encoder(tensor)
        inputs = morph_vocab.lookup_indices([SOS_TOKEN])
        last_index = None
        for _ in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
            predicted_token = output.argmax(-1).item()
            if predicted_token < dataset.morph_len:
                inputs.append(predicted_token)
                if predicted_token == morph_vocab[EOS_TOKEN]:
                    last_index = -1
                    break
                tokens = morph_vocab.lookup_tokens(inputs)
            else:
                morph_dict = {i: v for i,v in enumerate(morph_vocab.get_itos())}
                word_dict = {i: v for i,v in enumerate(word_vocab.get_itos())}
                logger.critical(f'MORPHS: {morph_dict}')
                logger.critical(f'WORDS: {word_dict}')
                logger.critical(f'word: {"".join(word)}, indices: {inputs} -> {predicted_token}')
                exit(1)

        prediction = "".join(tokens[1:last_index])
    return prediction

def main(config, file, use_cpu=False):
    data = MorphemeDataLoader(config)
    device = torch.device('cpu') if use_cpu else config.device()
    model = Seq2Seq(data.train.word_len, data.train.morph_len, device, config['encoder_decoder']['dropout']).to(device)
    model.load_from_file(file)
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

    pred_file = file.replace(config['model_ext'], config['predictions_ext'])
    write_predictions(dataset, pred_file)


if __name__ == "__main__":
    config = Config()
    parser = argparse.ArgumentParser(description='Generator')
    parser.add_argument("--file", type=str, help='path to model file', default=get_most_recent_model_file(config))
    parser.add_argument("--use-cpu", type=bool, default=True, help='Whether to load tensors to CPU')
    opt = parser.parse_args()
    main(config, opt.file, opt.use_cpu)
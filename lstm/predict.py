import torch
from config import Config
from morpheme_data import MorphemeDataLoader
from models import Seq2Seq
from helpers import get_logger, postprocess

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
            inputs.append(predicted_token)
            if predicted_token == morph_vocab[EOS_TOKEN]:
                last_index = -1
                break
        tokens = morph_vocab.lookup_tokens(inputs)
        prediction = "".join(tokens[1:last_index])
    return prediction

def main():
    config = Config()
    data = MorphemeDataLoader(config)
    device = config.device()
    model = Seq2Seq(data.train.word_len, data.train.morph_len, device).to(device)
    model.load_from_file(model.model_file)
    logger.info(model)

    # CHECK WORDS WE HAVE TRAINED
    logger.info("Checking first x trained examples")
    logger.info("=" * 80)
    dataset = data.test
    for word, morphs in dataset[:10]:
        pred = segment_word(word, model, dataset, device)
        pred = postprocess(pred)
        line = "".join(word[1:-1]) + '\t' + pred
        logger.info(line + " (" + "".join(morphs[1:-1]) + ")")

if __name__ == "__main__":
    main()
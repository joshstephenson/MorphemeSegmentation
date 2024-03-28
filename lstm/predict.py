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

def segment_word2(word, model, dataset, device):
    word_vocab = dataset.word_vocab
    morph_vocab = dataset.morph_vocab
    model.eval()
    with torch.no_grad():
        word_index = 1 # we are preloading with first 2 tokens (SOS + first_char)
        word_ids = word_vocab.lookup_indices(word)
        word_tensor = torch.LongTensor(word_ids).unsqueeze(-1).to(device)
        hidden, cell = model.encoder(word_tensor)
        input_ids = morph_vocab.lookup_indices(word[:1])
        last_index = None
        morph_separator_id = morph_vocab.lookup_indices(["|"])[0]
        while word_index < len(word)-1:
            # print("".join(morph_vocab.lookup_tokens(input_ids)))
            inputs_tensor = torch.LongTensor([input_ids[-1]]).to(device)
            output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
            predicted_token_id = output.argmax(-1).item()
            # print(f'{predicted_token_id}')
            if predicted_token_id != morph_separator_id:
                pred_char = morph_vocab.lookup_tokens([predicted_token_id])[0]
                char = word[word_index]
                if pred_char != char:
                    logger.info(f'Model predicted {pred_char}, using {char} instead.')
                    predicted_token_id = morph_vocab.lookup_indices([char])[0]
                # else:
                    # print('prediction matched')
                word_index += 1
            input_ids.append(predicted_token_id)
        tokens = morph_vocab.lookup_tokens(input_ids)
    prediction = ("".join(tokens[1:last_index]))
    return prediction

def main():
    config = Config()
    data = MorphemeDataLoader(config)
    device = config.device()
    model = Seq2Seq(data.train.word_len, data.train.morph_len, device).to(device)
    model.load_from_file(config.model_file)
    logger.info(model)

    # CHECK WORDS WE HAVE TRAINED
    logger.info("Checking first x trained examples")
    logger.info("=" * 80)
    dataset = data.test
    for word, morphs in dataset[:10]:
        pred = segment_word2(word, model, dataset, device)
        pred = postprocess(pred)
        line = "".join(word[1:-1]) + '\t' + pred
        logger.info(line + " (" + "".join(morphs[1:-1]) + ")")

if __name__ == "__main__":
    main()
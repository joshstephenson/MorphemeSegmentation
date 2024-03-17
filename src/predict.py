import torch
from config import Config
from morpheme_data import MorphemeDataLoader
from models import Seq2Seq

config = Config()
SOS_TOKEN = config['special_tokens']['sos_token']
EOS_TOKEN = config['special_tokens']['eos_token']

def segment_word(word, model, data, device, max_output_length=config['predictions']['max_output']):
    model.eval()
    word_vocab = data.word_vocab
    morph_vocab = data.morph_vocab
    with torch.no_grad():
        word_ids = word_vocab.lookup_indices(word)
        word_tensor = torch.LongTensor(word_ids).unsqueeze(-1).to(device)
        hidden, cell = model.encoder(word_tensor)
        input_ids = morph_vocab.lookup_indices([SOS_TOKEN])
        last_index = None
        for _ in range(max_output_length):
            # print("".join(morph_vocab.lookup_tokens(input_ids)))
            inputs_tensor = torch.LongTensor([input_ids[-1]]).to(device)
            output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
            predicted_token = output.argmax(-1).item()
            # print(predicted_token)
            input_ids.append(predicted_token)
            if predicted_token == morph_vocab[EOS_TOKEN]:
                last_index = -1
                break
        tokens = morph_vocab.lookup_tokens(input_ids)
    prediction = ("".join(tokens[1:last_index]))
    return prediction

def main():
    data = MorphemeDataLoader(config)

    # Look for Metal GPU device (for Silicon Macs) and default to CUDA (for hosted GPU service)
    device = config.device()

    model = Seq2Seq(data.train, device, config).to(device)
    model.load_from_file(config.model_file)
    print(model)

    # CHECK WORDS WE HAVE TRAINED
    print("Checking first 10 trained examples")
    print("=" * 80)
    count = 20
    for word, morph in zip(data.train.words[:count], data.train.morphs[:count]):
        pred = segment_word(word, model, data.train, device)
        print(f'word: {"".join(word[1:-1])}\n\t  pred: {pred}\n\tactual: {"".join(morph[1:-1])}')

    # print(", ".join(data.train.word_vocab.get_itos()))
    # print(", ".join(data.train.morph_vocab.get_itos()))

    exit(0)
    # WORDS WE HAVEN'T TRAINED
    print("Checking first 10 gold standard examples")
    print("=" * 80)
    f1 = find_f1(data.test.words[:1000], data.test.morphs[:1000], model, data.test.word_vocab, data.test.morph_vocab,
                 device)
    print(f'f1: {f1}')
    # for word, morph in zip(data.test.words[:10], data.test.morphs[:10]):
    #     actual, predict, predict_str = segment_word(word, morph, model, data.train.word_vocab, data.train.morph_vocab, '<SOS>', '<EOS>', device)
    #     print(f'word: {"".join(word[1:-1])}\n\t  pred: {pred}\n\tactual: {"".join(morph[1:-1])}')

if __name__ == "__main__":
    main()
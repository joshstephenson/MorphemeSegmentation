import torch
import logging
from config import Config
import numpy as np
from Levenshtein import distance
from sklearn.metrics import f1_score

config = Config()

SOS_TOKEN = config['special_tokens']['sos_token']
EOS_TOKEN = config['special_tokens']['eos_token']
def find_f1(words, expectations, model, word_vocab, morph_vocab, device,
            max_output_length=config['predictions']['max_output']):
    model.eval()
    distances = []
    with torch.no_grad():
        for i, (word, expectation) in enumerate(zip(words, expectations)):
            with torch.no_grad():
                word_ids = word_vocab.lookup_indices(list(word))
                word_tensor = torch.LongTensor(word_ids).unsqueeze(-1).to(device)
                hidden, cell = model.encoder(word_tensor)
                input_ids = morph_vocab.lookup_indices([SOS_TOKEN])
                last_index = None
                for _ in range(max_output_length):
                    inputs_tensor = torch.LongTensor([input_ids[-1]]).to(device)
                    output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
                    predicted_token = output.argmax(-1).item()
                    input_ids.append(predicted_token)
                    if predicted_token == morph_vocab[EOS_TOKEN]:
                        last_index = -1
                        break
                tokens = morph_vocab.lookup_tokens(input_ids)
                prediction = ("".join(tokens[1:last_index]))
            expectation = ("".join(expectation[1:-1]))
            d = distance(expectation, prediction) + 1
            if i % 10 == 0:
                logger.info(f'i: {i}, expectation: {expectation}, prediction: {prediction}, d: {d}')
            distances.append(d)
    return np.average(f1_score([1] * len(distances), distances, average='macro'))


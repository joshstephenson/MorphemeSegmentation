# -*- coding: utf-8 -*-
from morpheme_data import MorphemeDataLoader
from models import *
from torch.utils.data import DataLoader, TensorDataset
import logging
import string
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import string
import tqdm
import yaml
import os.path as pt
from Levenshtein import distance
from sklearn.metrics import f1_score

# Load the configuration
config_file = 'config.yaml'
try:
    path = pt.join(pt.dirname(pt.abspath(__file__)), 'config', config_file)
    with open(path, 'r') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
# On a hosted environment, this is likely to work, but the above will not
except Exception as e:
    path = 'config/' + config_file
    with open(path, 'r') as f:
        config = yaml.load(f, Loader = yaml.FullLoader)

assert config is not None

LANGUAGE = config['language']
UNKNOWN_TOKEN = config['special_tokens']['unknown_token']
PAD_TOKEN = config['special_tokens']['pad_token']
SOS_TOKEN = config['special_tokens']['sos_token']
EOS_TOKEN = config['special_tokens']['eos_token']

# Make sure we have a place to store the generated model
model_file = LANGUAGE + '-' + config['model_suffix']
try:
    file = pt.join(pt.dirname(pt.abspath(__file__)), config['model_dir'], model_file)
except Exception as e:
    file = config['model_dir'] + '/' + model_file
model_file = file
assert model_file is not None

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format = LOG_FORMAT, level = getattr(logging, 'INFO'))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def init_weights(m, initial = config['preprocessing']['initial_weights']):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -initial, initial)

def train_fn(model, data_loader, optimizer, criterion, device, **kwargs):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        # print(f"BATCH: {batch}")
        word = batch["word_ids"].to(device)
        morphs = batch["morph_ids"].to(device)
        # word = [word length, batch size]
        # morphs = [morphs length, batch size]
        optimizer.zero_grad()
        output = model(word, morphs, kwargs['teacher_forcing_ratio'])
        # output = [morphs length, batch size, morphs vocab size]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        # output = [(morphs length - 1) * batch size, morphs vocab size]
        morphs = morphs[1:].view(-1)
        # morphs = [(morphs length - 1) * batch size]
        loss = criterion(output, morphs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), kwargs['clip'])
        optimizer.step()
        epoch_loss += loss.item()
        if i % 100 == 0:
            print(f'i: {i}, loss: {loss.item()}: epoch: {epoch_loss}')
    return epoch_loss / len(data_loader)

def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            word = batch["word_ids"].to(device)
            morphs = batch["morph_ids"].to(device)
            # word = [word length, batch size]
            # morphs = [morphs length, batch size]
            output = model(word, morphs, 0)  # turn off teacher forcing
            # output = [morphs length, batch size, morphs vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            # output = [(morphs length - 1) * batch size, morphs vocab size]
            morphs = morphs[1:].view(-1)
            # morphs = [(morphs length - 1) * batch size]
            loss = criterion(output, morphs)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

def segment_word(word, model, word_vocab, morph_vocab, device, max_output_length = config['predictions']['max_output']):
    model.eval()
    with torch.no_grad():
        # if lower:
        #     tokens = [token.lower() for token in tokens]
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
    return prediction

def find_f1(words, expectations, model, word_vocab, morph_vocab, device, max_output_length = config['predictions']['max_output']):
    model.eval()
    distances = []
    with torch.no_grad():
        for i, (word, expectation) in enumerate(zip(words, expectations)):
            with torch.no_grad():
                word_ids = word_vocab.lookup_indices(list(word))
                expected_ids = morph_vocab.lookup_indices(list(expectation))
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
    return np.average(f1_score([1] * len(distances), distances, average = 'macro'))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    data = MorphemeDataLoader(LANGUAGE)
    input_dim = data.train.word_len
    output_dim = data.train.morph_len

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        print("GPU ENABLED √")
    else:
        print("NO GPU ACCESS. EXITING.")
        exit(1)

    encoder = Encoder(
        input_dim,
        **config['encoder']
    )
    decoder = Decoder(
        output_dim,
        **config['decoder']
    )
    model = Seq2Seq(encoder, decoder, device).to(device)

    if config['training']['enabled']:
        model.apply(init_weights)

    print(f"The model has {count_parameters(model):,} trainable parameters")

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=data.train.pad_index)

    # If training is enabled in config.yaml, we will actually train the model
    # Otherwise we'll just load the previously saved model from config['model_dir']
    if config["training"]["enabled"]:
        best_valid_loss = float("inf")
        for epoch in tqdm.tqdm(range(config['training']['epochs'])):
            train_loss = train_fn(
                model,
                data.train.loader,
                optimizer,
                criterion,
                device,
                **config['training']
            )
            valid_loss = evaluate_fn(
                model,
                data.validation.loader,
                criterion,
                device,
            )
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), model_file)
            print(f"\n\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
            print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")

    try:
        model.load_state_dict(torch.load(model_file))
    except Exception as e:
        model.load_state_dict(torch.load(model_file, map_location = torch.device('cpu')))

    assert model is not None

    if config['training']['enabled']:
        test_loss = evaluate_fn(model, data.test.loader, criterion, device)
        print(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")

    # CHECK WORDS WE HAVE TRAINED
    print("Checking first 10 trained examples")
    print("=" * 80)
    for word, morph in zip(data.train.words[:10], data.train.morphs[:10]):
        pred = segment_word(word, model, data.train.word_vocab, data.train.morph_vocab, device)
        print(f'word: {"".join(word[1:-1])}\n\t  pred: {pred}\n\tactual: {"".join(morph[1:-1])}')

    # WORDS WE HAVEN'T TRAINED
    print("Checking first 10 gold standard examples")
    print("=" * 80)
    f1 = find_f1(data.test.words[:1000], data.test.morphs[:1000], model, data.test.word_vocab, data.test.morph_vocab, device)
    print(f'f1: {f1}')
    # for word, morph in zip(data.test.words[:10], data.test.morphs[:10]):
    #     actual, predict, predict_str = segment_word(word, morph, model, data.train.word_vocab, data.train.morph_vocab, '<SOS>', '<EOS>', device)
    #     print(f'word: {"".join(word[1:-1])}\n\t  pred: {pred}\n\tactual: {"".join(morph[1:-1])}')

if __name__ == '__main__':
    main()
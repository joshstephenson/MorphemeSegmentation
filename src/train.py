# -*- coding: utf-8 -*-
from morpheme_data import MorphemeDataLoader
from models import *
import numpy as np
import torch
import tqdm
from config import Config

config = Config()

LANGUAGE = config['language']
UNKNOWN_TOKEN = config['special_tokens']['unknown_token']
PAD_TOKEN = config['special_tokens']['pad_token']
SOS_TOKEN = config['special_tokens']['sos_token']
EOS_TOKEN = config['special_tokens']['eos_token']

model_file = config.model_file

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

def main():
    data = MorphemeDataLoader(config)

    # Look for Metal GPU device (for Silicon Macs) and default to CUDA (for hosted GPU service)
    device = config.device()

    model = Seq2Seq(data.train, device, config).to(device)
    print(model)

    optimizer = config.optimizer(model)
    criterion = config.criterion(data)

    # If training is enabled in config.yaml, we will actually train the model
    # Otherwise we'll just load the previously saved model from config['model_dir']
    if config.training_enabled():
        best_valid_loss = float("inf")
        for _ in tqdm.tqdm(range(config['training']['epochs'])):
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
                model.save()
            print(f"\n\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
            print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")

    model.load_from_file(config.model_file)

    assert model is not None

    if config.training_enabled():
        test_loss = evaluate_fn(model, data.test.loader, criterion, device)
        print(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")

if __name__ == '__main__':
    main()

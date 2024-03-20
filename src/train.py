# -*- coding: utf-8 -*-
from morpheme_data import MorphemeDataLoader
from models import *
import numpy as np
import torch
import tqdm
from config import Config


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_fn(model, data_loader, optimizer, criterion, device, **kwargs):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        # logger.info(f"BATCH: {batch}")
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
            logger.info(f'i: {i}, loss: {loss.item()}: epoch: {epoch_loss}')
    return epoch_loss / len(data_loader)

def evaluate_fn(model, data_loader, criterion, scheduler, device):
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
            scheduler.step(loss)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

def main():
    config = Config()
    data = MorphemeDataLoader(config)
    device = config.device() # Look for Metal GPU device (for Silicon Macs) and default to CUDA (for hosted GPU service)
    model = Seq2Seq(data.train.word_len, data.train.morph_len, device).to(device)
    logger.info(model)

    # optimizer, scheduler = config.optimizer(model)
    optimizer, scheduler = config.optimizer(model)
    criterion = config.criterion(data.train.pad_index)

    # If training is enabled in config.yaml, we will actually train the model
    # Otherwise we'll just load the previously saved model from config['model_dir']
    if config.training_enabled():
        best_valid_loss = float("inf")
        early_stopping = EarlyStopping(patience = config['training']['early_stopping'], verbose = True, path = config.model_file)
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
                scheduler,
                device,
            )
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                logger.info("Early Stopping...")
                break

#            if valid_loss < best_valid_loss:
#                best_valid_loss = valid_loss
            logger.info(f"\n\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
            logger.info(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")

            model.load_from_file(config.model_file)

            test_loss = evaluate_fn(model, data.test.loader, criterion, scheduler, device)
            logger.info(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")
    else:
        logger.info("Training disabled in config.yaml")

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
from morpheme_data import MorphemeDataLoader
from models import *
import numpy as np
import torch
from tqdm import tqdm
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


class Trainer():
    def __init__(self):
        self.config = Config()
        self.logger = get_logger()
        self.data = MorphemeDataLoader(config)
        self.device = config.device()  # Look for Metal GPU device (for Silicon Macs) and default to CUDA (for hosted GPU service)
        self.model = Seq2Seq(self.data.train.word_len, self.data.train.morph_len, self.device).to(self.device)
        self.logger.info(self.model)

        # optimizer, scheduler = config.optimizer(model)
        self.optimizer = self.config.optimizer(self.model)
        self.criterion = self.config.criterion(self.data.train.pad_index)

        # For progress bar
        self.progress_bar = tqdm(total=len(self.data.train.words) * self.config['training']['epochs'])
        self.progress_count = 0

    def training_callback(self):
        self.progress_count += 1 #self.config['preprocessing']['batch_size']
        self.progress_bar.update(self.progress_count)

    def run(self):
        if not self.config.training_enabled():
            logger.info("Training disabled in config.yaml")
            exit(0)

        # best_valid_loss = float("inf")
        early_stopping = EarlyStopping(patience=config['training']['early_stopping'], verbose=True,
                                       path=config.model_file)

        for _ in range(config['training']['epochs']):
            train_loss = self.train_fn()
            valid_loss = self.validate_fn()
            early_stopping(valid_loss, self.model)

            learning_rate = self.optimizer.state_dict()["param_groups"][0]["lr"]
            self.logger.info(f'Learning rate is now: {learning_rate}')

            #            if valid_loss < best_valid_loss:
            #                best_valid_loss = valid_loss
            logger.info(f"\n\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
            logger.info(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")

            self.model.load_from_file(config.model_file)

            test_loss = self.validate_fn(use_test = True)
            logger.info(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")

            if early_stopping.early_stop:
                logger.info("Stopping early...")
                break

    def train_fn(self):
        kwargs = self.config['training']
        self.model.train()
        epoch_loss = 0
        for i, batch in enumerate(self.data.train.loader):
            word = batch["word_ids"].to(self.device)
            morphs = batch["morph_ids"].to(self.device)
            # word = [word length, batch size]
            # morphs = [morphs length, batch size]
            self.optimizer.zero_grad()
            output = self.model(word, morphs, kwargs['teacher_forcing_ratio'])
            # output = [morphs length, batch size, morphs vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            # output = [(morphs length - 1) * batch size, morphs vocab size]
            morphs = morphs[1:].view(-1)
            # morphs = [(morphs length - 1) * batch size]
            loss = self.criterion(output, morphs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), kwargs['clip'])
            self.optimizer.step()
            epoch_loss += loss.item()
            self.training_callback()
            # if i % 1000 == 0:
        logger.info(f'i: {i}, loss: {loss.item()}: epoch: {epoch_loss}')
        return epoch_loss / len(self.data.train.loader)

    def validate_fn(self, use_test = False):
        self.model.eval()
        epoch_loss = 0
        loader = self.data.test.loader if use_test else self.data.validation.loader
        with torch.no_grad():
            for i, batch in enumerate(loader):
                word = batch["word_ids"].to(self.device)
                morphs = batch["morph_ids"].to(self.device)
                # word = [word length, batch size]
                # morphs = [morphs length, batch size]
                output = self.model(word, morphs, 0)  # turn off teacher forcing
                # output = [morphs length, batch size, morphs vocab size]
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                # output = [(morphs length - 1) * batch size, morphs vocab size]
                morphs = morphs[1:].view(-1)
                # morphs = [(morphs length - 1) * batch size]
                loss = self.criterion(output, morphs)
                epoch_loss += loss.item()
        return epoch_loss / len(loader)

def main():
    trainer = Trainer()
    trainer.run()


if __name__ == '__main__':
    main()

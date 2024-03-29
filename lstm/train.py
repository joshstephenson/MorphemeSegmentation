# -*- coding: utf-8 -*-
from morpheme_data import MorphemeDataLoader
from models import *
import numpy as np
import torch
from tqdm import tqdm
from config import Config


class PlateauWithEarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, optimizer):
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
        self.optimizer = optimizer
        self.patience = config['training']['patience']
        self.counter = 0
        self.best_score = np.Inf
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = 0.5
        self.min_lr = float(config['training']['learning_rate_min'])
        self.current_lr = float(config['training']['learning_rate'])
        self.step_factor = float(config['training']['learning_rate_factor'])
    def __call__(self, val_loss, model):
        if not config['training']['early_stopping']:
            return
        score = val_loss

        if score < self.best_score - self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0 # reset the counter if we find a new best loss
        # elif self.current_lr > self.min_lr:
        #     self.set_learning_rate(self.current_lr * self.step_factor)
        else: # Now check for early stopping
            self.counter += 1
            logger.info(f'Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def set_learning_rate(self, new_lr):
        new_lr = max(self.min_lr, new_lr)
        for g in self.optimizer.param_groups:
            g['lr'] = new_lr
        self.current_lr = new_lr
        logger.info(f'Learning rate set to: {self.current_lr}')

        # if self.best_score is None:
        #     self.best_score = score
        #     self.save_checkpoint(val_loss, model)
        # elif score < self.best_score + self.delta:
        #     self.counter += 1
        #     self.trace_func(f'Early stopping counter: {self.counter} out of {self.patience}')
        #     if self.counter >= self.patience:
        #         self.early_stop = True
        # else:
        #     self.best_score = score
        #     self.save_checkpoint(val_loss, model)
        #     self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model.save()
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

    def run(self):
        early_stopping = PlateauWithEarlyStopping(self.optimizer)

        for i in range(config['training']['epochs']):
            logger.info(f"Epoch {i+1}")
            train_loss = self._train()
            valid_loss = self._validate()

            # learning_rate = self.scheduler.get_last_lr()
            # self.logger.info(f'Learning rate is now: {learning_rate}')

            #            if valid_loss < best_valid_loss:
            #                best_valid_loss = valid_loss
            logger.info(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
            logger.info(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")

            # self.model.load_from_file(config.model_file)

            test_loss = self._validate(use_test = True)
            logger.info(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")

            # Nothing will occur here if early_stopping is False in config.yaml
            early_stopping(valid_loss, self.model)
            if early_stopping.early_stop:
                self.model.load_from_file(config.model_file) # reload the lowest valid loss checkpoint
                logger.info(f"Stopping early after {i} epochs.")
                break

    def _train(self):
        kwargs = self.config['training']
        self.model.train()
        epoch_loss = 0
        # pbar = tqdm(self.data.train.word_count / config['preprocessing']['batch_size'], desc=self.data.train.label)
        pbar = tqdm(self.data.train.loader, leave=False)
        for i, batch in enumerate(pbar):
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
            pbar.set_description(f"{self.data.train.label}. Loss: {loss:.3f} (batch), {epoch_loss:.3f} (epoch)")
            # if i % 100 == 0:
            #     logger.info(f'i: {i}, loss: {loss.item()}: epoch: {epoch_loss}')
        logger.info(f'Batch {i}, loss: {loss.item()}: epoch: {epoch_loss}')
        return epoch_loss / len(self.data.train.loader)

    def _validate(self, use_test = False):
        self.model.eval()
        epoch_loss = 0
        dataset = self.data.test if use_test else self.data.validation
        pbar = tqdm(dataset.loader)
        with torch.no_grad():
            for i, batch in enumerate(pbar):
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
                pbar.set_description(f"{dataset.label}. Loss: {loss:.3f} (batch), {epoch_loss:.3f} (epoch)")
        return epoch_loss / len(dataset.loader)

def main():
    trainer = Trainer()
    trainer.run()


if __name__ == '__main__':
    main()

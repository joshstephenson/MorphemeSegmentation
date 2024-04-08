# -*- coding: utf-8 -*-
import argparse

from morpheme_data import MorphemeDataLoader
from models import *
import numpy as np
import torch
from tqdm import tqdm
from config import Config
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    def __init__(self, dropout):
        self.config = Config()
        self.logger = get_logger()
        self.data = MorphemeDataLoader(config)
        self.device = config.device()  # Look for Metal GPU device (for Silicon Macs) and default to CUDA (for hosted GPU service)
        self.model = Seq2Seq(self.data.train.word_len, self.data.train.morph_len, self.device, self.data.train.pad_index, dropout).to(self.device)
        self.logger.info(self.model)

        # optimizer, scheduler = config.optimizer(model)
        self.optimizer, self.scheduler = self.config.optimizer(self.model)
        self.criterion = self.config.criterion(self.data.train.pad_index)

    def run(self, args):
        writer = SummaryWriter()
        num_trainable_params = sum([p.numel() for p in self.model.parameters()])
        logger.info(f'Trainable params: {num_trainable_params}')
        lowest_valid_loss = float("inf")
        for i in range(config['training']['epochs']):
            logger.info(f"Epoch {i+1}")
            train_loss = self._train(i, writer, args)
            valid_loss = self._validate(i, writer, args)
            if valid_loss < lowest_valid_loss:
                lowest_valid_loss = valid_loss
                self.model.save()

            logger.info(f"\tTrain Loss: {train_loss:7.9f} | Train PPL: {np.exp(train_loss):7.3f}")
            logger.info(f"\tValid Loss: {valid_loss:7.9f} | Valid PPL: {np.exp(valid_loss):7.3f}")

            self.scheduler.step(valid_loss)
            logger.info(f"\tEpoch {i+1}, learning rate: {self.scheduler.get_last_lr()}")
            # self.model.load_from_file(self.model.model_file)

            test_loss = self._validate(i, writer, args, use_test = True)
            logger.info(f"| Test Loss: {test_loss:.9f} | Test PPL: {np.exp(test_loss):7.3f} |")

            # Nothing will occur here if early_stopping is False in config.yaml
        writer.close()
        # self.model.save()

    def _train(self, epoch, writer, args):
        kwargs = self.config['training']
        self.model.train()
        epoch_loss = 0
        # pbar = tqdm(self.data.train.word_count / config['preprocessing']['batch_size'], desc=self.data.train.label)
        pbar = tqdm(self.data.train.loader, leave=False, disable=args.disable_tqdm)
        for i, batch in enumerate(pbar):
            word = batch["word_ids"].to(self.device)
            morphs = batch["morph_ids"].to(self.device)
            # word = [word length, batch size]
            # morphs = [morphs length, batch size]
            output = self.model(word, morphs, kwargs['teacher_forcing_ratio'])
            # output = [morphs length, batch size, morphs vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            # output = [(morphs length - 1) * batch size, morphs vocab size]
            morphs = morphs[1:].view(-1)
            # morphs = [(morphs length - 1) * batch size]
            loss = self.criterion(output, morphs)
            epoch_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            if i % 100 == 0:
                writer.add_scalar('train-loss', loss, epoch_loss, epoch * len(self.data.train.loader) + i)
            if kwargs['clip'] > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), kwargs['clip'])
            self.optimizer.step()
            pbar.set_description(f"{self.data.train.label}. Loss: {loss:.3f} (batch), {epoch_loss:.3f} (epoch)")
            # if i % 100 == 0:
            #     logger.info(f'i: {i}, loss: {loss.item()}: epoch: {epoch_loss}')
        logger.info(f'Batch {i}, loss: {loss.item()}: epoch: {epoch_loss}')
        return epoch_loss / len(self.data.train.loader)

    def _validate(self, epoch, writer, args, use_test = False):
        self.model.eval()
        epoch_loss = 0
        dataset = self.data.test if use_test else self.data.validation
        pbar = tqdm(dataset.loader, disable=args.disable_tqdm)
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
                for param_group in self.optimizer.param_groups:
                    current_lr = param_group['lr']
                    logger.info(f"Epoch: {epoch + 1}, Current Learning Rate: {current_lr}")
                epoch_loss += loss.item()
                if i % 100 == 0:
                    if use_test:
                        writer.add_scalar('test-loss', loss, epoch_loss, epoch * len(self.data.train.loader) + i)
                    else:
                        writer.add_scalar('valid-loss', loss, epoch_loss, epoch * len(self.data.train.loader) + i)
                pbar.set_description(f"{dataset.label}. Loss: {loss:.3f} (batch), {epoch_loss:.3f} (epoch)")
        return epoch_loss / len(dataset.loader)

def main(args):
    trainer = Trainer(dropout = config['encoder_decoder']['dropout'])
    trainer.run(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM Trainer')
    parser.add_argument("--disable-tqdm", action="store_true", default=False, help='Whether to load tensors to CPU')
    opt = parser.parse_args()
    main(opt)

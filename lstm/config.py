import yaml
from helpers import project_file, get_logger
import os.path as pt
import torch
import torch.optim as optim
import torch.nn as nn
import random
import sys
import argparse
from entmax.losses import Entmax15Loss

logger = get_logger()

class Config():
    def __init__(self):
        try:
            # Load the configuration
            config_file = 'config.yaml'
            path = pt.join(pt.dirname(pt.abspath(__file__)), 'config', config_file)
            with open(path, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        # On a hosted environment, this is likely to work, but the above will not
        except Exception as _:
            path = 'config/' + config_file
            with open(path, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

        self.config = config
        torch.manual_seed(config['seed'])
        random.seed(config['seed'])


    def model_file(self):
        return project_file(self.config, self.config['model_ext'], self.config['model_suffix'])

    def device(self):
        """
        Look for Metal GPU device (for Silicon Macs) and default to CUDA (for hosted GPU service)
        :return: device
        """
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")
        if torch.backends.mps.is_available() or torch.cuda.is_available():
            logger.info("GPU ENABLED √")
        else:
            logger.exception("NO GPU ACCESS. EXITING.")
            exit(1)
        return device

    def optimizer(self, model):
        if self['training']['weight_decay'] > 0:
            return optim.Adam(model.parameters(), lr = self['training']['learning_rate'], weight_decay=self['training']['weight_decay'])
        else:
            return optim.Adam(model.parameters(), lr=self['training']['learning_rate'])

    def criterion(self, pad_index):
        match self['training']['loss_criterion']:
            case 'entmax':
                return Entmax15Loss(ignore_index=pad_index)
            case _:
                return nn.CrossEntropyLoss(ignore_index=pad_index)


    def __getitem__(self, item):
        return self.config[item]

    def info(self, message):
        """
        helper message because I can't help but continue writing 'config.info' instead of logger.info
        :param message:
        :return:
        """
        logger.info(message)

def main(args):
    config = Config()
    if args.l:
        sys.stdout.write(config['language'] + "\n")
    else:
        sys.stdout.write(config.model_file() + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generator')
    parser.add_argument("-l", action='store_true', help='return language name')
    opt = parser.parse_args()
    main(opt)

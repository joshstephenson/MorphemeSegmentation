import yaml
from helpers import project_file, get_logger
import os.path as pt
import torch
import torch.optim as optim
import torch.nn as nn
import random
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

        # Where we store the generated model
        self.model_file = project_file(self.config, self.config['model_ext'], self.config['model_suffix'])

        assert self.model_file is not None

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
        return optim.Adam(model.parameters(), lr = self['training']['learning_rate'])

    def criterion(self, pad_index):
        match self['training']['loss_criterion']:
            case 'entmax':
                return Entmax15Loss(ignore_index=pad_index)
            case _:
                return nn.CrossEntropyLoss(ignore_index=pad_index)


    def __getitem__(self, item):
        return self.config[item]



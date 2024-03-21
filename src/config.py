import yaml
from helpers import project_file, get_logger
import os.path as pt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

    def optimizer(self, model, include_scheduler = False):
        optimizer = optim.Adam(model.parameters(), lr = self['training']['learning_rate'])
        # optimizer = optim.SGD(model.parameters(), lr=self['training']['learning_rate'], momentum=0.9)
        scheduler = None
        if include_scheduler:
            scheduler = ReduceLROnPlateau(optimizer, 'min')
        return optimizer, scheduler

    def criterion(self, pad_index):
        return Entmax15Loss(ignore_index=pad_index) #nn.CrossEntropyLoss(ignore_index=pad_index)

    def __getitem__(self, item):
        return self.config[item]

    def training_enabled(self):
        return self['training']['enabled']


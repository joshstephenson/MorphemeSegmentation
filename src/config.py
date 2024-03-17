import yaml
import os.path as pt
import logging
import torch
import torch.optim as optim
import torch.nn as nn
from src.morpheme_data import MorphemeDataLoader


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

        # Logger
        LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
        logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        self.logger = logger

        # Where we store the generated model
        lang = self['language']
        model_file = lang + '-' + str(self['encoder_decoder']['hidden_dim']) + self['model_suffix']
        try:
            file = pt.join(pt.dirname(pt.abspath(__file__)), config['model_dir'], model_file)
        except Exception as _:
            file = config['model_dir'] + '/' + model_file
        self.model_file = file
        assert self.model_file is not None

    def device(self):
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")
        if torch.backends.mps.is_available() or torch.cuda.is_available():
            print("GPU ENABLED √")
        else:
            print("NO GPU ACCESS. EXITING.")
            exit(1)
        return device

    def optimizer(self, model):
        return optim.Adam(model.parameters())

    def criterion(self, data:MorphemeDataLoader):
        return nn.CrossEntropyLoss(ignore_index=data.train.pad_index)

    def __getitem__(self, item):
        return self.config[item]

    def training_enabled(self):
        return self['training']['enabled']


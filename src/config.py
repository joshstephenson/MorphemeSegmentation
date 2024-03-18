import yaml
from helpers import project_file
import os.path as pt
import torch
import torch.optim as optim
import torch.nn as nn
from morpheme_data import MorphemeDataLoader


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
        self.model_file = project_file(self, self.config['model_ext'], self.config['model_suffix'])
        assert self.model_file is not None

    def device(self):
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")
        if torch.backends.mps.is_available() or torch.cuda.is_available():
            print("GPU ENABLED √")
        else:
            print("NO GPU ACCESS. EXITING.")
            exit(1)
        return device

    def optimizer(self, model, include_scheduler = True):
        optimizer = optim.Adam(model.parameters(), lr = self['training']['learning_rate'])
        # optimizer = optim.Adam(model.parameters())
        scheduler = None
        if include_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=10,
                                                   threshold=0.0001,
                                                   threshold_mode='abs')
        return optimizer, scheduler

    def criterion(self, pad_index):
        return nn.CrossEntropyLoss(ignore_index=pad_index)

    def __getitem__(self, item):
        return self.config[item]

    def training_enabled(self):
        return self['training']['enabled']


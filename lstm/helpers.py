import os
import logging
from pathlib import Path
import glob

MORPH_SEPARATOR_INTERNAL = '|'
MORPH_SEPARATOR_EXTERNAL = ' @@'

def get_most_recent_model_file(config):
    model_dir = os.path.join(config['output_dir'], config['language'])
    dirs = list(filter(os.path.isdir, glob.glob(f'{model_dir}/*')))
    dirs.sort(key=lambda x: os.path.getmtime(x))
    lang = config['language']
    suffix = config['model_suffix']
    extension = config['model_ext']
    filename = f'{lang}{suffix}.{extension}'
    return os.path.join(dirs[-1], filename)

def preprocess(morphs):
    return morphs.replace(MORPH_SEPARATOR_EXTERNAL, MORPH_SEPARATOR_INTERNAL)

def postprocess(prediction):
    return prediction.strip().replace(MORPH_SEPARATOR_INTERNAL, MORPH_SEPARATOR_EXTERNAL)

def project_file(config, extension, suffix = None):
    lang = config['language']
    suffix = suffix if suffix is not None else ''
    config_description = ('embeddings_' + str(config['encoder_decoder']['embedding_dim']) +
                '-hidden_' + str(config['encoder_decoder']['hidden_dim']) +
                '-n_layers_' + str(config['encoder_decoder']['n_layers']) +
                '-dropout_' + str(config['encoder_decoder']['dropout']) +
                '-lr_' + str(config['training']['learning_rate'])).replace(' ', '')
    directory = Path(os.path.join(config['output_dir'], config['language'], config_description))
    filename = f'{lang}{suffix}.{extension}'
    directory.mkdir(parents=True, exist_ok=True)
    return os.path.join(directory, filename)

def get_logger():
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger

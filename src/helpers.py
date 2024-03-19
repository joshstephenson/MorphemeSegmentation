import os
import logging

MORPH_SEPARATOR_INTERNAL = '|'
MORPH_SEPARATOR_EXTERNAL = ' @@'

def preprocess(morphs):
    return morphs.replace(MORPH_SEPARATOR_EXTERNAL, MORPH_SEPARATOR_INTERNAL)

def postprocess(prediction):
    return prediction.strip().replace(MORPH_SEPARATOR_INTERNAL, MORPH_SEPARATOR_EXTERNAL)

def project_file(config, extension, suffix = None):
    lang = config['language']
    suffix = suffix if suffix is not None else ''
    filename = lang + '-' + str(config['encoder_decoder']['hidden_dim']) + suffix + '.' + extension
    directory = config['model_dir']
    if not os.path.isdir(directory):
        os.makedirs(directory)
    return os.path.join(directory, filename)

def get_logger():
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger

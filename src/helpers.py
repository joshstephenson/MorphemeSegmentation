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
    filename = (lang +
                '-embeddings_' + str(config['encoder_decoder']['embedding_dim']) +
                '-hidden_' + str(config['encoder_decoder']['hidden_dim']) +
                '-n_layers_' + str(config['encoder_decoder']['n_layers']) +
                '-dropout_' + str(config['encoder_decoder']['dropout']) +
                '-tfr_' + str(config['training']['teacher_forcing_ratio']) +
                '-lr_min_' + str(config['training']['learning_rate_min']) +
                '-clip_' + str(config['training']['clip']) +
                '-patience_' + str(config['training']['patience']) +
                '-loss_' + str(config['training']['loss_criterion']) +
                '-bi_' + str(config['encoder_decoder']['bidirectional']) +
                suffix +
                '.' +
                extension).replace(' ', '')
    directory = config['output_dir']
    if not os.path.isdir(directory):
        os.makedirs(directory)
    return os.path.join(directory, filename)

def get_logger():
    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, 'INFO'))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger

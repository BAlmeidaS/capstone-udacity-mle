import logging

import tensorflow as tf


def tf_config():
    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_visible_devices(gpu, 'GPU')
        tf.config.experimental.set_memory_growth(gpu, True)

    logical_gpus = tf.config.experimental.list_logical_devices('GPU')

    logging.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")


logging.debug('configuring tensorflow...')
tf_config()

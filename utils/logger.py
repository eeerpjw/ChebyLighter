# -*- coding: utf-8 -*-

import logging
import os
def get_logger(level=None, log_file=None):
    head = '[%(levelname)s] [%(asctime)-15s] %(funcName)s %(message)s'
    level_dit = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    logging.basicConfig(level=level_dit[level], format=head)
    logger = logging.getLogger()
    if log_file != None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)
    return logger



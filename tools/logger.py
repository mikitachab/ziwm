import logging
import os

from datetime import datetime

from constants import LOGS_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def create_logs_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def logger_init(logfile_name):
    create_logs_dir(LOGS_DIR)
    logfile_path = '{logs_dir_path}/{logfile_name}-{date}.log'.format(logs_dir_path=LOGS_DIR,
                                                                      logfile_name=logfile_name,
                                                                      date=datetime.now().strftime('%H_%M_%d_%m_%Y'))
    formatter = logging.Formatter(
        '[%(asctime)s %(levelname)-8s %(filename)15s:%(lineno)s - function: %(funcName)-15s]: %(message)2s',
        '%Y-%m-%d %H:%M:%S')
    for handler in (logging.StreamHandler(), logging.FileHandler(logfile_path, 'a', encoding='utf-8', delay='true')):
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.debug('Logger initialized.')

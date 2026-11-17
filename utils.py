import os
import sys
import json
import logging

import matplotlib.pyplot as plt

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%d-%b-%y %H:%M:%S'


def setup_logging(log_file='classifier.log'):
    """Log to the console AND append to one log file that keeps the history of every run."""
    root = logging.getLogger()
    if getattr(root, '_multiclass_configured', False):
        return
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')  # 'a' = append, never overwrite
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)
    root._multiclass_configured = True


def log_run_header(logger, run_type, config):
    """Write a clear separator plus the full configuration at the start of each run."""
    logger.info('=' * 80)
    logger.info(f'NEW {run_type.upper()} RUN')
    logger.info('Configuration:\n' + json.dumps(config, indent=4, ensure_ascii=False))
    logger.info('=' * 80)


def finish_figure(file_path, show=False):
    """Save the current figure, optionally display it, then close it to free memory.
    With show=False nothing pops up, so training never pauses waiting for a window to be closed."""
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    plt.savefig(file_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

# -*- coding: utf-8 -*-

from configparser import ConfigParser
from distutils.util import strtobool

utils_section = "UTILS"
metrics_section = "METRICS"


def str_to_bool(val):
    v = strtobool(val)
    if v == 0:
        return False
    elif v == 1:
        return True
    else:
        raise ValueError
 
def read_config(fpath):
    config = ConfigParser()
    with open(fpath, 'r') as f:
        config.read_file(f)
    return config._sections

def parse_model_config(config_path):
    config = read_config(config_path)
    return config

def parse_task_config(config_path):
    config = read_config(config_path)
    return config

def parse_tpot_config(config_path):
    config = read_config(config_path)
    config[utils_section]['n_jobs'] = int(config[utils_section]['n_jobs'])
    config[utils_section]['max_time_mins'] = int(config[utils_section]['max_time_mins'])
    config[utils_section]['minimal_number_of_models'] = int(config[utils_section]['minimal_number_of_models'])

    return config
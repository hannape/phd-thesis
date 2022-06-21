# -*- coding: utf-8 -*-

import os
import time
import logging 
import sys
import shutil
import numpy as np
import json
import pickle

from sklearn.calibration import CalibratedClassifierCV

def save_configs(cfgs_list, directory):
    # stores config files in the experiment dir
    timestamp = time.strftime('%Y-%m-%d-%H-%M')
    for config_file in cfgs_list:
        filename = f"{timestamp}-{os.path.basename(config_file)}"
        shutil.copyfile(config_file, os.path.join(directory, filename))


def save_as_json(obj, saving_dir, filename, nexp=None):
    # saves with json but uses a timestamp
    ## nexp = optional neptune experiment
    timestamp = time.strftime('%Y-%m-%d-%H-%M')
    if isinstance(obj, dict):
        for key in obj.keys():
            if isinstance(obj[key], np.ndarray):
                obj[key] = obj[key].tolist()

    with open(os.path.join(saving_dir, f'{timestamp}-{filename}'), 'w') as f:
        json.dump(obj, f, indent=2)
        
    if nexp:
        nexp.log_artifact(os.path.join(saving_dir, f'{timestamp}-{filename}'))


def pickle_and_log_artifact(obj, saving_dir, filename, nexp=None):
    timestamp = time.strftime('%Y-%m-%d-%H-%M')
    with open(os.path.join(saving_dir, f'{timestamp}-{filename}.pickle'), 'wb') as f:
        pickle.dump(obj, f)
    if nexp is not None:
        nexp.log_artifact(os.path.join(saving_dir, f'{timestamp}-{filename}.pickle'))


def save_predictions(x, y, cv_split, test_x, test_y, model, saving_dir):
    # saving predictions to files
    timestamp = time.strftime('%Y-%m-%d-%H-%M')

    def _save_to_file(true_label, predicted_probabilities, filename):

        try:
            os.makedirs(saving_dir)
        except FileExistsError:
            pass
        with open(os.path.join(saving_dir, f"{timestamp}-{filename}"), 'w') as fid:
            fid.write('has_bird\tbird_probability\n')
            for true, proba in zip(true_label, predicted_probabilities[:,1]):
                fid.write(f"{true}\t{proba}\n")            
    # test data

    try:
        proba = model.predict_proba(test_x)
    except (AttributeError, RuntimeError):
        #proba = None
        clf = CalibratedClassifierCV(model.fitted_pipeline_)  # for SVC
        clf.fit(x,y) 
        proba = clf.predict_proba(test_x)
        
    _save_to_file(test_y, proba, 'test.predictions')
    
    # training data
    for idx, (_, indices) in enumerate(cv_split):
        this_x = x[indices]
        this_y = y[indices]

        try:
            proba = model.predict_proba(this_x)
        except (AttributeError, RuntimeError):
            proba = clf.predict_proba(this_x) 
            
        _save_to_file(this_y, proba, f'valid-{idx}.predictions')

def my_save_to_file(true_label, predicted_probabilities, filename, saving_dir):

    try:
        os.makedirs(saving_dir)
    except FileExistsError:
        pass
    with open(os.path.join(saving_dir, f"{filename}"), 'w') as fid: 
        fid.write('has_bird\tbird_probability\n')
        for true, proba in zip(true_label, predicted_probabilities):
            fid.write(f"{true}\t{proba}\n")


class LoggerWrapper:
    def __init__(self, path='.'):
        """
        Wrapper for logging.
        Allows to replace sys.stderr.write so that error massages are redirected do sys.stdout and also saved in a file.
        use: logger = LoggerWrapper(); sys.stderr.write = logger.log_errors
        :param: path: directory to create log file
        """
        # count spaces so that the output is nicely indented
        self.trailing_spaces = 0

        # create the log file
        timestamp = time.strftime('%Y-%m-%d-%H-%M')
        self.filename = os.path.join(path, f'{timestamp}.log')
        try:
            with open(self.filename,"w") as f:
                pass
        except FileExistsError:
            pass

        # configure logging
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-6s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=self.filename,
                            filemode='w')
        formatter = logging.Formatter('%(name)-6s: %(levelname)-8s %(message)s')

        # make a handler to redirect to std.out
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)  
        self.console = logging.StreamHandler(sys.stdout)
        self.console.setLevel(logging.INFO)
        self.console.setFormatter(formatter)
        self.logger.addHandler(self.console)

    def log_errors(self, msg):
        msg = msg.strip('\n')  

        if msg == ' ' * len(msg):  
            self.trailing_spaces += len(msg)
        elif len(msg) > 1:
            self.logger.error(' ' * self.trailing_spaces + msg)
            self.trailing_spaces = 0
            
            

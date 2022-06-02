# -*- coding: utf-8 -*-

# IMPORTS

import pandas as pd
import os
import sys
import time
import torch
from tpot import TPOTClassifier

import config
import grid  
from config_tpot import parse_model_config, parse_tpot_config, parse_task_config
from config_tpot import utils_section
from useful_functions import make_cv_split 
from saving_utils import LoggerWrapper, save_configs, save_as_json, pickle_and_log_artifact 

''' Training and checking models from chosen families, with different hyperparameters, during specified time. Input data transformed to chosen representation.

 One representation at a time (out of five), one family of models at a time (out of 6)
 representations: ['spektrogram' , 'mel-spektrogram', 'multitaper', '8_classic', '8_classic_plus_MIR']
 models: ['nb', 'trees', 'SVM', 'logreg', 'kn', 'mlp']  - Naive Bayes, Trees, Support Vector Machines, Logistic Regression, k-nearest neighbours, Multilayer perceptron
 
 Running script: python script_name model_config tpot_config metrics_congfig representation_type saving_dir 
   e.g. python main_tpot.py ..\configs\trees.cfg ..\configs\tpot-pr-6h.cfg ..\configs\metrics.cfg 8_classic ..\results

'''
# %%

if __name__=='__main__':
    
    # Load configs
    model_cfg = parse_model_config(sys.argv[1])
    tpot_cfg = parse_tpot_config(sys.argv[2])
    metrics_cfg = parse_task_config(sys.argv[3])
    chosen_repr = sys.argv[4]
   
    exp_dir = os.path.join(sys.argv[-1],"6h-PR", f"{chosen_repr}-{model_cfg[utils_section]['model']}", time.strftime('%Y-%m-%d-%H-%M'))
   
    try:
        os.makedirs(exp_dir)
    except FileExistsError:
        pass
    
    # setup logger (everything that goes through logger or stderr will be saved in a file and sent to stdout)
    timestamp_start = time.strftime('%Y-%m-%d-%H-%M')
    logger_wrapper = LoggerWrapper(exp_dir)
    sys.stderr.write = logger_wrapper.log_errors
    logger_wrapper.logger.info(f'Running {sys.argv[1:-1]}')
    
    save_configs(sys.argv[1:-2], exp_dir)
    
    # load training and validation data   
    df1 = pd.read_hdf(os.path.join('..','..','jupyter','data','df_train_'+ str(chosen_repr) +'_norm.h5'),'repr4')
    df2 = pd.read_hdf(os.path.join('..','..','jupyter','data','df_valid_norm.h5'),'df')  # all representations at once
    all_repr = config.representation_1d + config.representation_2d
    all_repr.remove(chosen_repr)
    all_repr.remove('multitaper')  # not found in valid
    df2 = df2.drop(all_repr, axis=1)
    cv_split, dX, dy, di = make_cv_split(df1, df2, chosen_repr, model_cfg[utils_section]['model'])    
    print("Train + valid loaded")
       
# %% Grid space and TPOT configuration  
    print('Model type ',model_cfg[utils_section])
    grid_space = grid.get_grid(**model_cfg[utils_section])
    
    n_jobs = tpot_cfg[utils_section]['n_jobs']
    max_time_mins = tpot_cfg[utils_section]['max_time_mins']
    minimal_number_of_models = tpot_cfg[utils_section]['minimal_number_of_models']

    tpot_model_kwargs = { # constants
        'generations': None,
        'random_state': 667,
        'warm_start': True,
        'use_dask': True,
        'memory': 'auto',
        'verbosity': 3,
        'cv': cv_split,
        # general setup
        'n_jobs': n_jobs, 
        'max_time_mins': max_time_mins,
        'max_eval_time_mins': n_jobs * max_time_mins // minimal_number_of_models, 
        # per model setup
        'scoring': tpot_cfg[utils_section]['scoring'],
        'periodic_checkpoint_folder': os.path.join(exp_dir, "./tpot_checkpoints")
         }
    
    # run experiment
    print("running models")
    if torch.cuda.is_available():
        print(torch.device('cuda'))
    else:
        print(torch.device('cpu'))    
    
    timestamp1 = time.strftime('%Y-%m-%d-%H-%M')
    model = TPOTClassifier(config_dict=grid_space, **tpot_model_kwargs)
    _ = model.fit(dX, dy)
    print(model.fitted_pipeline_)
    
    # SAVING RESULTS
    timestamp = time.strftime('%Y-%m-%d-%H-%M')
    model.export(os.path.join(exp_dir, f'{timestamp_start}-best_model_script.py'))
    save_as_json(model.evaluated_individuals_, exp_dir, 'evaluated_individuals.json')
    
    # save the model itself
    pickle_and_log_artifact(model.fitted_pipeline_, exp_dir, 'model')

    # SAVING SCORES
    all_scores = {}
    all_scores['grid_mean_cv_score'] = model._optimized_pipeline_score

    print('valid:',all_scores['grid_mean_cv_score']) 
    
    print(all_scores)
    
    logger_wrapper.logger.info(all_scores)
    save_as_json(all_scores, exp_dir, 'best_model_scores.json')
    print('how many models checked:', len(model.evaluated_individuals_) )
    logger_wrapper.logger.info(len(model.evaluated_individuals_))
    print('start: ', timestamp1, ', end: ', time.strftime('%Y-%m-%d-%H-%M'))
    
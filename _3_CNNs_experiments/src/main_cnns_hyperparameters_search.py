# -*- coding: utf-8 -*-

import os
import time
import sys
import json
import pandas as pd
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
K.set_image_data_format('channels_last')
from sklearn.metrics import average_precision_score, make_scorer
from numpy.random import seed
from tensorflow.compat.v1.keras.backend import set_session

seed(1)
tf.random.set_seed(2)

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
experimentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(experimentdir)
sys.path.insert(0,parentdir)

import config
from saving_utils import LoggerWrapper
from sklearn.model_selection import RandomizedSearchCV 
from useful_func_cnn import load_my_data, make_cnn_grid_corr

# allocating fraction of GPU memory
config2 = tf.compat.v1.ConfigProto()
config2.gpu_options.per_process_gpu_memory_fraction = 0.3 
set_session(tf.compat.v1.Session(config=config2))

''' Random search for CNN hyperparameters. 

    Running script:python script_name chosen_representation how_many_architectures_checked
    e.g. python main_cnns_hyperparameters_search.py spektrogram 200
'''

#%%
if __name__=='__main__':

    # Load configs
    chosen_repr = sys.argv[1] 
    n_iter =  sys.argv[2]

    exp_dir = os.path.join("..","results","randomSearch", f"{chosen_repr}", time.strftime('%Y-%m-%d-%H-%M'))
    try:
        os.makedirs(exp_dir)
    except FileExistsError:
        pass
         
    start_time = time.time()
    logger_wrapper = LoggerWrapper(exp_dir)
    sys.stderr.write = logger_wrapper.log_errors
    logger_wrapper.logger.info(f'Running {sys.argv}')
    
    # cv - split to randomizedsearch, to avoid CV. dX - train + valid.separate as X_val & train_X
    cv_split, dX, dy, di, cv, X_val, y_val, train_X, train_y = load_my_data(chosen_repr, config) 
    
    print(chosen_repr)
    if chosen_repr=="spektrogram":
        in_shape = (63, 148, 1)
    if chosen_repr=="mel-spektrogram":
        in_shape = (60, 148, 1)    
    if chosen_repr=="multitaper":
        in_shape = (64, 149, 1)       
        
    dX_resh = dX.reshape(dX.shape[0], in_shape[0], in_shape[1], 1).astype('float32')
    input_shape = (np.shape(dX_resh)[1], np.shape(dX_resh)[2], np.shape(dX_resh)[3]) 
    X_val_resh = X_val.reshape(X_val.shape[0], in_shape[0], in_shape[1], 1).astype('float32') 
    
    # CNN random search hyperparameters
    params = {
    'lr': [0.0005, 0.0001, 0.0003, 0.0005, 0.001, 0.003], # learning rate
    'input_shape': [in_shape],                            # input shape
    'filters': [10, 25, 50, 100],                         # number of filters in first convolutional layer
    'layers': [3, 4],                                     # number of layers (convolution + pooling) 
    'drop_out': [0, 0.1, 0.2, 0.3, 0.4, 0.5],             # dropout after dense layers
    'if_dropout': [0, 0.1, 0.2],                          # parameters for spatialDropout2D, dropout after pooling layers
    'batch_size': [16, 32, 64, 128],                      # batch size
    'epochs': [100],                                      # max epochs number (early stopping is used)
    'dense_layer_sizes': [128, 256],                      # size of the first dense layer
    } 
     
    model = KerasClassifier(build_fn = make_cnn_grid_corr, verbose=1)
    logger_wrapper.logger.info('Hyperparameters matrix')
    logger_wrapper.logger.info(params)
    with open(os.path.join(exp_dir,"matrix_params.json"), "w") as json_file:
        json.dump(params, json_file)  
    
    my_callbacks = [ tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5, restore_best_weights = True), ]

    grid = RandomizedSearchCV(estimator=model, param_distributions=params, scoring = make_scorer(average_precision_score),
                        cv = cv, n_jobs=1, n_iter=int(n_iter), verbose = 3, random_state =667, refit = False) 
    grid_result = grid.fit(dX_resh, dy, validation_data=(X_val_resh, y_val.astype(np.float32)), callbacks = my_callbacks)
     
    print(grid_result.best_params_)
    
    logger_wrapper.logger.info('Random search took (in sec): ')
    logger_wrapper.logger.info(time.time() - start_time)
    
    best_params = grid_result.best_params_
    with open(os.path.join(exp_dir,"best_params.json"), "w") as json_file:
        json.dump(best_params, json_file, indent=1)    
    
        
    pd.DataFrame(grid_result.cv_results_['params']).to_pickle(os.path.join(exp_dir,"all_models.pkl")) 
   
    pd.DataFrame(grid_result.cv_results_['split0_test_score']).to_pickle(os.path.join(exp_dir,"all_models_scores.pkl") )
    
    def display_cv_results(search_results):
        print('Best score = {:.4f} using {}'.format(search_results.best_score_, search_results.best_params_))
        means = search_results.cv_results_['mean_test_score']
        stds = search_results.cv_results_['std_test_score']
        params = search_results.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print('mean test accuracy +/- std = {:.4f} +/- {:.4f} with: {}'.format(mean, stdev, param))    
    
    display_cv_results(grid_result)   
        
    logger_wrapper.logger.info('All params:')
    logger_wrapper.logger.info(params)
    logger_wrapper.logger.info('Best params:', best_params)
    logger_wrapper.logger.info( best_params)
    logger_wrapper.logger.info(grid_result.cv_results_)

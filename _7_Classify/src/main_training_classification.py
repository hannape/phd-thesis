# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import time
import keras
import sys


from sklearn.utils import class_weight

from numpy.random import seed
import tensorflow as tf
seed(1)
tf.random.set_seed(2)  

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
experimentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(experimentdir)
sys.path.insert(0,parentdir) # 

import config as cfg
from _2_TPOT_experiments.src.saving_utils import LoggerWrapper
from useful_func_classify import prepare_data_to_classify, run_train_ES
from _4_ResNets_experiments.src.useful_func_resnet import load_my_data_resnet_augment

# allocating fraction of GPU memory
from tensorflow.compat.v1.keras.backend import set_session
config2 = tf.compat.v1.ConfigProto()
config2.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.compat.v1.Session(config=config2))

"""
    Training classification modelson balanced training data. Validated on validation set.

    Running script: python script_name 
    chosen_representation  (must be 'mel-spektrogram')
    if_augmentation_used_and_which_type (-1: no augment, 1:15 augmentation types. 10 was chosen as best one) 
    architecture_classification_type (choice: leNet5', 'AlexNet', 'CNN_chapter5)
    if_class_weight (if class weighting is used during training. Default:0, no class weights)
    
    e.g. python main_training_classification.py mel-spektrogram 10 leNet5 0

"""



# %%
if __name__=='__main__':
    
    # Load configs
    dict_model_types = {'leNet5': 0, 'AlexNet':1, 'CNN_chapter5':2}

    chosen_repr = sys.argv[1]
    if_augment = int(sys.argv[2]) #10    
    print(sys.argv[3])
    print(type(sys.argv[3]))
    
    model_nr = dict_model_types[sys.argv[3]] #5
    if_class_weight = int(sys.argv[4])
    
    n_models = 2      # how many times model trained
    nb_of_classes = 6 # number of classes in classification - 6 in the examined case 
    if chosen_repr=='mel-spektrogram':
        in_shape = (60, 148, 1) 

    saving_dir = os.path.join(cfg.main_folder_path,"_7_Classify", "results","model_" + str(model_nr)+ time.strftime('-%Y-%m-%d-%H-%M'))
    os.makedirs(saving_dir, exist_ok=True)

    logger_wrapper = LoggerWrapper(saving_dir)
    sys.stderr.write = logger_wrapper.log_errors
    logger_wrapper.logger.info(f'Running {sys.argv}')    

    ## Load train and valid data
    start_time = time.time()

    if chosen_repr=="mel-spektrogram":

        df2 = pd.read_hdf(os.path.join('..','..','data','dataframe','df_valid_'+ str(chosen_repr) +'_norm.h5'),'df')
        print(if_augment)
        _, _, _, _, _, X_val, y_val, train_X, train_y, df_augmented = load_my_data_resnet_augment(chosen_repr, cfg, 'classify', if_augment) 
        in_shape = (60, 148, 1)  
      
    print(chosen_repr)
    print(np.shape(train_X), np.shape(train_y))
    print(np.shape(X_val), np.shape(y_val))
    
    ## Prepare training data to classification

    train_X_class_, train_y_class_bin, train_y_class_cat, _ =  prepare_data_to_classify(df_augmented, train_X, train_y, if_augment)
    X_train_resh = train_X_class_.reshape(train_X_class_.shape[0], in_shape[0], in_shape[1], 1).astype('float32') 
    
    my_trainY = keras.utils.to_categorical(np.floor(train_y_class_cat['bird_code'].to_numpy()).astype(int) , nb_of_classes) 
    my_trainX = X_train_resh   

    ## Prepare validation data to classification
    
    X_val_class, y_val_class_bin, y_val_class_cat, _ = prepare_data_to_classify(df2, X_val, y_val, -1)
    X_val_resh = X_val.reshape(X_val.shape[0], in_shape[0], in_shape[1], 1).astype('float32') 
    
    my_valY = keras.utils.to_categorical(np.floor(y_val_class_cat['bird_code'].to_numpy()).astype(int), nb_of_classes)
    my_valX = X_val_resh
    
    ## Training model    
    cw = []
    
    if if_class_weight ==1:
        cw = class_weight.compute_class_weight('balanced', np.unique(np.floor(train_y_class_cat['bird_code'].to_numpy()).astype(int)),
                                                np.floor(train_y_class_cat['bird_code'].to_numpy()).astype(int))
        cw = {i : cw[i] for i in range(nb_of_classes)}
        print(cw)

    for run_nr in range(0, n_models):
        run_train_ES(my_trainX, my_trainY, my_valX, my_valY, model_nr, if_augment, saving_dir, run_nr, if_class_weight, cw)       
    
    print('Models trained.')    

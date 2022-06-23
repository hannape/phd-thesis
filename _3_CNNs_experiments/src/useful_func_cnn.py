# -*- coding: utf-8 -*-

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D
from  sklearn.model_selection import PredefinedSplit


def make_cnn_grid_corr(input_shape = (63, 148, 1),  lr=0.001, filters = 10, drop_out = 0.5, layers = 3,
                  dense_layer_sizes=256, activation_function = 'relu', if_dropout = 0 ): 
  
    kernel_size = (3,3)
    pool_size = (2,2)
    
    model = Sequential()
    model.add(Conv2D(filters, kernel_size,input_shape=input_shape, activation=activation_function))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(SpatialDropout2D(if_dropout))
    
    for i in range(layers-1):
        model.add(Conv2D(filters, kernel_size, activation=activation_function))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(SpatialDropout2D(if_dropout))
       
    model.add(Flatten())
    model.add(Dense(dense_layer_sizes, activation=activation_function))
    model.add(Dropout(drop_out))
    model.add(Dense(32, activation=activation_function))
    model.add(Dropout(drop_out))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate = lr),
                  metrics=[tf.keras.metrics.AUC(curve='PR')])#  [av_precision]

    return model

def load_my_data(chosen_repr, config):
    
    df1 = pd.read_hdf(os.path.join('..','..','data','dataframe','df_train_'+ str(chosen_repr) +'_norm.h5'),'df')
    df2 = pd.read_hdf(os.path.join('..','..','data','dataframe','df_valid_'+ str(chosen_repr) +'_norm.h5'),'df')
    
    # all_repr = config.representation_1d + config.representation_2d
    # all_repr.remove(chosen_repr)
    # df2 = df2.drop(all_repr, axis=1)
    cv_split, dX, dy, di, cv = make_cv_split_cnn(df1, df2, chosen_repr)    
    print("Loaded train + valid")
    
    valid = df2.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
    train = df1.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
    valid_X = np.stack(valid[:, 2])
    valid_y = valid[:, 3]
    train_X = np.stack(train[:, 2])
    train_y = train[:, 3]
   
    return cv_split, dX, dy, di, cv, valid_X, valid_y, train_X, train_y

def make_cv_split_cnn(train, valid, chosen_repr, classifier=False):   
    
    train = train.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
    train_identifiers = train[:, 0:2]
    train_X = np.stack(train[:, 2])
    train_y = train[:, 3]
    
    valid = valid.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
    valid_identifiers = valid[:, 0:2]
    valid_X = np.stack(valid[:, 2])
    valid_y = valid[:, 3]

    dX = np.vstack((train_X, valid_X))
    print('min train+walid: ', np.amin(dX))
    dy = np.hstack((train_y, valid_y))
    di = np.vstack((train_identifiers, valid_identifiers))
    
    train_indices = np.array(range(0, train_X.shape[0]))
    val_indices = np.array(range(train_X.shape[0], dX.shape[0]))   
    cv_split = [(train_indices, val_indices)]   
    dX = np.reshape(dX, newshape=(dX.shape[0],-1))
    dy = dy.astype(int)
   
    test_fold = np.concatenate([
    np.full( train_X.shape[0],-1, dtype=np.int8),
    np.zeros(valid_X.shape[0], dtype=np.int8)
    ])
    cv = PredefinedSplit(test_fold)

    return cv_split, dX, dy, di, cv


def load_test(chosen_repr):
    import config as cfg
    from data_load import function_data_load
    from _2_TPOT_experiments.src.representations_all2 import function_representations_all2
    import numpy as np
    import time 

    if_scaler = True                   # if the scaler is used on testing data.
    nr_rec_start = 0                   # starting rec number, needed if we want to read many long recordings
    rec_batch_size = 18                # recording reading batch size
    test_new_only = 1                  # only test set is loaded 
    
    # DATA LOAD: Read data from recordings, read labels, create chunks, create testing set
    
    start_time = time.time()
    
    _,_,_, data_test_new = function_data_load(cfg.path_test1618_txt, cfg.path_train161718_txt, cfg.path_test1618_wav,\
                                              cfg.path_train161718_wav, cfg.balance_types, cfg.balance_ratios, cfg.chunk_length_ms,\
                                              cfg.chunk_overlap, cfg.calls_0, cfg.calls_1, cfg.calls_unknown, cfg.tolerance, \
                                              cfg.valid_set, cfg.test_rec_to_cut, cfg.columns_dataframe, test_new_only)
    print("--- Data load:: %s sec ---" % (time.time() - start_time))
    
    # REPRESENTATIONS: Create representations for test set
    
    start_time = time.time()
    
    file_names, indices, info_chunksy, repr_full_scaled = function_representations_all2(cfg.path_test1618_wav, data_test_new, if_scaler, \
                                                       cfg.repr_1d_summary, cfg.summary_1d, cfg.sr, cfg.chunk_length_ms, \
                                                       cfg.n_fft, cfg.win_length, cfg.hop_length, cfg.window, cfg.f_min, cfg.f_max,\
                                                       cfg.n_mels, cfg.N, cfg.step, cfg.Np, cfg.K, cfg.tm, cfg.flock, cfg.tlock,\
                                                       chosen_repr, nr_rec_start, rec_batch_size)    
        
    print("--- Data representation: %s min ---" % ((time.time() - start_time)/60))

    file_names_list, chunk_ids_list, has_bird_list, representation_list = [],[],[],[]
    
    
    for num,i in enumerate(indices[nr_rec_start:nr_rec_start+rec_batch_size]):    
      file_names_list.extend([file_names[num] for i in range(len(i))])
      chunk_ids_list.extend((info_chunksy[num][0])) 
      has_bird_list.extend((info_chunksy[num][3]))
      representation_list.extend([repr_full_scaled[num][i] for i in range(len(i))])      
      
    testing_target = has_bird_list
    
    testing_features = np.array(representation_list)    
    testing_features = np.reshape(testing_features, newshape=(testing_features.shape[0],-1))
    
    print("Test loaded")

    return testing_features, testing_target

def load_test_multitap(chosen_repr):
    import pandas as pd
    import os
   
    # datasets saved before (problems with library and python versions compatibility. Python 3.4 needed for library libtfr)
    k_range = [3,6,9,12,15]
    df_all = pd.read_hdf(os.path.join('..','..','data','dataframe','df_test_repr3_0_norm.h5'),'df')
    testy = df_all.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
    testy_identifiers_all = testy[:, 0:2]
    test_X_all = np.stack(testy[:, 2])
    test_y_all = testy[:, 3]
    
    for k in k_range:
        df_k = pd.read_hdf(os.path.join('..','..','data','df_test_repr3_'+ str(k) +'_norm.h5'),'df')
        testy = df_k.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
        testy_identifiers = testy[:, 0:2]
        test_X = np.stack(testy[:, 2])
        test_y = testy[:, 3]        
        print('stack: ', k)
        test_X_all = np.vstack((test_X_all, test_X))
        test_y_all = np.hstack((test_y_all, test_y))
        testy_identifiers_all = np.vstack((testy_identifiers_all, testy_identifiers))
    print(np.shape(test_X_all))   
    print(np.shape(test_y_all))  
    test_y_all = test_y_all.astype(int)
    print(np.shape(testy_identifiers_all))  
        
    return np.array(test_X_all), np.array(test_y_all), np.array(testy_identifiers)

def load_my_data_multitap(chosen_repr, config):

    # datasets saved before (problems with library and python versions compatibility. Python 3.4 needed for library libtfr)
    df1 = pd.read_hdf(os.path.join('..','..','data','dataframe','df_train_multitaper_norm.h5'),'df')
    df2 = pd.read_hdf(os.path.join('..','..','data','dataframe','df_valid_multitaper_norm.h5'),'df')

    cv_split, dX, dy, di, cv = make_cv_split_cnn(df1, df2, chosen_repr)    
    print("Loaded train + valid  multitaper")
    
    valid = df2.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
    train = df1.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
    valid_X = np.stack(valid[:, 2])
    valid_y = valid[:, 3]
    train_X = np.stack(train[:, 2])
    train_y = train[:, 3]
    
    return cv_split, dX, dy, di, cv, valid_X, valid_y, train_X, train_y 


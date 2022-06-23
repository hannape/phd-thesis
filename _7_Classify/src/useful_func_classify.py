# -*- coding: utf-8 -*-
    
import keras
import json
import numpy as np
import time 
import os
import tensorflow as tf
import warnings

from pandas.core.common import SettingWithCopyWarning

from tensorflow.keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, SpatialDropout2D


def define_leNet5(input_shape = (60, 148, 1)):
    model = keras.Sequential()
    
    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape= input_shape))
    model.add(AveragePooling2D()) 
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())
    
    model.add(Flatten())
    
    model.add(Dense(units=120, activation='relu')) 
    model.add(Dense(units=84, activation='relu')) 
    model.add(Dense(units=6, activation = 'softmax'))
    
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[tf.keras.metrics.AUC(curve='PR')])
    print('leNet5')

    return model

def define_CNN_chapter5(input_shape = (60, 148, 1)):
    
    dir_json = os.path.join('..','..',"_3_CNNs_experiments", "best_params.json")
    
    # best_params.json, for mel-spektrogram representation
    #     {
    #  "lr": 0.0005,
    #  "layers": 3,
    #  "input_shape": [
    #   60,
    #   148,
    #   1
    #  ],
    #  "if_dropout": 0.1,
    #  "filters": 10,
    #  "epochs": 100,
    #  "drop_out": 0.2,
    #  "dense_layer_sizes": 128,
    #  "batch_size": 16
    # }
    
    with open(dir_json, 'r') as fp:
        best_params = json.load(fp)
    
    lr = best_params['lr']
    filters = best_params['filters']
    drop_out = best_params['drop_out']
    layers = best_params['layers']
    if_dropout = best_params['if_dropout']
    dense_layer_sizes = best_params['dense_layer_sizes']
    activation_function = 'relu'
    
    kernel_size = (3,3)
    pool_size = (2,2)
    
    model = Sequential()
    model.add(Conv2D(filters, kernel_size, input_shape=input_shape, activation=activation_function))
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
    model.add(Dense(6, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate = lr),
                  metrics=[tf.keras.metrics.AUC(curve='PR')])
    print('CNN from chapter 5')
    return model


def define_AlexNet(input_shape = (60, 148, 1)):
    
    model = keras.models.Sequential()
    
    model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
    
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))) ## change from pool_size= (3,3)
    
    model.add(Flatten())
    
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[tf.keras.metrics.AUC(curve='PR')])
    print('AlexNet')
    return model

def choose_model(model_no):
    print('model ', model_no,':')
    if model_no == 0:
        model = define_leNet5() 
    if model_no == 1:    
        model = define_AlexNet()    
    if model_no == 2:    
        model = define_CNN_chapter5()
        
    return model 
     
def run_train_ES(my_trainX, my_trainY, my_valX, my_valY, model_nr=0, if_augment=-1, saving_dir ="", run_nr =0, if_class_weight = 0, cw = []):

    model = choose_model(model_nr)
    #model.fit(my_trainX, my_trainY, epochs=10, batch_size=32, verbose=0)
    my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights = True), ]
    cw_str = ''
    
    if if_class_weight==1:
        model.fit(my_trainX, my_trainY, validation_data = (my_valX, my_valY), epochs=200,
                    batch_size=32, callbacks = my_callbacks, verbose = 1, class_weight = cw)
        cw_str = '_cw'
    else:        
        model.fit(my_trainX, my_trainY, validation_data = (my_valX, my_valY), epochs=200,
                    batch_size=32, callbacks = my_callbacks, verbose = 1)
        
    hist = model.history.history['val_loss']
    n_epochs_best = np.argmin(hist)+1   
    print(n_epochs_best)
    
	# save model
    model_name_dir = os.path.join(saving_dir, "test_class_modelnr" + str(model_nr) +  "-run" + str(run_nr) 
                                  + "-aug" + str(if_augment) + "-ESepoch" + str(n_epochs_best) + cw_str + ".h5")
    model.save(model_name_dir)
    
    print("Saved model to disk")    
    

def prepare_data_to_classify(df1, _X, _y, if_augment=-1, if_drop = 1):    
    # Data preparations. Taking chunks species from has_bird=1 chunks
    # skip those which have length>1
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    df1 = df1.reset_index(drop=True)
    ind_len_over_1 = df1.index[(df1['has_bird']==1) & (df1['chunks_species'].str.len()>1)]  
    len_over_1 = df1['chunks_species'][ind_len_over_1]

    # check which have different labels
    result = []
    
    for i in ind_len_over_1:
        result.append(all(elem == len_over_1[i][0] for elem in len_over_1[i]))   
        
    result = [not elem for elem in result] 
    no_duplicates = len_over_1[result]
    
    # and if it's not the same species, but sure, and not sure labelling (e.g. s and s?) 
    # or one of them is 't' or '???' label
    result2, result3 = [], []
    
    for i in no_duplicates.index:
        result2.append([elem=='???' or elem=='t' for elem in len_over_1[i]])
        
    for k in range(0,len(result2)):
        result3.append(all(element == result2[k][0] for element in result2[k]))
   
    if if_drop==0:
        inds_to_drop=[]
        df_class = df1
        train_X_class = _X
    else:        
        inds_to_drop = no_duplicates[result3]
        df_class = df1.drop(index = inds_to_drop.index)
        train_X_class = np.delete(_X, inds_to_drop.index, axis = 0)
        
    # print('chunks dropped: ', len(inds_to_drop))  

    # hot-encoding for six classes: s (songthrush), k (blackbird), d (redwing), r (robin), ni/in (non-identified/others), no birds
    ind_len_over_1 = df_class.index[(df_class['has_bird']==1)]
    df_species = df_class[(df_class['has_bird']==1)]
    df_class['bird_code'] = 5.0
    for k in df_class.index:
        if df_class['has_bird'][k]==1:
            df_class['bird_code'][k] =6

    birds_codes = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, -1 ]  # adding 0.5 to labels which are 'not sure'
    birds_spec = ['s', 's?', 'k', 'k?', 'd', 'd?', 'r', 'r?', 'ni', '']
    zip_iterator = zip(birds_spec, birds_codes)
    bird_dict = dict(zip_iterator)
   
    for i in df_species.index:
        
        if len(df_class['chunks_species'][i])> 0:
            df_class['bird_code'][i] = bird_dict.get(df_class['chunks_species'][i][0])
   
            if np.isnan(df_class['bird_code'][i]): # nie zgadza się klucz
                
                if len(df_class['chunks_species'][i])==1:  # głosy innych ptaków
                    df_class['bird_code'][i] = 4
                else:
                    for s in range(1,len(df_class['chunks_species'][i])):
                            df_class['bird_code'][i] = bird_dict.get(df_class['chunks_species'][i][s])
                            if not np.isnan(df_class['bird_code'][i]):
                                break
                    if np.isnan(df_class['bird_code'][i]):
                        df_class['bird_code'][i] = 4   
        else:
            df_class['bird_code'][i]= -1

    inds_to_drop2 = []
    for i in df_class.index:
        if df_class['bird_code'][i]== -1:
            inds_to_drop2.append(i)
            
    #print('chunks dropped: ', len(inds_to_drop2))       
    if if_drop == 0:
        inds_to_drop2=[]  
          
    df_train_class =  df_class.drop(index = inds_to_drop2)
    
    X_class = np.delete(train_X_class, inds_to_drop2, axis = 0)
    y_class_bin = df_train_class['has_bird'].reset_index() 
    y_class_cat = df_train_class['bird_code'].reset_index()   
    
    return X_class, y_class_bin, y_class_cat, df_train_class


def load_test(chosen_repr):

  
    import config as cfg
    from data_load import function_data_load
    from _2_TPOT_experiments.src.representations_all2 import function_representations_all2
    import numpy as np
    
    if_scaler = True                   # if the scaler is used on testing data.
    nr_rec_start = 0                   # starting rec number, needed if we want to read many long recordings
    rec_batch_size = 18                # recording reading batch size
    test_new_only = 1                  # only test set is loaded 
    
    # DATA LOAD: Read data from recordings, read labels, create chunks, create testing set
    
    start_time = time.time()
    _,_,_, data_test_new = function_data_load(cfg.path_test1618_txt, cfg.path_train161718_txt, cfg.path_test1618_wav,\
                                              cfg.path_train161718_wav, cfg.balance_types, cfg.balance_ratios, cfg.chunk_length_ms,\
                                              cfg.chunk_overlap, cfg.calls_0, cfg.calls_1, cfg.calls_unknown, cfg.tolerance, cfg.valid_set,\
                                              cfg.test_rec_to_cut, cfg.columns_dataframe, test_new_only)
    print("--- Data load:: %s sec ---" % (time.time() - start_time))
    
    # REPRESENTATIONS: Create representations for test set
    
    start_time = time.time()
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        
    file_names, indices, info_chunksy, repr_full_scaled = function_representations_all2(cfg.path_test1618_wav, data_test_new, if_scaler, \
                                                       cfg.repr_1d_summary, cfg.summary_1d, cfg.sr, cfg.chunk_length_ms, \
                                                       cfg.n_fft, cfg.win_length, cfg.hop_length, cfg.window, cfg.f_min, cfg.f_max,\
                                                       cfg.n_mels, cfg.N, cfg.step, cfg.Np, cfg.K, cfg.tm, cfg.flock, cfg.tlock,\
                                                       chosen_repr, nr_rec_start, rec_batch_size)    
        
    print("--- Data representation: %s min ---" % ((time.time() - start_time)/60))
    
    return file_names, indices, info_chunksy, repr_full_scaled
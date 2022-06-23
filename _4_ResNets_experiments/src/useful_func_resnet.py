# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from  sklearn.model_selection import PredefinedSplit
from torchvision import models
import torch.nn as nn


def load_my_data_resnet_augment(chosen_repr, config, set_type, if_augment = -1):

    if set_type == 'wlasc' or set_type == 'classify':
        print("Main dataset")
        df1 = pd.read_hdf(os.path.join('..','..','data','dataframe','df_train_'+ str(chosen_repr) +'_norm.h5'),'df')
    ### augmentation part
        if if_augment!=-1:
            if if_augment<15 and if_augment!=5 and if_augment!=10:
                augment_type_names = ['timeWarp','freqMask','timeMask','allThree','mix','',
                                      'timeShift', 'pitchShift','timePitchShift', 'timePitchShiftMix','',
                                      'gaussianNoise','backgroundNoise','noiseNoise', 'noiseNoise1015', '']
                df = pd.read_hdf(os.path.join('..','..','data','augment','df_train_mel-spektrogram_'+ str(augment_type_names[if_augment]) +'.h5'),'augment')
                print('Loaded augmented '+ str(augment_type_names[if_augment]))
                train_aug = df.loc[:,['rec_name', 'chunk_ids', 'aug_specs', 'has_bird']].values
                print(np.shape(train_aug[:, 2]))
                print(type(train_aug[:, 2]))
                train_X_aug = np.stack(train_aug[:, 2])
                train_y_aug = train_aug[:, 3]
                print('aug train ', np.shape(train_X_aug), np.shape(train_y_aug))
            if if_augment==5:  ## read all 5 augmented versions and stack them
                dfA0 = pd.read_hdf(os.path.join('..','..','data','augment','df_train_mel-spektrogram_timeWarp.h5'),'augment')
                dfA1 = pd.read_hdf(os.path.join('..','..','data','augment','df_train_mel-spektrogram_freqMask.h5'),'augment')
                dfA2 = pd.read_hdf(os.path.join('..','..','data','augment','df_train_mel-spektrogram_timeMask.h5'),'augment')
                dfA3 = pd.read_hdf(os.path.join('..','..','data','augment','df_train_mel-spektrogram_allThree.h5'),'augment')
                print('Loaded augmentations x 4')
                train_aug0 = dfA0.loc[:,['rec_name', 'chunk_ids', 'aug_specs', 'has_bird']].values
                train_X_aug0 = np.stack(train_aug0[:, 2])
                train_aug1 = dfA1.loc[:,['rec_name', 'chunk_ids', 'aug_specs', 'has_bird']].values
                train_X_aug1 = np.stack(train_aug1[:, 2])
                train_aug2 = dfA2.loc[:,['rec_name', 'chunk_ids', 'aug_specs', 'has_bird']].values
                train_X_aug2 = np.stack(train_aug2[:, 2])
                train_aug3 = dfA3.loc[:,['rec_name', 'chunk_ids', 'aug_specs', 'has_bird']].values
                train_X_aug3 = np.stack(train_aug3[:, 2])
                train_X_aug_all = np.vstack((train_X_aug0, train_X_aug1, train_X_aug2, train_X_aug3))
                train_y_aug_all = np.hstack((train_aug0[:, 3], train_aug1[:, 3], train_aug2[:, 3], train_aug3[:, 3]))
                print('aug train ', np.shape(train_X_aug_all), np.shape(train_y_aug_all))
                
            if if_augment==10: ## read 3 augmented versions and stack them
                dfA0 = pd.read_hdf(os.path.join('..','..','data','augment','df_train_mel-spektrogram_timeShift.h5'),'augment')
                dfA1 = pd.read_hdf(os.path.join('..','..','data','augment','df_train_mel-spektrogram_pitchShift.h5'),'augment')
                dfA2 = pd.read_hdf(os.path.join('..','..','data','augment','df_train_mel-spektrogram_timePitchShift.h5'),'augment')
                print('Loaded augmentations x 3')
                train_aug0 = dfA0.loc[:,['rec_name', 'chunk_ids', 'aug_specs', 'has_bird']].values
                train_X_aug0 = np.stack(train_aug0[:, 2])
                train_aug1 = dfA1.loc[:,['rec_name', 'chunk_ids', 'aug_specs', 'has_bird']].values
                train_X_aug1 = np.stack(train_aug1[:, 2])
                train_aug2 = dfA2.loc[:,['rec_name', 'chunk_ids', 'aug_specs', 'has_bird']].values
                train_X_aug2 = np.stack(train_aug2[:, 2])
                train_X_aug_all = np.vstack((train_X_aug0, train_X_aug1, train_X_aug2))
                train_y_aug_all = np.hstack((train_aug0[:, 3], train_aug1[:, 3], train_aug2[:, 3]))
                print('aug train ', np.shape(train_X_aug_all), np.shape(train_y_aug_all))
                
            if if_augment==15:  ## read 6 best augmented versions and stack them
                dfA0 = pd.read_hdf(os.path.join('..','..','data','augment','df_train_mel-spektrogram_freqMask.h5'),'augment')
                dfA1 = pd.read_hdf(os.path.join('..','..','data','augment','df_train_mel-spektrogram_allThree.h5'),'augment')
                dfA2 = pd.read_hdf(os.path.join('..','..','data','augment','df_train_mel-spektrogram_timeShift.h5'),'augment')
                dfA3 = pd.read_hdf(os.path.join('..','..','data','augment','df_train_mel-spektrogram_pitchShift.h5'),'augment')
                dfA4 = pd.read_hdf(os.path.join('..','..','data','augment','df_train_mel-spektrogram_timePitchShift.h5'),'augment')
                dfA5 = pd.read_hdf(os.path.join('..','..','data','augment','df_train_mel-spektrogram_noiseNoise.h5'),'augment')
                print('Loaded augmentations x 6')
                train_aug0 = dfA0.loc[:,['rec_name', 'chunk_ids', 'aug_specs', 'has_bird']].values
                train_X_aug0 = np.stack(train_aug0[:, 2])
                train_aug1 = dfA1.loc[:,['rec_name', 'chunk_ids', 'aug_specs', 'has_bird']].values
                train_X_aug1 = np.stack(train_aug1[:, 2])
                train_aug2 = dfA2.loc[:,['rec_name', 'chunk_ids', 'aug_specs', 'has_bird']].values
                train_X_aug2 = np.stack(train_aug2[:, 2])
                train_aug3 = dfA3.loc[:,['rec_name', 'chunk_ids', 'aug_specs', 'has_bird']].values
                train_X_aug3 = np.stack(train_aug3[:, 2])
                train_aug4 = dfA4.loc[:,['rec_name', 'chunk_ids', 'aug_specs', 'has_bird']].values
                train_X_aug4 = np.stack(train_aug4[:, 2])
                train_aug5 = dfA5.loc[:,['rec_name', 'chunk_ids', 'aug_specs', 'has_bird']].values
                train_X_aug5 = np.stack(train_aug5[:, 2])
                
                train_X_aug_all = np.vstack((train_X_aug0, train_X_aug1, train_X_aug2, train_X_aug3, train_X_aug4, train_X_aug5))
                train_y_aug_all = np.hstack((train_aug0[:, 3], train_aug1[:, 3], train_aug2[:, 3], train_aug3[:, 3], train_aug4[:, 3], train_aug5[:, 3]))
                print('aug train ', np.shape(train_X_aug_all), np.shape(train_y_aug_all))
    ###
        df2 = pd.read_hdf(os.path.join('..','..','data','dataframe','df_valid_'+ str(chosen_repr) +'_norm.h5'),'df')
        cv_split, dX, dy, di, cv = make_cv_split_cnn(df1, df2, chosen_repr)    
        
        valid = df2.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
        train = df1.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
        valid_X = np.stack(valid[:, 2])
        valid_y = valid[:, 3]
        train_X = np.stack(train[:, 2])
        train_y = train[:, 3]
        print('normal train ', np.shape(train_X), np.shape(train_y))
        
        if if_augment!=-1:
            if if_augment<15 and if_augment!=5 and if_augment!=10:
                train_X = np.vstack((train_X, train_X_aug))
                train_y = np.hstack((train_y, train_y_aug))
            if if_augment==5 or if_augment==10 or if_augment==15:
                train_X = np.vstack((train_X, train_X_aug_all))
                train_y = np.hstack((train_y, train_y_aug_all))
    
        
    # df3 = pd.read_hdf(os.path.join('..','..','..','jupyter','data','df_test_3rec_'+ str(chosen_repr) +'_norm.h5'),'df')
    # dX_test, dy_test, di_test = make_test(df3, chosen_repr)
    if set_type == 'classify':
        if if_augment==10:
            frames = [df1, dfA0, dfA1, dfA2]
            df_augmented = pd.concat(frames)
            
            return cv_split, dX, dy, di, cv, valid_X, valid_y, train_X, train_y, df_augmented
    else:
        return cv_split, dX, dy, di, cv, valid_X, valid_y, train_X, train_y


def load_my_data_resnet_multitap(chosen_repr, config, set_type):
    
    if set_type == 'wlasc':
        print("Main dataset")
        df1 = pd.read_hdf(os.path.join('..','..','data','dataframe','df_train_multitaper_norm.h5'),'df')
        df2 = pd.read_hdf(os.path.join('..','..','data','dataframe','df_valid_multitaper_norm.h5'),'df')

        cv_split, dX, dy, di, cv = make_cv_split_cnn(df1, df2, chosen_repr)    
        print("Loaded train+valid multitaper")
        
        valid = df2.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
        train = df1.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
        #valid_identifiers = valid[:, 0:2]
        valid_X = np.stack(valid[:, 2])
        valid_y = valid[:, 3]
        train_X = np.stack(train[:, 2])
        train_y = train[:, 3]
  
    return cv_split, dX, dy, di, cv, valid_X, valid_y, train_X, train_y


def make_test(test, chosen_repr):  
    test = test.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
    test_identifiers = test[:, 0:2]
    test_X = np.stack(test[:, 2])
    test_y = test[:, 3]
    dX = np.reshape(test_X, newshape=(test_X.shape[0],-1))
    dy = test_y.astype(int)
    
    return dX, dy, test_identifiers

def make_cv_split_cnn(train, valid, chosen_repr, classifier=False):   
    
    # 
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
    # The training data
    np.full( train_X.shape[0],-1, dtype=np.int8),
    # The development data
    np.zeros(valid_X.shape[0], dtype=np.int8)
    ])
    cv = PredefinedSplit(test_fold)

    print(test_fold)
    return cv_split, dX, dy, di, cv

def load_test(chosen_repr):

    import time 
    
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
    
    
    file_names_list, chunk_ids_list, has_bird_list, representation_list = [],[],[],[]
    
    
    for num,i in enumerate(indices[nr_rec_start:nr_rec_start+rec_batch_size]):  
      file_names_list.extend([file_names[num] for i in range(len(i))])
      chunk_ids_list.extend((info_chunksy[num][0]))  
      has_bird_list.extend((info_chunksy[num][3]))
      representation_list.extend([repr_full_scaled[num][i] for i in range(len(i))])
     
    testing_features = np.array(representation_list) 
    testing_target = has_bird_list
    
    testing_features = np.reshape(testing_features, newshape=(testing_features.shape[0],-1))
    
    print("Test loaded")

    return testing_features, testing_target 

def load_tests_multitap(chosen_repr):
    import pandas as pd
    import os

    k_range = [3,6,9,12,15]
    df_all = pd.read_hdf(os.path.join('..','..','data','dataframe','df_test_repr3_0_norm.h5'),'df')
    testy = df_all.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
    testy_identifiers_all = testy[:, 0:2]
    test_X_all = np.stack(testy[:, 2])
    test_y_all = testy[:, 3]
    
    for k in k_range:
        df_k = pd.read_hdf(os.path.join('..','..','data','dataframe','df_test_repr3_'+ str(k) +'_norm.h5'),'df')
        testy = df_k.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
        testy_identifiers = testy[:, 0:2]
        test_X = np.stack(testy[:, 2])
        test_y = testy[:, 3]        
        test_X_all = np.vstack((test_X_all, test_X))
        test_y_all = np.hstack((test_y_all, test_y))
        testy_identifiers_all = np.vstack((testy_identifiers_all, testy_identifiers))
    print(np.shape(test_X_all))   
    print(np.shape(test_y_all))  
    test_y_all = test_y_all.astype(int)
    print(np.shape(testy_identifiers_all))  
        
    return np.array(test_X_all), np.array(test_y_all), np.array(testy_identifiers)

### RreNets Transfer Learning Scenarios:

######### OPCJA 1 - ResNet-18 architecture, no weight initialisation
def opcja1():
    model_arch = models.resnet18(pretrained=False)
    num_ftrs = model_arch.fc.in_features  # fc- fully connected, input features
    model_arch.fc = nn.Linear(num_ftrs, 2)
    return model_arch

######### OPCJA 2 - ResNet-18, Imagenet initialisation, learning from scratch
def opcja2():
    model_ft = models.resnet18(pretrained=True)

    num_ftrs = model_ft.fc.in_features  
    model_ft.fc = nn.Linear(num_ftrs, 2)

    return model_ft

######### OPCJA 3 - ResNet-18, Imagenet initialisation, freezing conv1 and layer1 
def opcja3():
    feature_extract_first = models.resnet18(pretrained=True)
    
    for name, child in feature_extract_first.named_children():
        if name in ['layer2','layer3', 'layer4']:
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False
    
    num_ftrs = feature_extract_first.fc.in_features
    feature_extract_first.fc = nn.Linear(num_ftrs, 2)
    return feature_extract_first

######### OPCJA 4 - ResNet-18, Imagenet initialisation and freezing all but last two layers, which will be trained 
def opcja4():
    feature_extract_few = models.resnet18(pretrained=True)
    
    for name, child in feature_extract_few.named_children():
        if name in ['layer3', 'layer4']:
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False
    
    num_ftrs = feature_extract_few.fc.in_features
    feature_extract_few.fc = nn.Linear(num_ftrs, 2)
    return feature_extract_few

######### OPCJA 5 - ResNet-18, Imagenet initialisation and freezing all weights but last layer, which will be trained 
def opcja5():
    feature_extract_last = models.resnet18(pretrained=True)
    for param in feature_extract_last.parameters():
        param.requires_grad = False
    
    num_ftrs = feature_extract_last.fc.in_features
    feature_extract_last.fc = nn.Linear(num_ftrs, 2)
    return feature_extract_last


######### OPCJA 6 - ResNet-50 architecture, no weight initialisation
def opcja6():
    model_arch = models.resnet50(pretrained=False)
    num_ftrs = model_arch.fc.in_features  
    model_arch.fc = nn.Linear(num_ftrs, 2)
    return model_arch

######### OPCJA 7 - ResNet-50, Imagenet initialisation, learning from scratch
def opcja7():
    model_ft50 = models.resnet50(pretrained=True)
    num_ftrs = model_ft50.fc.in_features  
    model_ft50.fc = nn.Linear(num_ftrs, 2)
    return model_ft50

######### OPCJA 8 - ResNet50, Imagenet initialisation, freezing conv1 and layer1 
def opcja8():
    feature_extract_first = models.resnet50(pretrained=True)
    
    for name, child in feature_extract_first.named_children():
        if name in ['layer2','layer3', 'layer4']:
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False
    
    num_ftrs = feature_extract_first.fc.in_features
    feature_extract_first.fc = nn.Linear(num_ftrs, 2)
    return feature_extract_first

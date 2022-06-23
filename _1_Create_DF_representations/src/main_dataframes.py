# -*- coding: utf-8 -*-

# IMPORTS

import os

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import joblib
#import libtfr
from sklearn.preprocessing import MinMaxScaler

import config as cfg
from data_load import function_data_load  
from representations_all import function_representations_all # _1_Create_DF_representations.src.

''' 
    Creating dataframe of representations, for one set type:
        set type: train, valid
        representations: '8_classic', '8_classic_plus_MIR', 'spektrogram', 'mel-spektrogram'
    Calculating and applying min-max scalers for each representations
    Saving dataframes as .h5 files (test not saved as a df - too much data)

'''

########## %% Parameters to set

chosen_set = 'valid' ## chosen set to create DF with representation, train or valid

##########

# %% DATA LOAD: Read data from recordings, read labels, create chunks, create (balanced or not) sets

start_time = time.time()

data_train, data_valid, data_test_old, data_test_new = function_data_load(cfg.path_test1618_txt, cfg.path_train161718_txt, cfg.path_test1618_wav,\
                                                                          cfg.path_train161718_wav, cfg.balance_types, cfg.balance_ratios, cfg.chunk_length_ms,\
                                                                          cfg.chunk_overlap, cfg.calls_0, cfg.calls_1, cfg.calls_unknown, cfg.tolerance, \
                                                                          cfg.valid_set, cfg.test_rec_to_cut, cfg.columns_dataframe)
print("--- Data load: %s sec ---" % (time.time() - start_time))


# %% REPRESENTATIONS: Create representations for chosen set 


if chosen_set =='train':
    chosen_set = data_train
    set_type_i = 0
    chosen_set_path = cfg.path_train161718_wav
if chosen_set =='valid':
    chosen_set = data_valid
    set_type_i = 1
    chosen_set_path = cfg.path_train161718_wav

# Scaling representations
if_scaler = False   # Min-Max scaled from the beginning or not 
first_pass = True
loop_count = 0

while (if_scaler or first_pass) and loop_count<2:

    first_pass = False   
    
    start_time = time.time()
    
    file_names, indices, info_chunksy, repr_full1, repr_full2, \
    repr_full4, repr_full5 = function_representations_all(chosen_set_path, chosen_set, if_scaler, \
                                                       cfg.repr_1d_summary, cfg.summary_1d, cfg.sr, cfg.chunk_length_ms, \
                                                       cfg.n_fft, cfg.win_length, cfg.hop_length, cfg.window, cfg.f_min, cfg.f_max,\
                                                       cfg.n_mels, cfg.N, cfg.step, cfg.Np, cfg.K, cfg.tm, cfg.flock, cfg.tlock)
    # repr_full3 
    print("--- Data representations: %s min ---" % ((time.time() - start_time)/60))
    
    #  DATAFRAME: All representations combined together in a dataframe
    
    file_names_list, chunk_ids_list = [],[]
    chunk_start_list, chunk_end_list, has_bird_list = [],[],[]
    chunks_species_list, call_id_list, has_unknown_list, has_noise_list = [],[],[],[]
    repr_name_list, representation1_list, norm_type_list = [],[],[]
    representation2_list,representation3_list,representation4_list,representation5_list = [],[],[],[]
    
    for num,i in enumerate(indices):
      
      file_names_list.extend([file_names[num] for i in range(len(i))])
      chunk_ids_list.extend((info_chunksy[num][0]).tolist()) 
      chunk_start_list.extend((info_chunksy[num][1]).tolist())
      chunk_end_list.extend((info_chunksy[num][2]).tolist())
      has_bird_list.extend((info_chunksy[num][3]).tolist())
      chunks_species_list.extend((info_chunksy[num][4]).tolist())
      call_id_list.extend(np.array((info_chunksy[num][5])).tolist()) 
      has_unknown_list.extend(np.array((info_chunksy[num][6])).tolist())
      has_noise_list.extend((info_chunksy[num][7]).tolist())
      representation1_list.extend([repr_full1[num][i] for i in range(len(i))]) 
      representation2_list.extend([repr_full2[num][i] for i in range(len(i))]) 
      #representation3_list.extend([repr_full3[num][i] for i in range(len(i))])
      representation4_list.extend([repr_full4[num][i] for i in range(len(i))])
      representation5_list.extend([repr_full5[num][i] for i in range(len(i))])
    
    data_df = [file_names_list, chunk_ids_list, chunk_start_list, chunk_end_list, has_bird_list, chunks_species_list, call_id_list, has_unknown_list, has_noise_list,\
               representation1_list, representation2_list,  representation4_list, representation5_list]  # representation3_list,
    
    data_df = np.array(data_df).T.tolist() 
    
    columns_df = ['rec_name','chunk_ids', 'chunk_start', 'chunk_end', 'has_bird', \
                  'chunks_species', 'call_id', 'has_unknown', 'has_noise',\
                  'spektrogram','mel-spektrogram','8_classic','8_classic_plus_MIR']  
    df = pd.DataFrame(data = data_df, columns =columns_df)
    
    #  Exemplary plots of 4 representations 
    
    ii = 52  # exemplary chunk with bird
    repr_type = ['spektrogram' , 'mel-spektrogram','8_classic', '8_classic_plus_MIR']
    
    print('Has bird? ',(df['has_bird'][ii]))
    
    for r in repr_type:
        
        plt.figure(figsize=(6,2))
        plt.imshow( (df[r][ii]) )
        plt.gca().invert_yaxis(),
        plt.colorbar()
        plt.show()
    
    #  Saving dataframes for further usage
    
    set_type = ['train','valid','test']
     
    
    if_norm = ''
    if if_scaler == False: 
        if_norm = "_no_normalization.h5"
    else:
        if_norm = '_norm.h5' 
        loop_count = 2
    
    data_path = os.path.join(cfg.main_folder_path,'data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    dataframe_path = os.path.join(data_path,'dataframe')
    if not os.path.exists(dataframe_path):
        os.makedirs(dataframe_path)
    
    for r in range(0,len(repr_type)):
    
        df_name = 'df_' + set_type[set_type_i] + '_'+ repr_type[r] + if_norm
        print(df_name)
        df_1repr = df.drop(columns= [x for x in repr_type if x != repr_type[r]] )
        df_1repr.to_hdf(os.path.join(dataframe_path,df_name), key='df', mode='w')
    
    # reading saved dataframe:
    #df_train_chosen_repr = pd.read_hdf(('df_train_8_classic_no_normalization.h5'),'df')
    
    loop_count = loop_count + 1
    
    #  Creating scalers for training data
    if if_scaler==False:
        
        reprs = [repr_full1, repr_full2, repr_full4, repr_full5]
        # 8 classic and 8_classic_plus_MIR parameters - scaling by columns. Parameters have different order of magnitude.
        # mel_spectrogram, spectrogram, multitaper - one min and max value in scalers, per training set
        
        scaler = MinMaxScaler()
        
        for ind,rep in enumerate(repr_type):
            print(rep)
            scaler_filename = "scaler_" + rep
            if rep =='8_classic' or rep =='8_classic_plus_MIR':
                if rep =='8_classic':
                    my_repr = (np.vstack(np.asarray(reprs[ind]))).reshape(-1,8)
                else:
                    my_repr = (np.vstack(np.asarray(reprs[ind]))).reshape(-1,39)  
                scaler.fit_transform(my_repr)
            else:
                scaler.fit_transform((np.vstack(np.asarray(reprs[ind]))).reshape(-1,1))
                
            print(scaler)
            joblib.dump(scaler, scaler_filename)
            
        if_scaler = True  
        # And now, as scalers are ready, the process of creating and saving dataframes 
        # (parts REPRESENTATION and DATAFRAME) may be repeated with variable if_scaler = True 
    
    #  Exemplary plots for 8_classic represenation, after scaling features
    
    for r in repr_type:
        scaler = joblib.load('scaler_'+ r)
        plt.figure(figsize=(6,2))
        plt.imshow(scaler.transform(df[r][ii]) )
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.show()

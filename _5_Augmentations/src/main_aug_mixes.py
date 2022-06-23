# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys
import numpy as np
import random
import librosa 
import joblib

from audiomentations import Compose 
from functions_my_audioment import my_melspektrogram 
from functions_my_audioment import AddBackgroundNoiseMy, AddGaussianSNRMy

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
experimentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(experimentdir)
sys.path.insert(0,parentdir) # 

from config import  n_fft, win_length, hop_length, window, sr, f_min, f_max, n_mels, chunk_length_ms, main_folder_path, path_train161718_wav

'''
   Creating augmented training datasets for mixes: noise and 2 samples mix
   11-13 augmentation types:
       11 - adding AWGN noise
       12 - mix with random negative chunks
       13 - mix with random negative chunks with labelled disturbances/noises
   (in phd thesis: numeration+1)
   
   only for mel-spektrogram
   Running script: python script_name representation_type augmentation_type
    e.g. python main_aug_mixes.py mel-spektrogram 11
'''

df_orig = pd.read_hdf(os.path.join('..','..','data','dataframe','df_train_mel-spektrogram_norm.h5'),'df')
path_train_wav = path_train161718_wav
path_scalers = os.path.join('..','..','_1_Create_DF_representations','scaler')
aug_type_names = ['gaussianNoise', 'backgroundNoise', 'noiseNoise1015']
scaler_mel = joblib.load(os.path.join(path_scalers,'scaler_mel_spektrogram')) 

augment_path = os.path.join(main_folder_path,'data', 'augment')
try:
    os.makedirs(augment_path)
except FileExistsError:
    pass

if __name__=='__main__':
    
    chosen_repr = sys.argv[1]   # mel-spektrogram
    aug_type = int(sys.argv[2]) # 11,12,13
    
    bg_chunks_list = df_orig.index[df_orig['has_bird']==0].tolist() 
    noise_chunks_list = df_orig[(df_orig.has_bird != 1) & (df_orig.has_noise != 0)].index.tolist()

    if aug_type==11:
        augmentG = Compose([ AddGaussianSNRMy(min_snr_in_db=5, max_snr_in_db=10, p=1) ])
       
    aug_types, masks, params, specs = [], [], [], []
    
    for i in range(0,np.shape(df_orig)[0]):
        
        samples, sr = librosa.load(os.path.join(path_train_wav, df_orig['rec_name'][i]), 
                                   sr = sr, offset = df_orig['chunk_start'][i]/sr, duration = chunk_length_ms/1000)
        s = i + 1
        if i%3000==0:
            print('sample nr:',i)  
            
        param = np.zeros([5])
        
        if aug_type==11:
            random.seed(s)
            augmented_samples, param[0] = augmentG(samples=samples, sample_rate=sr)
            
        elif aug_type==12:
            random.seed(s)
            param[1] = random.choice(bg_chunks_list)
            augmentB1 = Compose([ AddBackgroundNoiseMy(min_snr_in_db=5, max_snr_in_db=10,
                                                       p=1, i_bg_chunk = param[1], df_orig = df_orig) ])
            augmented_samples, param[2] = augmentB1(samples=samples, sample_rate=sr)
            
        elif aug_type==13:
            random.seed(s)
            param[3] = random.choice(noise_chunks_list)
            augmentB2 = Compose([ AddBackgroundNoiseMy(min_snr_in_db=10, max_snr_in_db=15,
                                                       p=1, i_bg_chunk = param[3], df_orig = df_orig) ])
            augmented_samples, param[4] = augmentB2(samples=samples, sample_rate=sr)
        
        spec = my_melspektrogram(augmented_samples, n_fft, win_length, hop_length, window, sr, f_min, f_max, n_mels, scaler = scaler_mel)
        aug_types.append(aug_type)
        params.append(param)
        specs.append(spec)
       

    df_orig['aug_specs'] = specs
    df_orig['aug_types'] = aug_types
    df_orig['params'] = params
       
    df_orig.drop(chosen_repr, inplace=True, axis=1)
    name_df = 'df_train_'+str(chosen_repr)+'_' + aug_type_names[aug_type-11] +'.h5'
    df_orig.to_hdf(os.path.join(augment_path, name_df), key='augment', mode='w')
    print("done")
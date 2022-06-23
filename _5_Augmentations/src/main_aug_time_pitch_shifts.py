# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys
import numpy as np
import random
import librosa 
import joblib

from audiomentations import Compose
from functions_my_audioment import PitchShiftMy, ShiftMy, my_melspektrogram

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
experimentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(experimentdir)
sys.path.insert(0,parentdir) # 

from config import n_fft, win_length, hop_length, window, sr, f_min, f_max, n_mels, chunk_length_ms, main_folder_path, path_train161718_wav


'''
   Creating augmented training datasets for time and/or pitch shifts.
   6-9 augmentation types:
       6 - time shift
       7 - pitch shift
       8 - time & pitch shift
       9 - mix of all above (random choice of augmentation type per sample)
   (in phd thesis: numeration+1)
   
   only for mel-spektrogram
   Running script: python script_name representation_type augmentation_type
    e.g. python main_aug_time_pitch_shifts.py mel-spektrogram 7
'''

#%% 

df_orig = pd.read_hdf(os.path.join('..','..','data','dataframe','df_train_mel-spektrogram_norm.h5'),'df')
path_train_wav = path_train161718_wav
path_scalers = os.path.join('..','..','_1_Create_DF_representations','scaler')


if __name__=='__main__':
        
    chosen_repr = sys.argv[1]
    aug_type_console = int(sys.argv[2]) # 6,7,8,9
    
    aug_type_names = ['timeShift', 'pitchShift','timePitchShift', 'timePitchShiftMix']
    scaler_mel = joblib.load(os.path.join(path_scalers,'scaler_mel_spektrogram')) 
    
    augmentT = Compose([ ShiftMy(min_fraction=-0.5, max_fraction=0.5, p=1)])   
    augmentP = Compose([ PitchShiftMy(min_semitones=-0.5, max_semitones=0.5, p=1)])   
    augment1 = Compose([ ShiftMy(min_fraction=-0.5, max_fraction=0.5, p=1)])
    augment2 = Compose([ PitchShiftMy(min_semitones=-0.5, max_semitones=0.5, p=1)])
       
    aug_types, masks, params, specs = [], [], [], []
    
    augment_path = os.path.join(main_folder_path,'data', 'augment')
    try:
        os.makedirs(augment_path)
    except FileExistsError:
        pass
    
    for i in range(0,np.shape(df_orig)[0]):
        
        samples, sr = librosa.load(os.path.join(path_train_wav, df_orig['rec_name'][i]), 
                                   sr = sr, offset = df_orig['chunk_start'][i]/sr, duration = chunk_length_ms/1000)
        s = i + 1

        param = np.zeros([4])
        
        if aug_type_console==9:
            random.seed(s)
            aug_type = random.randint(6,8)

        else:
            aug_type = aug_type_console
            
            
        if aug_type==6:
            random.seed(s)
            augmented_samples, param[0] = augmentT(samples=samples, sample_rate=sr)
            
        elif aug_type==7:
            random.seed(s)
            augmented_samples, param[1] = augmentP(samples=samples, sample_rate=sr)
            
        elif aug_type==8:   
            random.seed(s)
            augmented_samples_temp, param[0] = augment1(samples=samples, sample_rate=sr)
            random.seed(s)
            augmented_samples, param[1] = augment2(samples=augmented_samples_temp, sample_rate=sr)
            
        spec = my_melspektrogram(augmented_samples, n_fft, win_length, hop_length, window, sr, f_min, f_max, n_mels, scaler = scaler_mel)
        aug_types.append(aug_type)
        params.append(param)
        specs.append(spec)
        if i%3000==0:
            print('sample nr:',i)          
        

    df_orig['aug_specs'] = specs
    df_orig['aug_types'] = aug_types
    df_orig['params'] = params

    df_orig.drop(chosen_repr, inplace=True, axis=1)
    name_df = 'df_train_' + str(chosen_repr) + '_' + aug_type_names[aug_type_console-6] +'.h5'
    df_orig.to_hdf(os.path.join(augment_path, name_df), key='augment', mode='w')
    print("done")


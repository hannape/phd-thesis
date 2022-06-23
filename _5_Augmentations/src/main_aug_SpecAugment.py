# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys
import torch
import numpy as np
import random

from functions_SpecAugment import test_time_warp, test_freq_mask, test_time_mask   

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
experimentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(experimentdir)
sys.path.insert(0,parentdir) # 

import config as cfg

'''
   Creating augmented training datasets for SpecAugment and derived augmentations.
   0-4 augmentation types:
       0 - time warping
       1 - frequency masking
       2 - time masking
       3 - SpecAugment, all 3 above combined
       4 - mix of all above (random choice of augmentation type per sample)
   (in phd thesis: numeration+1)
   
   only for mel-spektrogram
   Running script: python script_name representation_type W_parameter F_parameter T_parameter if_show_figures augmentation_type
    e.g. python main_aug_SpecAugment.py mel-spektrogram 3 3 3 0 4
'''

#%% https://github.com/zcaceres/spec_augment/blob/master/SpecAugment.ipynb (MIT license) - modified code from spec_augment implementation. 

if __name__=='__main__':
    
    chosen_repr = sys.argv[1]       # should be mel-spectrogram, SpecAugment created for this representation
    W = int(sys.argv[2])            # time warp parameter W 
    F = int(sys.argv[3])            # frequency mask parameter F, max number of consecutive mel frequency channels masked
    T = int(sys.argv[4])            # time mask parameter T, max number of consecutive time steps masked
    if_fig = int(sys.argv[5])       # show plots or not
    augment_type = int(sys.argv[6]) # 0,1,2,3,4 - time warp, freq mask, time mask, all combined, mix
    
    df1 = pd.read_hdf(os.path.join('..','..','data','dataframe','df_train_'+ str(chosen_repr) +'_norm.h5'),'df')
    print(df1.shape)    
    augment_type_names = ['timeWarp','freqMask','timeMask','allThree','mix']
    aug_types, masks, params, specs = [], [], [], []
    sing =0
    aug_type = augment_type
    
    augment_path = os.path.join(cfg.main_folder_path,'data', 'augment')
    try:
        os.makedirs(augment_path)
    except FileExistsError:
        pass
        
    for i in range(0,df1.shape[0]):
        random.seed(i)
        mel_spectrogram = df1['mel-spektrogram'][i]
        mel_spectrogram = torch.Tensor(mel_spectrogram)
        
        # random choice of augmentation type, out of 4 (time warp, freq mask, time mask, SpecAugment)
        if augment_type == 4:
            aug_type = random.getrandbits(2)   
        
        param = np.zeros([7,2])
        num_mask_f = 0
        num_mask_t = 0
        replace_with_zero = 0
        
        if aug_type == 0:         
            # sometimes problems with singular matrix - indefinite solutions, and it's needed for interpolation.
            # solution - choose different seed - and thanks to that diffferent pt and w if such thing happend
            try: 
                spec, pt, w = test_time_warp(mel_spectrogram, W, i, if_fig)

            except:
                sing = sing+1
                spec, pt, w = test_time_warp(mel_spectrogram, W, i*667, if_fig)
            
            param[0]= [pt, w]
           
        if aug_type == 1:
            num_mask_f = random.randrange(1,F+1) 
            replace_with_zero = bool(random.getrandbits(1))
            spec, ff = test_freq_mask(mel_spectrogram, num_mask_f, replace_with_zero, F, i, if_fig)
            
            for j in range(0,num_mask_f):
                param[j+1] = ff[j]
            
        if aug_type == 2:
            num_mask_t = random.randrange(1,T+1) 
            replace_with_zero = bool(random.getrandbits(1))
            
            spec, tt = test_time_mask(mel_spectrogram, num_mask_t, replace_with_zero, T, i, if_fig)
            
            for j in range(0,num_mask_t):
                param[j+4] = tt[j]
            
        if aug_type == 3:
            num_mask_f = random.randrange(1,F+1) 
            num_mask_t = random.randrange(1,T+1) 
            replace_with_zero = bool(random.getrandbits(1))
            
            try:
                a1, pt, w = test_time_warp(mel_spectrogram, W, i, 0)
            except:
                a1, pt, w = test_time_warp(mel_spectrogram, W, i*667, 0)
                sing = sing+1
                
            a2, ff = test_freq_mask(a1[0], num_mask_f, replace_with_zero, F, i, 0)
            spec, tt = test_time_mask(a2[0], num_mask_t, replace_with_zero, T, i, if_fig)
            
            param[0]= [pt, w]
            
            for j in range(0,num_mask_f):
                param[j+1] = ff[j]
            
            for j in range(0,num_mask_t):
                param[j+4] = tt[j]
        spec = np.array(spec).squeeze(0)  
        aug_types.append(aug_type)
        masks.append([num_mask_f, num_mask_t, replace_with_zero])
        params.append(param)
        specs.append(spec)
       
    df1['aug_specs'] = specs
    df1['aug_types'] = aug_types
    df1['masks'] = masks
    df1['params'] = params

    
    df1.drop(chosen_repr, inplace=True, axis=1)
    name_df = 'df_train_' + str(chosen_repr)+'_' + augment_type_names[augment_type] +'.h5'
    df1.to_hdf(os.path.join(augment_path, name_df), key='augment', mode='w')
    print("done")

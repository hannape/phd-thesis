# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import time
import sys
import torch

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
experimentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(experimentdir)
sys.path.insert(0,parentdir) # 

from _2_TPOT_experiments.src.saving_utils import LoggerWrapper
from useful_func_applications import  load_test_new_data, dataframe_of_chunks
from _4_ResNets_experiments.src.useful_func_resnet import opcja1, opcja2, opcja3, opcja4, opcja5, opcja6, opcja7, opcja8


from config import main_folder_path
'''
   Saving model predictions for new, exemplary recording(s).
   Creating .txt file with chunks with predictions over certain chosen threshold, that can be imported and used e.g. in Audacity. 
   For example, extracting samples with predictions>0.5
    
   Running script: python script_name representation_type Resnet_scenario_chosen if_the_same_3layers directory_with_best_trained_model batch_size directory_with_test_data recordings_per_batch chosen_threshold
    e.g. python main_preds_new_data.py mel-spektrogram 5 1 2021-10-19-22-12-11 32 data_30min_test_preds 1 0.5
    
    
        e.g. python main_preds_new_data.py mel-spektrogram 1 1 2021-10-19-22-12-11 32 data_30min_test_preds 1 0.5

'''

if __name__=='__main__':
 
    chosen_repr = sys.argv[1]
    modelki = list(map(int, sys.argv[2].split(',')))  
    the_same_3layers= int(sys.argv[3])
    print(chosen_repr)
    dir_with_models = sys.argv[4]
    print(dir_with_models)
    start_time = time.time()
    batch_size = int(sys.argv[5])
    new_test_dir_name = sys.argv[6]
    rec_per_batch = int(sys.argv[7])
    chosen_thresh = float(sys.argv[8])

    exp_dir = os.path.join("..","..","_4_ResNets_experiments","results","no_grid", f"{chosen_repr}",f"{dir_with_models}")  # path to directory with best model
    path_recs = os.path.join(main_folder_path,'data', new_test_dir_name)  # whole path to directory with recordings to check. then split to subfolders 'labels' and 'rec', as in original recordings
    
    logger_wrapper = LoggerWrapper(exp_dir)
    sys.stderr.write = logger_wrapper.log_errors
    logger_wrapper.logger.info(f'Running {sys.argv}')
    
    # for best model with chosen_repr = mel-spektrogram, the_same_3layers=1. Other variations ignored
    if chosen_repr=="mel-spektrogram":
        in_shape = (60, 148, 1) 
        testing_features, y_tests, file_names_list, chunk_ids_list, info_chunksy, file_names = load_test_new_data(chosen_repr, path_recs, rec_per_batch)
    
    if the_same_3layers==1:
        testing_features_resh = testing_features.reshape(testing_features.shape[0], in_shape[0], in_shape[1], 1).astype('float32')
        image_test = (np.repeat(testing_features_resh, 3, axis=3)).transpose((0,3, 1, 2))
    
    # print("torch: ", torch.cuda.is_available() )
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(image_test))
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size)

    all_folders = [name for name in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, name))] 
    print(all_folders) # folders with models
    
    for option_no in modelki:
        
            predictions_all= []

            for file in os.listdir(exp_dir):
                if file.startswith(str(option_no)+'_'):
                    print('Chosen folder:', file)
                    print("---------------")
                    t = os.path.join(exp_dir, file)
                    all_models = [x for x in (os.listdir(t)) if '.pt' in x]
                    print(all_models)        
                    
            
                    for j in all_models: 
                    
                        model_name = j
                        opcje_modeli = [opcja1(), opcja2(), opcja3(), opcja4(), opcja5(),opcja6(), opcja7(), opcja8()]
                        clf = opcje_modeli[option_no-1]
                        clf.cuda()
                        clf.load_state_dict(torch.load(os.path.join(t, model_name)))
                      
                        print("Read model ",model_name)
                        clf.eval()
                        pred = []
                        for X in test_data_loader:
                            X = X[0]
                            out = clf(X.cuda())
                            pred.append(torch.softmax(out, dim = 1)[:, 1].cpu().detach().numpy())
                        pred = np.concatenate(pred, axis = 0)
                    
                        print('____________________________')
                        predictions_all.append(pred)
                        print(np.shape(predictions_all))
                        
                    file_name = file    
            
            print("Output vector size: ", np.shape(y_tests))
            print("5 models predictions sizes: ", np.shape(predictions_all))

            saving_t = os.path.join(exp_dir, file_name, new_test_dir_name, '_test'+str(start_time))
            print(saving_t)    
            os.makedirs(saving_t, exist_ok=True)
            pd.DataFrame(np.transpose(np.squeeze(predictions_all)),
                         columns = ['preds-1','preds-2','preds-3','preds-4','preds-5']).to_pickle(os.path.join(saving_t,"5_runs_preds.pkl") )
        
            averaged_preds = pd.DataFrame(np.transpose(np.squeeze(predictions_all))).mean(axis=1)
            pd.DataFrame(averaged_preds).to_pickle(os.path.join(saving_t,"5_runs_averaged_preds.pkl") )
            
# %%        Thresholding
            ind_1 = [i for i,v in enumerate(averaged_preds.to_numpy()) if v > chosen_thresh]
            print('how many samples over the threshold? ', np.shape(ind_1)[0])
            
            names_list_1 = [file_names_list[i] for i in ind_1]
            sample_size = len(names_list_1)
            names_id_list_1 = [file_names.index(names_list_1[i]) for i in range(0, sample_size)]          
            df_1 = dataframe_of_chunks(names_id_list_1, sample_size, info_chunksy, file_names, names_list_1, ind_1, averaged_preds)

            if type(info_chunksy) is list:
                i = 0
                df0 = df_1.loc[df_1['rec_id'] == i]
                file_audacity_name = os.path.join(saving_t,'prediction_0_' + str(int(chosen_thresh*100))+'_' + str(file_names[i]) + '.txt') 
                np.savetxt(file_audacity_name, np.transpose([np.array(df0["chunk_start"])/44100, np.array(df0["chunk_end"])/44100, np.array(df0['diff_pred_target'] )]), delimiter='\t', fmt='%.6f')
   
            else:     
                for i in range(0,np.shape(info_chunksy)[0]): 
                    df0 = df_1.loc[df_1['rec_id'] == i]
                    file_audacity_name = os.path.join(saving_t,'prediction_0_' + str(int(chosen_thresh*100))+'_' + str(file_names[i]) + '.txt')
                    print(file_audacity_name)
                    np.savetxt(file_audacity_name, np.transpose([np.array(df0["chunk_start"])/44100, np.array(df0["chunk_end"])/44100, np.array(df0['diff_pred_target'] )]), delimiter='\t', fmt='%.6f')
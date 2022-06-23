# -*- coding: utf-8 -*-

import os
import time
import logging 
import sys
import numpy as np

import config
import pandas as pd



# class LoggerWrapper:
#     def __init__(self, path='.'):
#         """
#         Wrapper for logging.
#         Allows to replace sys.stderr.write so that error massages are redirected do sys.stdout and also saved in a file.
#         use: logger = LoggerWrapper(); sys.stderr.write = logger.log_errors
#         :param: path: directory to create log file
#         """
#         # count spaces so that the output is nicely indented
#         self.trailing_spaces = 0

#         # create the log file
#         timestamp = time.strftime('%Y-%m-%d-%H-%M')
#         self.filename = os.path.join(path, f'{timestamp}.log')
#         try:
#             # os.mknod(self.filename)
#             with open(self.filename,"w") as f:
#                 pass
#         except FileExistsError:
#             pass

#         # configure logging
#         logging.basicConfig(level=logging.DEBUG,
#                             format='%(asctime)s %(name)-6s %(levelname)-8s %(message)s',
#                             datefmt='%m-%d %H:%M',
#                             filename=self.filename,
#                             filemode='w')
#         formatter = logging.Formatter('%(name)-6s: %(levelname)-8s %(message)s')

#         # make a handler to redirect stuff to std.out
#         self.logger = logging.getLogger('')
#         self.logger.setLevel(logging.INFO)  # bacause matplotlib throws lots of debug messages
#         self.console = logging.StreamHandler(sys.stdout)
#         self.console.setLevel(logging.INFO)
#         self.console.setFormatter(formatter)
#         self.logger.addHandler(self.console)

#     def log_errors(self, msg):
#         msg = msg.strip('\n')  # don't add extra enters

#         if msg == ' ' * len(msg):  # if you only get spaces: don't print them, but do remember
#             self.trailing_spaces += len(msg)
#         elif len(msg) > 1:
#             self.logger.error(' ' * self.trailing_spaces + msg)
#             self.trailing_spaces = 0

def load_test_new_data(chosen_repr, data_folder_name, rec_batch_size, calls_1=config.calls_1, calls_unknown = config.calls_unknown):

    import config as cfg
    from data_load import function_data_load
    from _2_TPOT_experiments.src.representations_all2 import function_representations_all2
    import numpy as np

    
    if_scaler = True                   # if the scaler is used on testing data.
    nr_rec_start = 0                   # starting rec number, needed if we want to read many long recordings
    test_new_only = 1                  # only test set is loaded 
    
    start_time = time.time()
    path_test1618_txt = os.path.join(data_folder_name, 'labels')
    path_train161718_txt = 0 
    path_test1618_wav = os.path.join(data_folder_name, 'rec')
    path_train161718_wav = 0 
    os.makedirs(path_test1618_txt, exist_ok=True)

    rec_files = sorted([file_name for file_name in os.listdir(path_test1618_wav) if file_name.endswith('.wav')])
    file_names = [[] for _ in range(np.size(rec_files))]
    for file_name in rec_files:
        recording_id = (file_name.split('.')[0]) 
        print(recording_id)
        file_name = os.path.join(path_test1618_txt, recording_id +'.txt')
        f = open(file_name, 'w+')  # open file in write mode, create epmty file if it doesn't exist
        f.close()
        

    _,_,_, data_test_new = function_data_load(path_test1618_txt, path_train161718_txt, path_test1618_wav,\
                                                                                     path_train161718_wav, config.balance_types, config.balance_ratios, config.chunk_length_ms,\
                                                                                     config.chunk_overlap, config.calls_0, calls_1, calls_unknown, config.tolerance, config.valid_set,\
                                                                                     config.test_rec_to_cut, config.columns_dataframe, test_new_only)
    print("--- Data load:: %s sec ---" % (time.time() - start_time))
    
    # REPRESENTATIONS: Create representations for test set
    
    start_time = time.time()
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
   
    # REPRESENTATIONS: Create representations for test set
    
    start_time = time.time()
    
    file_names, indices, info_chunksy, repr_full_scaled = function_representations_all2(path_test1618_wav, data_test_new, if_scaler, \
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
    
    print("Test loaded," + str(rec_batch_size) + " recordings from the directory " + data_folder_name)


    return testing_features, testing_target, file_names_list, chunk_ids_list, info_chunksy, file_names  
      

def dataframe_of_chunks(names_id_list_, sample_size, info_chunksy, file_names, names_list_, ind_, loaded_model,
                 columns_dataframe= ['rec_name','rec_id','chunk_ids', 'chunkrec_ids', 
                      'chunk_start', 'chunk_end', 'has_bird', 'chunks_species', 'call_id', 'has_unknown', 'has_noise', 'diff_pred_target']):

    names_id_list_ = [file_names.index(names_list_[i]) for i in range(0,sample_size)]
  
    chunk_rec_ids = []
    
    for i in range(0,sample_size):

        chunk_rec_ids.append(np.array(ind_)[i])
        
    chunks_start_sample, chunks_end_sample,has_bird_sample, chunks_species_sample = [],[],[],[]
    call_id_sample,has_unknown_sample,has_noise_sample, diff_pred_sample =[],[],[],[]
    
    for i in range(0,sample_size):
        chunks_start_sample.append(info_chunksy[names_id_list_[i]][1][chunk_rec_ids[i]])
        chunks_end_sample.append(info_chunksy[names_id_list_[i]][2][chunk_rec_ids[i]])
        has_bird_sample.append(info_chunksy[names_id_list_[i]][3][chunk_rec_ids[i]])
        chunks_species_sample.append(info_chunksy[names_id_list_[i]][4][chunk_rec_ids[i]])
        call_id_sample.append(info_chunksy[names_id_list_[i]][5][chunk_rec_ids[i]])
        has_unknown_sample.append(info_chunksy[names_id_list_[i]][6][chunk_rec_ids[i]])
        has_noise_sample.append(info_chunksy[names_id_list_[i]][7][chunk_rec_ids[i]])
        diff_pred_sample.append(np.squeeze(loaded_model)[np.array(ind_)[i]])
      
    df_ = pd.DataFrame(np.transpose((names_list_, names_id_list_, ind_, chunk_rec_ids,chunks_start_sample, chunks_end_sample,has_bird_sample,
                          chunks_species_sample,call_id_sample,has_unknown_sample,has_noise_sample,diff_pred_sample)), columns = columns_dataframe )
    return df_
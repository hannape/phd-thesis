# -*- coding: utf-8 -*-

## IMPORTS

import os
import contextlib
import numpy as np
import wave
import math
import pandas as pd
import random
import csv

def function_data_load(path_test1618_txt, path_train161718_txt, path_test1618_wav, path_train161718_wav, \
                             balance_types, balance_ratios, chunk_length_ms, chunk_overlap, calls_0, calls_1, \
                             calls_unknown, tolerance, valid_set, test_rec_to_cut, columns_dataframe, test_new_only=0):
  '''Main function to split recordings to chunks and decide if there's bird call / noise / unknown event in a specific chunk.
  
  Args: 
    Described in config.py
    test_new_only=1 : only test set is loaded 
  Returns:
    file_names_train_set, ind_for_train_set, result_dataframe_train: recording names, chunks indices for given rec chosen for train set (86), training dataframe 
    file_names_valid_set, ind_for_valid_set, result_dataframe_valid: recording names, chunks indices for given rec chosen for valid set (8), validation dataframe 
    file_names_test_old, ind_for_test_old, result_dataframe_test_old: recording names, chunks indices for given rec chosen for test set 3 (20), old test dataframe 
    file_names_test_new, ind_for_test_new, result_dataframe_test_new: recording names, chunks indices for given rec chosen for test set 3.1 (18), new test dataframe
    (list), (list), (dataframe)
  Dataframe structure:
    columns_dataframe = ['chunk_ids', 'chunk_start', 'chunk_end', 'has_bird', 'chunks_species', 'call_id', 'has_unknown', 'has_noise']
    result_dataframe_X = pd.DataFrame(data = X, index=file_names_X, columns = columns_dataframe)
  '''

  ### FUNCTIONS

  def my_read_labels(rec_name, path_txt):  
    ''' Reading labels from text file.
    
    Args: 
        rec_name (string): recording/annotation file name
        path_txt (string): path to folder with labels
    Returns:
        Table with columns:  label start time, label end time, label
    
    '''
    with open(os.path.join(path_txt, rec_name + '.txt'), 'r') as f:
      reader = csv.reader(f, delimiter='\t')
      data = [[float(row[0]), float(row[1]), str(row[2])] for row in reader]

    return np.array(data)


  def my_check_labels(second, chunk_length_s, labels, tol=0.):
    ''' Checking if in a chunk [second, second + chunk_length_s] there's an event marked in annotations.
      
    Args:
        second (float): Second of the recording
        chunk_length_s (float): length of a chunk, in seconds
        labels (ndarray): table where 1st column is a start and 2nd column is the end of the recording 
        tol (float): Tolerance on the edge of the chunk, in seconds 
        
    Returns:
        bool: If there's any label in a chunk (at least tolerance (4ms) of a label)
    ''' 

    # if starting time of label is in chunk, at the latest 4ms before end of chunk
    # if the ending time of the label is within the chunk, and at the earliest 4 ms after the start of the chunk
    # if the ing time of the label starts before start of chunk and ends after the end (for calls longer than chunk length, v.rare cases for 0.5 sec chunks)
    return (float(labels[0]) >= second and float(labels[0]) < second + chunk_length_s - tol) or \
            (float(labels[1]) < second + chunk_length_s and float(labels[1]) > second + tol) or \
            (float(labels[0]) < second and float(labels[1]) > second + chunk_length_s)              

  def my_map_seconds_to_y(labels, recording_duration, calls_of_interest, calls_to_cut, calls_unknown):  
    '''Creating binary labels for every chunk. 
    
    Args:
        labels (ndarray): Table with columns - label start time, label end time, label
        recording_duration (float): duration of the recording in seconds
        calls_of_interest (list), calls_to_cut (list), calls_unknown (list): list of labels of interest, negative and unknown (from config file)
    Returns:
        nb_of_chunks (int): number of chunks
        chunks_start, chunks_end (list): chunk start time, chunk end time
        has_bird (list): binary labels, bird/no bird in a chunk 
        chunks_species (list): labels within a chunk  
        call_id (list): list of IDs of labels 
        has_unknown (list), has_noise (list): binary labels, unknown/no unknown and noise/no noise in a chunk 
    '''
    
    duration_in_ms = recording_duration*1000
    nr_of_chunks =  1 + (duration_in_ms - chunk_length_ms) / (chunk_length_ms - chunk_overlap)
    
    y = [0] * math.ceil(nr_of_chunks)             
    y_restrictive = [0] * math.ceil(nr_of_chunks) 
    chunks_start, chunks_end = [0] * math.ceil(nr_of_chunks), [0] * math.ceil(nr_of_chunks)      
    has_unknown, has_noise = [0] * math.ceil(nr_of_chunks), [0] * math.ceil(nr_of_chunks)
    call_id = [[] for _ in range(math.ceil(nr_of_chunks))]
    chunks_species = [[] for _ in range(math.ceil(nr_of_chunks))]
    
    for s in range(math.ceil(nr_of_chunks)):
        chunks_start[s] = s * ((chunk_length_ms-chunk_overlap) / 1000)  
        chunks_end[s] = chunks_start[s] + chunk_length_ms/1000 
        
        for ind,l in enumerate(labels):
            if my_check_labels(chunks_start[s], chunk_length_ms/1000, l):               
                if l[2] in calls_to_cut:
                  has_noise[s] = 1
                if l[2] in calls_unknown:
                  has_unknown[s] = 1
                if l[2] in calls_of_interest:  
                  y[s] = 1 

            if my_check_labels(chunks_start[s], chunk_length_ms/1000, l, 0.004): 
                if l[2] in calls_to_cut:
                  chunks_species[s].append(l[2])
                if l[2] in calls_unknown:
                  has_unknown[s] = 1
                  chunks_species[s].append(l[2])
                if l[2] in calls_of_interest:  
                  y_restrictive[s] = 1  
                  chunks_species[s].append(l[2])  
                  call_id[s].append(ind)
              
                      
        if y[s] != y_restrictive[s] and l[2] in calls_of_interest:
            y[s] = 0    
            has_unknown[s] = 1
    has_bird = y
    nb_of_chunks = s+1
    return nb_of_chunks, chunks_start, chunks_end, has_bird, chunks_species, call_id, has_unknown, has_noise    

  def my_load_wav(path_wav, path_txt, recordings_incl = None, recordings_excl = None):  

    ''' Reading audio files.
    
    Args: 
        path_wav (string): path to folder with recordings
        path_txt (string): path to folder with labels
        recordings_incl (list): recordings only from given set (valid)
        recordings_excl (list): recordings apart from those from the given set (train)  
    Returns:
        X_matrix (list): chosen chunks data - matrix with chunk_ids, chunk_start, chunk_end, has_bird, chunks_species, call_id, has_unknown, has_noise
        file_names (list): recording names
    '''

    file_names = []

    if recordings_incl == None and recordings_excl == None:                                        
      rec_files = sorted([file_name for file_name in os.listdir(path_wav) if file_name.endswith('.wav')])

    if recordings_incl != None:                                                                     
      rec_files = sorted([s + '.wav' for s in recordings_incl])

    if recordings_excl != None:                                                                     
      rec_files_all = [file_name for file_name in os.listdir(path_wav) if file_name.endswith('.wav')]
      recordings_excl = [s + '.wav' for s in recordings_excl]
      rec_files = sorted(list(set(rec_files_all) - set(recordings_excl))  )
    
    X_matrix = [[] for _ in range(np.size(rec_files))]
    file_names = [[] for _ in range(np.size(rec_files))]

    for ind, file_name in enumerate(rec_files):
      
      print("------------Analysis of recording: " + file_name + "-----------")

      fname = os.path.join(path_wav, file_name) 
      with contextlib.closing(wave.open(fname,'r')) as f:
          frames = f.getnframes()
          rate = f.getframerate()
          duration = frames / float(rate)
    
      recording_id = (file_name.split('.')[0])      
      recording_labels = my_read_labels(recording_id, path_txt)    

      chunk_id, chunk_start, chunk_end, has_bird, chunks_species, call_id, has_unknown, has_noise  = my_map_seconds_to_y(recording_labels, duration, calls_1, calls_0, calls_unknown)
      chunk_ids = range(chunk_id)
      chunk_start = rate * np.array(chunk_start)
      chunk_start = np.int_(np.round(chunk_start)).tolist() 
      chunk_end = rate * np.array(chunk_end)
      chunk_end = np.int_(np.round(chunk_end)).tolist() 
      X_matrix[ind] = [chunk_ids, chunk_start, chunk_end, has_bird, chunks_species, call_id, has_unknown, has_noise]
      file_names[ind] = file_name

    return X_matrix , file_names 

  def create_set_ind(result_dataframe, balance, ratios = None):
 
    ''' Creating sets, balanced or not depending on a set type. 
    Args: 
        result_dataframe (dataframe): dataframe with all chunks data
        balance (string): balance type (from config)
        ratios (list): balance ratio (from config)  
    Returns:
        X_matrix (list): chosen chunks data - matrix with chunk_ids, chunk_start, chunk_end, has_bird, chunks_species, call_id, has_unknown, has_noise
        file_names (list): recording names
    '''     
 
    file_names = list(result_dataframe.index)
    ind_chosen = [[] for _ in range(len(file_names))]
    neg_random_numb = 0;
    all_pos = 0
    all_chunks = 0

    for file_i, file_name in enumerate(file_names):
      
      indices_hasbird = [i for i, j in enumerate(result_dataframe['has_bird'][file_i]) if j == 1]
      indices_hasnoise = [i for i, j in enumerate(result_dataframe['has_noise'][file_i]) if j == 1]
      indices_unknown = [i for i, j in enumerate(result_dataframe['has_unknown'][file_i]) if j == 1]
      indices_all = result_dataframe['chunk_ids'][file_i]
      pos = indices_hasbird

      if balance == 'full_rec':
        
        ind_chosen[file_i] = list(set(indices_all) - (set(indices_unknown) - set(indices_hasbird)) )  
        # those chunks with bird and unknown labels stay. In all other cases, unknown is not used in a set.
        all_pos += len(pos)
        all_chunks += len(ind_chosen[file_i])
        neg_labels = list(set(indices_hasnoise) - set(indices_hasbird +indices_unknown))

      if balance == 'balanced' or balance == 'valid':
                
        neg_labels = list(set(indices_hasnoise) - set(indices_hasbird +indices_unknown))

        if (len(pos) - len(neg_labels) >= 0):   
          neg_random_numb = len(pos) - len(neg_labels)
          if (neg_random_numb < ratios[1]):
            neg_random_numb = ratios[1]
          if neg_random_numb + len(neg_labels) < ratios[0]:
            neg_random_numb = ratios[0] - len(neg_labels)
        else:
          neg_random_numb = ratios[1]

        neg_random_set = list(set(indices_all) - (set(indices_unknown + indices_hasbird + indices_hasnoise)))
        random.seed(667)  # fixed
        neg_random = random.sample(neg_random_set, neg_random_numb)
        ind_chosen[file_i] = sorted(list(set(indices_hasbird + neg_labels + neg_random)))

        all_pos += len(pos)
        all_chunks += len(ind_chosen[file_i])  

    print('All chunks: ', all_chunks,', including ', all_pos, ' positives. Thus ', 100* round(all_pos/all_chunks,4),' % positive in set ', balance )
    return file_names, ind_chosen

  if test_new_only==0:
      print('*************  Train  ***********')
    
      X_train, file_names_train = my_load_wav(path_train161718_wav, path_train161718_txt, None, valid_set) 
      result_dataframe_train = pd.DataFrame(data = X_train, index = file_names_train, columns = columns_dataframe)
      file_names_train_set, ind_for_train_set = create_set_ind(result_dataframe_train, balance_types[1], balance_ratios[1])
    
      print('*************  Valid  ***********')
    
      X_valid, file_names_valid = my_load_wav(path_train161718_wav, path_train161718_txt, valid_set, None)   
      result_dataframe_valid = pd.DataFrame(data = X_valid, index = file_names_valid, columns = columns_dataframe)  
      file_names_valid_set, ind_for_valid_set = create_set_ind(result_dataframe_valid, balance_types[2], balance_ratios[2])
    
      print('*************  Test old  ***********')
    
      X_test_old, file_names_test_old = my_load_wav(path_test1618_wav, path_test1618_txt) 
      result_dataframe_test_old = pd.DataFrame(data = X_test_old, index = file_names_test_old, columns = columns_dataframe)
      file_names_test_set_old, ind_for_test_set_old = create_set_ind(result_dataframe_test_old, balance_types[0], balance_ratios[0])
  
  print('*************  Test new  ***********')
 
  X_test_new, file_names_test_new = my_load_wav(path_test1618_wav, path_test1618_txt, None, test_rec_to_cut)
  result_dataframe_test_new = pd.DataFrame(data = X_test_new, index = file_names_test_new, columns = columns_dataframe)
  file_names_test_set_new, ind_for_test_set_new = create_set_ind(result_dataframe_test_new, balance_types[0], balance_ratios[0])

  return [[file_names_train_set, ind_for_train_set, result_dataframe_train],[file_names_valid_set, ind_for_valid_set, result_dataframe_valid],
          [file_names_test_set_old, ind_for_test_set_old, result_dataframe_test_old], [file_names_test_set_new, ind_for_test_set_new, result_dataframe_test_new]]
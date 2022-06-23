# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os

import keras
import pickle
import sys

from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

from numpy.random import seed
import tensorflow as tf
seed(1)
tf.random.set_seed(2)  

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
experimentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(experimentdir)
sys.path.insert(0,parentdir) 

import config as cfg
from _2_TPOT_experiments.src.saving_utils import LoggerWrapper

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from useful_func_classify import prepare_data_to_classify, load_test  #  species_data_testy,

# allocating fraction of GPU memory
from tensorflow.compat.v1.keras.backend import set_session
config2 = tf.compat.v1.ConfigProto()
config2.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.compat.v1.Session(config=config2))

"""
    Testing classification models on test data after the detection step. Classification made not on all test data, 
    but on chunks chosen by the best detection model, on the basis of threshold from chosen recall value (e.g. 0.8 or 0.9)

    Running script: python script_name 
    chosen_representation  (must be 'mel-spektrogram')
    if_augmentation_used_and_which_type (-1: no augment, 1:15 augmentation types. 10 was chosen as best one) 
    architecture_classification_type (choice: leNet5', 'AlexNet', 'CNN_chapter5)
    chosen_recall_threshold (0.8, 0.9 recall, if not, threshold will be 0.5)
    dir_name_with_best_trained_models_for_detection (in resnet folder)
    Resnet_scenario_chosen 
    dir_name_with_classification_models (results from main_training_classification)
    
    e.g. python main_testing_classification.py mel-spektrogram 10 leNet5 0.8 2021-10-19-22-12-11 7 model_0-2022-01-23-20-40

"""
    


# %%
if __name__=='__main__':
    

    # Load configs
    dict_model_types = {'leNet5': 0, 'AlexNet':1, 'CNN_chapter5':2}

    chosen_repr = sys.argv[1]
    if_augment = int(sys.argv[2])     
    model_nr = dict_model_types[sys.argv[3]] 
    chosen_recall_thresh = float(sys.argv[4])   
    dir_with_detection_models = sys.argv[5]
    option_no = int(sys.argv[6])
    dir_with_classification_models = sys.argv[7]
    
    exp_dir = os.path.join("..","..","_4_ResNets_experiments","results","no_grid", f"{chosen_repr}",f"{dir_with_detection_models}")  # path to directory with best model
    model_dir = os.path.join(cfg.main_folder_path,"_7_Classify", "results")
    dir_classify_model = os.path.join(model_dir, dir_with_classification_models)
    
    for file in os.listdir(exp_dir):
            if file.startswith(str(option_no)+'_'):
                print('Chosen folder:', file)
                chosen_folder = file
    loaded_preds_test = pickle.load(open(os.path.join(exp_dir,chosen_folder, '5_runs_averaged_preds.pkl'), 'rb'))
    y_true_test = pickle.load(open(os.path.join("..","..","..","ResNet","results","grid", 'testing_target.pkl'), 'rb'))
        
    
    # Predefined parameters
    nb_of_classes = 6 # number of classes in classification - 6 in the examined case 
    if chosen_repr=='mel-spektrogram':
        in_shape = (60, 148, 1) 
        
    logger_wrapper = LoggerWrapper(dir_classify_model)
    sys.stderr.write = logger_wrapper.log_errors
    logger_wrapper.logger.info(f'Running {sys.argv}') 
    
    ## thresholds chosen the basis of test data, for recall=80% and 90% obtained for the test data
    if chosen_recall_thresh==0.8:     
        chosen_thresh = 0.44185
    elif chosen_recall_thresh==0.9:     
        chosen_thresh = 0.0971    
    else:
        chosen_thresh = 0.5
        
    if_print_aucpr = 1  # if you want to display images AUC PR of test data, with chosen 3 thresholds marked
    
    if if_print_aucpr == 1:  
        prec_av, recall_av, thresh = precision_recall_curve(y_true_test, loaded_preds_test)   

        fig = plt.figure(figsize=(10,10))
        plt.rcParams.update({'font.size': 22})
        plt.title('AUC PR')
        plt.plot(recall_av, prec_av, label = '4A, PR AUC = %0.2f%% ' % (100*average_precision_score(y_true_test, loaded_preds_test)))
        plt.plot(recall_av[np.argmin((thresh<0.5))],prec_av[np.argmin((thresh<0.5))],'ro', label = 'p = 0.5') 
        plt.plot(recall_av[np.argmin((thresh<0.44185))],prec_av[np.argmin((thresh<0.44185))],'bo', label = 'p = 0.44185 (recall = 80%)') 
        plt.plot(recall_av[np.argmin((thresh<0.0971))],prec_av[np.argmin((thresh<0.0971))],'ko', label = 'p = 0.0971 (recall = 90%)') 
        
        plt.legend(loc = 'lower left')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Precision')
        plt.xlabel('Recall')         
        print('p=0.5. recall: ', recall_av[np.argmin((thresh<0.5))],' precision: ',prec_av[np.argmin((thresh<0.5))])
        print('p=0.44185, (rec 80%). recall: ', recall_av[np.argmin((thresh<0.44185))],' precision: ',prec_av[np.argmin((thresh<0.44185))])
        print('p=0.0971, (rec 80%). recall: ', recall_av[np.argmin((thresh<0.0971))],' precision: ',prec_av[np.argmin((thresh<0.0971))])
# %%        
    #  indices chosen from test dataset
    binar = loaded_preds_test>chosen_thresh
    ind_chosen_by_model = list(binar.to_numpy().squeeze().nonzero())[0]
      
    file_names, indices, info_chunksy,repr_test = load_test(chosen_repr)
    
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
      representation1_list.extend([repr_test[num][i] for i in range(len(i))]) 

    data_df = [file_names_list, chunk_ids_list, chunk_start_list, chunk_end_list, has_bird_list,
               chunks_species_list, call_id_list, has_unknown_list, has_noise_list, representation1_list ]
    data_df = np.array(data_df).T.tolist() 
 
    columns_df = ['rec_name','chunk_ids', 'chunk_start', 'chunk_end', 'has_bird', \
                  'chunks_species', 'call_id', 'has_unknown', 'has_noise', 'mel-spektrogram'] 
    df_test_full = pd.DataFrame(data = data_df, columns =columns_df)     
    df_test = df_test_full.iloc[ind_chosen_by_model]
    
    temp_test = df_test.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
    X_test = np.stack(temp_test[:, 2])
    y_test = temp_test[:, 3]
    del temp_test   
    X_test_class, y_test_class_bin, y_test_class_cat, df_with_codes = prepare_data_to_classify(df_test, X_test, y_test)
 
    my_testX = X_test_class.reshape(X_test_class.shape[0], in_shape[0], in_shape[1], 1).astype('float32') 
    my_testY = keras.utils.to_categorical(np.floor(y_test_class_cat['bird_code'].to_numpy()).astype(int), nb_of_classes)
 
    # %% Model evaluation, saving predictions and averaged predictions

    predictions_all= []

    mdl_names = [x for x in os.listdir(dir_classify_model) if x.endswith(".h5")]

    for run_nr in range(0, len(mdl_names)):
        mdl_name = mdl_names[run_nr]
        model = load_model(os.path.join(dir_classify_model, mdl_name))  
        pred = model.predict_proba(my_testX)
        predictions_all.append(pred)
        print(np.shape(predictions_all))
    
    np.save(os.path.join(dir_classify_model,str(len(mdl_names))+'_runs_preds.npy'), np.array(predictions_all))
    averaged_preds = np.array(predictions_all).mean(axis=0)
    np.save(os.path.join(dir_classify_model,str(len(mdl_names))+"_runs_averaged_preds.npy"), averaged_preds)
    
    predictions_all = np.array(predictions_all)
    Y_true_test = np.argmax(my_testY, axis=1) # Convert one-hot to index
    
    if_av_preds = 0 # if you want to display images for averaged predictions (1) or for all N trained models (0).
    
    if if_av_preds == 1: # for averaged predictions (1)
        y_pred_test = averaged_preds
        y_pred_test = np.argmax(y_pred_test, axis=-1)
        y_pred_test_proba = averaged_preds 
        print("Accuracy score: ", accuracy_score(Y_true_test, y_pred_test))
        
        # Confusion matrix
        print(classification_report(Y_true_test, y_pred_test))
        confusion = confusion_matrix(Y_true_test, y_pred_test)
        print('Confusion Matrix\n')
        print(confusion)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion,
                                      display_labels=[0,1,2,3,4,5])
        disp = disp.plot(include_values=True,  cmap='viridis', ax=None, xticks_rotation='horizontal')
        plt.show()
    
    else: # for all N trained models (0).
        for n in range(0,len(predictions_all)):
            y_pred_test = predictions_all[n,:,:] 
            y_pred_test = np.argmax(y_pred_test, axis=-1)
            y_pred_test_proba = predictions_all[n,:,:]
            print("Accuracy score: ", accuracy_score(Y_true_test, y_pred_test))
            
            # Confusion matrix
            print(classification_report(Y_true_test, y_pred_test))
            confusion = confusion_matrix(Y_true_test, y_pred_test)
            print('Confusion Matrix\n')
            print(confusion)
            
            disp = ConfusionMatrixDisplay(confusion_matrix=confusion,
                                          display_labels=[0,1,2,3,4,5])
            disp = disp.plot(include_values=True,  cmap='viridis', ax=None, xticks_rotation='horizontal')
            plt.show()

# %% Displaying PR curves, method one class vs all
       
    precision = dict()
    recall = dict()
    plt.figure(1)
    for i in range(0,nb_of_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_true_test,
                                                            y_pred_test_proba[:, i], pos_label=i)
        aucpr = average_precision_score((Y_true_test==i).astype(int), y_pred_test_proba[:, i])
        print(aucpr)
        plt.plot(recall[i], precision[i], lw=2, label='class {}, AUC='.format(i) + str(np.round(aucpr*100,2))+'%')
        
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.title("PR Curves - one vs all")
    plt.show()
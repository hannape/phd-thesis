# -*- coding: utf-8 -*-

import os
import pickle
import time
import sys
import pandas as pd
import numpy as np
from keras import backend as K
K.set_image_data_format('channels_last')
from numpy.random import seed
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
from sklearn.metrics import average_precision_score, roc_auc_score 
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

seed(1)
tf.random.set_seed(2)

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
experimentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(experimentdir)
sys.path.insert(0,parentdir)

import config
from _2_TPOT_experiments.src.saving_utils import LoggerWrapper
from useful_func_cnn import load_my_data, load_my_data_multitap, make_cnn_grid_corr, load_test, load_test_multitap

# allocating fraction of GPU memory
config2 = tf.compat.v1.ConfigProto()
config2.gpu_options.per_process_gpu_memory_fraction = 0.5 
set_session(tf.compat.v1.Session(config=config2))

''' Training and testing models with parameters chosen by random search. 
    Saving models, creating predictions for test set, and drawing PR and ROC plots.
    
    Running script: python script_name representation_type directory_with_cnn_random_search_results which_model_checked(1st,2nd,3rd best) max_number_of_epochs number_of_models_trained 
     e.g. python main_cnns_training_testing.py mel-spektrogram ..\results\randomSearch\mel-spektrogram\2021-05-26-22-58\ 1 100 5
'''

# %%

if __name__=='__main__':

    # Load configs
    chosen_repr = sys.argv[1]
    with open(os.path.join(sys.argv[2],'all_models.pkl'), 'rb') as f:
        all_models = pickle.load(f)
        
    with open(os.path.join(sys.argv[2],'all_models_scores.pkl'), 'rb') as g:    
        all_models_scores = pickle.load(g)
        
    # list of indices of 3 best models
    best_Nth_idx = list((all_models_scores.sort_values(by=[0], ascending=False).nlargest(int(sys.argv[3] ), columns = 0)).index)[-1]
    best_params = all_models.iloc[best_Nth_idx]
    print(best_params)
    
    epochs = int(sys.argv[4] )
    n_models = int(sys.argv[5] )
    exp_dir = os.path.join("..","results","PR", f"{chosen_repr}", time.strftime('%Y-%m-%d-%H-%M'))
    try:
        os.makedirs(exp_dir)
    except FileExistsError:
        pass
    
    start_time = time.time()
    logger_wrapper = LoggerWrapper(exp_dir)
    sys.stderr.write = logger_wrapper.log_errors
    logger_wrapper.logger.info(f'Running {sys.argv}')
    
    if chosen_repr=="multitaper": 
        cv_split, dX, dy, di, cv, X_val, y_val, train_X, train_y = load_my_data_multitap(chosen_repr, config) 
    else:
        cv_split, dX, dy, di, cv, X_val, y_val, train_X, train_y = load_my_data(chosen_repr, config) 
    
    print(chosen_repr)
    if chosen_repr=="spektrogram":
        in_shape = (63, 148, 1)
    if chosen_repr=="mel-spektrogram":
        in_shape = (60, 148, 1)    
    if chosen_repr=="multitaper":
        in_shape = (64, 149, 1)  
           
    dX_resh = dX.reshape(dX.shape[0], in_shape[0], in_shape[1], 1).astype('float32')     
    input_shape = (np.shape(dX_resh)[1], np.shape(dX_resh)[2], np.shape(dX_resh)[3])
    
    # read test set
    if chosen_repr=="multitaper":
        testing_features, testing_target, testy_identifiers = load_test_multitap(chosen_repr)  
        testing_features_resh = testing_features.reshape(testing_features.shape[0], in_shape[0], in_shape[1], 1).astype('float32')
    else:         
        testing_features, testing_target = load_test(chosen_repr)     
        testing_features_resh = testing_features.reshape(testing_features.shape[0], in_shape[0], in_shape[1], 1).astype('float32')
    
    print("testing target ", np.shape(testing_target))
    print("testing features ", np.shape(testing_features))

    print("Fit model on training data")
    y_val = y_val.astype(int)
    train_y = train_y.astype(int)
    
    X_val_resh = X_val.reshape(X_val.shape[0],in_shape[0], in_shape[1], 1).astype('float32') 
    train_X_resh = train_X.reshape(train_X.shape[0], in_shape[0], in_shape[1], 1).astype('float32')
    
    my_callbacks = [ tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights = True), ]

    best_N_idx = list((all_models_scores.sort_values(by=[0], ascending=False).nlargest(int(sys.argv[3] ), columns = 0)).index)
    names= ["1st-best", "2nd-best","3rd-best"]
    n_plots = 0
    
    for num, k in enumerate(best_N_idx):
        
        best_params = all_models.iloc[k]
        logger_wrapper.logger.info('Model')
        logger_wrapper.logger.info(names[num])
        logger_wrapper.logger.info('chosen parameters to test:')
        logger_wrapper.logger.info(best_params)
        logger_wrapper.logger.info('Index of best model')
        logger_wrapper.logger.info(k)
        
        exp_dir_sub = os.path.join(exp_dir,"model-" + names[num])
        try:
            os.makedirs(exp_dir_sub)
        except FileExistsError:
            pass
        preds_data, aucroc_data, aucpr_data = [], [], []
        
        
        for n in range (0,n_models):
            print("*************  Run ", n, ", model ", num," with index ", k, "  ********************")
            model_refit = make_cnn_grid_corr(input_shape = best_params['input_shape'],
                                lr = best_params['lr'],
                                filters = best_params['filters'],
                                drop_out = best_params['drop_out'],
                                layers = best_params['layers'],
                                if_dropout = best_params['if_dropout'],
                                dense_layer_sizes = best_params['dense_layer_sizes']
                                )
        
            hist = model_refit.fit(train_X_resh, train_y.astype(np.float32), validation_data=(X_val_resh, y_val.astype(np.float32)), callbacks = my_callbacks,
                                           epochs=epochs, batch_size=best_params['batch_size']) 

            # save model
            model_refit.save(os.path.join(exp_dir_sub,"my-model"+str(n)+".h5"))
            print("Saved model to disk")       
         
            y_preds = model_refit.predict_proba(testing_features_resh) 
            
            preds_data.append(y_preds)       
            prec_nb, recall_nb, _ = precision_recall_curve(testing_target, y_preds)
            fpr_nb, tpr_nb, _ = roc_curve(testing_target, y_preds)
            print('AUC PR: ', average_precision_score(testing_target, y_preds))

            aucpr_data.append(average_precision_score(testing_target, y_preds))
            aucroc_data.append(roc_auc_score(testing_target, y_preds))
    
            fig = plt.figure(figsize=(10,10))
            plt.rcParams.update({'font.size': 22})
            plt.title('AUC PR')
            plt.plot(recall_nb, prec_nb,  'b', label = 'CNN AUC = %0.2f%% ' % (100*average_precision_score(testing_target, y_preds)))
            plt.legend(loc = 'upper right')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('Precyzja (Precision)')
            plt.xlabel('Czułość (Recall)')
            fig.savefig(os.path.join(exp_dir_sub,'AUC-PR-refit-'+ str(n) + time.strftime('-%Y-%m-%d-%H-%M')))
            
            fig = plt.figure(figsize=(10,10))
            plt.rcParams.update({'font.size': 22})
            plt.title('AUC ROC')
            plt.plot(fpr_nb, tpr_nb, 'b', label = 'CNN AUC = %0.2f%%' % (100*roc_auc_score(testing_target, y_preds)))
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('Wskaźnik czułości (True Positive Rate)')
            plt.xlabel('Odsetek fałszywie pozytywnych alarmów (False Positive Rate)')
            fig.savefig(os.path.join(exp_dir_sub,'AUC-ROC-refit-'+ str(n) +time.strftime('-%Y-%m-%d-%H-%M')))
          
            # summarize history for loss
            fig = plt.figure(figsize=(10,10))
            plt.plot(hist.history['loss'])
            plt.plot(hist.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Koszt (loss)')
            plt.xlabel('Liczba epok')
            plt.legend(['trening','walid'], loc='upper right')
            fig.savefig(os.path.join(exp_dir_sub,'uczenie-loss-refit-'+ str(n) +time.strftime('-%Y-%m-%d-%H-%M')))
            
            if n_plots==0:
                auc_nr = 'auc'
                val_auc_nr = 'val_auc'
            else:
                auc_nr = 'auc_'+str(n_plots)
                val_auc_nr = 'val_auc_'+ str(n_plots)
                
            n_plots = n_plots + 1
            
            fig = plt.figure(figsize=(10,10))
            plt.plot(hist.history[auc_nr])
            plt.plot(hist.history[val_auc_nr])
            plt.title('Model PR AUC')
            plt.ylabel('Pole pod krzywą PR')
            plt.xlabel('Liczba epok')
            plt.legend(['trening','walid'], loc='lower right')
            fig.savefig(os.path.join(exp_dir_sub,'uczenie-PR-AUC-refit-' + str(n) + time.strftime('-%Y-%m-%d-%H-%M')))
            
# %%  Printing & saving data for 5 models
        print('PR AUC mean: ' , np.mean(aucpr_data), '+/- std', np.std(aucpr_data) )
        print('ROC AUC mean: ' , np.mean(aucroc_data), '+/- std', np.std(aucroc_data) )
        
        logger_wrapper.logger.info('PR AUC mean i std:')
        logger_wrapper.logger.info(np.mean(aucpr_data))
        logger_wrapper.logger.info(np.std(aucpr_data))
        logger_wrapper.logger.info('ROC AUC mean i std:')
        logger_wrapper.logger.info(np.mean(aucroc_data))
        logger_wrapper.logger.info(np.std(aucroc_data))    
        
        pd.DataFrame(np.transpose(np.squeeze(preds_data)), columns = ['preds-1','preds-2','preds-3','preds-4','preds-5']).to_pickle(os.path.join(exp_dir_sub,"5_runs_preds.pkl") )   
        pd.DataFrame(aucpr_data).to_pickle(os.path.join(exp_dir_sub,"5_runs_aucpr.pkl") ) 
        pd.DataFrame(aucroc_data).to_pickle(os.path.join(exp_dir_sub,"5_runs_aucroc.pkl") ) 
        
#%% ensemble of 5 models, plots and saving   
        averaged_preds = pd.DataFrame(np.transpose(np.squeeze(preds_data))).mean(axis=1)
        pd.DataFrame(averaged_preds).to_pickle(os.path.join(exp_dir_sub,"5_runs_averaged_preds.pkl") )
        
        prec_av, recall_av, _ = precision_recall_curve(testing_target, averaged_preds)
        fpr_av, tpr_av, _ = roc_curve(testing_target, averaged_preds)
        print('ensemble AUC PR: ', average_precision_score(testing_target, averaged_preds))
        print('ensemble AUC ROC: ', roc_auc_score(testing_target, averaged_preds))
        logger_wrapper.logger.info('PR AUC ensembled:')
        logger_wrapper.logger.info(average_precision_score(testing_target, averaged_preds))
        logger_wrapper.logger.info('ROC AUC ensembled:')
        logger_wrapper.logger.info(roc_auc_score(testing_target, averaged_preds))
    
        fig = plt.figure(figsize=(10,10))
        plt.rcParams.update({'font.size': 22})
        plt.title('AUC PR')
        plt.plot(recall_av, prec_av,  'b', label = 'CNN AUC = %0.2f%% ' % (100*average_precision_score(testing_target, averaged_preds)))
        
        plt.legend(loc = 'upper right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Precyzja (Precision)')
        plt.xlabel('Czułość (Recall)')
        fig.savefig(os.path.join(exp_dir_sub,'AUC-PR-averaged5-' + time.strftime('-%Y-%m-%d-%H-%M')))

        fig, ax = plt.subplots(figsize=(10,10))
        plt.rcParams.update({'font.size': 22})
        plt.title('AUC ROC')
        plt.plot(fpr_av, tpr_av, 'b', label = 'CNN AUC = %0.2f%%' % (100*roc_auc_score(testing_target, averaged_preds)))
            
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Wskaźnik czułości (True Positive Rate)')
        plt.xlabel('Odsetek fałszywie pozytywnych alarmów (False Positive Rate)')
        fig.savefig(os.path.join(exp_dir_sub,'AUC-ROC-averaged5-'+time.strftime('-%Y-%m-%d-%H-%M')))
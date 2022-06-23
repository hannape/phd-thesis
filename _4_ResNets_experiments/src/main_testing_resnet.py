# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import sys
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
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

from _2_TPOT_experiments.src.saving_utils import LoggerWrapper
from useful_func_resnet import load_test, load_tests_multitap 
from useful_func_resnet import opcja1, opcja2, opcja3, opcja4, opcja5, opcja6, opcja7, opcja8

''' Testing models, creating predictions for test set, drawing PR and ROC plots.
    
    Running script: python script_name representation_type scenario_chosen if_the_same_3layers directory_with_trained_models batch_size  
     e.g. python main_testing_resnet.py spektrogram 2 1 2021-09-23-19-40 32
'''

if __name__=='__main__':
    
    # Load configs
    chosen_repr = sys.argv[1]
    modelki = list(map(int, sys.argv[2].split(',')))
    the_same_3layers= int(sys.argv[3])
    print(chosen_repr)
    dir_with_models = sys.argv[4]
    print(dir_with_models)
    start_time = time.time()
    batch_size = int(sys.argv[5])      
    
    exp_dir = os.path.join("..","results","no_grid", f"{chosen_repr}",f"{dir_with_models}")
    
    logger_wrapper = LoggerWrapper(exp_dir)
    sys.stderr.write = logger_wrapper.log_errors
    logger_wrapper.logger.info(f'Running {sys.argv}')
    
    # Read test set
    if chosen_repr=="spektrogram":
        in_shape = (63, 148, 1)
        testing_features, y_tests = load_test(chosen_repr)  
    if chosen_repr=="mel-spektrogram":
        in_shape = (60, 148, 1)  
        testing_features, y_tests = load_test(chosen_repr)  
    if chosen_repr=="multitaper":
        in_shape = (64, 149, 1)  
        testing_features, y_tests,_ = load_tests_multitap(chosen_repr)  
    if chosen_repr=="all":
        testing_features1, y_tests1 = load_test('spektrogram') 
        testing_features2, y_tests2 = load_test('mel-spektrogram') 
        testing_features3, y_tests3,_ = load_tests_multitap('multitaper') 
        in_shape = (64, 149, 1)
        y_tests = y_tests3
        print(np.shape(y_tests1),np.shape(y_tests2),np.shape(y_tests3))
            
# %%    
    if the_same_3layers==1:  # one representation tripled
        testing_features_resh = testing_features.reshape(testing_features.shape[0], in_shape[0], in_shape[1], 1).astype('float32')
        image_test = (np.repeat(testing_features_resh, 3, axis=3)).transpose((0,3, 1, 2))
    if the_same_3layers==0:  # 3 different representations in each channel. Only for my representation sizes, not universal. Padding to the same size
        print("Three different representations in 3 channels")       
        image_test = (np.stack(((np.pad(testing_features1.reshape(testing_features1.shape[0], 63, 148).astype('float32'), ((0,0),(0,1),(0,1)),'constant')),
                                (np.pad(testing_features2.reshape(testing_features2.shape[0], 60, 148).astype('float32'), ((0,0),(0,4),(0,1)),'constant')),
                                testing_features3.astype('float32')),
                                axis = -1)).transpose((0, 3, 1, 2))
     
    test_dataset = TensorDataset(torch.Tensor(image_test))
    test_data_loader = DataLoader(test_dataset, batch_size = batch_size)

    all_folders = [name for name in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, name))] 
    print(all_folders)

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
                    opcje_modeli = [opcja1(),opcja2(),opcja3(),opcja4(),opcja5(),opcja6(), opcja7(), opcja8()]
                    clf = opcje_modeli[option_no]
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
                    print("prediction by model: ",pred[-1],", baseline label: ",y_tests[-1])
                
                    print('____________________________')
                    predictions_all.append(pred)
                    print(np.shape(predictions_all))
        
        print(model_name)
        print("Output vector size: ", np.shape(y_tests))
        print("5 models predictions sizes: ", np.shape(predictions_all))
        
        # Printing & saving data for 5 models       
        pd.DataFrame(np.transpose(np.squeeze(predictions_all)), 
                     columns = ['preds-1','preds-2','preds-3','preds-4','preds-5']).to_pickle(os.path.join(t,"5_runs_preds.pkl") )

        averaged_preds = pd.DataFrame(np.transpose(np.squeeze(predictions_all))).mean(axis=1)
        pd.DataFrame(averaged_preds).to_pickle(os.path.join(t,"5_runs_averaged_preds.pkl") )
        
        for i in range(0,np.shape(predictions_all)[0]):
            print('model ', all_models[i])
            print('AUC PR: ', average_precision_score(y_tests, predictions_all[i]))
            print('AUC ROC: ', roc_auc_score(y_tests, predictions_all[i]))    
           
        prec_av, recall_av, _ = precision_recall_curve(y_tests, averaged_preds)
        fpr_av, tpr_av, _ = roc_curve(y_tests, averaged_preds)
        print('ensemble AUC PR: ', average_precision_score(y_tests, averaged_preds))
        print('ensemble AUC ROC: ', roc_auc_score(y_tests, averaged_preds))
        
        logger_wrapper.logger.info(option_no)
        logger_wrapper.logger.info('PR ensemble:')
        logger_wrapper.logger.info(average_precision_score(y_tests, averaged_preds))
        logger_wrapper.logger.info('ROC AUC ensemble:')
        logger_wrapper.logger.info(roc_auc_score(y_tests, averaged_preds))
        # %% ensemble of 5 models, plots and saving
        
        fig = plt.figure(figsize=(10,10))
        plt.rcParams.update({'font.size': 22})
        plt.title('AUC PR')
        plt.plot(recall_av, prec_av,  'b', label = 'ResNet AUC = %0.2f%% ' % (100*average_precision_score(y_tests, averaged_preds)))    
        plt.legend(loc = 'upper right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Precyzja (Precision)')
        plt.xlabel('Czułość (Recall)')
        fig.savefig(os.path.join(t,'AUC-PR-averaged5-' + time.strftime('-%Y-%m-%d-%H-%M')))
           
        fig, ax = plt.subplots(figsize=(10,10))
        plt.rcParams.update({'font.size': 22})
        plt.title('AUC ROC')
        plt.plot(fpr_av, tpr_av, 'b', label = 'ResNet AUC = %0.2f%%' % (100*roc_auc_score(y_tests, averaged_preds)))             
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Wskaźnik czułości (True Positive Rate)')
        plt.xlabel('Odsetek fałszywie pozytywnych alarmów (False Positive Rate)')
        fig.savefig(os.path.join(t,'AUC-ROC-averaged5-'+time.strftime('-%Y-%m-%d-%H-%M')))
               
        # %% plotting single models
        aucpr_data = []
        aucroc_data = []
        for n in range(0,5):
            y_preds = predictions_all[n] 
            prec_nb, recall_nb, _ = precision_recall_curve(y_tests, y_preds)
            fpr_nb, tpr_nb, _ = roc_curve(y_tests, y_preds)
            print('AUC PR: ', average_precision_score(y_tests, y_preds))
            
            fig = plt.figure(figsize=(10,10))
            plt.rcParams.update({'font.size': 22})
            plt.title('AUC PR')
            plt.plot(recall_nb, prec_nb,  'b', label = 'ResNet AUC = %0.2f%% ' % (100*average_precision_score(y_tests, y_preds)))
            plt.legend(loc = 'upper right')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('Precyzja (Precision)')
            plt.xlabel('Czułość (Recall)')
            fig.savefig(os.path.join(t,'AUC-PR-refit-'+ str(n) + time.strftime('-%Y-%m-%d-%H-%M')))
            
            fig = plt.figure(figsize=(10,10))
            plt.rcParams.update({'font.size': 22})
            plt.title('AUC ROC')
            plt.plot(fpr_nb, tpr_nb, 'b', label = 'ResNet AUC = %0.2f%%' % (100*roc_auc_score(y_tests, y_preds)))
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('Wskaźnik czułości (True Positive Rate)')
            plt.xlabel('Odsetek fałszywie pozytywnych alarmów (False Positive Rate)')
            fig.savefig(os.path.join(t,'AUC-ROC-refit-'+ str(n) +time.strftime('-%Y-%m-%d-%H-%M')))
            
            plt.clf()
            aucpr_data.append(average_precision_score(y_tests, y_preds))
            aucroc_data.append(roc_auc_score(y_tests, y_preds))
            
        print('PR AUC mean: ' , np.mean(aucpr_data), '+/- std', np.std(aucpr_data) )
        print('ROC AUC mean: ' , np.mean(aucroc_data), '+/- std', np.std(aucroc_data) )
        
        logger_wrapper.logger.info('PR AUC mean i std:')
        logger_wrapper.logger.info(np.mean(aucpr_data))
        logger_wrapper.logger.info(np.std(aucpr_data))
        logger_wrapper.logger.info('ROC AUC mean i std:')
        logger_wrapper.logger.info(np.mean(aucroc_data))
        logger_wrapper.logger.info(np.std(aucroc_data))      
        

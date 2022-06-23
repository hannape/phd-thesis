# -*- coding: utf-8 -*-
import os
import config as cfg
import numpy as np
import pandas as pd
import time 
import pickle 
import joblib

from tpot.export_utils import set_param_recursive
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.calibration import CalibratedClassifierCV

from data_load import function_data_load  # if problems with loading: make sure main folder is added to the python path
from representations_all2 import function_representations_all2
from useful_functions import make_cv_split, draw_pr_roc_charts   
from useful_functions import read_best_models_8_classic_only_params, read_best_models_8_classic_plus_MIR_only_params
from saving_utils import my_save_to_file

''' Checking best models from 6 classifier families on a test set, for 8_classic or 8_classic_plus_MIR representation.
    Creating predictions and drawing PR and ROC plots.
'''

########## %% Parameters to set
chosen_repr = '8_classic' # representation name, for which we want to check best models. 8_classic or 8_classic_plus_MIR
if_scaler = True                   # if the scaler is used on testing data.
nr_rec_start = 0                   # starting rec number, needed if we want to read many long recordings
rec_batch_size = 18                # recording reading batch size
#################

df1 = pd.read_hdf(os.path.join('..','..','data','dataframe','df_train_'+ str(chosen_repr) +'_norm.h5'),'df')
df2 = pd.read_hdf(os.path.join('..','..','data','dataframe','df_valid_'+ str(chosen_repr) +'_norm.h5'),'df') 

_, training_features, training_target, training_id = make_cv_split(df1, df2, chosen_repr) 
_, training_features_nb, training_target, training_id = make_cv_split(df1, df2, chosen_repr, 'nb') 

#%% DATA LOAD: Read data from recordings, read labels, create chunks, create testing set

start_time = time.time()
test_new_only = 1
_,_,_, data_test_new = function_data_load(cfg.path_test1618_txt, cfg.path_train161718_txt, cfg.path_test1618_wav,\
                                          cfg.path_train161718_wav, cfg.balance_types, cfg.balance_ratios, cfg.chunk_length_ms,\
                                          cfg.chunk_overlap, cfg.calls_0, cfg.calls_1, cfg.calls_unknown, cfg.tolerance, \
                                          cfg.valid_set, cfg.test_rec_to_cut, cfg.columns_dataframe, test_new_only)
print("--- Data load:: %s sec ---" % (time.time() - start_time))
# %% REPRESENTATIONS: Create representations for test set

start_time = time.time()

file_names, indices, info_chunksy, repr_full_scaled = function_representations_all2(cfg.path_test1618_wav, data_test_new, if_scaler, \
                                                   cfg.repr_1d_summary, cfg.summary_1d, cfg.sr, cfg.chunk_length_ms, \
                                                   cfg.n_fft, cfg.win_length, cfg.hop_length, cfg.window, cfg.f_min, cfg.f_max,\
                                                   cfg.n_mels, cfg.N, cfg.step, cfg.Np, cfg.K, cfg.tm, cfg.flock, cfg.tlock,\
                                                   chosen_repr, nr_rec_start, rec_batch_size)    
    
print("--- Data representation: %s min ---" % ((time.time() - start_time)/60))
# %%
file_names_list, chunk_ids_list, has_bird_list, representation_list = [],[],[],[]

for num,i in enumerate(indices[nr_rec_start:nr_rec_start+rec_batch_size]):

  file_names_list.extend([file_names[num] for i in range(len(i))])
  chunk_ids_list.extend((info_chunksy[num][0])) 
  has_bird_list.extend((info_chunksy[num][3]))
  representation_list.extend([repr_full_scaled[num][i] for i in range(len(i))])

testing_target = has_bird_list

testing_features = np.array(representation_list) 
testing_features = np.reshape(testing_features, newshape=(testing_features.shape[0],-1))

testing_features_nb = np.array(representation_list)
testing_features_nb[testing_features_nb < 0] = 0 # Naive bayes, non negative values
testing_features_nb = np.reshape(testing_features_nb, newshape=(testing_features_nb.shape[0],-1))

print("Test loaded")

# %% Reading best pipelines 
print("Read best models pipelines for 6 classifier families...")

# best pipelines as reported in the thesis. For including yours' calculated from main_tpot_hyperparameters_search,
# add their definitions taken from [...]-best_model_script.py 

if chosen_repr == '8_classic':
    exported_pipeline_nb, exported_pipeline_trees, exported_pipeline_svm, exported_pipeline_logreg,\
    exported_pipeline_kn, exported_pipeline_mlp = read_best_models_8_classic_only_params(training_features, training_target, training_features_nb)
if chosen_repr == '8_classic_plus_MIR':
    exported_pipeline_nb, exported_pipeline_trees, exported_pipeline_svm, exported_pipeline_logreg, \
    exported_pipeline_kn, exported_pipeline_mlp = read_best_models_8_classic_plus_MIR_only_params(training_features, training_target, training_features_nb)    
 
# %% Prediction and checking average ROC and PR AUC on 5 seeds

random_state = [665, 666, 667, 668, 669]
resultsy_names = ['nb', 'trees', 'svm', 'logreg', 'kn', 'mlp']
name_charts = ['NB', 'Drzewa', 'SVM', 'LogReg', 'KNN', 'MLP']
color_charts = ['b', 'r', 'g', 'y', 'k', 'orange']
exported_pipelines = [exported_pipeline_nb, exported_pipeline_trees, exported_pipeline_svm, 
                      exported_pipeline_logreg, exported_pipeline_kn, exported_pipeline_mlp]
ens_data = []

for model_type in range(0,len(resultsy_names)):
    print(model_type)
    preds_data = []
    pr_data,roc_data = [],[]
    classifier = resultsy_names[model_type]
    repr_classifier = chosen_repr + '-' + classifier
    saving_dir_folder_model = os.path.join('..','results','PR','charts_data_6_5', chosen_repr, classifier)
    try:
        os.makedirs(saving_dir_folder_model)
    except FileExistsError:
        pass
    
    for rs in (random_state):
    
        exported_pipeline = exported_pipelines[model_type]
        # Fix random state for all the steps in exported pipeline
        if resultsy_names[model_type]!="logreg":  #log reg solver newton doesn't have random_state
            set_param_recursive(exported_pipeline.steps, 'random_state', rs)

        if resultsy_names[model_type]=="svm":
           
            clf = CalibratedClassifierCV(exported_pipeline) 
            clf.fit(training_features, training_target)
            print('loaded model ' + chosen_repr +' ' + resultsy_names[model_type] + ', random seed: ' + str(rs))
            results_ = clf.predict_proba(testing_features) 
        
        else:
            exported_pipeline.fit(training_features, training_target)
            print('loaded model  ' + chosen_repr +' ' + resultsy_names[model_type] + ', random seed: ' + str(rs))
            results_ = exported_pipeline.predict_proba(testing_features)
        
        roc_auc = roc_auc_score(testing_target, results_[:,1])
        pr_auc = average_precision_score(testing_target, results_[:,1])
        print('ROC AUC on test set: ', roc_auc)
        print('PR AUC on test set: ', pr_auc)
        resultsy = results_[:,1]
        preds_data.append(resultsy)
        roc_data.append(roc_auc)
        pr_data.append(pr_auc)
        
        predicted_probabilities = resultsy
        filename = chosen_repr + '-' + classifier + '-' + str(rs)+'-test_new.predictions' 
        model_name = chosen_repr + '-' + classifier + '-' + str(rs)
        
        joblib.dump(exported_pipeline, os.path.join(saving_dir_folder_model, model_name)) 
        my_save_to_file(testing_target, predicted_probabilities, filename, saving_dir_folder_model)
        
        print('saved: ' + filename)
    
    pd.DataFrame(np.transpose(np.squeeze(preds_data)),
                  columns = ['preds-1','preds-2','preds-3','preds-4','preds-5']).to_pickle(os.path.join(saving_dir_folder_model,"5_runs_preds.pkl") )     
    averaged_preds = pd.DataFrame(np.transpose(np.squeeze(preds_data))).mean(axis=1)
    pd.DataFrame(averaged_preds).to_pickle(os.path.join(saving_dir_folder_model,"5_runs_averaged_preds.pkl") )
    
    ens_data.append(averaged_preds)
    prec_av, recall_av, _ = precision_recall_curve(testing_target, averaged_preds)
    fpr_av, tpr_av, _ = roc_curve(testing_target, averaged_preds)
    print('ensemble AUC PR: ', average_precision_score(testing_target, averaged_preds))
    print('ensemble AUC ROC: ', roc_auc_score(testing_target, averaged_preds))   
    print('ROC AUC ', np.mean(roc_data), '  ', np.std(roc_data))
    print('PR AUC ', np.mean(pr_data), '  ', np.std(pr_data))    
    
# %% Final ROC & PR charts for best models from 6 classifier families 

pred_results = np.zeros((len(has_bird_list), len(resultsy_names)))

saving_dir = os.path.join('..','results','PR','charts_data_6_5')
saving_dir_folder = os.path.join(saving_dir,chosen_repr)
results_files = os.listdir(os.path.join(saving_dir, chosen_repr))

for model_type in range(0,1):
   
    saving_dir_folder_model = os.path.join(saving_dir,chosen_repr,resultsy_names[model_type])    
    f = open(os.path.join(saving_dir_folder_model,'5_runs_averaged_preds.pkl'), 'rb')
    my_preds = pickle.load(f)
    f.close()        
    pred_results[:,model_type] = np.squeeze(my_preds.to_numpy()) 
 
testing_target = np.array(pd.read_csv(os.path.join(saving_dir_folder,'nb','8_classic-nb-665-test_new.predictions'),sep='\t')['has_bird'])
pd.DataFrame(testing_target).to_pickle(os.path.join(saving_dir,"testing_target_backup.pkl") )

draw_pr_roc_charts(testing_target, pred_results)
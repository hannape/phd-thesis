# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import os
import calendar
import time
import torch
import sys

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from datetime import date
from numpy.random import seed
seed(1)
tf.random.set_seed(2)

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
experimentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(experimentdir)
sys.path.insert(0,parentdir) 

import config
from _2_TPOT_experiments.src.saving_utils import LoggerWrapper
from useful_func_resnet import load_my_data_resnet_augment
from useful_func_resnet import opcja1, opcja2, opcja3, opcja4, opcja5, opcja6, opcja7, opcja8
    
# allocating fraction of GPU memory
from tensorflow.compat.v1.keras.backend import set_session
config2 = tf.compat.v1.ConfigProto()
config2.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.compat.v1.Session(config=config2))

"""
    Grid search for chosen ResNet option - transfer learning scenarios (opcja1-8). Hyperparameter search for learning rate and batch size.

    Running script:python script_name chosen_representation scenario_chosen if_the_same_3layers(=1) learning_rate_grid batch_size_grid
    e.g. python main_training_resnet_grid.py mel-spektrogram 7 1 0.00005,0.0001,0.0003,0.0005,0.001 8,16,32
    
    Scenarios described in useful_func_resnet
"""
 
# %%

epochs = 1      # max number of epochs (+ early stopping condition) 
n_models = 1      # how many times model is trained: one model calculated for grid search procedure
if_best_score = 1 # if_best_score = 1: we choose the best score. if_best_score = 0: we choose the last score

n_epochs_stop = 5
epochs_no_improve = 0
early_stop = False

if __name__=='__main__':

    # Load configs
    chosen_repr = sys.argv[1]
    option_nr = int(sys.argv[2])    
    the_same_3layers= int(sys.argv[3]) 
    exp_dir = os.path.join("..","results","grid_search", f"{chosen_repr}", time.strftime('%Y-%m-%d-%H-%M'))
    learning_rates = list(map(float, sys.argv[4].split(',')))
    batch_sizes = list(map(int, sys.argv[5].split(',')))

    try:
        os.makedirs(exp_dir)
    except FileExistsError:
        pass
 # %%  
 
    start_time = time.time()
    logger_wrapper = LoggerWrapper(exp_dir)
    sys.stderr.write = logger_wrapper.log_errors
    logger_wrapper.logger.info(f'Running {sys.argv}')
    
    if chosen_repr=="spektrogram" or chosen_repr=="mel-spektrogram":
        _, _, _, _, _, X_val, y_val, train_X, train_y = load_my_data_resnet_augment(chosen_repr, config, 'wlasc', -1) 
        if chosen_repr=="spektrogram":
            in_shape = (63, 148, 1)
        if chosen_repr=="mel-spektrogram":
            in_shape = (60, 148, 1)  

    if the_same_3layers==1:
        print("The same representation in 3 channels")
        X_val_resh = X_val.reshape(X_val.shape[0], in_shape[0], in_shape[1], 1).astype('float32') 
        X_train_resh = train_X.reshape(train_X.shape[0], in_shape[0], in_shape[1], 1).astype('float32') 
        image_train = (np.repeat(X_train_resh, 3, axis=3)).transpose((0,3, 1, 2))
        image_val = (np.repeat(X_val_resh, 3, axis=3)).transpose((0,3, 1, 2))
 
    X_train = torch.Tensor(image_train) 
    y_train = torch.LongTensor(train_y.astype(int)) 
    X_valid = torch.Tensor(image_val) 
    y_valid = torch.LongTensor(y_val.astype(int))     
    
    dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)

## Scenarios description (in polish)
# opis opcji = ["ResNet-18, nieinicjalizujemy wagami, bierzemy tylko architekturę", "ResNet-18, inicjalizujemy wagami, ale uczymy całą sieć od nowa", 
#               "ResNet-18,, inicjalizujemy i zamrażamy pierwszą warstwę, reszta retrenowane", "ResNet-18, inicjalizujemy i zamrażamy wszystkie wagi oprócz dwóch ostatnich warstw które będą retrenowane",
#               "ResNet-18, inicjalizujemy i zamrażamy wszystkie wagi oprócz ostatniej warstwy która będzie retrenowana",
#               "ResNet-50, nieinicjalizujemy wagami, bierzemy tylko architekturę", "ResNet-50, inicjalizujemy wagami, ale uczymy całą sieć od nowa", 
#               "ResNet-50,, inicjalizujemy i zamrażamy pierwszą warstwę, reszta retrenowane",]  

option_description = ["ResNet-18 architecture, no weight initialisation", "ResNet-18, Imagenet initialisation, learning from scratch", 
              "ResNet-18, Imagenet initialisation, freezing conv1 and layer1", "ResNet-18, Imagenet initialisation and freezing all but last two layers, which will be trained",
              "ResNet-18, Imagenet initialisation and freezing all weights but last layer, which will be trained",
              "ResNet-50 architecture, no weight initialisation", "ResNet-50, Imagenet initialisation, learning from scratch", 
              "ResNet50, Imagenet initialisation, freezing conv1 and layer1",]  

ts = calendar.timegm(time.gmtime())
#%%
today = date.today()
print("option "+ str(option_nr)+ " - "+ option_description[option_nr-1])

for my_bs in batch_sizes:    
  print('---BATCH: ', my_bs)  
  t1 = time.localtime()
  ts = time.strftime("%H-%M", t1)
  
  data_loader = DataLoader(dataset, batch_size = my_bs, shuffle = True)
  valid_data_loader = DataLoader(valid_dataset, batch_size = my_bs)

  for my_lr in learning_rates: 
  
      print('---LR: ', my_lr)   
      
      for run in range(0, n_models):
       
          print("batch size: ", my_bs, "learning rate: ", my_lr, "run: ", run)
          
          start_time =  time.time()
          losses, scores = [], []
          epoch = 0
          clf = []
          opcje_modeli2 = [opcja1(),opcja2(),opcja3(),opcja4(),opcja5(),opcja6(), opcja7(), opcja8()]
          clf = opcje_modeli2[option_nr-1]
    
          # Criterion and optimiser
          criterion = torch.nn.CrossEntropyLoss()
          optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, clf.parameters()), lr = my_lr)  
    
          best_preds, best_score = None, 100.
          torch.set_num_threads(55)
    
          for epoch in range(epochs):
              running_loss = 0
              valid_loss = 0
              clf.train()
              clf.cuda()
              batch = 0
              for X, y in data_loader:
                  batch = batch+1
                  optimizer.zero_grad()
    
                  outputs = clf(X.cuda())
                  loss = criterion(outputs, y.cuda())
                  loss.backward()
                  optimizer.step()
    
                  running_loss += loss.item()
    
              losses.append(running_loss)
    
              clf.eval()
              preds = []
              for X, y in valid_data_loader:
                  out = clf(X.cuda())
                  preds.append(torch.softmax(out, dim = 1)[:, 1].cpu().detach().numpy())
                  loss_val = criterion(out, y.cuda())
                  valid_loss += loss_val.item()
              preds = np.concatenate(preds, axis = 0)
    
              # loss valid is a testing metric
              roc_score = roc_auc_score(y_valid.numpy(), preds)
              pr_score = average_precision_score(y_valid.numpy(), preds)
              score = valid_loss
              scores.append(score)
              print('Epoch:', epoch+1, 'ROC Score:', np.round(roc_score,4), 'PR Score:', np.round(pr_score,4), 'Trains Loss', np.round(running_loss,4), 
                    'Valid Loss ', np.round(valid_loss,4))
              if (if_best_score == 1) & (score < best_score):
                  epochs_no_improve = 0
                  best_score = score
                  best_preds = preds
                  best_pr = pr_score
                  print('best score: ', np.round(best_score,4))
                  torch.save(clf.state_dict(), 'tmp_model_best.pt')
              else:
                  epochs_no_improve += 1
                  print('epochs_no_improve: ',epochs_no_improve)
                  
              torch.save(clf.state_dict(), 'tmp_model_last.pt')
    
              if (epochs_no_improve == n_epochs_stop):
                  print('Early stopping!' )
                  early_stop = True
                  break
              else:
                  continue
    
              # Check early stopping condition         
              if early_stop:
                  print("Stopped")
                  break
          
          last_score = score
          last_preds = preds        
          print(run, "best score", best_score)
          print(run, "last score", last_score)
          print('Saving')
          state_best = torch.load('tmp_model_best.pt')
          state_last = torch.load('tmp_model_last.pt')
          t = os.path.join(exp_dir , str(option_nr) + "_" + str(epochs) + "_"+ str(ts))
          os.makedirs(t, exist_ok=True)
          
          torch.save(state_best, os.path.join(t , 'run_' + str(run) + "_" + str(epoch+1-n_epochs_stop) + 
                                              'ep_valloss' + str(np.round(best_score,4)) + '_pr'+ str(np.round(best_pr,4)) +
                                              '_bs' + str(my_bs) + '_lr' + str(my_lr) + '.pt'))
          end = time.time()
          print(end - start_time)    
          del clf        
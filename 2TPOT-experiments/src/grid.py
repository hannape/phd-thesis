# -*- coding: utf-8 -*-

import numpy as np

from collections import namedtuple
from copy import deepcopy
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tpot.config.classifier import classifier_config_dict

def get_grid(model):
    model = model.lower().strip()
    print(model)
    if 'svm' == model:
        return get_svm_classification_grid()
    elif model in ['tree', 'trees']:
        return get_tree_classification_grid()
    elif model in ['nb', 'naive-bayes', 'naive_bayes']:
        return get_nb_classification_grid()
    elif model=='logreg':
        return get_logreg_classification_grid()
    elif model=='kn':
        return get_kn_classification_grid()
    elif model=='mlp':
        return get_mlp_classification_grid()
    else:
        raise ValueError(f'For classification `model` parameter must be `svm`, `tree`, `nb`, `logreg`\
                         `kn`,`mlp`, is {model}.')
    
# dummy classes
class SVC_rbf(SVC):
    pass

class SVC_poly(SVC):
    pass

class SVC_sigmoid(SVC):
    pass

class SVC_linear(SVC):
    pass

class logreg_elasticnet(LogisticRegression):
    pass

class logreg_solvers(LogisticRegression):
    pass

class logreg_saga(LogisticRegression):
    pass

def remove_classifiers(config_dict, remove_nb=False, remove_trees=False, remove_svms=False, remove_logreg=False,
                       remove_kn=False, remove_mlp=False, remove_other=True):
    nbs = []
    svms = []
    trees = []
    logreg = []
    kn = []
    mlp = []
    other_classifiers = []
    preprocessing = []
    
    for key in config_dict:
        if 'naive_bayes' in key:
            nbs.append(key)
        elif any(i in key for i in ['tree', 'ExtraTreesClassifier', 'RandomForestClassifier']):
            trees.append(key)
        elif 'svm' in key:
            svms.append(key)
        elif 'LogisticRegression' in key:   
            logreg.append(key)
        elif 'KNeighborsClassifier' in key:
            kn.append(key)  
        elif 'neural_network.MLPClassifier' in key:
            mlp.append(key) 
            
        elif any(i in key for i in
                 ['preprocessing', 'feature_selection', 'builtins.ZeroCount','builtins.OneHotEncoder',
                  'decomposition', 'kernel_approximation', 'FeatureAgglomeration']):
            preprocessing.append(key)
        else:
            other_classifiers.append(key)

    to_remove = []
    if remove_nb:
        to_remove.extend(nbs)

    if remove_trees:
        to_remove.extend(trees)

    if remove_svms:
        to_remove.extend(svms)
    
    if remove_logreg:
        to_remove.extend(logreg)
    
    if remove_kn:
        to_remove.extend(kn)    
    
    if remove_mlp:
        to_remove.extend(mlp)        

    if remove_other:
        to_remove.extend(other_classifiers)

    for key in to_remove:
        del config_dict[key]

    return config_dict


def get_logreg_config():
    logreg_config = namedtuple('logreg_config', ['c', 'tol', 'max_iter', 'solver', 'l1_ratio']) 
    cfg = logreg_config(c=[0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
                     tol=[1e-05, 0.0001, 0.001, 0.01, 0.1],
                     max_iter=[1000,], solver=['newton-cg', 'sag', 'lbfgs'], 
                     l1_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] )  
    return cfg

def get_logreg_classification_grid():
    cfg = get_logreg_config()

    logreg_solvers = {'C': cfg.c, 'tol': cfg.tol,
                   'penalty': ['l2', 'none'], 'max_iter': cfg.max_iter,
                   'solver': cfg.solver}
    
    logreg_saga = {'C': cfg.c, 'tol': cfg.tol,
                   'penalty': ['l1','l2', 'none'], 'max_iter': cfg.max_iter,
                   'solver': ['saga']}
    
    logreg_elasticnet = {'C': cfg.c, 'tol': cfg.tol,
                   'penalty': ['elasticnet'], 'max_iter': cfg.max_iter,
                   'solver': ['saga'],  'l1_ratio': cfg.l1_ratio}
    
    logreg_grid = deepcopy(classifier_config_dict)
    logreg_grid = remove_classifiers(logreg_grid, remove_svms=True, remove_nb=True, remove_trees=True,
                                     remove_logreg=True, remove_kn=True, remove_mlp=True, remove_other=True)
    

    logreg_grid['grid.logreg_solvers'] = logreg_solvers
    logreg_grid['grid.logreg_saga'] = logreg_saga
    logreg_grid['grid.logreg_elasticnet'] = logreg_elasticnet

    return logreg_grid
 
def get_kn_classification_grid():

    kn_grid = deepcopy(classifier_config_dict)
    kn_grid = remove_classifiers(kn_grid, remove_svms=True, remove_nb=True, remove_trees=True, 
                                         remove_kn=False, remove_logreg=True, remove_mlp=True, remove_other=True)
    return kn_grid

def get_mlp_config():
    mlp_config = namedtuple('mlp_config', ['learning_rate_init', 'learning_rate','batch_size', 'max_iter',
                                           'hidden_layer_sizes','alpha'])
      
    cfg = mlp_config(learning_rate_init= [1e-3, 1e-2, 1e-1, 0.5, 1.],
        learning_rate=['constant', 'adaptive'],
        batch_size=[4, 8, 16, 32, 64],
        max_iter=[10, 15, 25, 50, 100, 200],
        hidden_layer_sizes=[(100,),(200,),(500),(100,50),(200,100),(50,20,10),(10,10,10,10)],
        alpha = [0, 1e-4, 1e-3, 1e-2, 0.1])
       
    return cfg


def get_mlp_classification_grid():
    cfg = get_mlp_config()
        
    mlp_grid = deepcopy(classifier_config_dict) 
    mlp_grid = remove_classifiers(mlp_grid, remove_svms=True, remove_nb=True, remove_trees=True, 
                                  remove_kn=True, remove_logreg=True, remove_mlp=True, remove_other=True)
    mlp_grid['sklearn.neural_network.MLPClassifier'] = {'learning_rate': cfg.learning_rate,
                                                        'learning_rate_init': cfg.learning_rate_init,
                            'batch_size': cfg.batch_size, 'max_iter': cfg.max_iter,
                            'hidden_layer_sizes': cfg.hidden_layer_sizes,'alpha':cfg.alpha}
    return mlp_grid
    
def get_svm_config():
    SVM_config = namedtuple('SVM_config', ['c', 'gamma', 'coef0', 'degree', 'tol', 'epsilon', 'max_iter', 'probability','penalty','loss','dual'])
    cfg = SVM_config(c=[0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
                     gamma=['auto', 'scale'] + [10 ** i for i in range(-6, 0)],
                     coef0=[-10 ** i for i in range(-6, 0)] + [0.0] + [10 ** i for i in range(-1, -7, -1)],
                     degree=list(range(1, 3)),
                     tol=[1e-05, 0.0001, 0.001, 0.01, 0.1],
                     epsilon=[0.0001, 0.001, 0.01, 0.1, 1.0],
                     max_iter=[1000,], probability=[True,],
                     penalty=['l1','l2'], loss =['hinge', 'squared_hinge'], dual=[True, False])  
    return cfg

def get_svm_classification_grid():
    cfg = get_svm_config()

    cache_size = 100
    svc_rbf = {'C': cfg.c, 'gamma': cfg.gamma, 'tol': cfg.tol,
               'kernel': ['rbf'],
               'max_iter': cfg.max_iter, 'probability': cfg.probability, 'cache_size': [cache_size,]}
    svc_sigmoid = {'C': cfg.c, 'gamma': cfg.gamma, 'tol': cfg.tol,
                   'kernel': ['sigmoid'], 'coef0': cfg.coef0,
                   'max_iter': cfg.max_iter, 'probability': cfg.probability, 'cache_size': [cache_size,]}
    svc_linear = {'C': cfg.c, 'gamma': cfg.gamma, 'tol': cfg.tol,
                   'kernel': ['linear'], 
                   'max_iter': cfg.max_iter, 'probability': cfg.probability, 'cache_size': [cache_size,]}
    svc_linear2 = {'C': cfg.c, 'tol': cfg.tol,
                   'penalty': cfg.penalty, 'loss': cfg.loss, 
                   'max_iter': cfg.max_iter, 'dual':cfg.dual}

    svm_grid = deepcopy(classifier_config_dict)
    svm_grid = remove_classifiers(svm_grid, remove_svms=False, remove_nb=True, remove_trees=True,\
                                  remove_logreg=True, remove_kn=True, remove_mlp=True, remove_other=True)

    svm_grid['grid.SVC_rbf'] = svc_rbf
    svm_grid['grid.SVC_sigmoid'] = svc_sigmoid
    svm_grid['grid.SVC_linear'] = svc_linear
    svm_grid['sklearn.svm.LinearSVC'] = svc_linear2

    return svm_grid

def get_tree_config():
    Tree_config = namedtuple('Tree_config',
                             ['n_estimators', 'max_depth', 'max_samples', 'splitter', 'max_features', 'bootstrap'])
    cfg = Tree_config(n_estimators=[10, 50, 100, 500, 1000],
                      max_depth=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, None],
                      max_samples=[None, 0.5, 0.7, 0.9],
                      splitter=['best', 'random'],
                      max_features=np.arange(0.05, 1.01, 0.05),
                      bootstrap=[True, False])
    return cfg

def get_tree_classification_grid():
    cfg = get_tree_config()

    etc = 'sklearn.ensemble.ExtraTreesClassifier'
    dtc = 'sklearn.tree.DecisionTreeClassifier'
    rfc = 'sklearn.ensemble.RandomForestClassifier'

    tree_grid = deepcopy(classifier_config_dict)
    tree_grid = remove_classifiers(tree_grid, remove_trees=False, remove_nb=True, remove_svms=True, remove_logreg=True, remove_kn=True, remove_mlp=True, remove_other=True)

    tree_grid[etc]['n_estimators'] = cfg.n_estimators
    tree_grid[etc]['max_depth'] = cfg.max_depth
    tree_grid[etc]['max_samples'] = cfg.max_samples

    tree_grid[dtc]['splitter'] = cfg.splitter
    tree_grid[dtc]['max_depth'] = cfg.max_depth
    tree_grid[dtc]['max_features'] = cfg.max_features

    tree_grid[rfc]['n_estimators'] = cfg.n_estimators
    tree_grid[rfc]['max_depth'] = cfg.max_depth
    tree_grid[rfc]['bootstrap'] = cfg.bootstrap
    tree_grid[rfc]['max_samples'] = cfg.max_samples

    return tree_grid

def get_nb_config():
    NB_config = namedtuple('NB_config', ['alpha', 'fit_prior', 'norm', 'var_smoothing'])
    cfg = NB_config(alpha=[1e-3, 1e-2, 1e-1, 1., 10., 100.], fit_prior=[True, False], norm=[True, False],
                    var_smoothing=[1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4])

    return cfg

def get_nb_classification_grid():

    cfg = get_nb_config()

    bayes_grid = deepcopy(classifier_config_dict)
    bayes_grid = remove_classifiers(bayes_grid, remove_nb=True, remove_trees=True, remove_svms=True, remove_logreg=True, remove_kn=True, remove_mlp=True, remove_other=True)

    bayes_grid['sklearn.naive_bayes.BernoulliNB'] = {'alpha': cfg.alpha, 'fit_prior': cfg.fit_prior}
    bayes_grid['sklearn.naive_bayes.ComplementNB'] = {'alpha': cfg.alpha, 'fit_prior': cfg.fit_prior, 'norm': cfg.norm}
    bayes_grid['sklearn.naive_bayes.GaussianNB'] = {'var_smoothing': cfg.var_smoothing}
    bayes_grid['sklearn.naive_bayes.MultinomialNB'] = {'alpha': cfg.alpha, 'fit_prior': cfg.fit_prior}

    return bayes_grid
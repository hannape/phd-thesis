# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

def make_cv_split(train, valid, chosen_repr, classifier=False):   
    # making a cross-validation split of training and validation dataset
    train = train.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
    train_identifiers = train[:, 0:2]
    train_X = np.stack(train[:, 2])
    train_y = train[:, 3]
    
    valid = valid.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
    valid_identifiers = valid[:, 0:2]
    valid_X = np.stack(valid[:, 2])
    valid_y = valid[:, 3]
    if classifier=='nb': 
        valid_X[valid_X < 0] = 0 
    dX = np.vstack((train_X, valid_X))
    dy = np.hstack((train_y, valid_y))
    di = np.vstack((train_identifiers, valid_identifiers))
    
    train_indices = np.array(range(0, train_X.shape[0]))
    val_indices = np.array(range(train_X.shape[0], dX.shape[0]))
    
    cv_split = [(train_indices, val_indices)] 
    
    dX = np.reshape(dX, newshape=(dX.shape[0],-1))
    dy = dy.astype(int)

    return cv_split, dX, dy, di

def make_test(test, chosen_repr):
    # making a test set transformations and reshaping
    test = test.loc[:,['rec_name', 'chunk_ids', chosen_repr, 'has_bird']].values
    test_identifiers = test[:, 0:2]
    test_X = np.stack(test[:, 2])
    test_y = test[:, 3]
    dX = np.reshape(test_X, newshape=(test_X.shape[0],-1))
    dy = test_y.astype(int)
    
    return dX, dy, test_identifiers

def draw_pr_roc_charts(testing_target, pred_results):    
    # drawing PR and ROC charts for all six models
    
    results_nb = pred_results[:,0] 
    results_trees = pred_results[:,1] 
    results_svm = pred_results[:,2] 
    results_logreg = pred_results[:,3] 
    results_kn = pred_results[:,4] 
    results_mlp = pred_results[:,5] 
    
    prec_nb, recall_nb, _ = precision_recall_curve(testing_target, results_nb)
    fpr_nb, tpr_nb, _ = roc_curve(testing_target, results_nb)
    
    prec_trees, recall_trees, _ = precision_recall_curve(testing_target, results_trees)
    fpr_trees, tpr_trees, _ = roc_curve(testing_target, results_trees)
           
    prec_svm, recall_svm, _ = precision_recall_curve(testing_target, results_svm)
    fpr_svm, tpr_svm, _ = roc_curve(testing_target, results_svm)
    
    prec_logreg, recall_logreg, _ = precision_recall_curve(testing_target, results_logreg)
    fpr_logreg, tpr_logreg, _ = roc_curve(testing_target, results_logreg)
    
    prec_kn, recall_kn, _ = precision_recall_curve(testing_target, results_kn)
    fpr_kn, tpr_kn, _ = roc_curve(testing_target, results_kn)
    
    prec_mlp, recall_mlp, _ = precision_recall_curve(testing_target, results_mlp)
    fpr_mlp, tpr_mlp, _ = roc_curve(testing_target, results_mlp)
    
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10,10))
    plt.title('AUC ROC')
    plt.plot(fpr_nb, tpr_nb, 'b', label = 'NB AUC = %0.2f%%' % (100*roc_auc_score(testing_target, results_nb)))
    plt.plot(fpr_trees, tpr_trees, 'r', label = 'Drzewa AUC = %0.2f%% ' % (100*roc_auc_score(testing_target, results_trees)))
    plt.plot(fpr_svm, tpr_svm, 'g', label = 'SVM AUC = %0.2f%% ' % (100*roc_auc_score(testing_target, results_svm)))
    plt.plot(fpr_logreg, tpr_logreg, 'y', label = 'LogReg AUC = %0.2f%% ' % (100*roc_auc_score(testing_target, results_logreg)))
    plt.plot(fpr_kn, tpr_kn, 'k', label = 'KNN AUC = %0.2f%% ' % (100*roc_auc_score(testing_target, results_kn)))
    plt.plot(fpr_mlp, tpr_mlp, 'orange', label = 'MLP AUC = %0.2f%% ' % (100*roc_auc_score(testing_target, results_mlp)))
    
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Wskaźnik czułości (True Positive Rate)')
    plt.xlabel('Odsetek fałszywie pozytywnych alarmów (False Positive Rate)')
    plt.show()
    
    plt.figure(figsize=(10,10))
    plt.title('AUC PR')
    plt.plot(recall_nb, prec_nb,  'b', label = 'NB AUC = %0.2f%% ' % (100*average_precision_score(testing_target, results_nb)))
    plt.plot(recall_trees, prec_trees, 'r', label = 'Drzewa AUC = %0.2f%% ' % (100*average_precision_score(testing_target, results_trees)))
    plt.plot(recall_svm, prec_svm, 'g', label = 'SVM AUC = %0.2f%% ' % (100*average_precision_score(testing_target, results_svm)))
    plt.plot(recall_logreg, prec_logreg, 'y', label = 'LogReg AUC = %0.2f%% ' % (100*average_precision_score(testing_target, results_logreg)))
    plt.plot(recall_kn, prec_kn, 'k', label = 'KNN AUC = %0.2f%% ' % (100*average_precision_score(testing_target, results_kn)))
    plt.plot(recall_mlp, prec_mlp, 'orange', label = 'MLP AUC = %0.2f%% ' % (100*average_precision_score(testing_target, results_mlp)))
    
    plt.legend(loc = 'upper right')
    #plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precyzja (Precision)')
    plt.xlabel('Czułość (Recall)')
    plt.show()

def draw_pr_roc_chart(testing_target, pred_results, name_chart, color_chart):
    # drawing PR and ROC chart for one model
    
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, average_precision_score
         
    prec_, recall_, _ = precision_recall_curve(testing_target, pred_results)
    fpr_, tpr_, _ = roc_curve(testing_target, pred_results)
      
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(10,10))
    plt.title('AUC ROC')
    plt.plot(fpr_, tpr_, color_chart, label = name_chart+' AUC = %0.2f%%' % (100*roc_auc_score(testing_target, pred_results)))
     
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Wskaźnik czułości (True Positive Rate)')
    plt.xlabel('Odsetek fałszywie pozytywnych alarmów (False Positive Rate)')
    plt.show()
    
    plt.figure(figsize=(10,10))
    plt.title('AUC PR')
    plt.plot(recall_, prec_, color_chart, label = name_chart+' AUC = %0.2f%% ' % (100*average_precision_score(testing_target, pred_results)))
       
    plt.legend(loc = 'upper right')
    #plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precyzja (Precision)')
    plt.xlabel('Czułość (Recall)')
    plt.show()

def read_best_models_8_classic_only_params(training_features, training_target, training_features_nb):
    # reading best model pipelines from 6 models families, for representation 8_classic
    
    # 8-classic-nb
    
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import RFE, SelectPercentile, f_classif
    from sklearn.naive_bayes import ComplementNB
    from sklearn.preprocessing import MinMaxScaler, Normalizer
    from sklearn.pipeline import make_pipeline, make_union
    from sklearn.preprocessing import FunctionTransformer
    from copy import copy

    # Average CV score on the training set was: 0.48654211566957467
    exported_pipeline_nb = make_pipeline(
        make_union(
            FunctionTransformer(copy),
            make_pipeline(
                MinMaxScaler(),
                Normalizer(norm="max")
            )
        ),
        MinMaxScaler(),
        SelectPercentile(score_func=f_classif, percentile=8),
        RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.8500000000000001, n_estimators=100), step=0.4),
        ComplementNB(alpha=0.001, fit_prior=False, norm=False)
    ) 
    # Fix random state for all the steps in exported pipeline
# =============================================================================
#     set_param_recursive(exported_pipeline_nb.steps, 'random_state', 667)
#     exported_pipeline_nb.fit(training_features_nb, training_target)
# =============================================================================
    
    # 8-classic-trees
    
    from sklearn.decomposition import FastICA, PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer
    
    # Average CV score on the training set was: 0.5250095713407659
    exported_pipeline_trees = make_pipeline(
        PCA(iterated_power=5, svd_solver="randomized"),
        Normalizer(norm="l1"),
        FastICA(tol=0.1),
        RandomForestClassifier(bootstrap=True, criterion="entropy", max_depth=10, max_features=0.45, max_samples=0.9, min_samples_leaf=17, min_samples_split=6, n_estimators=1000)
    )
    # Fix random state for all the steps in exported pipeline
# =============================================================================
#     set_param_recursive(exported_pipeline_trees.steps, 'random_state', 667)
#     exported_pipeline_trees.fit(training_features, training_target)
# =============================================================================
    
    # 8_classic-svm
    
    from grid import SVC_linear, SVC_rbf
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer, PolynomialFeatures, RobustScaler
    from sklearn.svm import LinearSVC
    from tpot.builtins import StackingEstimator
    
    # Average CV score on the training set was: 0.5836075811189999
    exported_pipeline_svm = make_pipeline(
        RobustScaler(),
        StackingEstimator(estimator=SVC_linear(C=0.1, cache_size=100, gamma=0.0001, kernel="linear", max_iter=1000, probability=True, tol=0.1)),
        StackingEstimator(estimator=SVC_rbf(C=15.0, cache_size=100, gamma=0.01, kernel="rbf", max_iter=1000, probability=True, tol=0.1)),
        Normalizer(norm="l2"),
        PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
        RobustScaler(),
        LinearSVC(C=20.0, dual=False, loss="squared_hinge", max_iter=1000, penalty="l2", tol=1e-05)
    )
    # Fix random state for all the steps in exported pipeline
# =============================================================================
#     set_param_recursive(exported_pipeline_svm.steps, 'random_state', 667)
#     exported_pipeline_svm.fit(training_features, training_target)
# =============================================================================
    
    # 8-classic logreg
    
    from grid import logreg_solvers
    from sklearn.feature_selection import SelectPercentile, f_classif
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
    
    # Average CV score on the training set was: 0.5799180898543609
    exported_pipeline_logreg = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
        MinMaxScaler(),
        SelectPercentile(score_func=f_classif, percentile=91),
        SelectPercentile(score_func=f_classif, percentile=20),
        logreg_solvers(C=1.0, max_iter=1000, penalty="none", solver="newton-cg", tol=0.01)
    )
    # Fix random state for all the steps in exported pipeline
# =============================================================================
#     set_param_recursive(exported_pipeline_logreg.steps, 'random_state', 667)
#     exported_pipeline_logreg.fit(training_features, training_target)
# =============================================================================
    
    # 8-classic KN
    
    from sklearn.decomposition import FastICA
    from sklearn.feature_selection import SelectPercentile, f_classif
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    
    # Average CV score on the training set was: 0.5508968790883433
    exported_pipeline_kn = make_pipeline(
        SelectPercentile(score_func=f_classif, percentile=70),
        PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
        FastICA(tol=0.05),
        FastICA(tol=0.7000000000000001),
        KNeighborsClassifier(n_neighbors=74, p=2, weights="distance")
    )
    # Fix random state for all the steps in exported pipeline
# =============================================================================
#     set_param_recursive(exported_pipeline_kn.steps, 'random_state', 667)
#     exported_pipeline_kn.fit(training_features, training_target)
# =============================================================================
    
    # 8-classic MLP
    
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler
    
    # Average CV score on the training set was: 0.5606741529621232
    exported_pipeline_mlp = make_pipeline(
        RobustScaler(),
        MLPClassifier(alpha=0.001, batch_size=64, hidden_layer_sizes=(200,), learning_rate="constant", learning_rate_init=0.001, max_iter=200)
    )
    # Fix random state for all the steps in exported pipeline
# =============================================================================
#     set_param_recursive(exported_pipeline_mlp.steps, 'random_state', 667)
#     exported_pipeline_mlp.fit(training_features, training_target)
# 
# =============================================================================
    print('loaded 6 models for 8_classic')
    
    return exported_pipeline_nb, exported_pipeline_trees, exported_pipeline_svm,\
            exported_pipeline_logreg, exported_pipeline_kn, exported_pipeline_mlp

def read_best_models_8_classic_plus_MIR_only_params(training_features, training_target, training_features_nb):
    # reading best model pipelines from 6 models families, for representation 8_classic_plus_MIR
    
    # 8_classic_plus_MIR nb   

    from sklearn.decomposition import FastICA
    from sklearn.naive_bayes import BernoulliNB, GaussianNB
    from sklearn.pipeline import make_pipeline
    from tpot.builtins import StackingEstimator
    from sklearn.feature_selection import SelectFwe, f_classif    
    
    exported_pipeline_nb = make_pipeline(
        FastICA(tol=0.05),
        StackingEstimator(estimator=BernoulliNB(alpha=0.01, fit_prior=False)),
        StackingEstimator(estimator=GaussianNB(var_smoothing=1e-06)),
        SelectFwe(score_func=f_classif, alpha=0.039),
        GaussianNB(var_smoothing=0.0001)
    )
# =============================================================================
#     # Fix random state for all the steps in exported pipeline
#     set_param_recursive(exported_pipeline_nb.steps, 'random_state', 667)
#     
#     exported_pipeline_nb.fit(training_features_nb, training_target)
# 
# =============================================================================

    # 8_classic_plus_MIR trees
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    
    # Average CV score on the training set was: 0.5881723948164884
    exported_pipeline_trees = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
        RandomForestClassifier(bootstrap=False, criterion="entropy", max_depth=10, max_features=0.25, max_samples=None, min_samples_leaf=2, min_samples_split=17, n_estimators=50)
    )
    # Fix random state for all the steps in exported pipeline
# =============================================================================
#     set_param_recursive(exported_pipeline_trees.steps, 'random_state', 667)
#     exported_pipeline_trees.fit(training_features, training_target)    
# =============================================================================
    #  8_classic_plus_MIR svm
    
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, RobustScaler
    from sklearn.svm import LinearSVC
    
    # Average CV score on the training set was: 0.5697682299834397
    exported_pipeline_svm = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
        RobustScaler(),
        LinearSVC(C=0.5, dual=True, loss="squared_hinge", max_iter=1000, penalty="l2", tol=0.0001)
    )

    # Fix random state for all the steps in exported pipeline
# =============================================================================
#     set_param_recursive(exported_pipeline_svm.steps, 'random_state', 667)
#     exported_pipeline_svm.fit(training_features, training_target)
# =============================================================================
         
    # 8_classic_plus_MIR logreg
    from grid import logreg_solvers
    
    # Average CV score on the training set was: 0.5570700206957998
    exported_pipeline_logreg = logreg_solvers(C=0.5, max_iter=1000, penalty="none", solver="newton-cg", tol=0.1)
    # Fix random state in exported estimator
# =============================================================================
#     if hasattr(exported_pipeline_logreg, 'random_state'):
#         setattr(exported_pipeline_logreg, 'random_state', 667)
#     exported_pipeline_logreg.fit(training_features, training_target)
# =============================================================================
    
    # 8_classic_plus_MIR kn
    
    from sklearn.feature_selection import SelectPercentile, f_classif
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler
      
    # Average CV score on the training set was: 0.5530259243955804
    exported_pipeline_kn = make_pipeline(
        SelectPercentile(score_func=f_classif, percentile=23),
        RobustScaler(),
        KNeighborsClassifier(n_neighbors=97, p=1, weights="distance")
    )
    # Fix random state for all the steps in exported pipeline
# =============================================================================
#     set_param_recursive(exported_pipeline_kn.steps, 'random_state', 667)
#     exported_pipeline_kn.fit(training_features, training_target)
# =============================================================================

    # 8_classic_plus_MIR MLP

    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import RFE
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler

    # Average CV score on the training set was: 0.5711217296112318
    exported_pipeline_mlp = make_pipeline(
        RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.9000000000000001, n_estimators=100), step=0.25),
        RobustScaler(),
        MLPClassifier(alpha=0.0001, batch_size=16, hidden_layer_sizes=(200, 100), learning_rate="adaptive", learning_rate_init=0.01, max_iter=10)
    )
    # Fix random state for all the steps in exported pipeline
# =============================================================================
#     set_param_recursive(exported_pipeline_mlp.steps, 'random_state', 667)
#     exported_pipeline_mlp.fit(training_features, training_target)
# =============================================================================
    
    print('loaded 6 models for 8_classic_plus_MIR')
    
    return exported_pipeline_nb, exported_pipeline_trees, exported_pipeline_svm,\
            exported_pipeline_logreg, exported_pipeline_kn, exported_pipeline_mlp

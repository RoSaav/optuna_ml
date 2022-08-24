import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import (average_precision_score, roc_curve, roc_auc_score, precision_recall_curve, log_loss)
import seaborn as sns
from pylab import rcParams
from typing import List

import optuna
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import xgboost as xgb
import catboost as cb
import json

##################################################################### OPTUNA
###################################################################
###############################################################

############################################################### XGBOOST
def optuna_optimizer_xgb(trial, X, y, metric='auc', tol=0.1, random_state=666, n_splits=4):

    '''

    Function to optimize hyperparameter searching with OPTUNA - XGBoost

    '''
    with open("train/param_grid.json", 'r') as file:
            param_dict = json.load(file)["xgb"]
 
    param_grid = {   
            "objective": param_dict["objective"],          
            'random_state': random_state,
            'booster': trial.suggest_categorical("booster", choices=param_dict["booster"]),
            'num_parallel_tree': trial.suggest_int('num_parallel_tree', low=param_dict["num_parallel_tree"]["low"], high=param_dict["num_parallel_tree"]["high"], step=param_dict["num_parallel_tree"]["step"], log=param_dict["num_parallel_tree"]["log"]),
            "eval_metric": metric,
            'max_depth': trial.suggest_int('max_depth', low=param_dict["max_depth"]["low"], high=param_dict["max_depth"]["high"], step=param_dict["max_depth"]["step"] ,log=param_dict["max_depth"]["log"]), #less deep, less overfiting
            'learning_rate': trial.suggest_float('learning_rate', low=param_dict["learning_rate"]["low"], high=param_dict["learning_rate"]["high"], step=param_dict["learning_rate"]["step"], log=param_dict["learning_rate"]["log"]),
            'gamma': trial.suggest_float('gamma', low=param_dict["gamma"]["low"], high=param_dict["gamma"]["high"], step=param_dict["gamma"]["step"], log=param_dict["gamma"]["log"]), #Gamma specifies the minimum loss reduction required to make a split.
            'min_child_weight': trial.suggest_float('min_child_weight', low=param_dict["min_child_weight"]["low"], high=param_dict["min_child_weight"]["high"], step=param_dict["min_child_weight"]["step"], log=param_dict["min_child_weight"]["log"]), #Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
            'max_delta_step': trial.suggest_float('max_delta_step', low=param_dict["max_delta_step"]["low"], high=param_dict["max_delta_step"]["high"], step=param_dict["max_delta_step"]["step"], log=param_dict["max_delta_step"]["log"]),
            'subsample': trial.suggest_float('subsample', low=param_dict["subsample"]["low"], high=param_dict["subsample"]["high"], step=param_dict["subsample"]["step"], log=param_dict["subsample"]["log"]), #Denotes the fraction of observations to be randomly samples for each tree.
            'colsample_bytree': trial.suggest_float('colsample_bytree', low=param_dict["colsample_bytree"]["low"], high=param_dict["colsample_bytree"]["high"], step=param_dict["colsample_bytree"]["step"], log=param_dict["colsample_bytree"]["log"]), #Denotes the fraction of columns to be randomly samples for each tree.
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', low=param_dict["colsample_bylevel"]["low"], high=param_dict["colsample_bylevel"]["high"], step=param_dict["colsample_bylevel"]["step"], log=param_dict["colsample_bylevel"]["log"]), #Denotes the subsample ratio of columns for each split, in each level.
            'reg_alpha': trial.suggest_float('reg_alpha', low=param_dict["reg_alpha"]["low"], high=param_dict["reg_alpha"]["high"], step=param_dict["reg_alpha"]["step"], log=param_dict["reg_alpha"]["log"]),
            'reg_lambda': trial.suggest_float('reg_lambda', low=param_dict["reg_lambda"]["low"], high=param_dict["reg_lambda"]["high"], step=param_dict["reg_lambda"]["step"], log=param_dict["reg_lambda"]["log"]),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', low=param_dict["scale_pos_weight"]["low"], high=param_dict["scale_pos_weight"]["high"], step=param_dict["scale_pos_weight"]["step"], log=param_dict["scale_pos_weight"]["log"]),
            'n_jobs': param_dict["n_jobs"]
        }
    print("*"*100)
    print(param_grid)
    print("*"*100)
    skf = StratifiedKFold(n_splits=n_splits)

    cv_results = {
            f'{metric}_train': [],
            'logloss_train': [],
            f'{metric}_valid': [],
            'logloss_valid': []
        }

    fit_params = {
            'early_stopping_rounds': 100,
            'verbose_eval' : 0
        }
    
    for train_index, valid_index in skf.split(X, y):

        X_train, X_valid = X.iloc[train_index,], X.iloc[valid_index,]
        y_train, y_valid = y.iloc[train_index,], y.iloc[valid_index,]

        #trial
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, observation_key =  "validation-auc")
        bst = xgb.train(param_grid, dtrain, evals=[(dvalid, "validation")], callbacks=[pruning_callback], **fit_params)
        preds_train = bst.predict(dtrain)
        preds_valid = bst.predict(dvalid)

        cv_results[f'{metric}_train'].append(roc_auc_score(y_train, preds_train))
        cv_results['logloss_train'].append(log_loss(y_train, preds_train))
        cv_results[f'{metric}_valid'].append(roc_auc_score(y_valid, preds_valid))
        cv_results['logloss_valid'].append(log_loss(y_valid, preds_valid))

    metric_train = np.mean(cv_results[f'{metric}_train'])
    metric_valid = np.mean(cv_results[f'{metric}_valid'])

    loss_train = np.mean(cv_results['logloss_train'])
    loss_valid = np.mean(cv_results['logloss_valid'])

    print('Finished!')
    print(f'Train {metric}:{metric_train}')
    print(f'Valid {metric}:{metric_valid}')
    print('Train Loss:{}'.format(loss_train))
    print('Valid Loss:{}'.format(loss_valid))

    return metric_valid


def optuna_optimizer_lgb(trial, X, y, metric='auc', tol=0.1, random_state=666, n_splits=4):
    '''
    Function to optimize hyperparameter searching with OPTUNA - LightGBM
    '''
    
    with open("train/param_grid.json", 'r') as file:
            param_dict = json.load(file)["lgb"]
    param_grid = {
             'objective': param_dict["objective"],
             'boosting_type': param_dict["boosting_type"],
             'first_metric_only': param_dict["first_metric_only"],
             'is_unbalance': param_dict["is_unbalance"],
             'random_state': random_state,
             'verbose': param_dict["verbose"],
             'n_estimators': trial.suggest_int('n_estimators', low=param_dict["n_estimators"]["low"], high=param_dict["n_estimators"]["high"], step=param_dict["n_estimators"]["step"], log=param_dict["n_estimators"]["log"]),
             'max_depth': trial.suggest_int('max_depth', low=param_dict["max_depth"]["low"], high=param_dict["max_depth"]["high"], step=param_dict["max_depth"]["step"] ,log=param_dict["max_depth"]["log"]),
             'num_leaves': trial.suggest_int('num_leaves', low=param_dict["num_leaves"]["low"], high=param_dict["num_leaves"]["high"], step=param_dict["num_leaves"]["step"] ,log=param_dict["num_leaves"]["log"]),
             'min_split_gain': trial.suggest_float('min_split_gain', low=param_dict["min_split_gain"]["low"], high=param_dict["min_split_gain"]["high"], step=param_dict["min_split_gain"]["step"], log=param_dict["min_split_gain"]["log"]),
             'max_bin': trial.suggest_int('max_bin', low=param_dict["max_bin"]["low"], high=param_dict["max_bin"]["high"], step=param_dict["max_bin"]["step"], log=param_dict["max_bin"]["log"]),
             'path_smooth': trial.suggest_float('path_smooth', low=param_dict["path_smooth"]["low"], high=param_dict["path_smooth"]["high"], step=param_dict["path_smooth"]["step"], log=param_dict["path_smooth"]["log"]),
             'learning_rate': trial.suggest_float('learning_rate', low=param_dict["learning_rate"]["low"], high=param_dict["learning_rate"]["high"], step=param_dict["learning_rate"]["step"], log=param_dict["learning_rate"]["log"]),
             'lambda_l1': trial.suggest_float('lambda_l1', low=param_dict["lambda_l1"]["low"], high=param_dict["lambda_l1"]["high"], step=param_dict["lambda_l1"]["step"], log=param_dict["lambda_l1"]["log"]),
             'lambda_l2': trial.suggest_float('lambda_l2', low=param_dict["lambda_l2"]["low"], high=param_dict["lambda_l2"]["high"], step=param_dict["lambda_l2"]["step"], log=param_dict["lambda_l2"]["log"]),
             'bagging_fraction': trial.suggest_float('bagging_fraction', low=param_dict["bagging_fraction"]["low"], high=param_dict["bagging_fraction"]["high"], step=param_dict["bagging_fraction"]["step"], log=param_dict["bagging_fraction"]["log"]),
             'bagging_freq': trial.suggest_int('bagging_freq', low=param_dict["bagging_freq"]["low"], high=param_dict["bagging_freq"]["high"], step=param_dict["bagging_freq"]["step"], log=param_dict["bagging_freq"]["log"])
    }
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, metric=metric, valid_name='valid')
    fit_params = {
        'eval_names': ['train', 'valid'],
        'eval_class_weight': ['balanced'],
        'eval_metric': metric,
        'early_stopping_rounds': 100,
        'callbacks': [pruning_callback],
        'verbose': -1
    }
    skf = StratifiedKFold(n_splits=n_splits)
    cv_results = {
        f'{metric}_train': [],
        'loss_train': [],
        f'{metric}_valid': [],
        'loss_valid': []
    }
    for train_index, valid_index in skf.split(X, y):
        X_train, X_valid = X.iloc[train_index,], X.iloc[valid_index,]
        y_train, y_valid = y.iloc[train_index,], y.iloc[valid_index,]
        fit_params['eval_set'] = [(X_valid, y_valid), (X_train, y_train)]
        mdl = LGBMClassifier(**param_grid)
        mdl.fit(X_train, y_train, **fit_params)
        cv_results[f'{metric}_train'].append(mdl.best_score_['train'][metric])
        cv_results['loss_train'].append(mdl.best_score_['train']['binary_logloss'])
        cv_results[f'{metric}_valid'].append(mdl.best_score_['valid'][metric])
        cv_results['loss_valid'].append(mdl.best_score_['valid']['binary_logloss'])
    metric_train = np.mean(cv_results[f'{metric}_train'])
    metric_valid = np.mean(cv_results[f'{metric}_valid'])

    loss_train = np.mean(cv_results['loss_train'])
    loss_valid = np.mean(cv_results['loss_valid'])
    print('Finished!')
    print(f'Train {metric}:{metric_train}')
    print(f'Valid {metric}:{metric_valid}')
    print('Train Loss:{}'.format(loss_train))
    print('Valid Loss:{}'.format(loss_valid))

    return metric_valid

def optuna_optimizer_cat(trial, X, y, metric='AUC', tol=0.05, random_state = 666, n_splits=4):
    '''
    Function to optimize hyperparameter searching with OPTUNA - Catboost
    '''
    
    with open("train/param_grid.json", 'r') as file:
            param_dict = json.load(file)["cat"]
    
    param_grid = {
        'n_estimators': trial.suggest_int('n_estimators', low=param_dict["n_estimators"]["low"], high=param_dict["n_estimators"]["high"], step=param_dict["n_estimators"]["step"], log=param_dict["n_estimators"]["log"]),
        'random_state': random_state,
        'eval_metric' : metric,
        'max_depth': trial.suggest_int('max_depth', low=param_dict["max_depth"]["low"], high=param_dict["max_depth"]["high"], step=param_dict["max_depth"]["step"], log=param_dict["max_depth"]["log"]),      
        'learning_rate': trial.suggest_float('learning_rate', low=param_dict["learning_rate"]["low"], high=param_dict["learning_rate"]["high"], step=param_dict["learning_rate"]["step"], log=param_dict["learning_rate"]["log"]),
        'objective': trial.suggest_categorical("objective", choices=param_dict["objective"]),
        'colsample_bylevel': trial.suggest_float("colsample_bylevel", low=param_dict["colsample_bylevel"]["low"], high=param_dict["colsample_bylevel"]["high"], step=param_dict["colsample_bylevel"]["step"], log=param_dict["colsample_bylevel"]["log"]),
        'l2_leaf_reg' : trial.suggest_float("l2_leaf_reg", low=param_dict["l2_leaf_reg"]["low"], high=param_dict["l2_leaf_reg"]["high"], step=param_dict["l2_leaf_reg"]["step"], log=param_dict["l2_leaf_reg"]["log"]),
        'boosting_type': trial.suggest_categorical("boosting_type", choices=param_dict["boosting_type"]),
        'bootstrap_type': trial.suggest_categorical( "bootstrap_type", choices=param_dict["bootstrap_type"])
    }
    if param_grid["bootstrap_type"] == "Bayesian":
        param_grid["bagging_temperature"] = trial.suggest_float("bagging_temperature", low=param_dict["bagging_temperature"]["low"], high=param_dict["bagging_temperature"]["high"])
    elif param_grid["bootstrap_type"] == "Bernoulli":
        param_grid["subsample"] = trial.suggest_float("subsample", low=param_dict["subsample"]["low"], high=param_dict["subsample"]["high"])
    skf = StratifiedKFold(n_splits=n_splits)
    cv_results = {
        f'{metric}_train': [],
        'logloss_train': [],
        f'{metric}_valid': [],
        'logloss_valid': []
    }
    fit_params = {
        'early_stopping_rounds': 100,
        'verbose_eval' : 0
    }
    for train_index, valid_index in skf.split(X, y):
        X_train, X_valid = X.iloc[train_index,], X.iloc[valid_index,]
        y_train, y_valid = y.iloc[train_index,], y.iloc[valid_index,]
        fit_params['eval_set'] = [(X_valid, y_valid), (X_train, y_train)]
        mdl = cb.CatBoostClassifier(**param_grid)
        mdl.fit(
            X_train,
            y_train,
            **fit_params
        )
        preds_train = mdl.predict(X_train)
        preds_valid = mdl.predict(X_valid)
        cv_results[f'{metric}_train'].append(roc_auc_score(y_train, preds_train))
        cv_results['logloss_train'].append(log_loss(y_train, preds_train))
        cv_results[f'{metric}_valid'].append(roc_auc_score(y_valid, preds_valid))
        cv_results['logloss_valid'].append(log_loss(y_valid, preds_valid))
    metric_train = np.mean(cv_results[f'{metric}_train'])
    metric_valid = np.mean(cv_results[f'{metric}_valid'])

    loss_train = np.mean(cv_results['logloss_train'])
    loss_valid = np.mean(cv_results['logloss_valid'])
    print('Finished!')
    print(f'Train {metric}:{metric_train}')
    print(f'Valid {metric}:{metric_valid}')
    print('Train Loss:{}'.format(loss_train))
    print('Valid Loss:{}'.format(loss_valid))

    return metric_valid

###############################################################
###################################################################
##################################################################### OPTUNA

def plot_roc(labels, prediction_scores, legend, color):
    '''
    Function to plot ROC curve
    '''
    fpr, tpr, _   = roc_curve(labels, prediction_scores, pos_label=1)
    auc           = roc_auc_score(labels, prediction_scores)
    legend_string = legend + ' ($AUC = {:0.4f}$)'.format(auc)  
    plt.plot(fpr, tpr, label=legend_string, color=color)
    pass

def format_plot(title, xlabel, ylabel):
    '''
    Function to add format to plot
    '''
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid('on')
    plt.axis('square')
    plt.ylim((-0.05, 1.05))
    plt.legend()
    plt.tight_layout()
    pass
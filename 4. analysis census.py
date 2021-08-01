###############################################################################
##### 0. IMPORT PACAKGES AND LOAD DATA  #######################################
###############################################################################

# Import packages
import sys
sys.path.append('XXX/XXX/XXX')
import relabelling
import pandas as pd
import numpy as np
from numpy import argmax
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.metrics import recall_score
from aif360.sklearn.metrics import statistical_parity_difference
from aif360.sklearn.metrics import average_odds_difference
#from themis_ml.preprocessing import relabelling
from aif360.sklearn.preprocessing import Reweighing
from aif360.sklearn.inprocessing import AdversarialDebiasing
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

# Load data
df_2001 = pd.read_csv(r"XXX\XXX\df_2001.csv", usecols=range(1,30))
df_2011 = pd.read_csv(r"XXX\XXX\df_2011.csv", usecols=range(1,30))

###############################################################################
##### 1. DATA EXPLORATION            ##########################################
###############################################################################
# Data description
# Crosstab of default and gender
pd.crosstab(df_2001['HIGH_PROF'], df_2001['SEX'], margins=True)
pd.crosstab(df_2011['HIGH_PROF'], df_2011['SEX'], margins=True)

# Summary statistics
sum_stats_2001 = df_2001.describe()
sum_stast_2011 = df_2011.describe()

# Check correlation between the dependent and all other explanatory variables
grouped_2001 = df_2001.groupby('HIGH_PROF').mean()
grouped_2011 = df_2011.groupby('HIGH_PROF').mean()

###############################################################################
##### 2. MODEL TRAINING AND PREDICTIONS #######################################
###############################################################################

df_2001 = df_2001.set_index('SEX', drop=False) # Necessary for the bias mitigation techniques
df_2001.index.rename('MALE', inplace=True)

df_2011 = df_2011.set_index('SEX', drop=False) # Necessary for the bias mitigation techniques
df_2011.index.rename('MALE', inplace=True)

# Define dependent and independent variables for both data sets
y_2001 = df_2001['HIGH_PROF']
X_2001 = df_2001.drop('HIGH_PROF', axis=1)

y_2011 = df_2011['HIGH_PROF']
X_2011 = df_2011.drop('HIGH_PROF', axis=1)


def run_reweighing(different_sets, X_train_, X_test, y_train_, y_test):
    np.random.seed(42) # Set random seed in order to replicate results
    models = ['lr', 'rf', 'dt', 'xgb']
    
    # Initialize the vectors that need to be filled during the analyses
    accuracy = pd.DataFrame()
    recall = pd.DataFrame()
    spd = pd.DataFrame()
    aod = pd.DataFrame()
            
    accuracy_rew = pd.DataFrame()
    recall_rew = pd.DataFrame()
    spd_rew = pd.DataFrame()
    aod_rew = pd.DataFrame()
      
    diff_spd = pd.DataFrame()
    diff_aod = pd.DataFrame()
        
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    i=0
    for train_index, test_index in outer_cv.split(X_train_,y_train_):
        if different_sets == True:
            # We are going to split the train data in 5 folds, but are going to use the whole test set
            X_train, X_not_used = X_train_.iloc[train_index], X_train_.iloc[test_index]
            y_train, y_not_used = y_train_.iloc[train_index], y_train_.iloc[test_index]
        else:
            X_train, X_test = X_train_.iloc[train_index], X_train_.iloc[test_index]
            y_train, y_test = y_train_.iloc[train_index], y_train_.iloc[test_index]
       
        print(i)
        
        #### PRE-PROCESSING #######        
        # Apply reweighing
        rw = Reweighing('MALE')
        X_train, weights_rew = rw.fit_transform(X_train, y_train)        
        
        #### HYPERPARAMETER TUNING #######
        # Split data into 5 folds
        inner_cv = StratifiedKFold(n_splits=5, 
                                   shuffle=True, 
                                   random_state=42)    
        
        for model in models:
            print(model)
            if model=='rf':
                # Set parameters for hyperparameter tuning
                hyperparam_grid = {'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)],
                       'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                       'min_samples_split': [2, 5, 10],
                       'min_samples_leaf': [1, 2, 4]}
                rf = RandomForestClassifier(random_state=42)
                # No preprocessing
                search_rf = RandomizedSearchCV(estimator=rf, 
                                               n_jobs=-1,
                                               param_distributions=hyperparam_grid, 
                                               cv=inner_cv,
                                               scoring='recall')
                search_rf.fit(X_train, y_train)
                best_model = search_rf.best_estimator_ # Obtain best model (hyperparameters) based on average performance
                # Preprocessing
                search_rf_rew = RandomizedSearchCV(estimator=rf, 
                                               n_jobs=-1,
                                               param_distributions=hyperparam_grid, 
                                               cv=inner_cv,
                                               scoring='recall')
                search_rf_rew.fit(X_train, y_train, sample_weight=weights_rew)
                best_model_rew = search_rf_rew.best_estimator_  
            if model=='lr':
                # Set parameters for hyperparameter tuning
                hyperparam_grid = {'penalty': ['none','l2'],
                                   'C':[0.01,0.2,1,10,100]}       
                lr = LogisticRegression(random_state=42, max_iter=5000) 
                # No preprocessing
                search_lr = RandomizedSearchCV(estimator=lr, 
                                               n_jobs=-1, 
                                               param_distributions=hyperparam_grid, 
                                               cv=inner_cv, 
                                               scoring='recall')
                search_lr.fit(X_train, y_train)
                best_model = search_lr.best_estimator_  # Obtain best model (hyperparameters) based on average performance
                # Preprocessing
                search_lr_rew = RandomizedSearchCV(estimator=lr, 
                                                   n_jobs=-1, 
                                                   param_distributions=hyperparam_grid, 
                                                   cv=inner_cv, 
                                                   scoring='recall')
                search_lr_rew.fit(X_train, y_train, sample_weight=weights_rew)
                best_model_rew = search_lr_rew.best_estimator_  # Obtain best model (hyperparameters) based on average performance
            if model=='xgb':
                # Set parameters for hyperparameter tuning
                hyperparam_grid = {'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                                   'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15],
                                   'min_child_weight' : [ 1, 3, 5, 7 ],
                                   'gamma'            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
                                   'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ] }
                xgb = XGBClassifier(random_state=42, verbosity = 0, use_label_encoder=False)
                # No preprocessing
                search_xgb = RandomizedSearchCV(estimator=xgb, 
                                               n_jobs=-1, 
                                               param_distributions=hyperparam_grid, 
                                               cv=inner_cv, 
                                               scoring='recall')
                search_xgb.fit(X_train, y_train)
                best_model = search_xgb.best_estimator_  # Obtain best model (hyperparameters) based on average performance
                # Preprocessing
                search_xgb_rew = RandomizedSearchCV(estimator=xgb, 
                                                   n_jobs=-1, 
                                                   param_distributions=hyperparam_grid, 
                                                   cv=inner_cv, 
                                                   scoring='recall')
                search_xgb_rew.fit(X_train, y_train, sample_weight=weights_rew)
                best_model_rew = search_xgb_rew.best_estimator_  # Obtain best model (hyperparameters) based on average performance
            if model=='dt':
                # Set parameters for hyperparameter tuning
                hyperparam_grid = {"max_depth": [int(x) for x in np.linspace(10, 110, num = 11)],
                                   "max_features": [int(x) for x in np.linspace(1, 10, num = 5)],
                                   "min_samples_leaf": [1, 2, 4],
                                   "criterion": ["gini", "entropy"]}
                dt = DecisionTreeClassifier(random_state=42)
                # No preprocessing
                search_dt = RandomizedSearchCV(estimator=dt, 
                                               n_jobs=-1, 
                                               param_distributions=hyperparam_grid, 
                                               cv=inner_cv, 
                                               scoring='recall')
                search_dt.fit(X_train, y_train)
                best_model = search_dt.best_estimator_  # Obtain best model (hyperparameters) based on average performance
                # Preprocessing
                search_dt_rew = RandomizedSearchCV(estimator=dt, 
                                                   n_jobs=-1, 
                                                   param_distributions=hyperparam_grid, 
                                                   cv=inner_cv, 
                                                   scoring='recall')
                search_dt_rew.fit(X_train, y_train, sample_weight=weights_rew)
                best_model_rew = search_dt_rew.best_estimator_  # Obtain best model (hyperparameters) based on average performance
     
            ###### MAKE PREDICTIONS #######
           
            pred_proba_df = pd.DataFrame(best_model.predict_proba(X_test))
            precision_lr, recall_lr, thresholds = precision_recall_curve(y_test, pred_proba_df.iloc[:,1])
            fscore = (2 * precision_lr * recall_lr) / (precision_lr + recall_lr)
            # locate the index of the largest f score
            ix = argmax(fscore)
            threshold = thresholds[ix]
            y_pred = pred_proba_df.applymap(lambda x: 1 if x>threshold else 0).iloc[:,1]
            
            pred_proba_df_rew = pd.DataFrame(best_model_rew.predict_proba(X_test))
            precision_lr_rew, recall_lr_rew, thresholds_rew = precision_recall_curve(y_test, pred_proba_df_rew.iloc[:,1])
            fscore_rew = (2 * precision_lr_rew * recall_lr_rew) / (precision_lr_rew + recall_lr_rew)
            # locate the index of the largest f score
            ix_rew = argmax(fscore_rew)
            threshold_rew = thresholds_rew[ix_rew]
            
            y_pred_rew = pred_proba_df_rew.applymap(lambda x: 1 if x>threshold_rew else 0).iloc[:,1]         
    
            # Predict for test set
                #y_pred = best_model.predict(X_test)
                #y_pred_rew = best_model_rew.predict(X_test)
                            
            ###### FILL RESULTS IN DATA FRAMES #######
            ## NO PREPROCESSING ##
            # Performance metrics
            accuracy.loc[i, model] = accuracy_score(y_true=y_test, y_pred=y_pred)
            recall.loc[i, model] = recall_score(y_true=y_test, y_pred=y_pred)
            spd.loc[i, model] = statistical_parity_difference(y_test, y_pred, prot_attr='MALE')  
            aod.loc[i, model] = average_odds_difference(y_test, y_pred, prot_attr='MALE')
                    
            ## PREPROCESSING ##
            # Performance metrics
            accuracy_rew.loc[i, model] = accuracy_score(y_true=y_test, y_pred=y_pred_rew)
            recall_rew.loc[i, model] = recall_score(y_true=y_test, y_pred=y_pred_rew)
            spd_rew.loc[i, model] = statistical_parity_difference(y_test, y_pred_rew, prot_attr='MALE')  
            aod_rew.loc[i, model] = average_odds_difference(y_test, y_pred_rew, prot_attr='MALE')
              
            diff_spd.loc[i, model] = spd.loc[i, model] - spd_rew.loc[i, model]
            diff_aod.loc[i, model] = aod.loc[i, model] - aod_rew.loc[i, model]
            
        i = i+1
        
    
    spd_output = pd.concat([spd, spd_rew], axis=1)
    spd_output = spd_output.mean(axis=0)
    
    aod_output = pd.concat([aod, aod_rew], axis=1)
    aod_output = aod_output.mean(axis=0)

    metrics_output = pd.concat([recall, recall_rew, accuracy, accuracy_rew], axis=1)
    metrics_output = metrics_output.mean(axis=0)        
            
    return (spd_output, aod_output, metrics_output, diff_spd, diff_aod)

def run_massaging(different_sets, X_train_, X_test, y_train_, y_test):
    np.random.seed(42) # Set random seed in order to replicate results
    models = ['lr', 'rf', 'dt', 'xgb']
    
    # Initialize the vectors that need to be filled during the analyses
    accuracy = pd.DataFrame()
    recall = pd.DataFrame()
    spd = pd.DataFrame()
    aod = pd.DataFrame()
            
    accuracy_rew = pd.DataFrame()
    recall_rew = pd.DataFrame()
    spd_rew = pd.DataFrame()
    aod_rew = pd.DataFrame()
      
    diff_spd = pd.DataFrame()
    diff_aod = pd.DataFrame()
        
    outer_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    i=0
    for train_index, test_index in outer_cv.split(X_train_,y_train_):
        if different_sets == True:
            # We are going to split the train data in 5 folds, but are going to use the whole test set
            X_train, X_not_used = X_train_.iloc[train_index], X_train_.iloc[test_index]
            y_train, y_not_used = y_train_.iloc[train_index], y_train_.iloc[test_index]
        else:
            X_train, X_test = X_train_.iloc[train_index], X_train_.iloc[test_index]
            y_train, y_test = y_train_.iloc[train_index], y_train_.iloc[test_index]
       
        print(i)
        
        #### PRE-PROCESSING #######             
        # Apply massaging
        massager = relabelling.Relabeller(ranker=LogisticRegression(max_iter=5000))
        s = 1 - X_train['SEX']
        y_train_mas = massager.fit(X_train, y_train, s).transform(X_train)
        
        #### HYPERPARAMETER TUNING #######
        # Split data into 5 folds
        inner_cv = StratifiedKFold(n_splits=5, 
                                   shuffle=True, 
                                   random_state=42)    
        
        for model in models:
            print(model)
            if model=='rf':
                # Set parameters for hyperparameter tuning
                hyperparam_grid = {'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)],
                       'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                       'min_samples_split': [2, 5, 10],
                       'min_samples_leaf': [1, 2, 4]}
                rf = RandomForestClassifier(random_state=42)
                # No preprocessing
                search_rf = RandomizedSearchCV(estimator=rf, 
                                               n_jobs=-1,
                                               param_distributions=hyperparam_grid, 
                                               cv=inner_cv,
                                               scoring='recall')
                search_rf.fit(X_train, y_train)
                best_model = search_rf.best_estimator_ # Obtain best model (hyperparameters) based on average performance
                # Preprocessing
                search_rf_rew = RandomizedSearchCV(estimator=rf, 
                                               n_jobs=-1,
                                               param_distributions=hyperparam_grid, 
                                               cv=inner_cv,
                                               scoring='recall')
                search_rf_rew.fit(X_train, y_train_mas)
                best_model_rew = search_rf_rew.best_estimator_  
            if model=='lr':
                # Set parameters for hyperparameter tuning
                hyperparam_grid = {'penalty': ['none','l2'],
                                   'C':[0.01,0.2,1,10,100]}       
                lr = LogisticRegression(random_state=42, max_iter=5000) 
                # No preprocessing
                search_lr = RandomizedSearchCV(estimator=lr, 
                                               n_jobs=-1, 
                                               param_distributions=hyperparam_grid, 
                                               cv=inner_cv, 
                                               scoring='recall')
                search_lr.fit(X_train, y_train)
                best_model = search_lr.best_estimator_  # Obtain best model (hyperparameters) based on average performance
                # Preprocessing
                search_lr_rew = RandomizedSearchCV(estimator=lr, 
                                                   n_jobs=-1, 
                                                   param_distributions=hyperparam_grid, 
                                                   cv=inner_cv, 
                                                   scoring='recall')
                search_lr_rew.fit(X_train, y_train_mas)
                best_model_rew = search_lr_rew.best_estimator_  # Obtain best model (hyperparameters) based on average performance
            if model=='xgb':
                # Set parameters for hyperparameter tuning
                hyperparam_grid = {'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                                   'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15],
                                   'min_child_weight' : [ 1, 3, 5, 7 ],
                                   'gamma'            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
                                   'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ] }
                xgb = XGBClassifier(random_state=42, verbosity = 0, use_label_encoder=False)
                # No preprocessing
                search_xgb = RandomizedSearchCV(estimator=xgb, 
                                               n_jobs=-1, 
                                               param_distributions=hyperparam_grid, 
                                               cv=inner_cv, 
                                               scoring='recall')
                search_xgb.fit(X_train, y_train)
                best_model = search_xgb.best_estimator_  # Obtain best model (hyperparameters) based on average performance
                # Preprocessing
                search_xgb_rew = RandomizedSearchCV(estimator=xgb, 
                                                   n_jobs=-1, 
                                                   param_distributions=hyperparam_grid, 
                                                   cv=inner_cv, 
                                                   scoring='recall')
                search_xgb_rew.fit(X_train, y_train_mas)
                best_model_rew = search_xgb_rew.best_estimator_  # Obtain best model (hyperparameters) based on average performance
            if model=='dt':
                # Set parameters for hyperparameter tuning
                hyperparam_grid = {"max_depth": [int(x) for x in np.linspace(10, 110, num = 11)],
                                   "max_features": [int(x) for x in np.linspace(1, 10, num = 5)],
                                   "min_samples_leaf": [1, 2, 4],
                                   "criterion": ["gini", "entropy"]}
                dt = DecisionTreeClassifier(random_state=42)
                # No preprocessing
                search_dt = RandomizedSearchCV(estimator=dt, 
                                               n_jobs=-1, 
                                               param_distributions=hyperparam_grid, 
                                               cv=inner_cv, 
                                               scoring='recall')
                search_dt.fit(X_train, y_train)
                best_model = search_dt.best_estimator_  # Obtain best model (hyperparameters) based on average performance
                # Preprocessing
                search_dt_rew = RandomizedSearchCV(estimator=dt, 
                                                   n_jobs=-1, 
                                                   param_distributions=hyperparam_grid, 
                                                   cv=inner_cv, 
                                                   scoring='recall')
                search_dt_rew.fit(X_train, y_train_mas)
                best_model_rew = search_dt_rew.best_estimator_  # Obtain best model (hyperparameters) based on average performance
     
            ###### MAKE PREDICTIONS #######
            # Predict for test set
            pred_proba_df = pd.DataFrame(best_model.predict_proba(X_test))
            precision_lr, recall_lr, thresholds = precision_recall_curve(y_test, pred_proba_df.iloc[:,1])
            fscore = (2 * precision_lr * recall_lr) / (precision_lr + recall_lr)
            # locate the index of the largest f score
            ix = argmax(fscore)
            threshold = thresholds[ix]
            y_pred = pred_proba_df.applymap(lambda x: 1 if x>threshold else 0).iloc[:,1]
            
            pred_proba_df_rew = pd.DataFrame(best_model_rew.predict_proba(X_test))
            precision_lr_rew, recall_lr_rew, thresholds_rew = precision_recall_curve(y_test, pred_proba_df_rew.iloc[:,1])
            fscore_rew = (2 * precision_lr_rew * recall_lr_rew) / (precision_lr_rew + recall_lr_rew)
            # locate the index of the largest f score
            ix_rew = argmax(fscore_rew)
            threshold_rew = thresholds_rew[ix_rew]
            
            y_pred_rew = pred_proba_df_rew.applymap(lambda x: 1 if x>threshold_rew else 0).iloc[:,1]         
    
            ###### FILL RESULTS IN DATA FRAMES #######
            ## NO PREPROCESSING ##
            # Performance metrics
            accuracy.loc[i, model] = accuracy_score(y_true=y_test, y_pred=y_pred)
            recall.loc[i, model] = recall_score(y_true=y_test, y_pred=y_pred)
            spd.loc[i, model] = statistical_parity_difference(y_test, y_pred, prot_attr='MALE')  
            aod.loc[i, model] = average_odds_difference(y_test, y_pred, prot_attr='MALE')
                    
            ## PREPROCESSING ##
            # Performance metrics
            accuracy_rew.loc[i, model] = accuracy_score(y_true=y_test, y_pred=y_pred_rew)
            recall_rew.loc[i, model] = recall_score(y_true=y_test, y_pred=y_pred_rew)
            spd_rew.loc[i, model] = statistical_parity_difference(y_test, y_pred_rew, prot_attr='MALE')  
            aod_rew.loc[i, model] = average_odds_difference(y_test, y_pred_rew, prot_attr='MALE')
              
            diff_spd.loc[i, model] = spd.loc[i, model] - spd_rew.loc[i, model]
            diff_aod.loc[i, model] = aod.loc[i, model] - aod_rew.loc[i, model]
            
        i = i+1
        
    
    spd_output = pd.concat([spd, spd_rew], axis=1)
    spd_output = spd_output.mean(axis=0)
    
    aod_output = pd.concat([aod, aod_rew], axis=1)
    aod_output = aod_output.mean(axis=0)

    metrics_output = pd.concat([recall, recall_rew, accuracy, accuracy_rew], axis=1)
    metrics_output = metrics_output.mean(axis=0)        
            
    return (spd_output, aod_output, metrics_output, diff_spd, diff_aod)

def run_adv_debias(different_sets, X_train_, X_test, y_train_, y_test):
    np.random.seed(42) # Set random seed in order to replicate results
    # Initialize the vectors that need to be filled during the analyses
    accuracy = pd.DataFrame()
    recall = pd.DataFrame()
    spd = pd.DataFrame()
    aod = pd.DataFrame()
            
    accuracy_rew = pd.DataFrame()
    recall_rew = pd.DataFrame()
    spd_rew = pd.DataFrame()
    aod_rew = pd.DataFrame()
      
    diff_spd = pd.DataFrame()
    diff_aod = pd.DataFrame()
        
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    i=0
    for train_index, test_index in outer_cv.split(X_train_,y_train_):
        
        if different_sets == True:
            # We are going to split the train data in 5 folds, but are going to use the whole test set
            X_train, X_not_used = X_train_.iloc[train_index], X_train_.iloc[test_index]
            y_train, y_not_used = y_train_.iloc[train_index], y_train_.iloc[test_index]
        else:
            X_train, X_test = X_train_.iloc[train_index], X_train_.iloc[test_index]
            y_train, y_test = y_train_.iloc[train_index], y_train_.iloc[test_index]
       
        print(len(y_train))
        print(i)
        
        #### IN-PROCESSING #######
        no_adv_deb = AdversarialDebiasing(prot_attr='MALE',
                               debias=False,
                               adversary_loss_weight=0.05,
                               random_state=42)
        no_adv_deb.fit(X_train, y_train)
        
        adv_deb = AdversarialDebiasing(prot_attr='MALE',
                               debias=True,
                               adversary_loss_weight=0.05,
                               random_state=42)
        adv_deb.fit(X_train, y_train)
        
        ###### MAKE PREDICTIONS #######
        # Predict for test set
        y_pred = no_adv_deb.predict(X_test)
        y_pred_rew = adv_deb.predict(X_test)
        
        # Predict for test set
        pred_proba_df = pd.DataFrame(no_adv_deb.predict_proba(X_test))
        precision_lr, recall_lr, thresholds = precision_recall_curve(y_test, pred_proba_df.iloc[:,1])
        fscore = (2 * precision_lr * recall_lr) / (precision_lr + recall_lr)
        # locate the index of the largest f score
        ix = argmax(fscore)
        threshold = thresholds[ix]
        y_pred = pred_proba_df.applymap(lambda x: 1 if x>threshold else 0).iloc[:,1]
        
        pred_proba_df_rew = pd.DataFrame(adv_deb.predict_proba(X_test))     
        y_pred_rew = pred_proba_df_rew.applymap(lambda x: 1 if x>threshold else 0).iloc[:,1]         
       
        ###### FILL RESULTS IN DATA FRAMES #######
        model = 'adv_deb'
        
        ## NO PREPROCESSING ##
        # Performance metrics
        accuracy.loc[i, model] = accuracy_score(y_true=y_test, y_pred=y_pred)
        recall.loc[i, model] = recall_score(y_true=y_test, y_pred=y_pred)
        spd.loc[i, model] = statistical_parity_difference(y_test, y_pred, prot_attr='MALE')  
        aod.loc[i, model] = average_odds_difference(y_test, y_pred, prot_attr='MALE')
                
        ## PREPROCESSING ##
        # Performance metrics
        accuracy_rew.loc[i, model] = accuracy_score(y_true=y_test, y_pred=y_pred_rew)
        recall_rew.loc[i, model] = recall_score(y_true=y_test, y_pred=y_pred_rew)
        spd_rew.loc[i, model] = statistical_parity_difference(y_test, y_pred_rew, prot_attr='MALE')  
        aod_rew.loc[i, model] = average_odds_difference(y_test, y_pred_rew, prot_attr='MALE')
          
        diff_spd.loc[i, model] = spd.loc[i, model] - spd_rew.loc[i, model]
        diff_aod.loc[i, model] = aod.loc[i, model] - aod_rew.loc[i, model]
        
        i = i+1
        
    
    spd_output = pd.concat([spd, spd_rew], axis=1)
    spd_output = spd_output.mean(axis=0)
    
    aod_output = pd.concat([aod, aod_rew], axis=1)
    aod_output = aod_output.mean(axis=0)

    metrics_output = pd.concat([recall, recall_rew, accuracy, accuracy_rew], axis=1)
    metrics_output = metrics_output.mean(axis=0)        
            
    return (spd_output, aod_output, metrics_output, diff_spd, diff_aod)

###############################################################################
##### 3. RUNNING ANALAYSES   ##################################################
###############################################################################

# Function to write results to csv
def write_results_to_csv(): 
    spd_output_2.to_csv("spd_output_2.csv")
    spd_output_4.to_csv("spd_output_4.csv")
    
    aod_output_2.to_csv("aod_output_2.csv")
    aod_output_4.to_csv("aod_output_4.csv")
    
    metrics_output_2.to_csv("metrics_output_2.csv")
    metrics_output_4.to_csv("metrics_output_4.csv")
    
    diff_spd_2.to_csv("diff_spd_2.csv")
    diff_spd_4.to_csv("diff_spd_4.csv")
    
    diff_aod_2.to_csv("diff_aod_2.csv")
    diff_aod_4.to_csv("diff_aod_4.csv")

### REWEIGHING ###    
spd_output_2, aod_output_2, metrics_output_2, diff_spd_2, diff_aod_2 = run_reweighing(different_sets = True, X_train_ = X_2011, X_test=X_2001, y_train_=y_2011, y_test=y_2001)
spd_output_4, aod_output_4, metrics_output_4, diff_spd_4, diff_aod_4 = run_reweighing(different_sets = False, X_train_ = X_2011, X_test=X_2011, y_train_ = y_2011, y_test=y_2011)

os.chdir(r"XXX\XXX\Census_reweighing")
write_results_to_csv()


### MASSAGING ###
spd_output_2, aod_output_2, metrics_output_2, diff_spd_2, diff_aod_2 = run_massaging(different_sets = True, X_train_ = X_2011, X_test=X_2001, y_train_=y_2011, y_test=y_2001)
spd_output_4, aod_output_4, metrics_output_4, diff_spd_4, diff_aod_4 = run_massaging(different_sets = False, X_train_ = X_2011, X_test=X_2011, y_train_ = y_2011, y_test=y_2011)

os.chdir(r"XXX\XXX\Census_massaging")
write_results_to_csv()

### ADVERSARIAL DEBIASING ###
spd_output_2, aod_output_2, metrics_output_2, diff_spd_2, diff_aod_2 = run_adv_debias(different_sets = True, X_train_ = X_2011, X_test=X_2001, y_train_=y_2011, y_test=y_2001)
spd_output_4, aod_output_4, metrics_output_4, diff_spd_4, diff_aod_4 = run_adv_debias(different_sets = False, X_train_ = X_2011, X_test=X_2011, y_train_ = y_2011, y_test=y_2011)

os.chdir(r"XXX\XXX\Census_adv_debias")
write_results_to_csv()

###############################################################################
##### 4. PROCESSING THE RESULTS         #######################################
###############################################################################

## REWEIGHING AND MASSAGING ##

# Choose whether you want reweighing or massaging
os.chdir(r"XXX\XXX\Census_reweighing")
os.chdir(r"XXX\XXX\Census_massaging")

# Load results from csv
spd_output_2 = pd.read_csv("spd_output_2.csv", index_col=0)
spd_output_4 = pd.read_csv("spd_output_4.csv", index_col=0)
spd_output = pd.concat([spd_output_2, spd_output_4], axis=1).transpose()

# Load results from csv
aod_output_2 = pd.read_csv("aod_output_2.csv", index_col=0)
aod_output_4 = pd.read_csv("aod_output_4.csv", index_col=0)
aod_output = pd.concat([aod_output_2,aod_output_4], axis=1).transpose()

# Load results from csv
metrics_output_2 = pd.read_csv("metrics_output_2.csv", index_col=0)
metrics_output_4 = pd.read_csv("metrics_output_4.csv", index_col=0)
metrics_output = pd.concat([metrics_output_2, metrics_output_4], axis=1).transpose()

# Load results from csv
diff_spd_2 = pd.read_csv("diff_spd_2.csv", index_col=0)
diff_spd_4 = pd.read_csv("diff_spd_4.csv", index_col=0)

diff_spd_lr = pd.concat([diff_spd_2['lr'], diff_spd_4['lr']], axis=1)
diff_spd_rf = pd.concat([diff_spd_2['rf'], diff_spd_4['rf']], axis=1)
diff_spd_dt = pd.concat([diff_spd_2['dt'], diff_spd_4['dt']], axis=1)
diff_spd_xgb = pd.concat([diff_spd_2['xgb'], diff_spd_4['xgb']], axis=1)

diff_aod_2 = pd.read_csv("diff_aod_2.csv", index_col=0)
diff_aod_4 = pd.read_csv("diff_aod_4.csv", index_col=0)

diff_aod_lr = pd.concat([diff_aod_2['lr'], diff_aod_4['lr']], axis=1)
diff_aod_rf = pd.concat([diff_aod_2['rf'], diff_aod_4['rf']], axis=1)
diff_aod_dt = pd.concat([diff_aod_2['dt'], diff_aod_4['dt']], axis=1)
diff_aod_xgb = pd.concat([diff_aod_2['xgb'], diff_aod_4['xgb']], axis=1)

# Boxplot for the difference in SPD and AOD
diff_spd_lr = diff_spd_lr * 100
diff_spd_dt = diff_spd_dt * 100
diff_spd_rf = diff_spd_rf * 100
diff_spd_xgb = diff_spd_xgb * 100  
diff_aod_lr = diff_aod_lr * 100
diff_aod_dt = diff_aod_dt * 100
diff_aod_rf = diff_aod_rf * 100
diff_aod_xgb = diff_aod_xgb * 100 

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
bplot1 = ax[0].boxplot(diff_spd_lr, positions = [0, 5], patch_artist=True, boxprops=dict(facecolor="C0"))
bplot2 = ax[0].boxplot(diff_spd_rf, positions = [1, 6], patch_artist=True, boxprops=dict(facecolor="C1"))
bplot3 = ax[0].boxplot(diff_spd_dt, positions = [2, 7], patch_artist=True, boxprops=dict(facecolor="C2"))
bplot4 = ax[0].boxplot(diff_spd_xgb, positions = [3, 8], patch_artist=True, boxprops=dict(facecolor="C3"))
ax[0].axhline(y=0, color='gray',linestyle='--')
ax[0].set_ylabel('Difference in SPD for XXX (percentage point)') # Fill in reweighing or massaging on XXX
ax[0].set_xticks([1.5, 6.5])
ax[0].set_xticklabels(['Test 2001', 'Test 2011'])


bplot1 = ax[1].boxplot(diff_aod_lr, positions = [0, 5], patch_artist=True, boxprops=dict(facecolor="C0"))
bplot2 = ax[1].boxplot(diff_aod_rf, positions = [1, 6], patch_artist=True, boxprops=dict(facecolor="C1"))
bplot3 = ax[1].boxplot(diff_aod_dt, positions = [2, 7], patch_artist=True, boxprops=dict(facecolor="C2"))
bplot4 = ax[1].boxplot(diff_aod_xgb, positions = [3, 8], patch_artist=True, boxprops=dict(facecolor="C3"))
ax[1].axhline(y=0, color='gray',linestyle='--')
ax[1].set_ylabel('Difference in AOD for XXX (percentage point)') # Fill in reweighing or massaging on XXX
ax[1].set_xticks([1.5, 6.5])
ax[1].set_xticklabels(['Test 2001', 'Test 2011'])

fig.legend([bplot1["boxes"][0], bplot2["boxes"][0], bplot3["boxes"][0], bplot4["boxes"][0]], 
          ['Logistic regression', 'Random forest', 'Decision Tree', 'XGBoost'], ncol=2)



## ADVERSARIAL DEBIASING ##

os.chdir(r"XXX\XXX\Census_adv_debias")
# Load results from csv
spd_output_2 = pd.read_csv("spd_output_2.csv", index_col=0)
spd_output_4 = pd.read_csv("spd_output_4.csv", index_col=0)
spd_output = pd.concat([spd_output_2, spd_output_4], axis=1).transpose()

# Load results from csv
aod_output_2 = pd.read_csv("aod_output_2.csv", index_col=0)
aod_output_4 = pd.read_csv("aod_output_4.csv", index_col=0)
aod_output = pd.concat([aod_output_2, aod_output_4], axis=1).transpose()

# Load results from csv
metrics_output_2 = pd.read_csv("metrics_output_2.csv", index_col=0)
metrics_output_4 = pd.read_csv("metrics_output_4.csv", index_col=0)
metrics_output = pd.concat([metrics_output_2, metrics_output_4], axis=1).transpose()

# Load results from csv
diff_spd_2 = pd.read_csv("diff_spd_2.csv", index_col=0)
diff_spd_4 = pd.read_csv("diff_spd_4.csv", index_col=0)

diff_spd = pd.concat([diff_spd_2, diff_spd_4], axis=1)

diff_aod_2 = pd.read_csv("diff_aod_2.csv", index_col=0)
diff_aod_4 = pd.read_csv("diff_aod_4.csv", index_col=0)

diff_aod = pd.concat([diff_aod_2, diff_aod_4], axis=1)

# Boxplot for the difference in SPD and AOD
diff_spd = diff_spd * 100  
diff_aod = diff_aod * 100

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9,4))
bplot1 = ax[0].boxplot(diff_spd, positions = [0, 1], patch_artist=True, boxprops=dict(facecolor="C0"))
ax[0].axhline(y=0, color='gray',linestyle='--')
ax[0].set_ylabel('Difference in SPD for adv. debias. (percentage point)')
ax[0].set_xticklabels(['Test 2001', 'Test 2011'])


bplot1 = ax[1].boxplot(diff_aod, positions = [0, 1], patch_artist=True, boxprops=dict(facecolor="C1"))
ax[1].axhline(y=0, color='gray',linestyle='--')
ax[1].set_ylabel('Difference in AOD for adv. debias. (percentage point)')
ax[1].set_xticklabels(['Test 2001', 'Test 2011'])

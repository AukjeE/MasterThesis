###############################################################################
##### 0. IMPORT PACAKGES AND LOAD DATA  #######################################
###############################################################################

# Import packages
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from xgboost import XGBClassifier
from aif360.sklearn.metrics import statistical_parity_difference
from aif360.sklearn.metrics import average_odds_difference
from aif360.sklearn.preprocessing import Reweighing
from themis_ml.preprocessing import relabelling
from aif360.sklearn.inprocessing import AdversarialDebiasing
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

# Load data
df = pd.read_excel(r"XXX\XXX\default.xls", header=1)

###############################################################################
##### 1. DATA INITIALIZATION            #######################################
###############################################################################

# Perform data manipulations
df['MALE'] = np.where(df['SEX']==1, 1, 0)
df.rename(columns={'PAY_0':'PAY_1', 'default payment next month':'DEFAULT'}, inplace=True)

df = df.set_index(['ID', 'MALE']) # Necessary for the bias mitigation techniques

df['MALE_VAR'] = np.where(df['SEX']==1, 1, 0) # Retrieve variable again
df = df.drop('SEX', axis=1) # Drop the sex variable

####### Re-categorize variables #########
df['EDUCATION'].unique()
df['MARRIAGE'].unique()
df['PAY_1'].unique()

df.loc[(df['EDUCATION'] == 5) | (df['EDUCATION'] == 6) | 
       (df['EDUCATION'] == 0), 'EDUCATION'] = 4 # Group ambiguous values together as 'other' 
df.loc[(df['MARRIAGE'] == 0), 'MARRIAGE'] = 3 # Group ambiguous values together as other

# Group -2, -1 and 0 together and consider them as 'bill paid before deadline' 
df.loc[df['PAY_1']<=0,'PAY_1']=0          
df.loc[df['PAY_2']<=0,'PAY_2']=0
df.loc[df['PAY_3']<=0,'PAY_3']=0
df.loc[df['PAY_4']<=0,'PAY_4']=0
df.loc[df['PAY_5']<=0,'PAY_5']=0
df.loc[df['PAY_6']<=0,'PAY_6']=0

######## Transform categorical variables into dummy variables ######
# Education and marriage are nominal variables and therefore we will dummify them
# PAY_1 - PAY_6 are ordinal variables and therefore can be left the same
cat_vars = ['EDUCATION', 'MARRIAGE'] 

for var in cat_vars:
    cat_list = 'var'+'_'+var
    cat_list = pd.get_dummies(df[var], prefix=var)
    df1 = df.join(cat_list)
    df=df1

# Drop the original variables
df = df.drop(cat_vars, axis=1)

# Drop one category of each variable to prevent multicollinearity
df = df.drop('EDUCATION_4', axis=1)
df = df.drop('MARRIAGE_3', axis=1)

###############################################################################
##### 2. DATA EXPLORATION            ##########################################
###############################################################################
# Data description
# Crosstab of default and gender
pd.crosstab(df['DEFAULT'], df['MALE_VAR'], margins=True)

# Summary statistics
sum_stats = df.describe()

# Check correlation between the dependent and all other explanatory variables
correlation = df.corr(method="pearson")
cor = correlation['DEFAULT']

grouped = df.groupby('DEFAULT').mean()

###############################################################################
##### 3. MODEL TRAINING AND PREDICTIONS #######################################
###############################################################################

# Define dependent and independent variables
y = df['DEFAULT']
X = df.drop('DEFAULT', axis=1)

####### DATA MANIPULATIONS FUNCTIONS ##########################################

# Function to create the core test set. The samples in the core test will be used in each manipulation
def core_trainset(X_train, y_train, ratio):
    
    np.random.seed(42) # Set seed to replicate results
    sampling_amount = int(round(len(X_train) * ratio)) # Determine the sampling amount
    indices = X_train.index # Obtain the indices
    random_indices =  np.random.choice(indices, sampling_amount , replace=False) # Randomly sample subset of indices
    
    # Obtain the core test sets
    X_train_core = X_train.loc[random_indices]
    y_train_core = y_train.loc[random_indices]
    
    # Obtain rest of the test set, from which will be sampled
    X_train_other = X_train.loc[~X_train.index.isin(random_indices)]
    y_train_other = y_train.loc[~y_train.index.isin(random_indices)]
    
    return(X_train_core, y_train_core, X_train_other, y_train_other)

# Function to sample one subgroup relatively more based on which one is indicated. 
# Then adds the extra samples with the core train set to obtain the total train set
def add_samples(X_train_core, y_train_core, X_train_other, y_train_other, ratio_added=0.6, mult_factor=1, 
                default_male=False, no_default_male=False, default_female=False, no_default_female=False):
    
    np.random.seed(42) # Set seed to replicate results
    # Obtain indices of different subgroups
    default_males_indices = X_train_other[(X_train_other['MALE_VAR']==1) & (y_train_other==1)].index
    no_default_males_indices = X_train_other[(X_train_other['MALE_VAR']==1) & (y_train_other==0)].index
    default_females_indices = X_train_other[(X_train_other['MALE_VAR']==0) & (y_train_other==1)].index
    no_default_females_indices = X_train_other[(X_train_other['MALE_VAR']==0) & (y_train_other==0)].index
    
    # For the indicated subgroup: sample the subgroup as much as indicated and add the other subgroups randomly.
    # The selected subgroup will relatively be sampled more than the none selected subgroups
    if default_male==True:
        sampling_amount = int(round(len(default_males_indices)*mult_factor))
        selected_default_males_indices = np.random.choice(default_males_indices, sampling_amount, replace=False)
        
        sampling_amount_other = int(round((len(X_train_other) * ratio_added) - sampling_amount))
        other_indices = np.concatenate([no_default_males_indices,default_females_indices,no_default_females_indices])
        selected_other_indices = np.random.choice(other_indices, sampling_amount_other, replace=False)

        selected_indices = np.concatenate([selected_default_males_indices,selected_other_indices])       
    elif no_default_male==True:
        sampling_amount = int(round(len(no_default_males_indices)*mult_factor))
        selected_no_default_males_indices = np.random.choice(no_default_males_indices, sampling_amount, replace=False)
        
        sampling_amount_other = int(round((len(X_train_other) * ratio_added) - sampling_amount))
        other_indices = np.concatenate([default_males_indices,default_females_indices,no_default_females_indices])
        selected_other_indices = np.random.choice(other_indices, sampling_amount_other, replace=False)

        selected_indices = np.concatenate([selected_no_default_males_indices,selected_other_indices])               
    elif default_female==True:
        sampling_amount = int(round(len(default_females_indices)*mult_factor))
        selected_default_females_indices = np.random.choice(default_females_indices, sampling_amount, replace=False)
        
        sampling_amount_other = int(round((len(X_train_other) * ratio_added) - sampling_amount))
        other_indices = np.concatenate([no_default_males_indices,default_males_indices,no_default_females_indices])
        selected_other_indices = np.random.choice(other_indices, sampling_amount_other, replace=False)

        selected_indices = np.concatenate([selected_default_females_indices,selected_other_indices])  
    elif no_default_female==True:
        sampling_amount = int(round(len(no_default_females_indices)*mult_factor))
        selected_no_default_females_indices = np.random.choice(no_default_females_indices, sampling_amount, replace=False)
        
        sampling_amount_other = int(round((len(X_train_other) * ratio_added) - sampling_amount))
        other_indices = np.concatenate([no_default_males_indices,default_males_indices,default_females_indices])
        selected_other_indices = np.random.choice(other_indices, sampling_amount_other, replace=False)

        selected_indices = np.concatenate([selected_no_default_females_indices,selected_other_indices])
    else:
        sampling_amount = int(round((len(X_train_other) * ratio_added)))
        selected_indices = np.random.choice(X_train_other.index, sampling_amount, replace=False)
   
    # Obtain the selected samples 
    X_train_selected = X_train_other.loc[selected_indices]
    y_train_selected = y_train_other.loc[selected_indices]
    
    # Add the core test set with the extra samples and obtain the test set
    X_train_total = pd.concat([X_train_core, X_train_selected])
    y_train_total = pd.concat([y_train_core, y_train_selected])
    
    return(X_train_total, y_train_total)

def change_labels(X_train, y_train, ratio=0.8, mult_factor=1, orig=False,
                default_male=False, no_default_male=False, default_female=False, no_default_female=False):
    
    np.random.seed(42) # Set seed to replicate results
    
    # Keep part of observations   
    all_indices = X_train.index
    sampling_amount = int(round(len(X_train) * ratio))
    indices = np.random.choice(all_indices, sampling_amount, replace=False)
    X_train_core, y_train_core = X_train.loc[indices], y_train.loc[indices]
    
    
    default_males_indices = X_train_core[(X_train_core['MALE_VAR']==1) & (y_train_core==1)].index
    no_default_males_indices = X_train_core[(X_train_core['MALE_VAR']==1) & (y_train_core==0)].index
    default_females_indices = X_train_core[(X_train_core['MALE_VAR']==0) & (y_train_core==1)].index
    no_default_females_indices = X_train_core[(X_train_core['MALE_VAR']==0) & (y_train_core==0)].index

    
    if orig == True:
        X_total, y_total = X_train_core, y_train_core
    else:     
        if default_male == True:
            relabelling_amount = int(round(len(default_males_indices) * 0.12))
            selected_indices = np.random.choice(default_males_indices, relabelling_amount, replace=False)
        elif no_default_male == True:
            relabelling_amount = int(round(len(no_default_males_indices) * 0.07))
            selected_indices = np.random.choice(no_default_males_indices, relabelling_amount, replace=False)
        elif default_female == True:
            relabelling_amount = int(round(len(default_females_indices) * 0.12))
            selected_indices = np.random.choice(default_females_indices, relabelling_amount, replace=False)        
        elif no_default_female == True:
            relabelling_amount = int(round(len(no_default_females_indices) * 0.07))
            selected_indices = np.random.choice(no_default_females_indices, relabelling_amount, replace=False)
         
        y_not_relabelled = y_train_core.loc[~y_train_core.index.isin(selected_indices)]
        X_not_relabelled = X_train_core.loc[~X_train_core.index.isin(selected_indices)]
        
        y_selected = y_train_core.loc[y_train_core.index.isin(selected_indices)]
        X_selected = X_train_core.loc[X_train_core.index.isin(selected_indices)]
        
        y_relabelled = np.where(y_selected==1,0,1)
        y_relabelled = pd.Series(y_relabelled)
        y_relabelled.index = y_selected.index
        
        y_total = pd.concat([y_not_relabelled, y_relabelled])
        X_total = pd.concat([X_not_relabelled, X_selected])

    return(X_total, y_total)      

##### SET UP SENSITIVITY ANALYSIS #############################################

### REWEIGHING ###
# Function to perform sensitivity analysis
def sensitivity_analysis_rew(model, label_bias=False, sample_bias=False): # Model options: 'lr', 'rf', 'dt', 'xgb'
    np.random.seed(42) # Set random seed in order to replicate results
    
    # Set the different manipulations
    manipulations = {'male_no_default_1':1, 'female_default_0_9':0.9, 'orig':1, 'male_default_0_9':0.9,
                     'female_no_default_1':1}
    
    # Initialize the dataframes that need to be filled during the sensitivity analysis
    class_imbalance = pd.DataFrame()
    default_ratio_male = pd.DataFrame()
    default_ratio_female = pd.DataFrame()
    diff_default_ratio = pd.DataFrame()
    
    pr_default_male = pd.DataFrame()
    pr_default_female= pd.DataFrame()
    pr_default_male_rew = pd.DataFrame()
    pr_default_female_rew = pd.DataFrame()
    
    pr_true_default_male = pd.DataFrame()
    pr_true_default_female = pd.DataFrame()
    pr_false_default_male = pd.DataFrame()
    pr_false_default_female = pd.DataFrame()
    
    pr_true_default_male_rew = pd.DataFrame()
    pr_true_default_female_rew = pd.DataFrame()
    pr_false_default_male_rew = pd.DataFrame()
    pr_false_default_female_rew = pd.DataFrame()
    
    accuracy = pd.DataFrame()
    recall = pd.DataFrame()
    spd = pd.DataFrame()
    aod = pd.DataFrame()
    
    accuracy_rew = pd.DataFrame()
    recall_rew = pd.DataFrame()
    spd_rew = pd.DataFrame()
    aod_rew = pd.DataFrame()
    
    diff_accuracy = pd.DataFrame()
    diff_recall = pd.DataFrame()
    diff_spd = pd.DataFrame()
    diff_aod = pd.DataFrame()
    
    # Set up outer cross validation, split data into 5 folds
    # Stratified preserves the distribution of classes among the different folds
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    i=1
    for train_index, test_index in outer_cv.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        ###### DATA MANIPULATIONS #######
        # Core train set
        X_train_core, y_train_core, X_train_other, y_train_other = core_trainset(X_train, y_train, 0.5)
        
        for manip in manipulations:
            # Manipulate X_train/y_train
            if manip=='orig':
                if label_bias == True:
                    X_train_manipulated, y_train_manipulated = change_labels(X_train, y_train, orig=True)
                elif sample_bias == True:
                    X_train_manipulated, y_train_manipulated = add_samples(X_train_core, y_train_core, X_train_other, y_train_other)
            elif manip=='male_default_0_9':
                if label_bias == True:
                    X_train_manipulated, y_train_manipulated = change_labels(X_train, y_train, default_male=True)
                elif sample_bias == True:
                    mult = manipulations[manip]
                    X_train_manipulated, y_train_manipulated = add_samples(X_train_core, y_train_core, X_train_other, y_train_other, default_male=True, mult_factor=mult)
            elif manip=='male_no_default_1':
                if label_bias == True:
                    X_train_manipulated, y_train_manipulated = change_labels(X_train, y_train, no_default_male=True)
                elif sample_bias == True:
                    mult = manipulations[manip]
                    X_train_manipulated, y_train_manipulated = add_samples(X_train_core, y_train_core, X_train_other, y_train_other, no_default_male=True, mult_factor=mult)
            elif manip=='female_default_0_9':
                if label_bias == True:
                    X_train_manipulated, y_train_manipulated = change_labels(X_train, y_train, default_female=True)
                elif sample_bias == True:
                    mult = manipulations[manip]
                    X_train_manipulated, y_train_manipulated = add_samples(X_train_core, y_train_core, X_train_other, y_train_other, default_female=True, mult_factor=mult)
            elif manip=='female_no_default_1':
                if label_bias == True:
                    X_train_manipulated, y_train_manipulated = change_labels(X_train, y_train, no_default_female=True)
                elif sample_bias == True:
                    mult = manipulations[manip]
                    X_train_manipulated, y_train_manipulated = add_samples(X_train_core, y_train_core, X_train_other, y_train_other, no_default_female=True, mult_factor=mult)
               
            ###### FILL RESULTS IN DATA FRAMES #######
            # Class imbalance and ratio defaults
            class_imbalance.loc[i, manip] = sum(y_train_manipulated)/len(y_train_manipulated)
            default_ratio_male.loc[i, manip] = sum(y_train_manipulated[y_train_manipulated.index.get_level_values(1)==1])/len(y_train_manipulated[y_train_manipulated.index.get_level_values(1)==1])
            default_ratio_female.loc[i, manip] = sum(y_train_manipulated[y_train_manipulated.index.get_level_values(1)==0])/len(y_train_manipulated[y_train_manipulated.index.get_level_values(1)==0])
            diff_default_ratio.loc[i, manip] = default_ratio_male.loc[i, manip] - default_ratio_female.loc[i, manip]
            
            #### PRE-PROCESSING #######
            # Apply reweighing
            rw = Reweighing('MALE')
            X_train_manipulated, weights_rew = rw.fit_transform(X_train_manipulated, y_train_manipulated)
            
            # Scale train data
            scaler = StandardScaler()
            scaler.fit(X_train_manipulated)
            X_train_manipulated = scaler.fit_transform(X_train_manipulated)
            
            #### HYPERPARAMETER TUNING #######
            # Split data into 5 folds
            inner_cv = StratifiedKFold(n_splits=5, 
                                       shuffle=True, 
                                       random_state=42)
        
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
                search_rf.fit(X_train_manipulated, y_train_manipulated)
                best_model = search_rf.best_estimator_ # Obtain best model (hyperparameters) based on average performance
                # Preprocessing
                search_rf_rew = RandomizedSearchCV(estimator=rf, 
                                               n_jobs=-1,
                                               param_distributions=hyperparam_grid, 
                                               cv=inner_cv,
                                               scoring='recall')
                search_rf_rew.fit(X_train_manipulated, y_train_manipulated, sample_weight=weights_rew)
                best_model_rew = search_rf_rew.best_estimator_
            if model=='lr':
                # Set parameters for hyperparameter tuning
                hyperparam_grid = {'penalty': ['none','l2'],
                                   'C':[0.01,0.2,1,10,100]}       
                lr = LogisticRegression(random_state=42, max_iter=500) 
                # No preprocessing
                search_lr = RandomizedSearchCV(estimator=lr, 
                                               n_jobs=-1, 
                                               param_distributions=hyperparam_grid, 
                                               cv=inner_cv, 
                                               scoring='recall')
                search_lr.fit(X_train_manipulated, y_train_manipulated)
                best_model = search_lr.best_estimator_  # Obtain best model (hyperparameters) based on average performance
                # Preprocessing
                search_lr_rew = RandomizedSearchCV(estimator=lr, 
                                                   n_jobs=-1, 
                                                   param_distributions=hyperparam_grid, 
                                                   cv=inner_cv, 
                                                   scoring='recall')
                search_lr_rew.fit(X_train_manipulated, y_train_manipulated, sample_weight=weights_rew)
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
                search_xgb.fit(X_train_manipulated, y_train_manipulated)
                best_model = search_xgb.best_estimator_  # Obtain best model (hyperparameters) based on average performance
                # Preprocessing
                search_xgb_rew = RandomizedSearchCV(estimator=xgb, 
                                                   n_jobs=-1, 
                                                   param_distributions=hyperparam_grid, 
                                                   cv=inner_cv, 
                                                   scoring='recall')
                search_xgb_rew.fit(X_train_manipulated, y_train_manipulated, sample_weight=weights_rew)
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
                search_dt.fit(X_train_manipulated, y_train_manipulated)
                best_model = search_dt.best_estimator_  # Obtain best model (hyperparameters) based on average performance
                # Preprocessing
                search_dt_rew = RandomizedSearchCV(estimator=dt, 
                                                   n_jobs=-1, 
                                                   param_distributions=hyperparam_grid, 
                                                   cv=inner_cv, 
                                                   scoring='recall')
                search_dt_rew.fit(X_train_manipulated, y_train_manipulated, sample_weight=weights_rew)
                best_model_rew = search_dt_rew.best_estimator_  # Obtain best model (hyperparameters) based on average performance
     
            ###### MAKE PREDICTIONS #######
            # Scale test data
            X_test_sc = scaler.transform(X_test)
    
            # Predict for each manipulated test set
            y_pred = best_model.predict(X_test_sc)
            y_pred_rew = best_model_rew.predict(X_test_sc)
            
            ###### FILL RESULTS IN DATA FRAMES #######
            ## NO PREPROCESSING ##
            # Performance metrics
            accuracy.loc[i, manip] = accuracy_score(y_true=y_test, y_pred=y_pred)
            recall.loc[i, manip] = recall_score(y_true=y_test, y_pred=y_pred)
            spd.loc[i, manip] = statistical_parity_difference(y_test, y_pred, prot_attr='MALE', priv_group=0)  
            aod.loc[i, manip] = average_odds_difference(y_test, y_pred, prot_attr='MALE', priv_group=0)
            
            # Intermediate steps to calculate spd
            y_pred = pd.Series(y_pred)
            y_pred.index = y_test.index
            pr_default_male.loc[i, manip] = sum(y_pred[y_pred.index.get_level_values(1)==1])/len(y_pred[y_pred.index.get_level_values(1)==1])
            pr_default_female.loc[i, manip] = sum(y_pred[y_pred.index.get_level_values(1)==0])/len(y_pred[y_pred.index.get_level_values(1)==0])
            
            # Intermediate steps to calculate aod
            pr_true_default_male.loc[i, manip] = sum(y_pred[(y_pred.index.get_level_values(1)==1)&(y_test==1)])/len(y_pred[(y_pred.index.get_level_values(1)==1)&(y_test==1)])
            pr_true_default_female.loc[i, manip] = sum(y_pred[(y_pred.index.get_level_values(1)==0)&(y_test==1)])/len(y_pred[(y_pred.index.get_level_values(1)==0)&(y_test==1)])
            pr_false_default_male.loc[i, manip] = sum(y_pred[(y_pred.index.get_level_values(1)==1)&(y_test==0)])/len(y_pred[(y_pred.index.get_level_values(1)==1)&(y_test==0)])
            pr_false_default_female.loc[i, manip] = sum(y_pred[(y_pred.index.get_level_values(1)==0)&(y_test==0)])/len(y_pred[(y_pred.index.get_level_values(1)==0)&(y_test==0)])
            
            ## PREPROCESSING ##
            # Performance metrics
            accuracy_rew.loc[i, manip] = accuracy_score(y_true=y_test, y_pred=y_pred_rew)
            recall_rew.loc[i, manip] = recall_score(y_true=y_test, y_pred=y_pred_rew)
            spd_rew.loc[i, manip] = statistical_parity_difference(y_test, y_pred_rew, prot_attr='MALE', priv_group=0)  
            aod_rew.loc[i, manip] = average_odds_difference(y_test, y_pred_rew, prot_attr='MALE', priv_group=0)
          
            # Intermediate steps to calculate spd
            y_pred_rew = pd.Series(y_pred_rew)
            y_pred_rew.index = y_test.index
            pr_default_male_rew.loc[i, manip] = sum(y_pred_rew[y_pred_rew.index.get_level_values(1)==1])/len(y_pred_rew[y_pred_rew.index.get_level_values(1)==1])
            pr_default_female_rew.loc[i, manip] = sum(y_pred_rew[y_pred_rew.index.get_level_values(1)==0])/len(y_pred_rew[y_pred_rew.index.get_level_values(1)==0])
            
            # Intermediate steps to calculate aod
            pr_true_default_male_rew.loc[i, manip] = sum(y_pred_rew[(y_pred_rew.index.get_level_values(1)==1)&(y_test==1)])/len(y_pred_rew[(y_pred_rew.index.get_level_values(1)==1)&(y_test==1)])
            pr_true_default_female_rew.loc[i, manip] = sum(y_pred_rew[(y_pred_rew.index.get_level_values(1)==0)&(y_test==1)])/len(y_pred_rew[(y_pred_rew.index.get_level_values(1)==0)&(y_test==1)])
            pr_false_default_male_rew.loc[i, manip] = sum(y_pred_rew[(y_pred_rew.index.get_level_values(1)==1)&(y_test==0)])/len(y_pred_rew[(y_pred_rew.index.get_level_values(1)==1)&(y_test==0)])
            pr_false_default_female_rew.loc[i, manip] = sum(y_pred_rew[(y_pred_rew.index.get_level_values(1)==0)&(y_test==0)])/len(y_pred_rew[(y_pred_rew.index.get_level_values(1)==0)&(y_test==0)])
           
            ## COMPARE NO PREPROCESSING WITH PREPROCESSING ##
            # Difference in performance metrics
            diff_accuracy.loc[i, manip] = accuracy.loc[i, manip] - accuracy_rew.loc[i, manip] 
            diff_recall.loc[i, manip] = recall.loc[i, manip] - recall_rew.loc[i, manip]
            diff_spd.loc[i, manip] = spd.loc[i, manip] - spd_rew.loc[i, manip]
            diff_aod.loc[i, manip] = aod.loc[i, manip] - aod_rew.loc[i, manip]
        
        i = i+1
    
    # Combine performance metrics into one dataframe
    metrics_output = pd.concat([np.mean(class_imbalance), np.mean(default_ratio_male), np.mean(default_ratio_female),
                            np.mean(diff_default_ratio), np.mean(accuracy),np.mean(recall),
                            np.mean(accuracy_rew),np.mean(recall_rew),
                            np.mean(diff_accuracy), np.mean(diff_recall),
                            np.mean(spd), np.mean(aod), np.mean(spd_rew), np.mean(aod_rew),
                            np.std(spd), np.std(aod), np.std(spd_rew), np.std(aod_rew)], axis=1)
    metrics_output.columns=['class_imbalance', 'default_ratio_male', 'default_ratio_female', 'diff_default_ratio', 
                            'accuracy', 'recall','accuracy_rew', 'recall_rew', 'diff_accuracy', 'diff_recall',
                            'spd', 'aod', 'spd_rew', 'aod_rew', 'std_spd', 'std_aod', 'std_spd_rew', 'std_aod_rew']
    metrics_output = metrics_output.sort_values(by='diff_default_ratio')
    
    # Combine spd output into one dataframe
    spd_output = pd.concat([np.mean(diff_default_ratio), np.mean(pr_default_male), np.mean(pr_default_female), np.mean(spd), 
                            np.mean(pr_default_male_rew), np.mean(pr_default_female_rew), np.mean(spd_rew), 
                            np.mean(diff_spd)], axis=1)
    spd_output.columns=['diff_default_ratio', 'pr_default_male', 'pr_default_female', 'spd', 'pr_default_male_rew',
                        'pr_default_female_rew', 'spd_rew', 'diff_spd']
    spd_output = spd_output.sort_values(by='diff_default_ratio')
    
    # Combine aod output into one dataframe
    aod_output = pd.concat([np.mean(diff_default_ratio), np.mean(pr_true_default_male), np.mean(pr_true_default_female),
                            np.mean(pr_false_default_male), np.mean(pr_false_default_female), np.mean(aod), 
                            np.mean(pr_true_default_male_rew), np.mean(pr_true_default_female_rew), 
                            np.mean(pr_false_default_male_rew), np.mean(pr_false_default_female_rew), np.mean(aod_rew), 
                            np.mean(diff_aod)], axis=1)
    aod_output.columns=['diff_default_ratio', 'pr_true_default_male', 'pr_true_default_female', 'pr_false_default_male',
                        'pr_false_default_female', 'aod', 'pr_true_default_male_rew', 'pr_true_default_female_rew',
                        'pr_false_default_male_rew', 'pr_false_default_female_rew', 'aod_rew', 'diff_aod']
    aod_output = aod_output.sort_values(by='diff_default_ratio')
   
    return(metrics_output, spd_output, diff_spd, aod_output, diff_aod)

### MASSAGING ###
# Function to perform sensitivity analysis
def sensitivity_analysis_mas(model, label_bias=False, sample_bias=False): # Model options: 'lr', 'rf', 'dt', 'xgb'
    np.random.seed(42) # Set random seed in order to replicate results
    
    # Set the different manipulations
    manipulations = {'male_no_default_1':1, 'female_default_0_9':0.9, 'orig':1, 'male_default_0_9':0.9,
                     'female_no_default_1':1}
    
    # Initialize the dataframes that need to be filled during the sensitivity analysis
    class_imbalance = pd.DataFrame()
    default_ratio_male = pd.DataFrame()
    default_ratio_female = pd.DataFrame()
    diff_default_ratio = pd.DataFrame()
    
    pr_default_male = pd.DataFrame()
    pr_default_female= pd.DataFrame()
    pr_default_male_mas = pd.DataFrame()
    pr_default_female_mas = pd.DataFrame()
    
    pr_true_default_male = pd.DataFrame()
    pr_true_default_female = pd.DataFrame()
    pr_false_default_male = pd.DataFrame()
    pr_false_default_female = pd.DataFrame()
    
    pr_true_default_male_mas = pd.DataFrame()
    pr_true_default_female_mas = pd.DataFrame()
    pr_false_default_male_mas = pd.DataFrame()
    pr_false_default_female_mas = pd.DataFrame()
    
    accuracy = pd.DataFrame()
    recall = pd.DataFrame()
    spd = pd.DataFrame()
    aod = pd.DataFrame()
    
    accuracy_mas = pd.DataFrame()
    recall_mas = pd.DataFrame()
    spd_mas = pd.DataFrame()
    aod_mas = pd.DataFrame()
    
    diff_accuracy = pd.DataFrame()
    diff_recall = pd.DataFrame()
    diff_spd = pd.DataFrame()
    diff_aod = pd.DataFrame()
    
    # Set up outer cross validation, split data into 5 folds
    # Stratified preserves the distribution of classes among the different folds
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    i=1
    for train_index, test_index in outer_cv.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        ###### DATA MANIPULATIONS #######
    
        # Core train set
        X_train_core, y_train_core, X_train_other, y_train_other = core_trainset(X_train, y_train, 0.5)
        
        for manip in manipulations:
            # Manipulate X_train/y_train
            if manip=='orig':
                if label_bias == True:
                    X_train_manipulated, y_train_manipulated = change_labels(X_train, y_train, orig=True)
                elif sample_bias == True:
                    X_train_manipulated, y_train_manipulated = add_samples(X_train_core, y_train_core, X_train_other, y_train_other)
            elif manip=='male_default_0_9':
                if label_bias == True:
                    X_train_manipulated, y_train_manipulated = change_labels(X_train, y_train, default_male=True)
                elif sample_bias == True:
                    mult = manipulations[manip]
                    X_train_manipulated, y_train_manipulated = add_samples(X_train_core, y_train_core, X_train_other, y_train_other, default_male=True, mult_factor=mult)
            elif manip=='male_no_default_1':
                if label_bias == True:
                    X_train_manipulated, y_train_manipulated = change_labels(X_train, y_train, no_default_male=True)
                elif sample_bias == True:
                    mult = manipulations[manip]
                    X_train_manipulated, y_train_manipulated = add_samples(X_train_core, y_train_core, X_train_other, y_train_other, no_default_male=True, mult_factor=mult)
            elif manip=='female_default_0_9':
                if label_bias == True:
                    X_train_manipulated, y_train_manipulated = change_labels(X_train, y_train, default_female=True)
                elif sample_bias == True:
                    mult = manipulations[manip]
                    X_train_manipulated, y_train_manipulated = add_samples(X_train_core, y_train_core, X_train_other, y_train_other, default_female=True, mult_factor=mult)
            elif manip=='female_no_default_1':
                if label_bias == True:
                    X_train_manipulated, y_train_manipulated = change_labels(X_train, y_train, no_default_female=True)
                elif sample_bias == True:
                    mult = manipulations[manip]
                    X_train_manipulated, y_train_manipulated = add_samples(X_train_core, y_train_core, X_train_other, y_train_other, no_default_female=True, mult_factor=mult)
            
            ###### FILL RESULTS IN DATA FRAMES #######
            # Class imbalance and ratio defaults
            class_imbalance.loc[i, manip] = sum(y_train_manipulated)/len(y_train_manipulated)
            default_ratio_male.loc[i, manip] = sum(y_train_manipulated[y_train_manipulated.index.get_level_values(1)==1])/len(y_train_manipulated[y_train_manipulated.index.get_level_values(1)==1])
            default_ratio_female.loc[i, manip] = sum(y_train_manipulated[y_train_manipulated.index.get_level_values(1)==0])/len(y_train_manipulated[y_train_manipulated.index.get_level_values(1)==0])
            diff_default_ratio.loc[i, manip] = default_ratio_male.loc[i, manip] - default_ratio_female.loc[i, manip]
            
            s = X_train_manipulated['MALE_VAR']
            
            total = float(len(s))
            s1 = s.sum()
            s0 = total - s1
            s1_positive = ((s == 1) & (y_train_manipulated == 1)).sum()
            s0_positive = ((s == 0) & (y_train_manipulated == 1)).sum()
            number = int(math.ceil(((s1 * s0_positive) - (s0 * s1_positive)) / total))
            
            #### PRE-PROCESSING #######
            # Apply massaging
            if number < 0:
                y_rev = np.where(y_train_manipulated==1,0,1)
            else:
                y_rev = y_train_manipulated
            massager = relabelling.Relabeller(ranker=LogisticRegression(max_iter=500))
            s = X_train_manipulated['MALE_VAR']
            y_train_rev = massager.fit(X_train_manipulated, y_rev, s).transform(X_train_manipulated)
            if number < 0:
                y_train_new = np.where(y_train_rev==1,0,1)
            else:
                y_train_new = y_train_rev
                
            sum(y_train_new != y_train_manipulated)
            
            # Scale train data
            scaler = StandardScaler()
            scaler.fit(X_train_manipulated)
            X_train_manipulated = scaler.fit_transform(X_train_manipulated)
            
            #### HYPERPARAMETER TUNING #######
            # Split data into 5 folds
            inner_cv = StratifiedKFold(n_splits=5, 
                                       shuffle=True, 
                                       random_state=42)
        
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
                search_rf.fit(X_train_manipulated, y_train_manipulated)
                best_model = search_rf.best_estimator_ # Obtain best model (hyperparameters) based on average performance
                # Preprocessing
                search_rf_mas = RandomizedSearchCV(estimator=rf, 
                                               n_jobs=-1,
                                               param_distributions=hyperparam_grid, 
                                               cv=inner_cv,
                                               scoring='recall')
                search_rf_mas.fit(X_train_manipulated, y_train_new)
                best_model_mas = search_rf_mas.best_estimator_
            if model=='lr':
                # Set parameters for hyperparameter tuning
                hyperparam_grid = {'penalty': ['none','l2'],
                                   'C':[0.01,0.2,1,10,100]}       
                lr = LogisticRegression(random_state=42, max_iter=500) 
                # No preprocessing
                search_lr = RandomizedSearchCV(estimator=lr, 
                                               n_jobs=-1, 
                                               param_distributions=hyperparam_grid, 
                                               cv=inner_cv, 
                                               scoring='recall')
                search_lr.fit(X_train_manipulated, y_train_manipulated)
                best_model = search_lr.best_estimator_  # Obtain best model (hyperparameters) based on average performance
                # Preprocessing
                search_lr_mas = RandomizedSearchCV(estimator=lr, 
                                                   n_jobs=-1, 
                                                   param_distributions=hyperparam_grid, 
                                                   cv=inner_cv, 
                                                   scoring='recall')
                search_lr_mas.fit(X_train_manipulated, y_train_new)
                best_model_mas = search_lr_mas.best_estimator_  # Obtain best model (hyperparameters) based on average performance
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
                search_xgb.fit(X_train_manipulated, y_train_manipulated)
                best_model = search_xgb.best_estimator_  # Obtain best model (hyperparameters) based on average performance
                # Preprocessing
                search_xgb_mas = RandomizedSearchCV(estimator=xgb, 
                                                   n_jobs=-1, 
                                                   param_distributions=hyperparam_grid, 
                                                   cv=inner_cv, 
                                                   scoring='recall')
                search_xgb_mas.fit(X_train_manipulated, y_train_new)
                best_model_mas = search_xgb_mas.best_estimator_  # Obtain best model (hyperparameters) based on average performance
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
                search_dt.fit(X_train_manipulated, y_train_manipulated)
                best_model = search_dt.best_estimator_  # Obtain best model (hyperparameters) based on average performance
                # Preprocessing
                search_dt_mas = RandomizedSearchCV(estimator=dt, 
                                                   n_jobs=-1, 
                                                   param_distributions=hyperparam_grid, 
                                                   cv=inner_cv, 
                                                   scoring='recall')
                search_dt_mas.fit(X_train_manipulated, y_train_new)
                best_model_mas = search_dt_mas.best_estimator_  # Obtain best model (hyperparameters) based on average performance
     
            ###### MAKE PREDICTIONS #######
            # Scale test data
            X_test_sc = scaler.transform(X_test)
    
            # Predict for each manipulated test set
            y_pred = best_model.predict(X_test_sc)
            y_pred_mas = best_model_mas.predict(X_test_sc)
            
            ###### FILL RESULTS IN DATA FRAMES #######
            ## NO PREPROCESSING ##
            # Performance metrics
            accuracy.loc[i, manip] = accuracy_score(y_true=y_test, y_pred=y_pred)
            recall.loc[i, manip] = recall_score(y_true=y_test, y_pred=y_pred)
            spd.loc[i, manip] = statistical_parity_difference(y_test, y_pred, prot_attr='MALE', priv_group=0)  
            aod.loc[i, manip] = average_odds_difference(y_test, y_pred, prot_attr='MALE', priv_group=0)
            
            # Intermediate steps to calculate spd
            y_pred = pd.Series(y_pred)
            y_pred.index = y_test.index
            pr_default_male.loc[i, manip] = sum(y_pred[y_pred.index.get_level_values(1)==1])/len(y_pred[y_pred.index.get_level_values(1)==1])
            pr_default_female.loc[i, manip] = sum(y_pred[y_pred.index.get_level_values(1)==0])/len(y_pred[y_pred.index.get_level_values(1)==0])
            
            # Intermediate steps to calculate aod
            pr_true_default_male.loc[i, manip] = sum(y_pred[(y_pred.index.get_level_values(1)==1)&(y_test==1)])/len(y_pred[(y_pred.index.get_level_values(1)==1)&(y_test==1)])
            pr_true_default_female.loc[i, manip] = sum(y_pred[(y_pred.index.get_level_values(1)==0)&(y_test==1)])/len(y_pred[(y_pred.index.get_level_values(1)==0)&(y_test==1)])
            pr_false_default_male.loc[i, manip] = sum(y_pred[(y_pred.index.get_level_values(1)==1)&(y_test==0)])/len(y_pred[(y_pred.index.get_level_values(1)==1)&(y_test==0)])
            pr_false_default_female.loc[i, manip] = sum(y_pred[(y_pred.index.get_level_values(1)==0)&(y_test==0)])/len(y_pred[(y_pred.index.get_level_values(1)==0)&(y_test==0)])
            
            ## PREPROCESSING ##
            # Performance metrics
            accuracy_mas.loc[i, manip] = accuracy_score(y_true=y_test, y_pred=y_pred_mas)
            recall_mas.loc[i, manip] = recall_score(y_true=y_test, y_pred=y_pred_mas)
            spd_mas.loc[i, manip] = statistical_parity_difference(y_test, y_pred_mas, prot_attr='MALE', priv_group=0)  
            aod_mas.loc[i, manip] = average_odds_difference(y_test, y_pred_mas, prot_attr='MALE', priv_group=0)
          
            # Intermediate steps to calculate spd
            y_pred_mas = pd.Series(y_pred_mas)
            y_pred_mas.index = y_test.index
            pr_default_male_mas.loc[i, manip] = sum(y_pred_mas[y_pred_mas.index.get_level_values(1)==1])/len(y_pred_mas[y_pred_mas.index.get_level_values(1)==1])
            pr_default_female_mas.loc[i, manip] = sum(y_pred_mas[y_pred_mas.index.get_level_values(1)==0])/len(y_pred_mas[y_pred_mas.index.get_level_values(1)==0])
            
            # Intermediate steps to calculate aod
            pr_true_default_male_mas.loc[i, manip] = sum(y_pred_mas[(y_pred_mas.index.get_level_values(1)==1)&(y_test==1)])/len(y_pred_mas[(y_pred_mas.index.get_level_values(1)==1)&(y_test==1)])
            pr_true_default_female_mas.loc[i, manip] = sum(y_pred_mas[(y_pred_mas.index.get_level_values(1)==0)&(y_test==1)])/len(y_pred_mas[(y_pred_mas.index.get_level_values(1)==0)&(y_test==1)])
            pr_false_default_male_mas.loc[i, manip] = sum(y_pred_mas[(y_pred_mas.index.get_level_values(1)==1)&(y_test==0)])/len(y_pred_mas[(y_pred_mas.index.get_level_values(1)==1)&(y_test==0)])
            pr_false_default_female_mas.loc[i, manip] = sum(y_pred_mas[(y_pred_mas.index.get_level_values(1)==0)&(y_test==0)])/len(y_pred_mas[(y_pred_mas.index.get_level_values(1)==0)&(y_test==0)])
           
            ## COMPARE NO PREPROCESSING WITH PREPROCESSING ##
            # Difference in performance metrics
            diff_accuracy.loc[i, manip] = accuracy.loc[i, manip] - accuracy_mas.loc[i, manip] 
            diff_recall.loc[i, manip] = recall.loc[i, manip] - recall_mas.loc[i, manip]
            diff_spd.loc[i, manip] = spd.loc[i, manip] - spd_mas.loc[i, manip]
            diff_aod.loc[i, manip] = aod.loc[i, manip] - aod_mas.loc[i, manip]
        
        i = i+1
    
    # Combine performance metrics into one dataframe
    metrics_output = pd.concat([np.mean(class_imbalance), np.mean(default_ratio_male), np.mean(default_ratio_female),
                            np.mean(diff_default_ratio), np.mean(accuracy),np.mean(recall),
                            np.mean(accuracy_mas),np.mean(recall_mas),
                            np.mean(diff_accuracy), np.mean(diff_recall),
                            np.mean(spd), np.mean(aod), np.mean(spd_mas), np.mean(aod_mas),
                            np.std(spd), np.std(aod), np.std(spd_mas), np.std(aod_mas)], axis=1)
    metrics_output.columns=['class_imbalance', 'default_ratio_male', 'default_ratio_female', 'diff_default_ratio', 
                            'accuracy', 'recall','accuracy_mas', 'recall_mas', 'diff_accuracy', 'diff_recall',
                            'spd', 'aod', 'spd_mas', 'aod_mas', 'std_spd', 'std_aod', 'std_spd_mas', 'std_aod_mas']
    metrics_output = metrics_output.sort_values(by='diff_default_ratio')
    
    # Combine spd output into one dataframe
    spd_output = pd.concat([np.mean(diff_default_ratio), np.mean(pr_default_male), np.mean(pr_default_female), np.mean(spd), 
                            np.mean(pr_default_male_mas), np.mean(pr_default_female_mas), np.mean(spd_mas), 
                            np.mean(diff_spd)], axis=1)
    spd_output.columns=['diff_default_ratio', 'pr_default_male', 'pr_default_female', 'spd', 'pr_default_male_mas',
                        'pr_default_female_mas', 'spd_mas', 'diff_spd']
    spd_output = spd_output.sort_values(by='diff_default_ratio')
    
    # Combine aod output into one dataframe
    aod_output = pd.concat([np.mean(diff_default_ratio), np.mean(pr_true_default_male), np.mean(pr_true_default_female),
                            np.mean(pr_false_default_male), np.mean(pr_false_default_female), np.mean(aod), 
                            np.mean(pr_true_default_male_mas), np.mean(pr_true_default_female_mas), 
                            np.mean(pr_false_default_male_mas), np.mean(pr_false_default_female_mas), np.mean(aod_mas), 
                            np.mean(diff_aod)], axis=1)
    aod_output.columns=['diff_default_ratio', 'pr_true_default_male', 'pr_true_default_female', 'pr_false_default_male',
                        'pr_false_default_female', 'aod', 'pr_true_default_male_mas', 'pr_true_default_female_mas',
                        'pr_false_default_male_mas', 'pr_false_default_female_mas', 'aod_mas', 'diff_aod']
    aod_output = aod_output.sort_values(by='diff_default_ratio')
   
    return(metrics_output, spd_output, diff_spd, aod_output, diff_aod)

### ADVERSARIAL DEBIASING ###

# Function to perform sensitivity analysis
def sensitivity_analysis_ad(label_bias=False, sample_bias=False):
    np.random.seed(42) # Set random seed in order to replicate results
    
    # Set the different manipulations
    manipulations = {'male_no_default_1':1, 'female_default_0_9':0.9, 'orig':1, 'male_default_0_9':0.9,
                     'female_no_default_1':1}
    
    # Initialize the dataframes that need to be filled during the sensitivity analysis
    class_imbalance = pd.DataFrame()
    default_ratio_male = pd.DataFrame()
    default_ratio_female = pd.DataFrame()
    diff_default_ratio = pd.DataFrame()
    
    pr_default_male = pd.DataFrame()
    pr_default_female= pd.DataFrame()
    pr_default_male_ad = pd.DataFrame()
    pr_default_female_ad = pd.DataFrame()
    
    pr_true_default_male = pd.DataFrame()
    pr_true_default_female = pd.DataFrame()
    pr_false_default_male = pd.DataFrame()
    pr_false_default_female = pd.DataFrame()
    
    pr_true_default_male_ad = pd.DataFrame()
    pr_true_default_female_ad = pd.DataFrame()
    pr_false_default_male_ad = pd.DataFrame()
    pr_false_default_female_ad = pd.DataFrame()
    
    accuracy = pd.DataFrame()
    recall = pd.DataFrame()
    spd = pd.DataFrame()
    aod = pd.DataFrame()
    
    accuracy_ad = pd.DataFrame()
    recall_ad = pd.DataFrame()
    spd_ad = pd.DataFrame()
    aod_ad = pd.DataFrame()
    
    diff_accuracy = pd.DataFrame()
    diff_recall = pd.DataFrame()
    diff_spd = pd.DataFrame()
    diff_aod = pd.DataFrame()
    
    # Set up outer cross validation, split data into 5 folds
    # Stratified preserves the distribution of classes among the different folds
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    i=1
    for train_index, test_index in outer_cv.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        print(i)
        
       # (X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=1234567)
        
        ###### DATA MANIPULATIONS #######
        # Core train set
        X_train_core, y_train_core, X_train_other, y_train_other = core_trainset(X_train, y_train, 0.5)
        
        for manip in manipulations:
            # Manipulate X_train/y_train
            if manip=='orig':
                if label_bias == True:
                    X_train_manipulated, y_train_manipulated = change_labels(X_train, y_train, orig=True)
                elif sample_bias == True:
                    X_train_manipulated, y_train_manipulated = add_samples(X_train_core, y_train_core, X_train_other, y_train_other)
            elif manip=='male_default_0_9':
                if label_bias == True:
                    X_train_manipulated, y_train_manipulated = change_labels(X_train, y_train, default_male=True)
                elif sample_bias == True:
                    mult = manipulations[manip]
                    X_train_manipulated, y_train_manipulated = add_samples(X_train_core, y_train_core, X_train_other, y_train_other, default_male=True, mult_factor=mult)
            elif manip=='male_no_default_1':
                if label_bias == True:
                    X_train_manipulated, y_train_manipulated = change_labels(X_train, y_train, no_default_male=True)
                elif sample_bias == True:
                    mult = manipulations[manip]
                    X_train_manipulated, y_train_manipulated = add_samples(X_train_core, y_train_core, X_train_other, y_train_other, no_default_male=True, mult_factor=mult)
            elif manip=='female_default_0_9':
                if label_bias == True:
                    X_train_manipulated, y_train_manipulated = change_labels(X_train, y_train, default_female=True)
                elif sample_bias == True:
                    mult = manipulations[manip]
                    X_train_manipulated, y_train_manipulated = add_samples(X_train_core, y_train_core, X_train_other, y_train_other, default_female=True, mult_factor=mult)
            elif manip=='female_no_default_1':
                if label_bias == True:
                    X_train_manipulated, y_train_manipulated = change_labels(X_train, y_train, no_default_female=True)
                elif sample_bias == True:
                    mult = manipulations[manip]
                    X_train_manipulated, y_train_manipulated = add_samples(X_train_core, y_train_core, X_train_other, y_train_other, no_default_female=True, mult_factor=mult)
             
            print(manip)
            ###### FILL RESULTS IN DATA FRAMES #######
            # Class imbalance and ratio defaults
            class_imbalance.loc[i, manip] = sum(y_train_manipulated)/len(y_train_manipulated)
            default_ratio_male.loc[i, manip] = sum(y_train_manipulated[y_train_manipulated.index.get_level_values(1)==1])/len(y_train_manipulated[y_train_manipulated.index.get_level_values(1)==1])
            default_ratio_female.loc[i, manip] = sum(y_train_manipulated[y_train_manipulated.index.get_level_values(1)==0])/len(y_train_manipulated[y_train_manipulated.index.get_level_values(1)==0])
            diff_default_ratio.loc[i, manip] = default_ratio_male.loc[i, manip] - default_ratio_female.loc[i, manip]
                
            # Scale train data
            scaler = StandardScaler()
            scaler.fit(X_train_manipulated)
            X_train_manipulated = scaler.fit_transform(X_train_manipulated)
            X_train_manipulated = pd.DataFrame(X_train_manipulated)
            X_train_manipulated = X_train_manipulated.set_index(y_train_manipulated.index)

            #### IN-PROCESSING #######
            no_adv_deb = AdversarialDebiasing(prot_attr='MALE',
                                   debias=False,
                                   adversary_loss_weight=0.05,
                                   random_state=42)
            no_adv_deb.fit(X_train_manipulated, y_train_manipulated)
          
            
            adv_deb = AdversarialDebiasing(prot_attr='MALE',
                                   debias=True,
                                   adversary_loss_weight=0.05,
                                   random_state=42)
            adv_deb.fit(X_train_manipulated, y_train_manipulated)
            
            ###### MAKE PREDICTIONS #######
            # Scale test data
            X_test_sc = scaler.transform(X_test)
            X_test_sc = pd.DataFrame(X_test_sc)
            X_test_sc = X_test_sc.set_index(y_test.index)
            
            # Predict for each manipulated test set
            y_pred = no_adv_deb.predict(X_test_sc)
            y_pred_ad = adv_deb.predict(X_test_sc)
            
            ###### FILL RESULTS IN DATA FRAMES #######
            ## NO PREPROCESSING ##
            # Performance metrics
            accuracy.loc[i, manip] = accuracy_score(y_true=y_test, y_pred=y_pred)
            recall.loc[i, manip] = recall_score(y_true=y_test, y_pred=y_pred)
            spd.loc[i, manip] = statistical_parity_difference(y_test, y_pred, prot_attr='MALE', priv_group=0)  
            aod.loc[i, manip] = average_odds_difference(y_test, y_pred, prot_attr='MALE', priv_group=0)
            
            # Intermediate steps to calculate spd
            y_pred = pd.Series(y_pred)
            y_pred.index = y_test.index
            pr_default_male.loc[i, manip] = sum(y_pred[y_pred.index.get_level_values(1)==1])/len(y_pred[y_pred.index.get_level_values(1)==1])
            pr_default_female.loc[i, manip] = sum(y_pred[y_pred.index.get_level_values(1)==0])/len(y_pred[y_pred.index.get_level_values(1)==0])
            
            # Intermediate steps to calculate aod
            pr_true_default_male.loc[i, manip] = sum(y_pred[(y_pred.index.get_level_values(1)==1)&(y_test==1)])/len(y_pred[(y_pred.index.get_level_values(1)==1)&(y_test==1)])
            pr_true_default_female.loc[i, manip] = sum(y_pred[(y_pred.index.get_level_values(1)==0)&(y_test==1)])/len(y_pred[(y_pred.index.get_level_values(1)==0)&(y_test==1)])
            pr_false_default_male.loc[i, manip] = sum(y_pred[(y_pred.index.get_level_values(1)==1)&(y_test==0)])/len(y_pred[(y_pred.index.get_level_values(1)==1)&(y_test==0)])
            pr_false_default_female.loc[i, manip] = sum(y_pred[(y_pred.index.get_level_values(1)==0)&(y_test==0)])/len(y_pred[(y_pred.index.get_level_values(1)==0)&(y_test==0)])
            
            ## INPROCESSING ##
            # Performance metrics
            accuracy_ad.loc[i, manip] = accuracy_score(y_true=y_test, y_pred=y_pred_ad)
            recall_ad.loc[i, manip] = recall_score(y_true=y_test, y_pred=y_pred_ad)
            spd_ad.loc[i, manip] = statistical_parity_difference(y_test, y_pred_ad, prot_attr='MALE', priv_group=0)  
            aod_ad.loc[i, manip] = average_odds_difference(y_test, y_pred_ad, prot_attr='MALE', priv_group=0)
          
            # Intermediate steps to calculate spd
            y_pred_ad = pd.Series(y_pred_ad)
            y_pred_ad.index = y_test.index
            pr_default_male_ad.loc[i, manip] = sum(y_pred_ad[y_pred_ad.index.get_level_values(1)==1])/len(y_pred_ad[y_pred_ad.index.get_level_values(1)==1])
            pr_default_female_ad.loc[i, manip] = sum(y_pred_ad[y_pred_ad.index.get_level_values(1)==0])/len(y_pred_ad[y_pred_ad.index.get_level_values(1)==0])
            
            # Intermediate steps to calculate aod
            pr_true_default_male_ad.loc[i, manip] = sum(y_pred_ad[(y_pred_ad.index.get_level_values(1)==1)&(y_test==1)])/len(y_pred_ad[(y_pred_ad.index.get_level_values(1)==1)&(y_test==1)])
            pr_true_default_female_ad.loc[i, manip] = sum(y_pred_ad[(y_pred_ad.index.get_level_values(1)==0)&(y_test==1)])/len(y_pred_ad[(y_pred_ad.index.get_level_values(1)==0)&(y_test==1)])
            pr_false_default_male_ad.loc[i, manip] = sum(y_pred_ad[(y_pred_ad.index.get_level_values(1)==1)&(y_test==0)])/len(y_pred_ad[(y_pred_ad.index.get_level_values(1)==1)&(y_test==0)])
            pr_false_default_female_ad.loc[i, manip] = sum(y_pred_ad[(y_pred_ad.index.get_level_values(1)==0)&(y_test==0)])/len(y_pred_ad[(y_pred_ad.index.get_level_values(1)==0)&(y_test==0)])
           
            ## COMPARE NO PREPROCESSING WITH PREPROCESSING ##
            # Difference in performance metrics
            diff_accuracy.loc[i, manip] = accuracy.loc[i, manip] - accuracy_ad.loc[i, manip] 
            diff_recall.loc[i, manip] = recall.loc[i, manip] - recall_ad.loc[i, manip]
            diff_spd.loc[i, manip] = spd.loc[i, manip] - spd_ad.loc[i, manip]
            diff_aod.loc[i, manip] = aod.loc[i, manip] - aod_ad.loc[i, manip]
        
        i = i+1
    
    # Combine performance metrics into one dataframe
    metrics_output = pd.concat([np.mean(class_imbalance), np.mean(default_ratio_male), np.mean(default_ratio_female),
                            np.mean(diff_default_ratio), np.mean(accuracy),np.mean(recall),
                            np.mean(accuracy_ad),np.mean(recall_ad),
                            np.mean(diff_accuracy), np.mean(diff_recall),
                            np.mean(spd), np.mean(aod), np.mean(spd_ad), np.mean(aod_ad),
                            np.std(spd), np.std(aod), np.std(spd_ad), np.std(aod_ad)], axis=1)
    metrics_output.columns=['class_imbalance', 'default_ratio_male', 'default_ratio_female', 'diff_default_ratio', 
                            'accuracy', 'recall','accuracy_ad', 'recall_ad', 'diff_accuracy', 'diff_recall',
                            'spd', 'aod', 'spd_ad', 'aod_ad', 'std_spd', 'std_aod', 'std_spd_ad', 'std_aod_ad']
    metrics_output = metrics_output.sort_values(by='diff_default_ratio')
    
    # Combine spd output into one dataframe
    spd_output = pd.concat([np.mean(diff_default_ratio), np.mean(pr_default_male), np.mean(pr_default_female), np.mean(spd), 
                            np.mean(pr_default_male_ad), np.mean(pr_default_female_ad), np.mean(spd_ad), 
                            np.mean(diff_spd)], axis=1)
    spd_output.columns=['diff_default_ratio', 'pr_default_male', 'pr_default_female', 'spd', 'pr_default_male_ad',
                        'pr_default_female_ad', 'spd_ad', 'diff_spd']
    spd_output = spd_output.sort_values(by='diff_default_ratio')
    
    # Combine aod output into one dataframe
    aod_output = pd.concat([np.mean(diff_default_ratio), np.mean(pr_true_default_male), np.mean(pr_true_default_female),
                            np.mean(pr_false_default_male), np.mean(pr_false_default_female), np.mean(aod), 
                            np.mean(pr_true_default_male_ad), np.mean(pr_true_default_female_ad), 
                            np.mean(pr_false_default_male_ad), np.mean(pr_false_default_female_ad), np.mean(aod_ad), 
                            np.mean(diff_aod)], axis=1)
    aod_output.columns=['diff_default_ratio', 'pr_true_default_male', 'pr_true_default_female', 'pr_false_default_male',
                        'pr_false_default_female', 'aod', 'pr_true_default_male_ad', 'pr_true_default_female_ad',
                        'pr_false_default_male_ad', 'pr_false_default_female_ad', 'aod_ad', 'diff_aod']
    aod_output = aod_output.sort_values(by='diff_default_ratio')
   
    return(metrics_output, spd_output, diff_spd, aod_output, diff_aod)


###############################################################################
##### 4. RUNNING SENSITIVITY ANALYSIS   #######################################
###############################################################################
  
# Function to write results to csv
def write_results_to_csv(): 
    metrics_output_lr.to_csv("metrics_output_lr.csv")
    metrics_output_rf.to_csv("metrics_output_rf.csv")
    metrics_output_dt.to_csv("metrics_output_dt.csv")
    metrics_output_xgb.to_csv("metrics_output_xgb.csv")
    
    spd_output_lr.to_csv("spd_output_lr.csv")
    spd_output_rf.to_csv("spd_output_rf.csv")
    spd_output_dt.to_csv("spd_output_dt.csv")
    spd_output_xgb.to_csv("spd_output_xgb.csv")
    
    diff_spd_lr.to_csv("diff_spd_lr.csv")
    diff_spd_rf.to_csv("diff_spd_rf.csv")
    diff_spd_dt.to_csv("diff_spd_dt.csv")
    diff_spd_xgb.to_csv("diff_spd_xgb.csv")
    
    aod_output_lr.to_csv("aod_output_lr.csv")
    aod_output_rf.to_csv("aod_output_rf.csv")
    aod_output_dt.to_csv("aod_output_dt.csv")
    aod_output_xgb.to_csv("aod_output_xgb.csv")
    
    diff_aod_lr.to_csv("diff_aod_lr.csv")
    diff_aod_rf.to_csv("diff_aod_rf.csv")
    diff_aod_dt.to_csv("diff_aod_dt.csv")
    diff_aod_xgb.to_csv("diff_aod_xgb.csv")
    
# Function to write results to csv
def write_results_to_csv_ad(): 
    metrics_output.to_csv("metrics_output.csv")
    spd_output.to_csv("spd_output.csv")
    diff_spd.to_csv("diff_spd.csv")
    aod_output.to_csv("aod_output.csv")
    diff_aod.to_csv("diff_aod.csv")

## REWEIGHING ##   
metrics_output_lr, spd_output_lr, diff_spd_lr, aod_output_lr, diff_aod_lr = sensitivity_analysis_rew(model='lr', label_bias=True)
metrics_output_rf, spd_output_rf, diff_spd_rf, aod_output_rf, diff_aod_rf = sensitivity_analysis_rew(model='rf', label_bias=True)
metrics_output_dt, spd_output_dt, diff_spd_dt, aod_output_dt, diff_aod_dt = sensitivity_analysis_rew(model='dt', label_bias=True)
metrics_output_xgb, spd_output_xgb, diff_spd_xgb, aod_output_xgb, diff_aod_xgb = sensitivity_analysis_rew(model='xgb', label_bias=True)

os.chdir(r"XXX\XXX\Reweighing_training_sample") # If label_bias = False
os.chdir(r"XXX\XXX\Reweighing_training_label") # If label_bias = True
write_results_to_csv()

## MASSAGING ##
metrics_output_lr, spd_output_lr, diff_spd_lr, aod_output_lr, diff_aod_lr = sensitivity_analysis_mas(model='lr', label_bias=True)
metrics_output_rf, spd_output_rf, diff_spd_rf, aod_output_rf, diff_aod_rf = sensitivity_analysis_mas(model='rf', label_bias=True)
metrics_output_dt, spd_output_dt, diff_spd_dt, aod_output_dt, diff_aod_dt = sensitivity_analysis_mas(model='dt', label_bias=True)
metrics_output_xgb, spd_output_xgb, diff_spd_xgb, aod_output_xgb, diff_aod_xgb = sensitivity_analysis_mas(model='xgb', label_bias=True)

os.chdir(r"XXX\XXX\Massaging_training_sample") # If label_bias = False
os.chdir(r"XXX\XXX\Massaging_training_label") # If label_bias = True
write_results_to_csv()

## ADVERSARIAL DEBIASING ##
metrics_output, spd_output, diff_spd, aod_output, diff_aod = sensitivity_analysis_ad(label_bias=True)

os.chdir(r"XXX\XXX\Adversarial_debiasing_training_sample") # If label_bias = False
os.chdir(r"XXX\XXX\Adversarial_debiasing_training_label") # If label_bias = True
write_results_to_csv_ad()

###############################################################################
##### 5. PROCESSING THE RESULTS         #######################################
###############################################################################

##### MASSAGING AND REWEIGHING #####

# Choose whether you want reweighing or massaging results 
# and whether you want the sample or label bias results
os.chdir(r"XXX\XXX\Reweighing_training_sample")
os.chdir(r"XXX\XXX\Reweighing_training_label")
os.chdir(r"XXX\XXX\Massaging_training_sample")
os.chdir(r"XXX\XXX\Massaging_training_label")

# Load metrics results and spd/aod output
metrics_output_lr = pd.read_csv("metrics_output_lr.csv", index_col=0)
metrics_output_rf = pd.read_csv("metrics_output_rf.csv", index_col=0)
metrics_output_dt = pd.read_csv("metrics_output_dt.csv", index_col=0)
metrics_output_xgb = pd.read_csv("metrics_output_xgb.csv", index_col=0)

spd_output_lr = pd.read_csv("spd_output_lr.csv", index_col=0)
spd_output_rf = pd.read_csv("spd_output_rf.csv", index_col=0)
spd_output_dt = pd.read_csv("spd_output_dt.csv", index_col=0)
spd_output_xgb = pd.read_csv("spd_output_xgb.csv", index_col=0)

aod_output_lr = pd.read_csv("aod_output_lr.csv", index_col=0)
aod_output_rf = pd.read_csv("aod_output_rf.csv", index_col=0)
aod_output_dt = pd.read_csv("aod_output_dt.csv", index_col=0)
aod_output_xgb = pd.read_csv("aod_output_xgb.csv", index_col=0)

# Load diff results of spd and aod for both sample and label bias together
# Choose whether you want reweighing or massaging
os.chdir(r"XXX\XXX\Reweighing_training_sample")
os.chdir(r"XXX\XXX\Massaging_training_sample")

diff_spd_lr = pd.read_csv("diff_spd_lr.csv", index_col=0)
diff_spd_rf = pd.read_csv("diff_spd_rf.csv", index_col=0)
diff_spd_dt = pd.read_csv("diff_spd_dt.csv", index_col=0)
diff_spd_xgb = pd.read_csv("diff_spd_xgb.csv", index_col=0)

diff_aod_lr = pd.read_csv("diff_aod_lr.csv", index_col=0)
diff_aod_rf = pd.read_csv("diff_aod_rf.csv", index_col=0)
diff_aod_dt = pd.read_csv("diff_aod_dt.csv", index_col=0)
diff_aod_xgb = pd.read_csv("diff_aod_xgb.csv", index_col=0)

# Choose whether you want reweighing or massaging
os.chdir(r"XXX\XXX\Reweighing_training_label")
os.chdir(r"XXX\XXX\Massaging_training_label")

diff_spd_lr_l = pd.read_csv("diff_spd_lr.csv", index_col=0)
diff_spd_rf_l = pd.read_csv("diff_spd_rf.csv", index_col=0)
diff_spd_dt_l = pd.read_csv("diff_spd_dt.csv", index_col=0)
diff_spd_xgb_l = pd.read_csv("diff_spd_xgb.csv", index_col=0)

diff_aod_lr_l = pd.read_csv("diff_aod_lr.csv", index_col=0)
diff_aod_rf_l = pd.read_csv("diff_aod_rf.csv", index_col=0)
diff_aod_dt_l = pd.read_csv("diff_aod_dt.csv", index_col=0)
diff_aod_xgb_l = pd.read_csv("diff_aod_xgb.csv", index_col=0)

# Reorder columns for label bias manipulations
diff_spd_lr_l = diff_spd_lr_l[['female_no_default_1', 'male_default_0_9', 'orig', 'female_default_0_9', 'male_no_default_1']]
diff_spd_dt_l = diff_spd_dt_l[['female_no_default_1', 'male_default_0_9', 'orig', 'female_default_0_9', 'male_no_default_1']]
diff_spd_rf_l = diff_spd_rf_l[['female_no_default_1', 'male_default_0_9', 'orig', 'female_default_0_9', 'male_no_default_1']]
diff_spd_xgb_l = diff_spd_xgb_l[['female_no_default_1', 'male_default_0_9', 'orig', 'female_default_0_9', 'male_no_default_1']]

# Boxplot for the difference in SPD
diff_spd_lr = diff_spd_lr * 100
diff_spd_dt = diff_spd_dt * 100
diff_spd_rf = diff_spd_rf * 100
diff_spd_xgb = diff_spd_xgb * 100   
diff_spd_lr_l = diff_spd_lr_l * 100
diff_spd_dt_l = diff_spd_dt_l * 100
diff_spd_rf_l = diff_spd_rf_l * 100
diff_spd_xgb_l = diff_spd_xgb_l * 100

diff_spd_lr = pd.concat([diff_spd_lr, diff_spd_lr_l], axis=1)
diff_spd_dt = pd.concat([diff_spd_dt, diff_spd_dt_l], axis=1)
diff_spd_rf = pd.concat([diff_spd_rf, diff_spd_rf_l], axis=1)
diff_spd_xgb = pd.concat([diff_spd_xgb, diff_spd_xgb_l], axis=1)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,4))
bplot1 = ax.boxplot(diff_spd_lr, widths=0.3, positions = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5], patch_artist=True, boxprops=dict(facecolor="C0"))
bplot2 = ax.boxplot(diff_spd_rf, widths=0.3, positions = [0.5, 3, 5.5, 8, 10.5, 13, 15.5, 18, 20.5, 23], patch_artist=True, boxprops=dict(facecolor="C1"))
bplot3 = ax.boxplot(diff_spd_dt, widths=0.3, positions = [1, 3.5, 6, 8.5, 11, 13.5, 16, 18.5, 21, 23.5], patch_artist=True, boxprops=dict(facecolor="C2"))
bplot4 = ax.boxplot(diff_spd_xgb, widths=0.3, positions = [1.5, 4, 6.5, 9, 11.5, 14, 16.5, 19, 21.5, 24], patch_artist=True, boxprops=dict(facecolor="C3"))
ax.axhline(y=0, color='gray',linestyle='--')
ax.set_ylabel('Difference in SPD for XXX (percentage point)') # Fill in reweighing or massaging on XXX
ax.set_xticks([0.75, 3.25, 5.75, 8.25, 10.75, 13.25, 15.75, 18.25, 20.75, 23.25])
ax.set_xticklabels(['Train A', 'Train B', 'Train C', 'Train D', 'Train E', 'Train F', 'Train G', 'Train H', 'Train I', 'Train J'])
ax.legend([bplot1["boxes"][0], bplot2["boxes"][0], bplot3["boxes"][0], bplot4["boxes"][0]], 
          ['Logistic regression', 'Random forest', 'Decision Tree', 'XGBoost'], loc='upper left', ncol=2)


# Reorder columns for label bias manipulations
diff_aod_lr_l = diff_aod_lr_l[['female_no_default_1', 'male_default_0_9', 'orig', 'female_default_0_9', 'male_no_default_1']]
diff_aod_dt_l = diff_aod_dt_l[['female_no_default_1', 'male_default_0_9', 'orig', 'female_default_0_9', 'male_no_default_1']]
diff_aod_rf_l = diff_aod_rf_l[['female_no_default_1', 'male_default_0_9', 'orig', 'female_default_0_9', 'male_no_default_1']]
diff_aod_xgb_l = diff_aod_xgb_l[['female_no_default_1', 'male_default_0_9', 'orig', 'female_default_0_9', 'male_no_default_1']]

# Boxplot for the difference in aod
diff_aod_lr = diff_aod_lr * 100
diff_aod_dt = diff_aod_dt * 100
diff_aod_rf = diff_aod_rf * 100
diff_aod_xgb = diff_aod_xgb * 100   
diff_aod_lr_l = diff_aod_lr_l * 100
diff_aod_dt_l = diff_aod_dt_l * 100
diff_aod_rf_l = diff_aod_rf_l * 100
diff_aod_xgb_l = diff_aod_xgb_l * 100

diff_aod_lr = pd.concat([diff_aod_lr, diff_aod_lr_l], axis=1)
diff_aod_dt = pd.concat([diff_aod_dt, diff_aod_dt_l], axis=1)
diff_aod_rf = pd.concat([diff_aod_rf, diff_aod_rf_l], axis=1)
diff_aod_xgb = pd.concat([diff_aod_xgb, diff_aod_xgb_l], axis=1)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,4))
bplot1 = ax.boxplot(diff_aod_lr, widths=0.3, positions = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5], patch_artist=True, boxprops=dict(facecolor="C0"))
bplot2 = ax.boxplot(diff_aod_rf, widths=0.3, positions = [0.5, 3, 5.5, 8, 10.5, 13, 15.5, 18, 20.5, 23], patch_artist=True, boxprops=dict(facecolor="C1"))
bplot3 = ax.boxplot(diff_aod_dt, widths=0.3, positions = [1, 3.5, 6, 8.5, 11, 13.5, 16, 18.5, 21, 23.5], patch_artist=True, boxprops=dict(facecolor="C2"))
bplot4 = ax.boxplot(diff_aod_xgb, widths=0.3, positions = [1.5, 4, 6.5, 9, 11.5, 14, 16.5, 19, 21.5, 24], patch_artist=True, boxprops=dict(facecolor="C3"))
ax.axhline(y=0, color='gray',linestyle='--')
ax.set_ylabel('Difference in AOD for XXX (percentage point)') # Fill in reweighing or massaging on XXX
ax.set_xticks([0.75, 3.25, 5.75, 8.25, 10.75, 13.25, 15.75, 18.25, 20.75, 23.25])
ax.set_xticklabels(['Train A', 'Train B', 'Train C', 'Train D', 'Train E', 'Train F', 'Train G', 'Train H', 'Train I', 'Train J'])
ax.legend([bplot1["boxes"][0], bplot2["boxes"][0], bplot3["boxes"][0], bplot4["boxes"][0]], 
          ['Logistic regression', 'Random forest', 'Decision Tree', 'XGBoost'], loc='upper left', ncol=2)


### ADVERSARIAL DEBIASING ###

# Choose whether you want the sample or label bias results
os.chdir(r"XXX\XXX\Adversarial_debiasing_training_sample")
os.chdir(r"XXX\XXX\Adversarial_debiasing_training_label")
# Load metrics results and spd/aod output
metrics_output = pd.read_csv("metrics_output.csv", index_col=0)
spd_output = pd.read_csv("spd_output.csv", index_col=0)
aod_output = pd.read_csv("aod_output.csv", index_col=0)

# Load diff results of spd and aod for both sample and label bias together
os.chdir(r"XXX\XXX\Adversarial_debiasing_training_sample")
diff_spd = pd.read_csv("diff_spd.csv", index_col=0)
diff_aod = pd.read_csv("diff_aod.csv", index_col=0)

os.chdir(r"XXX\XXX\Adversarial_debiasing_training_label")
diff_spd_l = pd.read_csv("diff_spd.csv", index_col=0)
diff_aod_l = pd.read_csv("diff_aod.csv", index_col=0)

# Reorder columns for label bias manipulations
diff_spd_l = diff_spd_l[['female_no_default_1', 'male_default_0_9', 'orig', 'female_default_0_9', 'male_no_default_1']]
diff_aod_l = diff_aod_l[['female_no_default_1', 'male_default_0_9', 'orig', 'female_default_0_9', 'male_no_default_1']]

diff_spd = pd.concat([diff_spd, diff_spd_l], axis=1)
diff_aod = pd.concat([diff_aod, diff_aod_l], axis=1)

# Boxplot for the difference in SPD and AOD
diff_spd = diff_spd * 100
diff_aod = diff_aod * 100

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,4))
bplot1 = ax[0].boxplot(diff_spd, positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], patch_artist=True, boxprops=dict(facecolor="C0"))
ax[0].axhline(y=0, color='gray',linestyle='--')
ax[0].set_ylabel('Difference in SPD for adv. deb. (percentage point)')
ax[0].set_xticklabels(['Train A', 'Train B', 'Train C', 'Train D', 'Train E', 'Train F', 'Train G', 'Train H', 'Train I', 'Train J'])

bplot2 = ax[1].boxplot(diff_aod, positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], patch_artist=True, boxprops=dict(facecolor="C1"))
ax[1].axhline(y=0, color='gray',linestyle='--')
ax[1].set_ylabel('Difference in AOD for adv. deb. (percentage point)')
ax[1].set_xticklabels(['Train A', 'Train B', 'Train C', 'Train D', 'Train E', 'Train F', 'Train G', 'Train H', 'Train I', 'Train J'])

###############################################################################
##### 0. IMPORT PACAKGES AND LOAD DATA  #######################################
###############################################################################

# Import packages
import pandas as pd
import numpy as np
import os

# Load data
df_2001 = pd.read_csv(r"XXX\XXX\census_2001.csv")
df_2011 = pd.read_csv(r"XXX\XXX\census_2011.csv")

###############################################################################
##### 1. DATA INITIALIZATION            #######################################
###############################################################################

#### 2001 and 2011 #####

# Drop columns we do not need in our analyses
df_2001 = df_2001.drop(columns=['COUNTRY', 'YEAR', 'SAMPLE', 'SERIAL', 'HHWT', 'NL2001A_DWNUM', 'PERNUM', 'PERWT', 'NL2001A_PERNUM', 'NL2001A_WEIGHT'])
df_2011 = df_2011.drop(columns=['COUNTRY', 'YEAR', 'SAMPLE', 'SERIAL', 'HHWT', 'NL2011A_DWNUM', 'PERNUM', 'PERWT', 'NL2011A_PERN', 'NL2011A_PERNUM', 'NL2011A_WEIGHT'])

# Rename columns to have the same names
df_2001.rename(columns={'NL2001A_SEX':'SEX',
                        'NL2001A_AGE':'AGE',
                        'NL2001A_RELATE':'RELATE',
                        'NL2001A_HHSIZE':'HHSIZE',
                        'NL2001A_RESPREV':'RESPREV',
                        'NL2001A_CITZ':'CITIZ',
                        'NL2001A_BPL':'BPL',
                        'NL2001A_EDUC': 'EDUC',
                        'NL2001A_CLASSWK':'CLASSWK',
                        'NL2001A_OCC': 'OCC',
                        'NL2001A_IND': 'IND',
                        'NL2001A_MARST':'MARST',}, 
                           inplace=True)
df_2011.rename(columns={'NL2011A_HHSIZE':'HHSIZE',
                        'NL2011A_SEX':'SEX',
                        'NL2011A_AGE':'AGE',
                        'NL2011A_RELATE':'RELATE',
                        'NL2011A_RES1YR':'RESPREV',
                        'NL2011A_CITIZEN':'CITIZ',
                        'NL2011A_BPLCNTRY':'BPL',
                        'NL2011A_EDATTAIN': 'EDUC',
                        'NL2011A_EMPSTAT':'CLASSWK',
                        'NL2011A_OCC': 'OCC',
                        'NL2011A_IND': 'IND',
                        'NL2011A_MARST':'MARST',}, 
                           inplace=True)

# Drop people underaged
df_2001 = df_2001[df_2001['AGE']>4]
df_2011 = df_2011[df_2011['AGE']>4]

# Drop missing values
df_2001 = df_2001[(df_2001['AGE']!=99) & (df_2001['RELATE']!=9) & (df_2001['CITIZ']!=9) & (df_2001['EDUC']!=9) & 
                  (df_2001['CLASSWK']!=9) & (df_2001['OCC']!=98) & (df_2001['IND']!=98) & (df_2001['MARST']!=9)]
df_2011 = df_2011[(df_2011['AGE']!=98) & (df_2011['RELATE']!=8) & (df_2011['RESPREV']!=8) & (df_2011['CITIZ']!=8) & 
                  (df_2011['BPL']!=8) & (df_2011['EDUC']!=8) & 
                  (df_2011['CLASSWK']!=8) & (df_2011['OCC']!=98) & (df_2011['IND']!=98) & (df_2011['MARST']!=8)]


# Create dependent variable; whether an individual as a high profession or not
df_2001['HIGH_PROF'] = np.where((df_2001['OCC']==1) | (df_2001['OCC']==2), 1, 0)
df_2011['HIGH_PROF'] = np.where((df_2011['OCC']==1) | (df_2011['OCC']==2), 1, 0)

# Equalize the coding of 2001 and 2011
df_2001.loc[(df_2001['RELATE']==1), 'RELATE_NEW'] = 1
df_2001.loc[(df_2001['RELATE']==2) | (df_2001['RELATE']==3), 'RELATE_NEW'] = 2
df_2001.loc[(df_2001['RELATE']==4) | (df_2001['RELATE']==5), 'RELATE_NEW'] = 3
df_2001.loc[(df_2001['RELATE']==6), 'RELATE_NEW'] = 4
df_2001.loc[(df_2001['RELATE']==7), 'RELATE_NEW'] = 5
df_2001.loc[(df_2001['RELATE']==8), 'RELATE_NEW'] = 6
df_2001 = df_2001.drop(columns='RELATE')
df_2001.rename(columns={'RELATE_NEW':'RELATE'}, inplace=True)

df_2001.loc[(df_2001['EDUC']==0) | (df_2001['EDUC']==6), 'EDUC_NEW'] = 0
df_2001.loc[(df_2001['EDUC']==1), 'EDUC_NEW'] = 1
df_2001.loc[(df_2001['EDUC']==2), 'EDUC_NEW'] = 2
df_2001.loc[(df_2001['EDUC']==3), 'EDUC_NEW'] = 3
df_2001.loc[(df_2001['EDUC']==4), 'EDUC_NEW'] = 4
df_2001.loc[(df_2001['EDUC']==5), 'EDUC_NEW'] = 5
df_2001 = df_2001.drop(columns='EDUC')
df_2001.rename(columns={'EDUC_NEW':'EDUC'}, inplace=True)

df_2011.loc[(df_2011['EDUC']==5) | (df_2011['EDUC']==6), 'EDUC_NEW'] = 5
df_2011.loc[(df_2011['EDUC']==0), 'EDUC_NEW'] = 0
df_2011.loc[(df_2011['EDUC']==1), 'EDUC_NEW'] = 1
df_2011.loc[(df_2011['EDUC']==2), 'EDUC_NEW'] = 2
df_2011.loc[(df_2011['EDUC']==3), 'EDUC_NEW'] = 3
df_2011.loc[(df_2011['EDUC']==4), 'EDUC_NEW'] = 4
df_2011 = df_2011.drop(columns='EDUC')
df_2011.rename(columns={'EDUC_NEW':'EDUC'}, inplace=True)

# Other manipulations
df_2001['SEX'] = df_2001['SEX'].replace(2,0) 
df_2011['SEX'] = df_2011['SEX'].replace(2,0) 

# Transform categorical variables into dummy variable
cat_vars = ['CITIZ', 'BPL', 'IND', 'EDUC', 'MARST'] 

# 2001
for var in cat_vars:
    cat_list = 'var'+'_'+var
    cat_list = pd.get_dummies(df_2001[var], prefix=var)
    df1 = df_2001.join(cat_list)
    df_2001=df1
    
# Drop the original variables
df_2001 = df_2001.drop(cat_vars, axis=1)

# Drop one category of each variable to prevent multicollinearity
df_2001 = df_2001.drop('CITIZ_3', axis=1)
df_2001 = df_2001.drop('BPL_3', axis=1)
df_2001 = df_2001.drop('IND_99', axis=1)
df_2001 = df_2001.drop('EDUC_5.0', axis=1)
df_2001 = df_2001.drop('MARST_4', axis=1)

# 2011
for var in cat_vars:
    cat_list = 'var'+'_'+var
    cat_list = pd.get_dummies(df_2011[var], prefix=var)
    df1 = df_2011.join(cat_list)
    df_2011=df1
    
# Drop the original variables
df_2011 = df_2011.drop(cat_vars, axis=1)

# Drop one category of each variable to prevent multicollinearity
df_2011 = df_2011.drop('CITIZ_3', axis=1)
df_2011 = df_2011.drop('BPL_3', axis=1)
df_2011 = df_2011.drop('IND_99', axis=1)
df_2011 = df_2011.drop('EDUC_5.0', axis=1)
df_2011 = df_2011.drop('MARST_4', axis=1)


# Drop other columns we not need in the X and y sets
df_2001 = df_2001.drop(columns=['CLASSWK', 'OCC', 'RELATE'])
df_2011 = df_2011.drop(columns=['CLASSWK', 'OCC', 'RELATE'])


# Write newly created data frames to csv files
os.chdir(r"XXX\XXX")

df_2001.to_csv("df_2001.csv")
df_2011.to_csv("df_2011.csv")








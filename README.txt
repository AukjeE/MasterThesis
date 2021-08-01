STRUCTURE OF CODE

1. sensitivity analysis test.py
Input: Taiwan Default data (default.xls)
Goal: Perform sensitivity analyses for differentiation in test sets Taiwan Default data
Output: Results in Section 6.1

2. senstivity analysis training.py
Input: Taiwan Default data (default.xls)
Goal: Perform sensitivity analyses for differentation in training sets Taiwan Default data
Output: Results in Section 6.2

3. create data sets census.py
Input: Dutch Census data sets of 2001 and 2011 (census_2001.csv and census_2011.csv)
Goal: Create data sets suitable voor analyses census data: filter, drop missing values, match coding, etc.
Output: df_2001.csv and df_2011.csv

4. analysis census.py
Input: Dutch Census data of 2001 and 2011 suitable for analyses (df_2001.csv and df_2011.csv)
Goal: Perform analyses for Dutch census data 
Output: Results in Section 6.3

5. relabelling.py
Goal: Small changes made to the code of the themis-ml relabelling code in order to have it work for exceptional cases 

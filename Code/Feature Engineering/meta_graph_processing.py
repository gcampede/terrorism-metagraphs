# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:10:46 2020

@author: Gian Maria
"""


import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt
import pandas as pd
import math
import patsy
import seaborn as sns
import os
from scipy.stats import kde
from functools import reduce
import networkx as nx


''' Import necessary .csv files, i.e. those created with gtd_preprocess.py

and use the functions formalized in meta_graph_functions.py for obtaining

multivariate graph-derived time series for both datasets '''


os.getcwd()


os.chdir('C:\\Users\\Gian Maria\\Desktop\\AAAI_21\\datasets_for_submission')
afg_unique_complete = pd.read_csv('afg_unique_complete.csv', delimiter=',', index_col=0)
ira_unique_complete = pd.read_csv('ira_unique_complete.csv', delimiter=',', index_col=0)



''' AFGHANISTAN DATASET: Temporal Meta-Graph Extraction and 
Derivation of Graph-Derived Time Series '''


# select theoretical dimensions

tactics = afg_unique_complete.iloc[:,np.r_[0:9]]
weapons = afg_unique_complete.iloc[:,np.r_[10:17]]
targets = afg_unique_complete.iloc[:,np.r_[18:39]]

# split in units such that u=2t
split_tactics = split(tactics, 2)
split_weapons = split(weapons, 2)
split_targets = split(targets, 2)

# obtain square matrix
split_tactics_sq=list(map(sq_mul, split_tactics))
split_weapons_sq=list(map(sq_mul, split_weapons))
split_targets_sq = list(map(sq_mul, split_targets))




# obtain centralities over time by using normalize and normalized_cen functions

    
list_centralities_tactics=list(map(normalized_cen, split_tactics_sq))  
list_centralities_weapons=list(map(normalized_cen, split_weapons_sq))  
list_centralities_targets=list(map(normalized_cen, split_targets_sq))


# obtain first mode of temporal centralities


centrality_db_tactics = pd.DataFrame(list_centralities_tactics)
centrality_db_weapons = pd.DataFrame(list_centralities_weapons)
centrality_db_targets = pd.DataFrame(list_centralities_targets)




# create and finalize dataset

afg_modes = [centrality_db_tactics, centrality_db_weapons, centrality_db_targets]
afg_graph_derived = pd.concat(afg_modes, axis=1)
afg_graph_derived.fillna(0, inplace=True)

afg_graph_derived.to_csv('afghanistan_time_series01.csv', header=True)




''' IRAQ DATASET: Temporal Meta-Graph Extraction and 
Derivation of Graph-Derived Time Series '''



# select theoretical dimensions

tactics = ira_unique_complete.iloc[:,np.r_[0:9]]
weapons = ira_unique_complete.iloc[:,np.r_[10:17]]
targets = ira_unique_complete.iloc[:,np.r_[18:39]]



# split in units such that u=2t
split_tactics = split(tactics, 2)
split_weapons = split(weapons, 2)
split_targets = split(targets, 2)

# obtain square matrix
split_tactics_sq=list(map(sq_mul, split_tactics))
split_weapons_sq=list(map(sq_mul, split_weapons))
split_targets_sq = list(map(sq_mul, split_targets))



# obtain centralities over time by using normalize and normalized_cen functions

    
list_centralities_tactics=list(map(normalized_cen, split_tactics_sq))  
list_centralities_weapons=list(map(normalized_cen, split_weapons_sq))  
list_centralities_targets=list(map(normalized_cen, split_targets_sq))


# obtain of temporal centralities


centrality_db_tactics = pd.DataFrame(list_centralities_tactics)
centrality_db_weapons = pd.DataFrame(list_centralities_weapons)
centrality_db_targets = pd.DataFrame(list_centralities_targets)




# create and finalize dataset

ira_modes = [centrality_db_tactics, centrality_db_weapons, centrality_db_targets]
ira_graph_derived = pd.concat(ira_modes, axis=1)
ira_graph_derived.fillna(0, inplace=True)

ira_graph_derived.to_csv('iraq_time_series01.csv', header=True)




# convert to np array to count zeros
afg_graph_array = afg_graph_derived.to_numpy()
ira_graph_array = ira_graph_derived.to_numpy()


np.count_nonzero(afg_graph_array==0)
np.count_nonzero(ira_graph_array==0)
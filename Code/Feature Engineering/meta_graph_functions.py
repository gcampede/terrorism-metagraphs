# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:53:53 2020

@author: Gian Maria
"""



import numpy as np
import pandas as pd
import math
import patsy
import os
from functools import reduce
import networkx as nx

''' This script contains the necessary functions 

to transform a dataset in the form (days x features) into
 
a dataset containing multivariate time-series mapping the

centrality value of each feature in its dimension (e.g. Firearms

in Weapons) in two-day based units'''




### Functions




'''This function first divides a given dataset into two-day slices (u=2t). 

The chunkSize argument can be modified based on the use case: in this 

experiments we have set a chunkSize = 2. Default is 3'''


def split(df, chunkSize = 3):
    
    numberChunks = len(df) // chunkSize + 1
    
    return np.array_split(df, numberChunks, axis=0)




''' This function multiplies the transpose of the two-mode matrix (2t x features)

by the original matrix, obtaining a new square matrix (features x features)'''
    


def sq_mul(df):
    
    df_trans=df.T
    
    sqr_df=df_trans.dot(df)
    
    return(sqr_df)




    
''' This function takes as input the centrality values stored in 

a dictionary to normalize them by dividing each value in each u 

by the max value in that same dictionary, so that the highest 

centrality will become equal to 1 and all the others will be bounded in the

[0,0.999] interval '''


def normalize(dict):
    
    m=max(dict.values()) if dict.values() else 0
    
    dict.update({n:dict[n]/max(dict.values()) for n in dict.keys()})
    
    return(dict)


''' Finally, this function uses networkx to calculate centrality values of each 

temporal meta-graph, and exploits the normalize(dict) function to compute

normalized values'''
    

def normalized_cen(df):
    
    g=nx.Graph(df)
    edge_g=nx.to_edgelist(g)
    GN= nx.from_edgelist(edge_g)
    #GN.remove_edges_from(nx.selfloop_edges(GN))
    
    degree_centrality = GN.degree(weight='weight')
    degree_access = dict(GN.degree(weight='weight'))
    norm_degree_access=normalize(degree_access)
    
    return norm_degree_access


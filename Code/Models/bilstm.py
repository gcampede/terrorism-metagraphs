# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:35:32 2020

@author: Gian Maria
"""

import numpy as np
import tensorflow as tf
import random
import warnings
import os
from support_functions import prepare_data, create_lookback, model_compile, model_eval 

warnings.filterwarnings("ignore")

# random seed set for reproducability
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(5)
random.seed(1254)
tf.random.set_seed(5)


''' import dataset: the script is based on the Afghanistan dataset,
however, one can perform the models using the Iraq one. 
When using the Iraq dataset, target_column_start 
should be as well: 'Private Citizens & Property' '''

path = 'afghanistan_time_series01.csv' # alternatively: 'iraq_time_series01.csv'
target_column_start =  'Private Citizens & Property'


df,target_columns, target_indices,train_df,val_df,test_df, num_features = prepare_data(path,target_column_start)

# define the lookback. This is set to 15, but the experiments in the project  have also used 1, 5, and 30
lookback_frame = create_lookback(15,train_df,val_df,test_df,target_columns)

# define the deep learning model 
bilstm = tf.keras.models.Sequential([
    # lstm layer
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.5)),
    # output layer
    tf.keras.layers.Dense(units=num_features)
])

# build and evaluate model
bilstm, _ = model_compile(bilstm, lookback_frame)
model_eval(bilstm,lookback_frame)
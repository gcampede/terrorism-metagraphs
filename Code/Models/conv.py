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

# define the deep learning model (conv_width has to be modified in order to be matched with lookback)
conv_width = 15
conv = tf.keras.Sequential([
    # convolutional layer
    tf.keras.layers.Conv1D(filters=32, kernel_size=(conv_width,), activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=num_features),
])

# build and evaluate model
conv, _ = model_compile(conv, lookback_frame)
model_eval(conv,lookback_frame)
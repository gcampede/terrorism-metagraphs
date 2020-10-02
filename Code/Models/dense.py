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
dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=num_features),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])

# build and evaluate model
dense, _ = model_compile(dense, lookback_frame)
model_eval(dense,lookback_frame)
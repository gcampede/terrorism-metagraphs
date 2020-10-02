import numpy as np
import tensorflow as tf
import random
from support_functions import model_eval, create_lookback, prepare_data, Baseline
import warnings
import os
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


# define the lookback 
loockback_frame = create_lookback(1,train_df,val_df,test_df,target_columns)


baseline = Baseline(target_indices,len(df.columns))
model_eval(baseline,loockback_frame)
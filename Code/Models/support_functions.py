import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping





def prepare_data(path,target_column_start,remove_threshold = 10):

    



    df = pd.read_csv(path, index_col=0)

    

    # delete all feature columns that have less than remove_threshold number of entries

    for col in df.columns:

        if df[col].astype(bool).sum() < remove_threshold:

            df.drop(col,inplace=True,axis=1)

            

    # get all target columns

    target_columns = list(df.columns[list(df.columns).index(target_column_start):])

            

    # get column indices

    col_indices = {col: i for i, col in enumerate(df.columns)}

    target_indices = [col_indices[i] for i in target_columns] 

    

    # split data into train, validation and test

    data_len = len(df)

    train_data = df[0:int(data_len*0.7)]

    val_data = df[int(data_len*0.7):int(data_len*0.9)]

    test_data = df[int(data_len*0.9):]

    num_features = len(target_columns)

    

    return df,target_columns, target_indices,train_data,val_data,test_data, num_features





class create_lookback():

    

    # create lookback class that gives the deep learning model size_input number of elements to predict next eleement from

    

    def __init__(self, size_input,train_data, val_data, test_data,cols=None):



        # initialization of all variables

        self.train_data = train_data

        self.val_data = val_data

        self.test_data = test_data

        self.cols = cols

        

        if cols is not None:

          self.cols_indices = {name: i for i, name in enumerate(cols)}

        self.column_indices = {name: i for i, name in enumerate(train_data.columns)}

        

        self.size_input = size_input

        self.size_lookback = size_input + 1

        self.input_slice = slice(0, size_input)

        self.input_idx = np.arange(self.size_lookback)[self.input_slice]

        self.label_start = self.size_lookback - 1

        self.labels_slice = slice(self.label_start, None)

        self.label_idx = np.arange(self.size_lookback)[self.labels_slice]





    def split(self, features):

        

        # split and reshape dataset  to correct inputs and labels for deep learnin model

        

        inputs = features[:, self.input_slice, :]

        labels = features[:, self.labels_slice, :]

        

        if self.cols is not None:

            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.cols],axis=-1)

    

        inputs.set_shape([None, self.size_input, None])

        labels.set_shape([None, 1, None])



        return inputs, labels







    def create_keras_data(self, data):

        

        # create final data with size_lookback elements, stride movement of 1 and batch_size 16 

        

        data = np.array(data, dtype=np.float32)

        keras_data = tf.keras.preprocessing.timeseries_dataset_from_array( data=data, targets=None, sequence_length=self.size_lookback,

                sequence_stride=1, shuffle=True, batch_size=16,)

        keras_data = keras_data.map(self.split)

        return keras_data



    

    @property

    def train(self):

        return self.create_keras_data(self.train_data)

    

    @property

    def val(self):

        return self.create_keras_data(self.val_data)

    

    @property

    def test(self):

        return self.create_keras_data(self.test_data)







class Baseline(tf.keras.Model):

    #create baseline class that returns back values from previous steps

  def __init__(self, label_idx= [], label_max_count = 9999):

    super().__init__()

    self.label_idx = label_idx

    self.label_max_count = label_max_count



  

  def call(self, inputs):

    if len(self.label_idx) == 0:

      return inputs

    if len(self.label_idx) > 0 and len(self.label_idx)<self.label_max_count:

        result = inputs[:,:, self.label_idx[0]-1:self.label_idx[-1]]

        return result

    result = inputs[:, :, self.label_idx]

    return result[:, :, tf.newaxis]







def model_compile(model, window, patience=10,max_epochs=100):

    # define early stopping

    stop_fcn = EarlyStopping(monitor='val_loss', patience=patience, mode='min')

    # compile deep learning model 

    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics= [tf.metrics.MeanSquaredError(), tf.metrics.MeanAbsoluteError()])

    # train deep learning model with predefined max epoch, and stopping function 

    out = model.fit(window.train, epochs = max_epochs, validation_data=window.val, callbacks= stop_fcn)

    return model,out





def accuracy_fcn(Y_predicted, Y_groundtruth, threshold = 0.1):



    # threshold  determines which values are considered small enough to be  considered as "not attacked" targets

    # accuracy function for elementwise and setwise accuracy for targer prediction



    cnt = 0

    nonzero_cnt = 0

    cnt_setwise = 0

    nonzero_setwise = 0



    Y_predicted = [[0 if i < threshold else i for i in j] for j in Y_predicted]



    for i in range(len(Y_predicted)):



        Y_pred_curr = Y_predicted[i]

        Y_gt_curr = Y_groundtruth[i]

        

        correct_set = {}

        

        # logic to account for no attack days and to handle target attacked days

                

        if np.count_nonzero(Y_gt_curr) == 0:

            if np.count_nonzero(Y_pred_curr)==0:

                correct_set = [1,1]

                idx_predicted = [None,None]

                idx_groundtruth = [None,None]

            

        elif np.count_nonzero(Y_gt_curr) == 1:

            idx_predicted = np.array(Y_pred_curr).argsort()[-1:][::-1]

            idx_groundtruth = np.array(Y_gt_curr).argsort()[-1:][::-1]

            if np.count_nonzero(Y_pred_curr) == 1:

                idx_predicted = [list(idx_predicted)[0],99]

                idx_groundtruth = [list(idx_groundtruth)[0],99]

            if np.count_nonzero(Y_pred_curr) == 0:

                idx_predicted = [99,99]

                idx_groundtruth = [list(idx_groundtruth)[0],99]

            else:

                idx_predicted = np.array(Y_pred_curr).argsort()[-2:][::-1]

                idx_groundtruth = [list(idx_groundtruth)[0],99]

            correct_set = set(idx_groundtruth).intersection(set(idx_predicted))



        # otherwise just compare top 2 entries 

            

        else:

            idx_predicted = np.array(Y_pred_curr).argsort()[-2:][::-1]

            idx_groundtruth = np.array(Y_gt_curr).argsort()[-2:][::-1]



            correct_set = set(idx_groundtruth).intersection(set(idx_predicted))

            

        if len(correct_set) >= 1:



            cnt_setwise += 1;



        nonzero_setwise += 1;

        cnt += len(correct_set)

        nonzero_cnt += 2



    print('| setwise accuracy = %f' % (cnt / nonzero_cnt))

    print('| eventwise accuracy = %f' % (cnt_setwise / nonzero_setwise))

    return float(cnt / nonzero_cnt),float(cnt_setwise / nonzero_setwise)





def model_eval(model,frame):

    

    # evalation over test data for elementiwse and setwise accuracy 

    predicted_vals = []

    groundtruth_vals = []

    for test_X,test_Y in frame.test:

        out = model(test_X).numpy()

        for i in range(test_Y.shape[0]):

            predicted_vals.append(out[i][0])

            groundtruth_vals.append(test_Y[i][0])

    accuracy_fcn(predicted_vals,groundtruth_vals)

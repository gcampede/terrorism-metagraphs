# terrorism-metagraphs
This repository contains the data and code for replicating the analyses presented in the paper "Learning Future Terrorist Targets Trhough Temporal Meta-Graphs" authored with Mihovil Bartulovic and Kathleen M. Carley (School of Computer Science - Carnegie Mellon University).


The repo contains two folders: the (1) Datasets and the (2) Code one. 

The (1) Dataset folder contains all the datasets that are necessary to replicate the analyses. 

The ***datasets*** in this folder are: 

- "gtd7018.csv": this is the original GTD dataset as downloaded from the START dedicated website (https://www.start.umd.edu/gtd/). The GTD file user has to comply with the end user agreement specified here: https://www.start.umd.edu/gtd/end-user-agreement/. No public reproduction without permission and it is here attached solely to facilitate the potential replication of the experiments. 


- "afg_unique_complete.csv": this is the Afghanistan dedicated dataset after the preprocessing operations performed in "gtd_preprocess.py". It contains dates (days) as observations and the count of occurrences of each feature in each day, given the attacks reported.

- "ira_unique_complete.csv": this is the Iraq dedicated dataset after the preprocessing operations performed in "gtd_preprocess.py". It contains dates (days) as observations and the count of occurrences of each feature in each day, given the attacks reported.

- "afghanistan_time_series01.csv": this is the final Afghanistan dataset containing the multivariate time series mapping the centrality of each feature in each theoretical dimension. It is the result of the "meta_graph_processing.py" code. 

- "iraq_time_series01.csv": this is the final Iraq dataset containing the multivariate time series mapping the centrality of each feature in each theoretical dimension. It is the result of the "meta_graph_processing.py" code.


The ***Python scripts*** in the (2) "Code" folder are divided into two subfolders: 
(A) 'Feature Engineering' and (B) 'Models'. All the deep learning models have been performed using TensorFlow 2.5.

In the (A) Feature Engineering folder are stored all the necessary scripts to produce meta-graphs and graph-derived time series from the original GTD dataset. these are:

- "gtd_preprocess.py": the code in this file takes the original gtd .csv file and perform pre-processing operations (e.g., missing date handling, exclusion of doubtful events) and generate two dedicated datasets for Iraq and Afghanistan ("afg_unique_complete.csv" and "ira_unique_complete.csv").

- "meta_graph_functions.py": this .py file contains the basic functions that are used to process "afg_unique_complete.csv" and "ira_unique_complete.csv" in order to extrapolate, for each country, the multivariate time series mapping graph-derived feature centralities. 

- "meta_graph_processing.py": this file performs the core operations to obtain the graph-derived time series, taking as input both "afg_unique_complete.csv" and "ira_unique_complete.csv" and exploiting the basic functions created in "meta_graph_functions.py".

In the (B) 'Models' subfolder are stored all the necessary scripts to run the modeling experiments presented in the paper. These scripts are: 

- "support_functions.py": this script contains all the fundamental functions to process the two time series datasets ("afghanistan_time_series01.csv" and "iraq_time_series01.csv") in order to run each DL model. The file also contains the code for generating element-wise and set-wise accuracy functions. 

-"baseline.py": code for running the baseline model. It makes use of the functions contained in "support_functions.py". The provided code sets "afghanistan_time_series01.csv" as the default dataset, and 10 as input_width/lookback. 

-"dense.py": code for running the baseline (BASE) model. It makes use of the functions contained in "support_functions.py". The provided code sets"afghanistan_time_series01.csv" as the default dataset, and 10 as input_width/lookback. 

-"lstm.py": code for running the LSTM model. It makes use of the functions contained in "support_functions.py". The provided code sets"afghanistan_time_series01.csv" as the default dataset, and 10 as input_width/lookback. 

- "bilstm.py": code for running the Bi-LSTM model. It makes use of the functions contained in "support_functions.py". The provided code sets"afghanistan_time_series01.csv" as the default dataset, and 10 as input_width/lookback. 

- "conv.py": code for running the CNN model. It makes use of the functions contained in "support_functions.py". The provided code sets"afghanistan_time_series01.csv" as the default dataset, and 10 as input_width/lookback. 

- "conv_lstm.py": code for running the CLDNN/CNN-LSTM model. It makes use of the functions contained in "support_functions.py". The provided code sets"afghanistan_time_series01.csv" as the default dataset, and 10 as input_width/lookback. 


import pandas as pd


''' Import necessary .csv files, i.e. those created with gtd_preprocess.py

and use create a shallow version of the dataset in which targets are still calculated as centrality 

values of meta-graphs, but weapons and tactics are instead the simple sum of each feature

over two day time units '''




afg_unique_complete = pd.read_csv('afg_unique_complete.csv', delimiter=',', index_col=0)
ira_unique_complete = pd.read_csv('ira_unique_complete.csv', delimiter=',', index_col=0)


# reset and drop index

afg_unique_complete.reset_index(drop=True, inplace=True)
ira_unique_complete.reset_index(drop=True, inplace=True)


afg_shallow = afg_unique_complete.groupby(afg_unique_complete.index // 2).sum()
ira_shallow = ira_unique_complete.groupby(ira_unique_complete.index // 2).sum()




# Import entirely graph-derived datasets for further processing (substitute tactics and weapons as centralities with their counts

ira_graph_derived = pd.read_csv('iraq_time_series01.csv', index_col=0)
afg_graph_derived = pd.read_csv('afghanistan_time_series01.csv', index_col=0)


''' AFGHANISTAN '''
# drop count target columns
afg_shallow.drop(['Airports & Aircraft', 'Business', 'Educational Institution',
       'Food or Water Supply', 'Government (Diplomatic)',
       'Government (General)', 'Journalists & Media', 'Maritime', 'Military',
       'NGO', 'Other.1', 'Police', 'Private Citizens & Property',
       'Religious Figures/Institutions', 'Telecommunication',
       'Terrorists/Non-State Militia', 'Tourists', 'Transportation',
       'Unknown.2', 'Utilities', 'Violent Political Party'], axis=1, inplace=True)



# merge columns from the centrality-based dataset
afg_shallow_hy = pd.merge(afg_shallow, afg_graph_derived[['Airports & Aircraft', 'Business', 'Educational Institution',
       'Food or Water Supply', 'Government (Diplomatic)',
       'Government (General)', 'Journalists & Media', 'Maritime', 'Military',
       'NGO', 'Other.1', 'Police', 'Private Citizens & Property',
       'Religious Figures/Institutions', 'Telecommunication',
       'Terrorists/Non-State Militia', 'Tourists', 'Transportation',
       'Unknown.2', 'Utilities', 'Violent Political Party']], left_index=True, right_index=True)


''' IRAQ ''' 
# drop count target columns
ira_shallow.drop(['Airports & Aircraft', 'Business', 'Educational Institution',
       'Food or Water Supply', 'Government (Diplomatic)',
       'Government (General)', 'Journalists & Media', 'Maritime', 'Military',
       'NGO', 'Other.1', 'Police', 'Private Citizens & Property',
       'Religious Figures/Institutions', 'Telecommunication',
       'Terrorists/Non-State Militia', 'Tourists', 'Transportation',
       'Unknown.2', 'Utilities', 'Violent Political Party'], axis=1, inplace=True)


# merge columns from the centrality-based dataset
ira_shallow_hy = pd.merge(ira_shallow, ira_graph_derived[['Airports & Aircraft', 'Business', 'Educational Institution',
       'Food or Water Supply', 'Government (Diplomatic)',
       'Government (General)', 'Journalists & Media', 'Maritime', 'Military',
       'NGO', 'Other.1', 'Police', 'Private Citizens & Property',
       'Religious Figures/Institutions', 'Telecommunication',
       'Terrorists/Non-State Militia', 'Tourists', 'Transportation',
       'Unknown.2', 'Utilities', 'Violent Political Party']], left_index=True, right_index=True)



afg_shallow_hy.to_csv('afg_shallow.csv', header=True)
ira_shallow_hy.to_csv('ira_shallow.csv', header=True)

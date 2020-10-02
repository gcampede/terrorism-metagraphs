# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:16:47 2020

@author: Gian Maria
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 11:29:02 2020

@author: Gian Maria
"""



import numpy as np
import pandas as pd
import os
from functools import reduce




os.getcwd()




''' IMPORTING THE ENTIRE GTD FIRST'''


os.chdir('C:\\Users\\Gian Maria\\Desktop\\AAAI_21\\datasets_for_submission')
gtd = pd.read_csv('gtd7018.csv',delimiter=',', index_col=0)


gtd = gtd.rename({'iyear':'year', 'imonth': 'month', 'iday': 'day'}, axis=1)


''' substituting days and months = 0 with random numbers 
between 1 and 29 (for days) and 1 and 12 (for months) will 
produce slightly different results in the inputation of 
missing dates. It has to be noted however that out of the 34,893
attacks in the two datasets, only 10 have "0" as days, thus the random
procedure does not have a considerable effect on the creation of
the dataset'''

gtd.loc[gtd['month'] == 0,'month'] = np.random.randint(1,12, 
                                                       len(gtd.loc[gtd['month'] == 0]))
gtd.loc[gtd['day'] == 0,'day'] = np.random.randint(1,29, 
                                                   len(gtd.loc[gtd['day'] == 0]))

gtd['date']=pd.to_datetime(gtd[['year', 'month', 'day']])




# remove doubtful terrorist attacks

gtd = gtd[gtd.doubtterr !=1]


# create Afghanistan and Iran data

afg = gtd[gtd.country_txt == 'Afghanistan']
ira = gtd[gtd.country_txt == 'Iraq']


cols = ['provstate', 'attacktype1_txt', 'attacktype2_txt'
        'attacktype3_txt', 'targtype1_txt', 'targtype2_txt',
        'targtype3_txt', 'gname', 'gname2', 'gname3',
        'weaptype1_txt', 'weaptype2_txt', 'weaptype3_txt', 'weaptype4_txt']


# AFGHANISTAN 


# remove events happened before 2001 (too sparse)

afg['date']=pd.to_datetime(afg['date'], utc=True)
print(afg.date.dtype)

afg = afg[(afg['date'].dt.year >= 2001)]
afg['date']=afg['date'].dt.date


''' For each dimension (i.e., tactics, weapons and targets) we now 
process the dataset in order to have date as observations (rather than events)
and columns mapping the number of times a certain feature has been 
utilized in attacks occurring in that same day'''

# tactics

tactic1 = afg.groupby(['date','attacktype1_txt']).size().unstack(fill_value=0)
tactic2 = afg.groupby(['date','attacktype2_txt']).size().unstack(fill_value=0)
tactic3 = afg.groupby(['date','attacktype3_txt']).size().unstack(fill_value=0)

tacs = [tactic1, tactic2, tactic3]

tactic_total=reduce(lambda x,y: x.add(y, fill_value=0), tacs)


# weapons
weapon1 = afg.groupby(['date','weaptype1_txt']).size().unstack(fill_value=0)
weapon2 = afg.groupby(['date','weaptype2_txt']).size().unstack(fill_value=0)
weapon3 = afg.groupby(['date','weaptype3_txt']).size().unstack(fill_value=0)
weapon4 = afg.groupby(['date','weaptype4_txt']).size().unstack(fill_value=0)



weaps= [weapon1, weapon2, weapon3, weapon4]

weapon_total=reduce(lambda x,y: x.add(y, fill_value=0), weaps)

#targets
target1 = afg.groupby(['date', 'targtype1_txt']).size().unstack(fill_value=0)
target2 = afg.groupby(['date', 'targtype2_txt']).size().unstack(fill_value=0)
target3 = afg.groupby(['date', 'targtype3_txt']).size().unstack(fill_value=0)

target = [target1, target2, target3]

target_total=reduce(lambda x,y: x.add(y, fill_value=0), target)

''' We now combine tactics, weapons, targets in unique df that
incorporates the three dimensions having days as observations and
we fill in missing dates to have a complete series of observations
from January 1st 2000 to December 31st 2018'''

afg_unique=pd.concat([tactic_total, weapon_total, target_total], axis=1)


#fill in missing dates
datelist = pd.date_range(start='01-01-2001', end='31-12-2018', freq='1D')


afg_unique_complete = afg_unique.reindex(datelist).fillna(0.0)

# save and export dataset
afg_unique_complete.to_csv('afg_unique_complete.csv', header=True)



############ IRAQ 



ira['date']=pd.to_datetime(ira['date'], utc=True)
print(ira.date.dtype)

ira = ira[(ira['date'].dt.year >= 2001)]
ira['date']=ira['date'].dt.date




# tactics

tactic1 = ira.groupby(['date','attacktype1_txt']).size().unstack(fill_value=0)
tactic2 = ira.groupby(['date','attacktype2_txt']).size().unstack(fill_value=0)
tactic3 = ira.groupby(['date','attacktype3_txt']).size().unstack(fill_value=0)

tacs = [tactic1, tactic2, tactic3]

tactic_total=reduce(lambda x,y: x.add(y, fill_value=0), tacs)


# weapons subtypes 
weapon1 = ira.groupby(['date','weaptype1_txt']).size().unstack(fill_value=0)
weapon2 = ira.groupby(['date','weaptype2_txt']).size().unstack(fill_value=0)
weapon3 = ira.groupby(['date','weaptype3_txt']).size().unstack(fill_value=0)
weapon4 = ira.groupby(['date','weaptype4_txt']).size().unstack(fill_value=0)



weaps= [weapon1, weapon2, weapon3, weapon4]

weapon_total=reduce(lambda x,y: x.add(y, fill_value=0), weaps)

#targets
target1 = ira.groupby(['date', 'targtype1_txt']).size().unstack(fill_value=0)
target2 = ira.groupby(['date', 'targtype2_txt']).size().unstack(fill_value=0)
target3 = ira.groupby(['date', 'targtype3_txt']).size().unstack(fill_value=0)

target = [target1, target2, target3]

target_total=reduce(lambda x,y: x.add(y, fill_value=0), target)

# combining tactics, weapons, targets in unique df

ira_unique=pd.concat([tactic_total, weapon_total, target_total], axis=1)


#fill in missing dates


datelist = pd.date_range(start='01-01-2001', end='31-12-2018', freq='1D')


ira_unique_complete = ira_unique.reindex(datelist).fillna(0.0)

ira_unique_complete.to_csv('ira_unique_complete.csv', header=True)
#ira_unique_complete.index = ira_unique.index.date


ira_unique_complete.columns



























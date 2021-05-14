
# importing pandas package 
import pandas as pd 

 
# making data frame from csv file 
data = pd.read_csv("ps_locations.csv") 

data['freq'] = data.groupby('position')['position'].transform('count')
data= data.drop_duplicates('position')
data.to_csv('ps_unique_locations_freq.csv')


'''
#remove duplicates based on position column
mydf = data.drop_duplicates('position')

#count frequency of values in position column 
mydf2=data['position'].value_counts()
mydf3=data.groupby('position').count()
data['freq'] = data.groupby('position')['position'].transform('count')

#https://stackoverflow.com/questions/21558999/pandas-how-can-i-remove-duplicate-rows-from-dataframe-and-calculate-their-frequ  
mydf = data.groupby(['contig','position']).size().reset_index()
mydf.rename(columns = {0: 'frequency'}, inplace = True)

  
mydf.to_csv('ps_unique_locations.csv')
mydf2.to_csv('freq_value_counts.csv')
mydf3.to_csv('freq_groupby_count.csv')
data.to_csv('freq_groupby_transform.csv')
'''
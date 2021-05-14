# To set seed random number in order to reproducable results in keras
from numpy.random import seed
seed(4)
#import tensorflow
#tensorflow.random.set_seed(1234)
########################################
import pandas as pd
from pandas import *
import numpy as np
import random
import pickle
import joblib
from sklearn import svm
classifier =svm.SVC(gamma='scale',C=1,probability=True)
import plot_learning_curves as plc
from sklearn.preprocessing import MinMaxScaler #For feature normalization
scaler = MinMaxScaler()

from sklearn.model_selection import train_test_split


# Evaluate the model: Model Accuracy, how often is the classifier correct
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report #for classifier evaluation
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score # for printing AUC
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle


dataset = pd.read_csv('all_hela_data.csv', index_col=0)  #Pandas: read csv file: https://realpython.com/pandas-read-write-files
#dataset = shuffle(dataset)
#df = pd.read_csv('chunk.csv', index_col=0,nrows=1000)
#df.to_csv('chunk_1000.csv')
df = pd.DataFrame()      

columns=['event_level_mean','event_stdv','event_length']
#X_train = dataset[columns]

#columns=['event_level_mean']
#columns=['event_stdv']
#columns=['event_length']

#############################################

#line_count=0
for chunk in pd.read_csv('unlabeled_dataset_hela.csv', index_col=0, chunksize=100000):
    print(type(chunk))
    dataset_test=chunk
    X1 = dataset_test.iloc[:,0:2]
    #print("777777", X1)
    #print(";;;;;", type(X1))
    print("88888", X1.head())
    #shuffle the test and train datasets
    #dataset_test=shuffle(dataset_test)
    #combine onehot_encoding of train and test
    union_reference_kmer_set=set(dataset.iloc[:, 2]).union(set(dataset_test.iloc[:, 2]))
    union=list(union_reference_kmer_set)
    print(len(union))
    dataset['reference_kmer']=pd.Categorical(dataset['reference_kmer'], categories=list(union))
    dataset_test['reference_kmer']=pd.Categorical(dataset_test['reference_kmer'], categories=list(union))
    ##############################################
    X_train = dataset[columns]
    #insert onehot encoding of reference-kmer in train data
    Onehot=pd.get_dummies(dataset['reference_kmer'], prefix='reference_kmer')
    X_train= pd.concat([X_train,Onehot],axis=1)
    print("#############",X_train.shape)
    print(X_train.head())
    #scale training data
    X_train= scaler.fit_transform(X_train)
    y_train = dataset['label'] 
    print(",,,,,,,,",X_train.shape)
    ###################################
    X_test = dataset_test[columns]
    #insert onehot encoding of reference-kmer in test data
    Onehot=pd.get_dummies(dataset_test['reference_kmer'], prefix='reference_kmer')
    X_test= pd.concat([X_test,Onehot],axis=1)
    #X_test= Onehot
    print("#############",X_test.shape)
    print(X_test.head())
    #scale training data
    X_test= scaler.fit_transform(X_test)
    print(",,,,,,,,",X_test.shape)
    ###############################################
    clf = classifier.fit(X_train,y_train.ravel())
    y_pred = classifier.predict(X_test)     ################################ 515 features in testing against 339 features in training
    #create new dataframe to store  predicted values 
    newDF = pd.DataFrame() #creates a new dataframe that's empty
    newDF['y_predict'] = y_pred
    print(newDF.head())
    ##################
    print("MMMMMMMMMMMMMMMMMMMMMMMM")
    X1.reset_index(drop=True, inplace=True) #https://stackoverflow.com/questions/32801806/pandas-concat-ignore-index-doesnt-work
    newDF['y_predict'].reset_index(drop=True, inplace=True)
    pred_locations= pd.concat([X1,newDF['y_predict'] ],axis=1)
    print("ffffff",pred_locations.head())
    #filter ps locations from predicted locations
    is_ps =  pred_locations['y_predict']==1
    print(is_ps.head())
    ps_pred_locations = pred_locations[is_ps]
    print(ps_pred_locations.shape)
    print(ps_pred_locations.head())
    # to append ps_pred_locations at the end of df dataframe 
    df = pd.concat([df,ps_pred_locations])
df.to_csv('ps_locations.csv')
print("MMMMMMMMMMMMMMMMMMM")


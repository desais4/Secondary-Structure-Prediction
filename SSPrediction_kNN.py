# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:01:36 2020

@author: Shivani
"""
#importing libraries 
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics 
import numpy as np
from sklearn.model_selection import KFold
import statistics as st
#-----------------------------------------------------------------------------

# reading csv file 
data = pd.read_csv('Dataset.csv')

# shape of dataset 
print("Shape:", data.shape) 
# column names 
print("\nFeatures:", data.columns) 

# storing the feature matrix (x) and response vector (y) 
x = data[data.columns[0]] 
y = data[data.columns[1]] 
#-----------------------------------------------------------------------------

#function to perform one-hot encoding of protein sequence
def one_hot_encode(sw): 
    O=[]
    P=[]
    #loop over every letter in the sliding window
    for j in range(0,len(sw)):
        seq=sw[j]
        
        #defining the alphabet for encoding
        aminocode ='ACDEFGHIKLMNPQRSTVWY'
        char_to_int = dict((c, i) for i, c in enumerate(aminocode))
        integer_encoded = [char_to_int[char] for char in seq]
        onehot_encoded = list()
        for value in integer_encoded:
        	letter = [0 for _ in range(len(aminocode))]
        	letter[value] = 1
        	onehot_encoded.append(letter)
        P.append(onehot_encoded)
    O.append(P)
    return O
#-----------------------------------------------------------------------------

#arrays to hold processed feature matrix and response vectors
X=[]
Y=[]

#function to generate feature vectors and corresponding response vectors according to the value of sliding window
def create_vectors(w):
    for xi,yj in zip(x,y):
        for i in range(0, len(xi)):
            sw=xi[i:i+w]
            #encode only if the number of elements in sw equals the length of sliding window
            if(len(sw)==w):
                O=one_hot_encode(sw)  
                Z=np.reshape(O,-1)
                X.append(Z)          
                if(i<len(xi)-(w/2)):                
                    Y.append(yj[i+int(w/2)])
    return X,Y
#-----------------------------------------------------------------------------

#looping over different values of sliding window
for sw in [5,7,9,11]:
    print("\nSliding window=" + str(sw))
    
    #create feature matrix and response vectors
    X,Y=create_vectors(sw)    
    X=np.array(X)    
    Y=np.array(Y)    
    
    #declare a KFold object with number of splits for cross validation
    kf = KFold(n_splits=5, random_state=0, shuffle=True)    
    kf.get_n_splits(X)
    
    #array to hold accuracy for each split
    A=[]
    
    #looping over different values of k-nearest neighbors
    for k in [1,3,5,7,9]:
        
        #looping over train, test indices for each split
        for train, test in kf.split(X):
            
            X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test] 
            
            #building a kNN classifier with given value of k
            knn = KNeighborsClassifier(n_neighbors=k, metric='hamming')
            
            #fitting training data to the classifier
            knn.fit(X_train, y_train) 
            
            #making predictions on testing data
            y_pred = knn.predict(X_test) 
            
            #measuring accuracy of prediction
            acc=metrics.accuracy_score(y_test, y_pred)
            
            #adding the accuracy value to A
            A.append(acc)
        
        #displaying the maximum accuracy over all 5 splits
        print("For k=" + str(k))
        print("Average accuracy= " + str(st.mean(A)))
        print("Standard deviation= "+ str(st.stdev(A)))
        A.clear()
    
    #converting numpy arrays to list and deleting their contents before entering the next loop
    X=X.tolist()
    Y=Y.tolist()
    X.clear()
    Y.clear()
#-----------------------------------------------------------------------------




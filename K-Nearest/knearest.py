# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:15:46 2017

@author: Della
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data = pd.read_csv('https://s3.amazonaws.com/demo-datasets/wine.csv') #Read data from csv file hosted online
numeric_data = data.drop('color', axis = 1) #Drop redundant column
numeric_data = (numeric_data - np.mean(numeric_data))/numeric_data.std(ddof=0) #Normalize data

#Reduce the amount of variables using Principal Component Analysis 
pca = PCA(n_components = 2) 
principal_components = pca.fit(numeric_data).transform(numeric_data)

#Plot 2 principal components 
observation_colormap = ListedColormap(['red', 'blue'])
x = principal_components[:,0]
y = principal_components[:,1]
plt.title("Principal Components of Wine")
plt.scatter(x, y, alpha = 0.2,
    c = data['high_quality'], cmap = observation_colormap, edgecolors = 'none')
plt.xlim(-8, 8); plt.ylim(-8, 8)
plt.xlabel("Principal Component 1"); plt.ylabel("Principal Component 2")
plt.show()

def accuracy(predictions, outcomes):
    """Calculate accuracy of a prediction"""
    return np.mean(predictions == outcomes)*100

#This implementation omits splitting data into training and preficting sets
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(numeric_data, data['high_quality'])
library_predictions = knn.predict(numeric_data)

#Print out accuracy of prediction
acc = accuracy(library_predictions, data['high_quality'])
print ("Prediction accuracy is ","{0:.2f}".format(acc),"%", sep="")


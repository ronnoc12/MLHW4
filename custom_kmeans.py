#  =========================================================
#  HW 4: Unsupervised Learning, K-Means Clustering
#  CS 4824 / ECE 4484, Spring '21
#  Written by Matt Harrington, Haider Ali
#  =========================================================

# Standard imports
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA

class CustomKMeans():

    # Initialize all attributes
    def __init__(self, k):
        self.k_ = k       # Number of clusters
        self.labels_ = 0  # Each sample's cluster label
        self.inertia_ = 0 # Sum of all samples' distances from their centroids

    # Find K cluster centers & label all samples
    def fit(self, data, plot_steps=False):
        # Fit the PCA module & Transform our data for later graphing
        self.pca = PCA(2).fit(data)
        self.data = pd.DataFrame(data)
        self.data_pca = pd.DataFrame(self.pca.transform(data))
        self.data_pca.columns = ['PC1', 'PC2']
        
        # Initialize variables
        self.iteration = 1
        n = data.shape[0]
        
        # Initialize centroids to random datapoints
        self.centroids = data.iloc[np.random.choice(range(n), self.k_, replace=False)].copy()
        self.centroids.index = np.arange(self.k_)
        
        #  =====================================================================
        #  ====================== IMPLEMENT KMEANS ======================= 
        self.labels_ = [random.randint(0,self.k_-1) for i in range(n)] # Update
        self.inertia_ = np.sum(np.arange(n))                           # Update
    
        self.isNOTConverged = True

        while (self.isNOTConverged):
            
            self.helper(data, n)

            #update our centroids
            if (self.isNOTConverged):
                for i in range(self.k_):
                    self.centroids.iloc[i] = data.iloc[self.centroidPointList[i]].mean()

            # show data & centroids at each iteration when testing performance
            if plot_steps:
                self.plot_state() 
                self.iteration += 1
            #  =====================================================================
            
        return self
    
    def helper(self, data, n):
        distanceArray = np.zeros((n, self.k_)) #sets distance to an array of zeros of size n by k 

        for j in range(self.k_):
            distanceArray[:, j] = np.linalg.norm(data - self.centroids.iloc[j], axis=1) #calculate our distance

        minDistanceLocation = np.argmin(distanceArray, axis=1) #goes across the rows to return the assigned centroid for each data sample 

        self.centroidPointList = [] #declare a list to hold more lists 

        for i in range(self.k_):
            self.centroidPointList.append(np.where(i == minDistanceLocation)[0]) #goes through all the elements in the array finds all the points equal to a given centroid and adds them to a list  

        oldLables = self.labels_.copy() #copy the old lables before we change them so we can test for convergence

        for i in range(n): 
            self.labels_[i] = minDistanceLocation[i] #assign all the lables 

        #check for convergence
        self.isNOTConverged = (not oldLables == self.labels_)

        #get the minimum distances then sum them to update inertia
        minDistanceArray = np.min(distanceArray, axis=1)
        self.inertia_ = np.sum(minDistanceArray)
        
        return self

    # Plot projection of data and centroids in 2D
    def plot_state(self):
        # Project the centroids along the principal components
        centroid_pca = self.pca.transform(self.centroids)
        
        # Draw the plot
        plt.figure(figsize=(8,8))
        plt.scatter(self.data_pca['PC1'], self.data_pca['PC2'], c=self.labels_)
        plt.scatter(centroid_pca[0], centroid_pca[1], marker = '*', s=1000)
        plt.title("Clusters and Centroids After step {}".format(self.iteration))
        plt.show()
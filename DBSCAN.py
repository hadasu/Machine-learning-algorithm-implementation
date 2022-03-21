# -*- coding: utf-8 -*-
"""

@author: Hadas Unger
DBSCAN
"""
import numpy as np
from sklearn.datasets import load_iris

class DBSCAN:
    def __init__(self, dataSet, epsilon, min_points):
        self.dataset = dataSet
        self.epsilon = epsilon
        self.min_points = min_points
        self.clusters = dict.fromkeys(range(len(self.dataset)), None)
        self.points_dist=np.zeros((len(self.dataset),len(self.dataset))) 
        
        for i in range(len(self.dataset)):
            for j in range(len(self.dataset)):
                self.points_dist[i][j]=np.linalg.norm(self.dataset[i]-self.dataset[j])
    
#   Distance function between two points
    def distance(self, point1, point2):
        return self.points_dist[point1][point2]
    
#    the function calculate how many neighbours are in an epsilon envirement of a point
    def regionQuery(self, point):
        neighbours = []
        size = len(self.dataset)
        for i in range(size):
            dis = self.distance(i, point)
            if dis <= self.epsilon:
                 neighbours.append((self.dataset[i], i))
        return neighbours
    
#    A function that begins in a core point and expands it till it can’t be expanded anymore
    def expandCluster(self,neighbours, cluster_num):
        cluster = dict()
        for point in neighbours:
            new_neighbours = self.regionQuery(point[1])
            
            if len(new_neighbours) < self.min_points:
                cluster[point[1]] = 'Border'
                self.clusters[point[1]] = cluster_num
            else:
                cluster[point[1]] = 'Core'
                self.clusters[point[1]] = cluster_num
                neighbours.extend(new_neighbours)
            neighbours = [item for item in neighbours if item[1] not in cluster or cluster[item[1]] != 'Core']

            
#     A function that iterates over all points in the db and if they are core points expands them       
    def DBSCAN(self, cluster_num = 1): 
        random_point = None   
        while random_point is None or self.clusters[random_point] is not None:
            random_point = np.random.randint(0,len(self.dataset))
            
        neighbours = self.regionQuery(random_point)
        
        if len(neighbours) < self.min_points:
            self.clusters[random_point] = 'Outlier'
            
            if None in self.clusters.values():
                self.DBSCAN(cluster_num)
        else:
            self.clusters[random_point] = cluster_num 
            self.expandCluster(neighbours, cluster_num)
            
            if None in self.clusters.values():
                self.DBSCAN(cluster_num + 1)



       
data = load_iris()
DBSCAN = DBSCAN(data.data, 0.5 ,10)   
DBSCAN.DBSCAN()       
        
print(DBSCAN.clusters)

print(DBSCAN.points_dist)

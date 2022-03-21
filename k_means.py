# -*- coding: utf-8 -*-
"""

@author: Hadas
k-means
"""
import numpy as np
import matplotlib.pyplot as plt

def import_data():
    from sklearn.datasets import load_iris
    data = load_iris()
        
    return data.data


def distance_data_to_point(datasets, point):
    distance = 0
    size = len(datasets) - 1
    for i in range(size):
        distance += np.power((datasets[i] - point[i]), 2)
    return np.sqrt(distance)

def closes_point(datasets, k_point):
    size = len(k_point)
    min_distance_from_point = 9999
    cluster_num = -1
    for i in range (size):
        distance = distance_data_to_point(datasets, k_point[i])
        
        if distance < min_distance_from_point:
            min_distance_from_point = distance
            cluster_num = i
    return cluster_num, datasets

def point_to_cluster(datasets, k_point):
    dict_cluster = dict()
    size = len(datasets)
    for i in range (size):
        key, data = closes_point(datasets[i], k_point)
        if key in dict_cluster:
            dict_cluster[key].append(data)
        else:
            dict_cluster[key] = [data]
    return dict_cluster

def cluster_mean(cluster):
    size = len(cluster)
    sum_cluster = [0,0,0,0]
    for i in range (size):
        sum_cluster = [sum(x) for x in zip(sum_cluster, cluster[1])]
        
    return [i/size for i in sum_cluster]

        
def recalculate_points(dict_cluster, k_point):
    k_new_point = np.empty_like(k_point)
    for k, v in dict_cluster.items():
        k_new_point[k] = cluster_mean(v)
    return k_new_point

def check_chenge_of_points(k_new_point, k_point):
    k = len(k_new_point)
    size = len(k_new_point[0])
    diff = 0
    for i in range (k):
        for j in range (size):
            diff += k_new_point[i][j] - k_point[i][j]
        
    if diff == 0:
        return False
    return True

def k_means(data, start_k_points):
    k_points = start_k_points
    itration = 1
    dict_cluster = point_to_cluster(data,k_points)
    k_new_point = recalculate_points(dict_cluster, k_points)
    show_cluster(dict_cluster)
    
    while check_chenge_of_points(k_new_point, k_points):
        k_points= k_new_point
        itration += 1
        dict_cluster = point_to_cluster(data,k_points)
        k_new_point = recalculate_points(dict_cluster, k_points)
        show_cluster(dict_cluster)
    return dict_cluster, k_points, itration

def show_cluster(dict_cluster):
    colors = ['red','green', 'blue', 'green']
    
    for k, v in dict_cluster.items():
        plt.scatter(dict_cluster[k], dict_cluster[k],color=colors[k])

    plt.show()        
   
data = import_data()
#k =  [[6.3,3.3,4.7,1.6],[4.9,2.4,3.3,1],[6.6,2.9,4.6,1.3]]
k =  [[3.,3.3,4.7,1.6],[4.9,2.4,5.,1],[6.6,2.9,4.6,1]]
dict_cluster, k_points, itration = k_means(data, k)





























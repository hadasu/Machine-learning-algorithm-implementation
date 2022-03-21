# -*- coding: utf-8 -*-
"""

@author: Hadas Unger
KNN
"""

import numpy as np


#Handle the data:
class DataSet:
    
    def __init__(self, test_percentage):
        self.test_percentage = test_percentage
        self.dataset = self.import_data()

    def split_train_test(self):
        np.random.shuffle(self.dataset)
        per_index=int(len(self.dataset)*(1-self.test_percentage))
        return self.dataset[:per_index,:],self.dataset[per_index:,:]
    
    def import_data(self):
        from sklearn.datasets import load_iris
        data = load_iris()
        
        return np.c_[data.data, data.target]


class KNN_algorithm:
    
    def __init__(self, DataSet, K,):
        self.dataset = DataSet
        self.K = K
        self.train, self.test = DataSet.split_train_test()
    
    #Distance function:
    def distance_between_two_points(self, datasets1, datasets2):
        distance = 0
        size = len(datasets1) - 1
        for i in range(size):
            distance += np.power((datasets1[i] - datasets2[i]), 2)
        return np.sqrt(distance)
#        return np.linalg(datasets1-datasets2)
    
    def distance_between_dataset_to_point(self, mainDatasets, datasets):
        distances = []
        size = len(mainDatasets)
        for i in range(size):
            dis = self.distance_between_two_points(mainDatasets[i], datasets)
            distances.append((mainDatasets[i], dis))
        return distances
    
    #Nearest neighbours:
    def KNN(self, dataset, K):
        dataset.sort(key=lambda tup: tup[1])
        return dataset[:K]
    
    #Predict from k nearst neighbours:
    def predict_for_one_point(self, knn):
        Vote = dict()
        size = len(knn)
        for i in range(size):
            key = ((knn[i])[0])[len((knn[i])[0]) - 1]
            if key in Vote:
                Vote[key] += 1
            else:
                Vote[key] = 1
                
        return max(Vote.items())[0]
    
    def predict_for_dataset(self):
        prediction = []
        size = len(self.test)
        for i in range(size):
            distances = self.distance_between_dataset_to_point(self.train, self.test[i])
            knn = self.KNN(distances, self.K)
            prediction.append(self.predict_for_one_point(knn))
                
        return np.c_[self.test, prediction]
    
    #Calculate the accuracy on the test data:
    def accuracy(self, predict_dataset):
        correct = 0
        size = len(predict_dataset)
        for i in range(size):
            if predict_dataset[i][-2] == predict_dataset[i][-1]:
                correct += 1
                
        return (correct/float(len(predict_dataset))) * 100.0
        

def main():
    
    percentage_test = 0.1
    data = DataSet(percentage_test)

    K = 10
    knn = KNN_algorithm(data, K,)
    predict_dataset = knn.predict_for_dataset()
    accuracy = knn.accuracy(predict_dataset)
    
    print(predict_dataset)
    print(accuracy)

  
if __name__== "__main__":
  main()
















import numpy as np
from operator import itemgetter

class KNearestNeighbours :
# constructr
    def __init__(self,data,target,test_point,k) :
        self.data = data
        self.target= target
        self.test_point =test_point
        self.k =k
        self.distances =list()
        self.categories =list()
        self.indices =list()
        self.counts=list()
        self.category_assigned =None

# static method only works between class
    @staticmethod
    def dist(p1,p2) :
        # method returns the euclidean distance between 2 points
        return np.linalg.norm(np.array(p1)-np.array(p2))

    def fit(self) :
        # method that perform KNN classification
        # crete a list of (distance,index) tuple from test int to each point indexes
        self.distances.extend([(self.dist(self.test_point,point),i) for point , i in zip(self.data, [i for i in range(len(self.data))])])
         # Sort the distances in ascending order
        sorted_li = sorted(self.distances, key=itemgetter(0))
        # Fetch the indices of the k nearest point from the data
        self.indices.extend([index for (val, index) in sorted_li[:self.k]])
        # Fetch the categories from the train data target
        for i in self.indices:
            self.categories.append(self.target[i])
        # Fetch the count for each category from the K nearest neighbours
        self.counts.extend([(i, self.categories.count(i)) for i in set(self.categories)])
        # Find the highest repeated category among the K nearest neighbours
        self.category_assigned = sorted(self.counts, key=itemgetter(1), reverse=True)[0][0]

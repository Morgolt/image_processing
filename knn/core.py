import operator

import numpy as np


class KNearestNeighbors:
    def __init__(self, train_set):
        self.train_set = train_set

    def find_nearest(self, instance, k):
        distances = np.array([], dtype=[('distance', 'f4'), ('class', 'S1')])
        for x in self.train_set:
            dist = np.linalg.norm(x[0] - instance)
            distances = np.concatenate((distances, np.array([(dist, x[1],)], dtype=distances.dtype)))
        distances.sort(kind='mergesort', order='distance')
        neighbors = np.array([])
        for i in range(k):
            neighbors = np.append(neighbors, distances[i][1])
        return neighbors

    def classify(self, instance, k=3):
        neighbors = self.find_nearest(instance, k)
        votes = {}
        for x in neighbors:
            response = x
            if response in votes:
                votes[response] += 1
            else:
                votes[response] = 1
        sorted_votes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_votes[0][0]

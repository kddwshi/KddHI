import numpy as np

class L2():
    def __init__(self):
        self.name = 'L2'

    def compute(self, y_true, y_pred):
        return np.linalg.norm(y_true-y_pred, ord=2)
    
class EuclideanDistance():
    def __init__(self):
        self.name = 'EuclideanDistance'

    def compute(self, interval1, interval2):
        return np.sqrt(np.sum((interval1 - interval2)**2))
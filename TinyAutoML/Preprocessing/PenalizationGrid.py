import numpy as np

class PenalizationGrid:

    def __init__(self, minCoef=1e-10, maxCoef=1, length=200):
        self.values = np.linspace(maxCoef, minCoef, length).tolist()

    def isEmpty(self)->bool:
        return len(self.values)==0

    def getNextKCoeffs(self,k):
        penalizationCoeffsForBatch = self.values[:k]
        del self.values[:k]
        return penalizationCoeffsForBatch

    def size(self):
        return len(self.values)

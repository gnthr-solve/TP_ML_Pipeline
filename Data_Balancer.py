from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import numpy as np

class DataBalancer:
    def __init__(self, balancer=None, random_state=42):
        self.balancer = balancer
        self.random_state = random_state

    def balance_data(self, X, y):
        if self.balancer is None:
            raise ValueError("Balancer object is not provided. Please initialize the balancer.")
        
        X_resampled, y_resampled = self.balancer.fit_resample(X, y)
        return X_resampled, y_resampled
from sklearn.metrics import precision_score
import numpy as np

class Assessor():

    def __init__(self, y_test_dict, y_predicted_dict, metrics_dict, balanced = False):
        """
        Input: dictionaries with the parameters of generation, balancing and classification as keys or
        other key value pairs
        """
        self.y_test = y_test_dict
        self.y_predicted = y_predicted_dict
        #self.y_test_dict = y_test_dict
        #self.y_predicted_dict = y_predicted_dict
        self.metrics_dict = metrics_dict
        self.balanced = balanced
        
    def confusion_values(self):
        y_test = self.y_test
        y_predicted = self.y_predicted

        self.TP = np.sum((y_test == 1) & (y_predicted == 1))
        self.FP = np.sum((y_test == 0) & (y_predicted == 1))
        self.TN = np.sum((y_test == 0) & (y_predicted == 0))
        self.FN = np.sum((y_test == 1) & (y_predicted == 0))

    def net_benefit(self, harm_to_benefit):

        self.NB = self.TP - harm_to_benefit * self.FP

    def evaluate():
        pass



#precision_score(study.y_test,study.y_predicted_no_balancing)
#precision_score(study.y_test,study.y_predicted_balanced)
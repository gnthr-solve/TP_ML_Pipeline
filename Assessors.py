from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from parameters import default_test_dict
import numpy as np




class Study():
    def __init__(self, 
                 data_generator=None, 
                 data_balancer=None,
                 data_classifier=None
                 ):
        self.data_generator = data_generator
        self.data_balancer = data_balancer
        self.data_classifier = data_classifier
        self.results = {
            "no balancing": None,
            "balanced": None
        }
        
    def run(self):
        X_train, X_test, y_train, y_test = self.data_generator.generate_data()
        self.y_test = y_test
        self.data_classifier.fit(X_train, y_train)
        self.y_predicted_no_balancing = self.data_classifier.predict(X_test)
        self.X_train_balanced, self.y_train_balanced = self.data_balancer.balance_data(X_train, y_train)
        self.data_classifier.fit(self.X_train_balanced, self.y_train_balanced)
        self.y_predicted_balanced = self.data_classifier.predict(X_test)
        
    def calculate_metrics(self):
        imbalanced_results = {
            "accuracy": accuracy_score(self.y_test,self.y_predicted_no_balancing),
            "precision": precision_score(self.y_test,self.y_predicted_no_balancing), 
            "recall":  recall_score(self.y_test,self.y_predicted_no_balancing),
            "F1 score": f1_score(self.y_test,self.y_predicted_no_balancing),
            "ROC AUC Score": roc_auc_score(self.y_test,self.y_predicted_no_balancing), 
            #"Confusion Matrix": confusion_matrix(self.y_test,self.y_predicted_no_balancing)
        }
        balanced_results = {
            "accuracy": accuracy_score(self.y_test,self.y_predicted_balanced),
            "precision": precision_score(self.y_test,self.y_predicted_balanced), 
            "recall":  recall_score(self.y_test,self.y_predicted_balanced),
            "F1 score": f1_score(self.y_test,self.y_predicted_balanced),
            "ROC AUC Score": roc_auc_score(self.y_test,self.y_predicted_balanced), 
            #"Confusion Matrix": confusion_matrix(self.y_test,self.y_predicted_balanced)
        }
        
        return {
            "imbalanced_results": imbalanced_results,
            "balanced_results": balanced_results
        }






class IterAssessor():

    def __init__(self, 
                 data_generator = None, 
                 data_balancer = None,
                 data_classifier = None,
                 metrics = None
                 ):
        self.data_generator = data_generator
        self.data_balancer = data_balancer
        self.data_classifier = data_classifier
    
        self.results = {}


    def run(self):

        X_train, X_test, y_train, y_test = self.data_generator.generate_data()
        self.y_test = y_test

        self.balanced_data = self.data_balancer.balance_data(X_train, y_train)
        self.data_classifier.fit(self.X_train_balanced, self.y_train_balanced)
        self.predictions_list_balanced = self.data_classifier.predict(X_test)


    def calculate_metrics(self):

        balanced_results = {
            "accuracy": accuracy_score(self.y_test,self.y_predicted_balanced),
            "precision": precision_score(self.y_test,self.y_predicted_balanced), 
            "recall":  recall_score(self.y_test,self.y_predicted_balanced),
            "F1 score": f1_score(self.y_test,self.y_predicted_balanced),
            "ROC AUC Score": roc_auc_score(self.y_test,self.y_predicted_balanced), 
            #"Confusion Matrix": confusion_matrix(self.y_test,self.y_predicted_balanced)
        }
        
        return {
            "balanced_results": balanced_results
        }





class Metrics():

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
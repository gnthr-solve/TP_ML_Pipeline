from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from parameters import default_test_dict
import numpy as np
import plotly.express as px




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

    def __init__(self, 
                 X_test, 
                 y_test,
                 #test_data, 
                 predictions_dict, 
                 #metrics_dict, 
                 #balanced = False
                 ):
        """
        Input: dictionaries with the parameters of generation, balancing and classification as keys or
        other key value pairs
        """

        self.X_test = X_test
        self.y_test = y_test
        self.classes = predictions_dict['classes']
        #print(self.classes)
        #print(np.where(self.classes == 1))
        #print(np.shape(np.where(self.classes == 1)))
        #print(np.where(self.classes == 1)[0])
        all_classes_proba_array = predictions_dict['probabilities']
        minority_proba_2darray = all_classes_proba_array[:, np.where(self.classes == 1)[0]]
        self.test_probabilities = minority_proba_2darray.flatten()
        self.y_predicted = predictions_dict['predictions']
        #self.predictions_dict = predictions_dict,
        #self.metrics_dict = metrics_dict
        #self.balanced = balanced

        #print(predictions_dict['probabilities'][:20])
        #print(minority_proba_2darray[:20])
        #print(self.test_probabilities[:20])
        #print(np.sum(self.test_probabilities[:20], axis = 1))
        #print(len(self.test_probabilities))
        #print(self.y_predicted)
        
    def confusion_values(self):
        y_test = self.y_test
        y_predicted = self.y_predicted

        self.TP = np.sum((y_test == 1) & (y_predicted == 1))
        self.FP = np.sum((y_test == 0) & (y_predicted == 1))
        self.TN = np.sum((y_test == 0) & (y_predicted == 0))
        self.FN = np.sum((y_test == 1) & (y_predicted == 0))

    def net_benefit(self, harm_to_benefit):

        self.NB = self.TP - harm_to_benefit * self.FP


    def calibration_curve(self, k):

        sorted_indices = np.argsort(self.test_probabilities)
        #print(sorted_indices)

        sorted_probabilities = self.test_probabilities[sorted_indices]
        #print(sorted_probabilities[:20])

        sorted_y_test = self.y_test[sorted_indices]
        #sorted_y_predicted = self.y_predicted[sorted_indices]

        binned_probabilities = np.array_split(sorted_probabilities, k)
        binned_y_test = np.array_split(sorted_y_test, k)

        #print(len(binned_probabilities))
        #print(np.shape(binned_probabilities))
        #print(binned_probabilities[:1])
        #print(binned_y_test[0])

        mean_pred_proba = [bin.sum()/len(bin) for bin in binned_probabilities]
        mean_freq = [bin.sum()/len(bin) for bin in binned_y_test]

        #print(mean_pred_proba)
        #print(mean_freq)

        fig = px.line(x = mean_pred_proba, y = mean_freq, title = 'Calibration Curve')
        fig.show()


    def evaluate():
        pass



#precision_score(study.y_test,study.y_predicted_no_balancing)
#precision_score(study.y_test,study.y_predicted_balanced)






if __name__=="__main__":

    import pandas as pd
    from loguru import logger
    from Classifier import Classifier, DictIterClassifier
    from Data_Generator import Multi_Modal_Dist_Generator
    from Visualiser import RawVisualiser
    from parameters import mixed_3d_test_dict
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier


    data_generator = Multi_Modal_Dist_Generator(**mixed_3d_test_dict)
    X_train, X_test, y_train, y_test = data_generator.prepare_data(0.2)

    visualiser = RawVisualiser()

    classifiers_dict = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    #"SVC": SVC,
    #"Naive Bayes": GaussianNB,
    #"XGboost": XGBClassifier,
    #"Lightgbm": LGBMClassifier
    }

    """
    Classify to obtain Metrics Test Case
    -------------------------------------------------------------------------------------------------------------------------------------------
    """
    dict_iter_classifier = DictIterClassifier(classifiers_dict = classifiers_dict)
    
    dict_iter_classifier.fit(X_train, y_train)
    #Make predictions
    predictions_dict_list = dict_iter_classifier.predict(X_test)

    # Make predictions
    y_predictions = [pred_dict['predicted_y'] for pred_dict in predictions_dict_list]

    prob_predictions = [pred_dict['predicted_proba'] for pred_dict in predictions_dict_list]

    classes = [pred_dict['classes'] for pred_dict in predictions_dict_list]

    predictions_dict = {'predictions': y_predictions[0], 'probabilities': prob_predictions[0], 'classes': classes[0]}
    
    metr = Metrics(X_test, y_test, predictions_dict)

    metr.calibration_curve(10)

    
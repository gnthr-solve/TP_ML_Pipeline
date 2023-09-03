from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np





class CorStudy():
    def __init__(self, 
                 data_generator=None, 
                 data_balancer=None,
                 data_classifier=None
                 ):
        self.data_generator = data_generator
        self.data_balancer = data_balancer
        self.data_classifier = data_classifier
        
        
    def run(self):

        X_train, X_test, y_train, y_test = self.data_generator.generate_data()
        self.y_test = y_test

        self.X_train_balanced, self.y_train_balanced = self.data_balancer.balance_data(X_train, y_train)

        self.data_classifier.fit(self.X_train_balanced, self.y_train_balanced)
        self.y_predicted = self.data_classifier.predict(X_test)
    

    def calculate_metrics(self):
        
        results = {
            "accuracy": accuracy_score(self.y_test,self.y_predicted),
            "precision": precision_score(self.y_test,self.y_predicted), 
            "recall":  recall_score(self.y_test,self.y_predicted),
            "F1 score": f1_score(self.y_test,self.y_predicted),
            "ROC AUC Score": roc_auc_score(self.y_test,self.y_predicted), 
            #"Confusion Matrix": confusion_matrix(self.y_test,self.y_predicted)
        }
        
        return results







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
    "XGboost": XGBClassifier,
    #"Lightgbm": LGBMClassifier
    }

    """
    Assessor Testcase
    -------------------------------------------------------------------------------------------------------------------------------------------
    """
    dict_iter_classifier = DictIterClassifier(classifiers_dict = classifiers_dict)
    
    dict_iter_classifier.fit(X_train, y_train)
    #Make predictions
    predictions_dict_list = dict_iter_classifier.predict(X_test)
    
    
    

    
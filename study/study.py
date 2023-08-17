import sys
sys.path.append("C:/Users/Artemii/Desktop/teamproject/TP_ML_Pipeline/study")
from generator import ImbalancedDataGenerator
from balancer import DataBalancer
from classifier import Classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

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
        
    def calculatete_matrics(self):
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
        
        
if __name__=="__main__":
    from imblearn.over_sampling import ADASYN,RandomOverSampler,KMeansSMOTE,SMOTE,BorderlineSMOTE,SVMSMOTE,SMOTENC, RandomOverSampler
    import pandas as pd
    from sklearn.svm import SVC

    class_ratios = [0.1, 0.01, 0.01]
    n_samples = [10e4, 10e5, 10e6]
    n_features = [10,20,100]
    data_generator = ImbalancedDataGenerator(class_ratio=0.1, n_samples=1000, n_features=10, distance=1, flip_y=0.1)
    classifier = SVC(random_state=123)
    data_classifier = Classifier(classifier=classifier)
    smote_balancer = SMOTE(sampling_strategy='auto', random_state=123)
    data_balancer = DataBalancer(balancer=smote_balancer)
    
    study = Study(
        data_generator=data_generator,
        data_balancer=data_balancer,
        data_classifier=data_classifier
    )
    study.run()
    
    results = study.calculatete_matrics()
    
    pd.DataFrame(results['balanced_results'], index=[0])
    
    study.y_predicted_no_balancing
    study.y_predicted_balanced
    study.y_test
    from sklearn.metrics import precision_score
    precision_score(study.y_test,study.y_predicted_no_balancing)
    precision_score(study.y_test,study.y_predicted_balanced)
        

import sys
sys.path.append("C:/Users/Artemii/Desktop/teamproject/TP_ML_Pipeline/study")
from generator import ImbalancedDataGenerator
from balancer import DataBalancer
from classifier import Classifier

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
        
if __name__=="__main__":
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
    data_generator = ImbalancedDataGenerator(class_ratio=0.1, n_samples=1000, n_features=10, distance=1, flip_y=0.1)
    classifier = SVC(random_state=123)
    data_classifier = Classifier(classifier=classifier)
    smote_balancer = SMOTE(sampling_strategy='auto', random_state=42)
    data_balancer = DataBalancer(balancer=smote_balancer)
    
    study = Study(
        data_generator=data_generator,
        data_balancer=data_balancer,
        data_classifier=data_classifier
    )
    study.run()
    
    study.y_predicted_no_balancing
    study.y_predicted_balanced
    study.y_test
    from sklearn.metrics import precision_score
    precision_score(study.y_test,study.y_predicted_no_balancing)
    precision_score(study.y_test,study.y_predicted_balanced)
        

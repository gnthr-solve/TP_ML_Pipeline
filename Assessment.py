import pandas as pd
from loguru import logger
from imblearn.over_sampling import ADASYN,RandomOverSampler,KMeansSMOTE,SMOTE,BorderlineSMOTE,SVMSMOTE,SMOTENC, RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import sys
from Data_Generator import ImbalancedDataGenerator
from Data_Balancer import DataBalancer
from Classifier import Classifier
from Assessors import Study


balancing_methods = {
"ADASYN": ADASYN,
"RandomOverSampler": RandomOverSampler,
#"KMeansSMOTE": KMeansSMOTE,
"SMOTE": SMOTE,
"BorderlineSMOTE": BorderlineSMOTE,
"SVMSMOTE": SVMSMOTE,
#"SMOTENC": SMOTENC,
}

classifiers = {
    "Logistic Regression": LogisticRegression,
    #"Decision Tree": DecisionTreeClassifier,
    #"Random Forest": RandomForestClassifier,
    #"SVC": SVC,
    #"Naive Bayes": GaussianNB,
    #"XGboost": XGBClassifier,
    #"Lightgbm": LGBMClassifier
}
class_ratio_list = [0.1, 0.01, 0.01]
n_samples_list = [10e2, 10e3, 10e4]
n_features_list = [10,20,100]

def run_Study_experiment(class_ratio_list, n_samples_list, n_features_list, balancing_methods, classifiers):
    results_balanced = pd.DataFrame()
    results_imbalanced = pd.DataFrame()

    for class_ratio in class_ratio_list:
        for n_samples in n_samples_list:
            for n_features in n_features_list:

                data_generator = ImbalancedDataGenerator(class_ratio=class_ratio, n_samples=int(n_samples), n_features=n_features, distance=1, flip_y=0.1)

                for method in balancing_methods:

                    balancing_method = balancing_methods[method](sampling_strategy='auto', random_state=123)
                    data_balancer = DataBalancer(balancer=balancing_method)
                    
                    for classifier in classifiers:

                        meta_df = pd.DataFrame({
                            "class_ratio": class_ratio, 
                            "n_samples": n_samples, 
                            "n_features": n_features, 
                            "balancing_method": method, 
                            "classifier": classifier
                        }, index=[0])

                        logger.info(f"""({class_ratio},{int(n_samples)}) | {n_features} | {method} | {classifier}""")
                        
                        classification_method = classifiers[classifier](random_state=123)
                        data_classifier = Classifier(classifier=classification_method)
                        

                        study = Study(
                            data_generator=data_generator,
                            data_balancer=data_balancer,
                            data_classifier=data_classifier
                        )
                        study.run()
                        results = study.calculate_metrics()
                        
                        results_imbalanced = pd.concat(
                            [
                                results_imbalanced,
                                meta_df.join(pd.DataFrame(results['imbalanced_results'], index=[0]))
                            ]
                        )
                        results_imbalanced = results_imbalanced.reset_index(drop=True)
                        
                        results_balanced = pd.concat(
                            [
                                results_balanced,
                                meta_df.join(pd.DataFrame(results['balanced_results'], index=[0]))
                            ]
                        )
                        results_balanced = results_balanced.reset_index(drop=True)

                        #print('Done: ', class_ratio, n_samples, n_features, method, classifier)
                        
                #results_balanced.to_csv('results_balanced.csv')
                #results_imbalanced.to_csv('results_imbalanced.csv')
                        
    print(results_balanced)
    print(results_imbalanced)
import pandas as pd
import numpy as np
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
from Data_Generator import ImbalancedDataGenerator, Multi_Modal_Dist_Generator
from Data_Balancer import DataBalancer, IterDataBalancer, DictIterDataBalancer
from Classifier import Classifier, IterClassifier, DictIterClassifier
from Assessors import Study, Metrics
from parameters import mixed_3d_test_dict



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








def run_iter_experiment(generator_dict, balancing_methods, classifiers_dict, results_df = None):
    
    data_generator = Multi_Modal_Dist_Generator(**generator_dict)
    X_train, X_test, y_train, y_test = data_generator.prepare_data()

    balancers = [(name, method(sampling_strategy='auto', random_state=123)) 
                 if method != None else (name, method)
                 for name, method in balancing_methods.items()]
    

    iter_data_balancer = IterDataBalancer(balancers = [balancer for name, balancer in balancers])
    
    balanced_data = iter_data_balancer.balance_data(X_train, y_train)

    for X_bal, y_bal in balanced_data:
        # Initialize the classifiers, e.g., Support Vector Machine (SVC)
        classifiers = [(name, classifier(random_state = 42))
                       for name, classifier in classifiers_dict.items()]
        
        iter_classifier = IterClassifier(classifiers = [classifier for name, classifier in classifiers])
        # Fit the model on the data
        iter_classifier.fit(X_bal, y_bal)

        # Make predictions
        predictions_list = iter_classifier.predict(X_test)

        for ind, predictions in enumerate(predictions_list):
            print(f'Accuracy of {classifiers[ind][0]}:', np.sum(y_test == predictions)/ len(y_test))







def run_dict_iter_experiment(generator_dict, balancing_methods, classifiers_dict, results_df = None):
    
    data_generator = Multi_Modal_Dist_Generator(**generator_dict)
    X_train, X_test, y_train, y_test = data_generator.prepare_data()


    dict_iter_data_balancer = DictIterDataBalancer(balancers_dict = balancing_methods)
    
    balanced_data = dict_iter_data_balancer.balance_data(X_train, y_train)

    for bal_name, X_bal, y_bal in balanced_data:
        
        dict_iter_classifier = DictIterClassifier(classifiers_dict = classifiers_dict)
        # Fit the model on the data
        dict_iter_classifier.fit(X_bal, y_bal)

        # Make predictions
        predictions_list = dict_iter_classifier.predict(X_test)

        for clsf_name, predictions in predictions_list:
            print(f'Accuracy of {clsf_name} after balancing with {bal_name}:', np.sum(y_test == predictions)/ len(y_test))



"""
Execute
-------------------------------------------------------------------------------------------------------------------------------------------
"""
balancing_methods = {
#"Unbalanced": None,
"ADASYN": ADASYN,
#"RandomOverSampler": RandomOverSampler,
#"KMeansSMOTE": KMeansSMOTE,
"SMOTE": SMOTE,
#"BorderlineSMOTE": BorderlineSMOTE,
#"SVMSMOTE": SVMSMOTE,
#"SMOTENC": SMOTENC,
}

classifiers = {
    #"Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    #"SVC": SVC,
    #"Naive Bayes": GaussianNB,
    "XGboost": XGBClassifier,
    "Lightgbm": LGBMClassifier
}
class_ratio_list = [0.1, 0.01, 0.01]
n_samples_list = [10e2, 10e3, 10e4]
n_features_list = [10,20,50]


run_Study_experiment(class_ratio_list[:1], n_samples_list[:1], n_features_list[:1], balancing_methods, classifiers)
#run_iter_experiment(mixed_3d_test_dict, balancing_methods, classifiers)
#run_dict_iter_experiment(mixed_3d_test_dict, balancing_methods, classifiers)
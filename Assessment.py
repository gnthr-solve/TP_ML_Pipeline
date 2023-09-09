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
from Metrics import IterMetrics
from Assessors import CorStudy
from gen_parameters import extract_table_info, create_simple_normal_dict_list,  mixed_3d_test_dict



def run_CorStudy_experiment(class_ratio_list, n_samples_list, n_features_list, balancing_methods, classifiers, exp_title):

    results_df = pd.DataFrame()

    for class_ratio in class_ratio_list:
        for n_samples in n_samples_list:
            for n_features in n_features_list:

                data_generator = ImbalancedDataGenerator(class_ratio=class_ratio, n_samples=int(n_samples), n_features=n_features, distance=1, flip_y=0.1)

                for method in balancing_methods:

                    if method != 'Unbalanced':
                        balancing_method = balancing_methods[method](sampling_strategy='auto', random_state=123)
                    else:
                        balancing_method = balancing_methods[method]

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
                        

                        study = CorStudy(
                            data_generator=data_generator,
                            data_balancer=data_balancer,
                            data_classifier=data_classifier
                        )
                        study.run()
                        results = study.calculate_metrics()
                        
                        results_df = pd.concat(
                            [
                                results_df,
                                meta_df.join(pd.DataFrame(results, index=[0]))
                            ]
                        )
                        results_df = results_df.reset_index(drop=True)
                        
                    

                        #print('Done: ', class_ratio, n_samples, n_features, method, classifier)

    print(results_df)
    results_df.to_csv(f'{exp_title}.csv')
                
                        
    
    







def run_dict_iter_experiment(generator_dict_list, balancing_methods, classifiers_dict, results_df = pd.DataFrame()):

    experiment_sizes = [len(generator_dict_list), len(balancing_methods), len(classifiers_dict)]
    
    for generator_dict in generator_dict_list:

        metrics_dfs = []
        
        dataset_table_info = extract_table_info(generator_dict)
        meta_dict = {
        'n_features': [dataset_table_info[0] for _ in range(np.prod(experiment_sizes[1:]))],
        'n_samples': [dataset_table_info[1] for _ in range(np.prod(experiment_sizes[1:]))],
        'class_ratio': [dataset_table_info[2] for _ in range(np.prod(experiment_sizes[1:]))],
        'balancing_method': [],
        'classifier': [],
        }

        data_generator = Multi_Modal_Dist_Generator(**generator_dict)
        X_train, X_test, y_train, y_test = data_generator.prepare_data()

        
        dict_iter_data_balancer = DictIterDataBalancer(balancers_dict = balancing_methods)
        
        balanced_data = dict_iter_data_balancer.balance_data(X_train, y_train)

        for bal_name, X_bal, y_bal in balanced_data:

            meta_dict["balancing_method"].extend([bal_name for _ in range(experiment_sizes[2])])
            
            dict_iter_classifier = DictIterClassifier(classifiers_dict = classifiers_dict)
            
            dict_iter_classifier.fit(X_bal, y_bal)
            predictions_dict_list = dict_iter_classifier.predict(X_test)
            

            meta_dict["classifier"].extend([predictions_dict['name'] for predictions_dict in predictions_dict_list])

            metrics = IterMetrics(X_test, y_test, predictions_dict_list)

            metrics_dfs.append(pd.DataFrame(metrics.confusion_metrics()))
            
    
        meta_df = pd.DataFrame(meta_dict)

        metrics_df = pd.concat(metrics_dfs).reset_index(drop=True)

        results_df = pd.concat(
                        [
                            results_df,
                            meta_df.join(metrics_df)
                        ]
                    )
        results_df = results_df.reset_index(drop=True)
        #print(results_df)
    
    return results_df





"""
Execute
-------------------------------------------------------------------------------------------------------------------------------------------
"""
balancing_methods = {
"Unbalanced": None,
"ADASYN": ADASYN,
#"RandomOverSampler": RandomOverSampler,
#"KMeansSMOTE": KMeansSMOTE,
"SMOTE": SMOTE,
"BorderlineSMOTE": BorderlineSMOTE,
#"SVMSMOTE": SVMSMOTE,
#"SMOTENC": SMOTENC,
}

classifiers = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    #"SVC": SVC,
    #"Naive Bayes": GaussianNB,
    "XGboost": XGBClassifier,
    #"Lightgbm": LGBMClassifier
}
class_ratio_list = [0.1, 0.01, 0.001]
n_samples_list = [10e2, 10e3, 10e4]
n_features_list = range(5, 50, 5)
class_distance_list = [3,2,1]


#Uncomment to run first pipeline
#run_CorStudy_experiment(class_ratio_list[:1], n_samples_list[1:2], n_features_list[:], balancing_methods, classifiers, 'feature_range_experiment')


#Create a dictionary list for the generator with experiment parameters for a standard multivariate normal with variance 1*I
gen_dict_list = create_simple_normal_dict_list(n_samples_list, n_features_list, class_ratio_list, class_distance_list)

#Run second pipeline approach with dictionary list for alternative generator
results_df = run_dict_iter_experiment(gen_dict_list, balancing_methods, classifiers)

print(results_df)
#results_df.to_csv('YourTitleHere.csv')
import pandas as pd
import numpy as np
import sys

from loguru import logger
from imblearn.over_sampling import ADASYN, RandomOverSampler, KMeansSMOTE, SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC, RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from Assessors import Assessor
from gen_parameters import presentation_experiment_dict
from helper_tools import extract_table_info, create_simple_normal_dict_list




"""
Methods
-------------------------------------------------------------------------------------------------------------------------------------------
"""
balancing_methods = {
"Unbalanced": None,
"ADASYN": ADASYN,
"RandomOverSampler": RandomOverSampler,
"KMeansSMOTE": KMeansSMOTE,
"SMOTE": SMOTE,
"BorderlineSMOTE": BorderlineSMOTE,
"SVMSMOTE": SVMSMOTE,
#"SMOTENC": SMOTENC,
}

classifiers_dict = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    #"SVC": SVC,
    #"Naive Bayes": GaussianNB,
    "XGboost": XGBClassifier,
    "Lightgbm": LGBMClassifier
}



"""
Parameters
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class_ratio_list = [0.1, 0.01, 0.001]
n_samples_list = [10e3, 10e4, 10e5]
n_features_list = range(2, 10, 2)
class_distance_list = [5, 4, 3, 2.5, 2, 1.5, 1]


"""
Class distance
-------------------------------------------------------------------------------------------------------------------------------------------


for distance in class_distance_list[6:]:

    results_df = pd.read_csv('Experiments/cls_dist_std_mv_normal.csv', index_col=0)
    #print(results_df)

    gen_dict_list = create_simple_normal_dict_list(n_samples_list[1:2], n_features_list[:2], class_ratio_list[:2], [distance])
    print([extract_table_info(gen_dict) for gen_dict in gen_dict_list])

    assessor = Assessor(0.2, gen_dict_list, balancing_methods, classifiers_dict)

    assessor.generate()
    assessor.balance()
    assessor.clsf_pred()

    new_results_df = assessor.calc_std_metrics()
    
    new_results_df['cluster distance'] = [distance for _ in range(len(new_results_df))]

    
    #print(new_results_df)
    results_df = pd.concat([results_df, new_results_df],
                           ignore_index=True,
                           axis = 0).reset_index(drop=True)
    
    print(results_df)
    results_df.to_csv('Experiments/cls_dist_std_mv_normal.csv')

"""



"""
Presentation Experiment
-------------------------------------------------------------------------------------------------------------------------------------------

assessor = Assessor(0.2, [presentation_experiment_dict], balancing_methods, classifiers_dict)

assessor.generate()
assessor.balance()
assessor.clsf_pred()

results_df = assessor.calc_std_metrics()
print(results_df)


results_df.to_csv('Experiments/presentation_results.csv')


#All calibration curves
#-------------------------------------------------------------------------------------------------------------------------------------------
doc_dict = {
        "n_features": False,
        "n_samples": False,
        "class_ratio": False, 
        "doc_string": False,
        "balancer": True, 
        "classifier": True
    }

assessor.create_confusion_plots(doc_dict, feature1=0, feature2=2)
assessor.create_calibration_curves(doc_dict, spline = True, save = False, title = f'All Calibration Curves')
assessor.create_decision_curves(doc_dict = doc_dict, m=20, save = False, title = f'All Decision Curves')

"""


"""
Presentation Experiment - Balancer Specific Calibration Curves
-------------------------------------------------------------------------------------------------------------------------------------------
"""

balancing_methods = {
"Unbalanced": None,
"ADASYN": ADASYN,
"RandomOverSampler": RandomOverSampler,
"KMeansSMOTE": KMeansSMOTE,
"SMOTE": SMOTE,
"BorderlineSMOTE": BorderlineSMOTE,
"SVMSMOTE": SVMSMOTE,
#"SMOTENC": SMOTENC,
}

classifiers_dict = {
"Decision Tree": DecisionTreeClassifier,
"Random Forest": RandomForestClassifier,
"Logistic Regression": LogisticRegression,
#"SVC": SVC,
#"Naive Bayes": GaussianNB,
"XGboost": XGBClassifier,
"Lightgbm": LGBMClassifier
}

m = 20
for name, bal in balancing_methods.items():

    balancer_dict = {name: bal}
    #print(balancer_dict)
    assessor = Assessor(0.2, [presentation_experiment_dict], balancer_dict, classifiers_dict)
    
    assessor.generate()
    assessor.balance()
    assessor.clsf_pred()

    #calibration curves test
    #-------------------------------------------------------------------------------------------------------------------------------------------
    doc_dict = {
            "n_features": False,
            "n_samples": False,
            "class_ratio": False, 
            "doc_string": False,
            "balancer": False, 
            "classifier": True
        }
    
    feature_map = {
        0: 'Normal Feature',
        1: 'Normal Feature',
        2: 'Beta Feature',
        3: 'Poisson Feature',
        4: 'Gamma Feature',
    }
    
    assessor.create_confusion_plots(doc_dict, feature1=0, feature2=4, feature_map = feature_map, save = False)
    assessor.create_pred_proba_plots(doc_dict, feature1=0, feature2=2, feature_map = feature_map, save = False)
    assessor.create_calibration_curves(doc_dict, spline = True, save = False, title = f'Calibration Curves for {name}')
    assessor.create_decision_curves(doc_dict = doc_dict, m = 20, save = False, title = f'Decision Curves for {name}')


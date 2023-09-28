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
from gen_parameters import mixed_3d_test_dict, mixed_test_dict
from helper_tools import extract_table_info, create_simple_normal_dict_list




"""
Methods
-------------------------------------------------------------------------------------------------------------------------------------------
"""
balancing_methods = {
"Unbalanced": None,
"ADASYN": ADASYN,
"RandomOverSampler": RandomOverSampler,
#"KMeansSMOTE": KMeansSMOTE,
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
First run standard MV-Normal class distance
-------------------------------------------------------------------------------------------------------------------------------------------

#gen_dict_list = [mixed_3d_test_dict, mixed_test_dict]

#for n_samples in n_samples_list:

gen_dict_list = create_simple_normal_dict_list(n_samples_list[:1], n_features_list[:], class_ratio_list[:2], class_distance_list[:1])
#print(gen_dict_list)

assessor = Assessor(0.2, gen_dict_list, balancing_methods, classifiers_dict)

assessor.generate()
assessor.balance()
assessor.clsf_pred()

results_df = assessor.calc_metrics()
results_df['cluster distance'] = [class_distance_list[0] for _ in range(len(results_df))]
print(results_df)
results_df.to_csv('Experiments/cls_dist_std_mv_normal.csv')
"""


"""
First Run class distance
-------------------------------------------------------------------------------------------------------------------------------------------
"""

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


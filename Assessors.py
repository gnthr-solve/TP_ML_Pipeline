from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
from itertools import product
from helper_tools import Data, extract_table_info, calculate_no_samples
from Data_Generator import FMLP_Generator
from Data_Balancer import FMLP_DataBalancer
from Classifier import FMLP_DataClassifier
from Metrics import FMLP_Metrics




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













class Assessor(Data):

    def __init__(self, test_size, generation_dict_list, balancers_dict, classifiers_dict):

        Data.data_dict = {}

        self.test_size = test_size
        self.generation_dict_list = generation_dict_list

        balancer_list = [(name, balancer) for name, balancer in balancers_dict.items()]
        
        clsf_list = [(name, classifier) for name, classifier in classifiers_dict.items()]

        self.exp_dim = (len(generation_dict_list), len(balancers_dict), len(classifiers_dict))
        
        self.data_dict['assignment_dict'] = {(a, b, c): [gen_dict, bal, clsf]
                                             for (a, gen_dict), (b, bal), (c, clsf)
                                             in product(enumerate(generation_dict_list), 
                                                        enumerate(balancer_list), 
                                                        enumerate(clsf_list)
                                                        )
                                            }


    def generate(self):     

        test_size = self.test_size
        table_infos = [extract_table_info(generation_dict) for generation_dict in self.generation_dict_list]

        self.d = max([info[0] for info in table_infos])
        n = max([info[1] for info in table_infos])
        a = self.exp_dim[0]
        
        trainset_size = int((1-test_size)*n)
        testset_size = int(test_size*n)
        print('Size X array: \n', a*n*self.d)

        self.data_dict['org_X_train'] = np.full(shape = (a, trainset_size, self.d), fill_value = np.nan)
        self.data_dict['org_y_train'] = np.full(shape = (a, trainset_size,), fill_value = np.nan)

        self.data_dict['org_X_test'] = np.full(shape = (a, testset_size, self.d), fill_value = np.nan)
        self.data_dict['org_y_test'] = np.full(shape = (a, testset_size,), fill_value = np.nan)

        for i, generation_dict in enumerate(self.generation_dict_list):
            generation_dict['gen_index'] = i
            generator = FMLP_Generator(**generation_dict)
            generator.prepare_data(self.test_size)

        #print(self.data_dict)
        #print({key: np.shape(value) for key, value in self.data_dict.items()})

        #print((n,d))


    def balance(self, bal_params_dicts = {}):

        a, b, c = self.exp_dim
        y_train = self.data_dict['org_y_train']

        default_strategy = 'auto'
        max_c1 = max([np.sum(y_train[data_ind] == 1) for data_ind in range(a)])
        max_total_samples = 0

        for data_ind in range(a):
            #check if a balancer parameter dict is given
            if bal_params_dicts:
                #iterate over the parameter dictionaries that are given
                for bal_dict in bal_params_dicts.values():

                    max_total_samples = max(max_total_samples, sum(calculate_no_samples(y_train[data_ind], bal_dict['sampling_strategy']).values()))

            #compare to default strategy if not every balancer has a dict
            if len(bal_params_dicts) < b:
                max_total_samples = max(max_total_samples, sum(calculate_no_samples(y_train[data_ind], default_strategy).values()))
        
        k = max_c1 + max_total_samples

        #k = 2 * max([np.sum(self.data_dict['org_y_train'][data_ind] == 0) for data_ind in range(a)]) + 1
        #print(k)
        #print(np.shape(self.data_dict['org_y_train']))
        print('Number of individual balancing steps: \n', a*b,
              'Size balance array X: \n', a*b*k*self.d)

        self.data_dict['bal_X_train'] = np.full(shape = (a, b, k, self.d), fill_value = np.nan)
        self.data_dict['bal_y_train'] = np.full(shape = (a, b, k, ), fill_value = np.nan)

        data_balancer = FMLP_DataBalancer(bal_params_dicts)
        
        data_balancer.balance_data()

        #print(self.data_dict['bal_y_train'])
        #print(self.data_dict['pos_doc'])
        #print({key: np.shape(value) for key, value in self.data_dict.items()})
        


    def clsf_pred(self):

        a, n = np.shape(self.data_dict['org_y_test'])

        self.data_dict['clsf_predictions_y'] = np.full(shape = self.exp_dim + (n,), fill_value = np.nan)
        self.data_dict['clsf_predictions_proba'] = np.full(shape = self.exp_dim + (n, 2), fill_value = np.nan)
        self.data_dict['classes_order'] = np.full(shape = self.exp_dim + (2,), fill_value = np.nan)

        print('Size classifier array: \n', self.exp_dim[0]*self.exp_dim[1]*self.exp_dim[2]*n*2)
        data_classifier = FMLP_DataClassifier()

        data_classifier.fit()

        data_classifier.predict()

        #print({key: np.shape(value) for key, value in self.data_dict.items()})
        #print(self.data_dict['classes_order'])



    def calc_std_metrics(self, std_metrics_dict = {}):

        default_metrics = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'F1 score': f1_score,
            'ROC AUC Score': roc_auc_score,
        }

        metrics_dict = std_metrics_dict or default_metrics


        self.data_dict['std_metrics_res'] = np.full(shape = self.exp_dim + (len(metrics_dict),), fill_value = np.nan)
        
        metrics = FMLP_Metrics(metrics_dict)

        metrics.confusion_metrics()

        std_metrics_res = self.data_dict['std_metrics_res'].reshape(-1, len(metrics_dict))

        results_df = pd.DataFrame(std_metrics_res, columns= [name for (name, metr_func) in metrics.std_metric_list])

        reference_list = [self.data_dict['assignment_dict'][(i, j, k)] 
                          for i in range(self.exp_dim[0]) 
                          for j in range(self.exp_dim[1]) 
                          for k in range(self.exp_dim[2])]
        
        #print(reference_list)

        reference_list = [extract_table_info(alist[0])+[alist[1][0], alist[2][0]] for alist in reference_list]
        #print(reference_list)

        reference_df = pd.DataFrame(reference_list, columns= ['n_features', 
                                                              'n_samples', 
                                                              'class_ratio', 
                                                              'distributions', 
                                                              'balancer', 
                                                              'classifier'])

        results_df = pd.concat([reference_df, results_df], axis = 1)
        
        #print({key: np.shape(value) for key, value in self.data_dict.items()})
        #print(results_df)
        #print(self.data_dict['classes_order'])

        return results_df
    







if __name__=="__main__":

    import pandas as pd
    from imblearn.over_sampling import ADASYN,RandomOverSampler,KMeansSMOTE,SMOTE,BorderlineSMOTE,SVMSMOTE,SMOTENC, RandomOverSampler
    from Classifier import Classifier, DictIterClassifier
    from Data_Generator import Multi_Modal_Dist_Generator
    from Visualiser import RawVisualiser
    from gen_parameters import mixed_3d_test_dict, mixed_test_dict
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
    #"Lightgbm": LGBMClassifier
    }

    """
    Assessor Testcase
    -------------------------------------------------------------------------------------------------------------------------------------------
    """
    assessor = Assessor(0.2, [mixed_3d_test_dict, mixed_test_dict], balancing_methods, classifiers_dict)

    assessor.generate()
    assessor.balance()
    assessor.clsf_pred()
    assessor.calc_metrics()
    
    
    

    
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np
import scipy.stats as st
import plotly.express as px
from scipy.stats import linregress
#from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from helper_tools import extract_table_info, calculate_no_samples
from itertools import product
import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds")
        return result
    return wrapper


class Data():

    data_dict = {}




class Assessor(Data):

    @timing_decorator
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

    @timing_decorator
    def generate(self):     

        test_size = self.test_size
        table_infos = [extract_table_info(generation_dict) for generation_dict in self.generation_dict_list]

        self.d = max([info[0] for info in table_infos])
        n = max([info[1] for info in table_infos])
        a = self.exp_dim[0]
        
        trainset_size = int((1-test_size)*n)
        testset_size = int(test_size*n)

        self.data_dict['org_X_train'] = np.full(shape = (a, trainset_size, self.d), fill_value = np.nan)
        self.data_dict['org_y_train'] = np.full(shape = (a, trainset_size,), fill_value = np.nan)

        self.data_dict['org_X_test'] = np.full(shape = (a, testset_size, self.d), fill_value = np.nan)
        self.data_dict['org_y_test'] = np.full(shape = (a, testset_size,), fill_value = np.nan)

        for i, generation_dict in enumerate(self.generation_dict_list):
            generation_dict['gen_index'] = i
            generator = Generator(**generation_dict)
            generator.prepare_data(self.test_size)

        #print(self.data_dict)
        #print({key: np.shape(value) for key, value in self.data_dict.items()})

        #print((n,d))


    @timing_decorator
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
        print('Number of individual balancing steps: \n', a*b, '\n',
              'Size balance array X: \n', a*b*k*self.d)

        self.data_dict['bal_X_train'] = np.full(shape = (a, b, k, self.d), fill_value = np.nan)
        self.data_dict['bal_y_train'] = np.full(shape = (a, b, k, ), fill_value = np.nan)

        data_balancer = DataBalancer(bal_params_dicts)

        data_balancer.balance_data()

        #print(self.data_dict['bal_y_train'])
        #print(self.data_dict['pos_doc'])
        #print({key: np.shape(value) for key, value in self.data_dict.items()})
        

    @timing_decorator
    def clsf_pred(self):

        a, n = np.shape(self.data_dict['org_y_test'])

        self.data_dict['clsf_predictions_y'] = np.full(shape = self.exp_dim + (n,), fill_value = np.nan)
        self.data_dict['clsf_predictions_proba'] = np.full(shape = self.exp_dim + (n, 2), fill_value = np.nan)
        self.data_dict['classes_order'] = np.full(shape = self.exp_dim + (2,), fill_value = np.nan)

        print('Number of classifiers to train: \n', self.exp_dim[0]*self.exp_dim[1]*self.exp_dim[2], '\n',
              'Size classifier array: \n', self.exp_dim[0]*self.exp_dim[1]*self.exp_dim[2]*n*2)
        data_classifier = DataClassifier()

        data_classifier.fit()

        data_classifier.predict()

        #print({key: np.shape(value) for key, value in self.data_dict.items()})
        #print(self.data_dict['classes_order'])


    @timing_decorator
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
        
        metrics = Metrics(metrics_dict)

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


    @timing_decorator
    def create_calibration_curves(self, m = 10, save = False, title = f'Calibration Curves'):

        metrics = Metrics({})

        metrics.calibration_curve_fmlp(m = 10, save = False, title = f'Calibration Curves')





class Generator(Data):
    
    def __init__(self, distributions, params_dict_list, sizes, gen_index, random_state = 1234):
        self.sizes = sizes
        self.dists = distributions
        self.params_dict_list = params_dict_list
        self.gen_index = gen_index
        self.random_state = random_state


    def create_data(self):
        
        self.dists_sample_lists = {'c0': [], 'c1': []}
        
        for l, parameters_dict in enumerate(self.params_dict_list):
            #print(parameters_dict.values())

            for i in range(2):

                if (modes:=parameters_dict[f'modes_c{i}']) > 1:
                    
                    self.create_multimodal_features(c_id = i, 
                                                    dist = self.dists[l], 
                                                    params_dict = parameters_dict[f'params_c{i}'], 
                                                    modes = modes, 
                                                    mixing_weights = parameters_dict[f'mixing_weights_c{i}'])

                else:
                    self.create_unimodal_features(c_id = i, 
                                                  dist = self.dists[l], 
                                                  params_dict = parameters_dict[f'params_c{i}'] )
        

        self.dists_sample_lists = {key: [array for array in sample_features_list]
                                   for key, sample_features_list in self.dists_sample_lists.items()}
        
        X_c0 = np.concatenate(self.dists_sample_lists['c0'], axis = 1)
        X_c1 = np.concatenate(self.dists_sample_lists['c1'], axis = 1)
        #print(X_c1)

        y_c0 = np.zeros(self.sizes[0])
        y_c1 = np.ones(self.sizes[1])

        self.X = np.concatenate( (X_c0, X_c1), axis = 0)
        self.y = np.concatenate( (y_c0, y_c1), axis = 0)

        # Generate a random permutation of indices
        permuted_indices = np.random.permutation(len(self.X))

        self.X = self.X[permuted_indices]
        self.y = self.y[permuted_indices]
                


    def create_multimodal_features(self, c_id, dist, params_dict, modes, mixing_weights):

        size = self.sizes[c_id]

        k = modes

        multinomial = st.multinomial(size, mixing_weights)
        comp_sizes = multinomial.rvs(size = 1)[0]

        acc_list = []

        for i in range(k):

            params = {key: value[i] for key, value in params_dict.items()}

            frozen_dist = dist(**params)

            acc_list.append(frozen_dist.rvs(size = comp_sizes[i]))

        feature_samples = np.concatenate( acc_list, axis = 0)
        feature_samples = feature_samples.reshape(size, -1)

        self.dists_sample_lists[f'c{c_id}'].append(feature_samples)
        


    def create_unimodal_features(self, c_id, dist, params_dict):

        size = self.sizes[c_id]

        k = len(next(iter(params_dict.values())))

        sample_features_list = []

        for i in range(k):

            params = {key: value[i] for key, value in params_dict.items()}

            frozen_dist = dist(**params)

            sample_features_list.append(frozen_dist.rvs(size = size))

        sample_features_list = [sample.reshape(size, -1) for sample in sample_features_list]

        self.dists_sample_lists[f'c{c_id}'].extend(sample_features_list)

            
    
    def prepare_data(self, test_size = 0.2):
        
        self.create_data()
        
        X_train, X_test, y_train, y_test= train_test_split(
                                                            self.X, 
                                                            self.y, 
                                                            test_size = test_size, 
                                                            random_state=self.random_state
                                                            )
        
        n_train, d = np.shape(X_train)
        n_test = np.shape(y_test)[0]
        #print(n_train, d, n_test)

        self.data_dict['org_X_train'][self.gen_index, :n_train, :d] = X_train
        self.data_dict['org_y_train'][self.gen_index, :n_train] = y_train

        self.data_dict['org_X_test'][self.gen_index, :n_test, :d] = X_test
        self.data_dict['org_y_test'][self.gen_index, :n_test] = y_test

        







class DataBalancer(Data):

    def __init__(self, bal_params_dict = {}):

        self.balancer_dict = {(i,j): assign_list[1] for (i,j,k), assign_list in self.data_dict['assignment_dict'].items()}
        
        default_dict = {'sampling_strategy': 'auto', 'random_state': 42}
        self.bal_params_dict = {name: bal_params_dict[name]
                                if name in bal_params_dict else default_dict
                                for (name, bal) in self.balancer_dict.values()}
        
        

    def balance_data(self):

        X = self.data_dict['org_X_train']
        y = self.data_dict['org_y_train']

        for (data_ind, bal_ind), (name, balancer) in self.balancer_dict.items():

            #n_train, d_train = train_shape_list[data_ind]

            X_bal = X[data_ind]
            y_bal = y[data_ind]
            #print("Original X shape: \n", np.shape(X_bal))
            #print("Original y shape: \n", np.shape(y_bal))
            #print("Original X: \n", X_bal[:5])
            #print("Original y: \n", y_bal[:5])
            
            # Drop rows with NaN values
            X_bal = X_bal[~np.isnan(X_bal).all(axis = 1)]
            # Drop columns with NaN values
            X_bal = X_bal[: , ~np.isnan(X_bal).all(axis = 0)]
            y_bal = y_bal[~np.isnan(y_bal)]
            #print("Filtered X shape: \n", np.shape(X_bal))
            #print("Filtered y shape: \n", np.shape(y_bal))
            #print("Filtered X: \n", X_bal[:5])
            #print("Filtered y: \n", y_bal[:5])


            if balancer == None:
                resample = (X_bal,y_bal)

            else:
                balancer = balancer(**self.bal_params_dict[name])
                resample = balancer.fit_resample(X_bal, y_bal)

            n, d = np.shape(resample[0])
            #self.data_dict['bal_shape_dict'][(data_ind, bal_ind)] = (n, d)
            #print(n, d)
            
            self.data_dict['bal_X_train'][data_ind, bal_ind, :n, :d] = resample[0]
            self.data_dict['bal_y_train'][data_ind, bal_ind, :n] = resample[1]

            




class DataClassifier(Data):

    def __init__(self, clsf_params_dict = {}):

        default_dict = {'random_state': 42}
        classifier_dict = {key: assign_list[2] for key, assign_list in self.data_dict['assignment_dict'].items()}
        
        classifier_dict = {key: (name, clsf(**clsf_params_dict[name]))
                           if name in clsf_params_dict else (name, clsf(**default_dict))
                           for key, (name, clsf) in classifier_dict.items()}
        
        self.classifier_dict = classifier_dict



    def fit(self):

        X = self.data_dict['bal_X_train']
        y = self.data_dict['bal_y_train']
        
        for (i,j,k), (name, clsf) in self.classifier_dict.items():
            
            X_fit = X[i, j, :, :]
            y_fit = y[i, j, :]

            # Drop rows with NaN values
            X_fit = X_fit[~np.isnan(X_fit).all(axis = 1)]
            # Drop columns with NaN values
            X_fit = X_fit[: , ~np.isnan(X_fit).all(axis = 0)]
            
            y_fit = y_fit[~np.isnan(y_fit)]
            
            self.classifier_dict[(i,j,k)] = (name, clsf.fit(X_fit, y_fit))


        return self
    

    def predict(self):

        X = self.data_dict['org_X_test']

        for (i,j,k), (name, clsf) in self.classifier_dict.items():
            
            X_test = X[i, :, :]

            # Drop rows with NaN values
            X_test = X_test[~np.isnan(X_test).all(axis = 1)]
            # Drop columns with NaN values
            X_test = X_test[: , ~np.isnan(X_test).all(axis = 0)]

            n_i = len(X_test)

            self.data_dict['clsf_predictions_y'][i, j, k, : n_i] = clsf.predict(X_test)
            self.data_dict['clsf_predictions_proba'][i, j, k, : n_i, :] = clsf.predict_proba(X_test) 
            self.data_dict['classes_order'][i, j, k, :] = clsf.classes_

    

    





class Metrics(Data):

    def __init__(self, std_metrics_dict):

        self.std_metric_list = [(name, metr_func) for name, metr_func in std_metrics_dict.items()]

        
    def confusion_metrics(self):

        y_test = self.data_dict['org_y_test']
        y_pred = self.data_dict['clsf_predictions_y']

        for (i,j,k) in self.data_dict['assignment_dict']:
            
            y_i_test = y_test[i]
            y_i_test = y_i_test[~np.isnan(y_i_test)]

            y_clsf_pred = y_pred[i, j, k]
            y_clsf_pred = y_clsf_pred[~np.isnan(y_clsf_pred)]

            evaluation = np.array([metr_func(y_i_test, y_clsf_pred) for (name, metr_func) in self.std_metric_list])

            self.data_dict['std_metrics_res'][i, j, k, :] = evaluation



    def net_benefit(self, harm_to_benefit):

        self.NB = self.TP - harm_to_benefit * self.FP



    def calibration_curve_fmlp(self, m = 10, save = False, title = f'Calibration Curves'):
        predicted_proba_raw = self.data_dict['clsf_predictions_proba']
        class_orders = self.data_dict['classes_order']

        names_dict = {key: [*list(map(str, extract_table_info(assign_list[0]))), 
                            assign_list[1][0], 
                            assign_list[2][0]]
                      for key, assign_list in self.data_dict['assignment_dict'].items()}
        
        names_dict = {key: ', '.join(doc_list) for key, doc_list in names_dict.items()}

        creation_dict ={
        'mean_pred_proba': [np.arange(0,1, 1/m)],
        'mean_freq': [np.arange(0,1, 1/m)],
        'name': [['Optimum' for _ in range(m)]]
        }
        for (i,j,k) in self.data_dict['assignment_dict']:

            pred_probabilities = predicted_proba_raw[i, j, k]
            corr_classes = class_orders[i, j, k]

            pred_proba = pred_probabilities[:, np.where(corr_classes == 1)[0]].flatten()
            
            sorted_indices = np.argsort(pred_proba)
            sorted_probabilities = pred_proba[sorted_indices]
            sorted_y_test = self.y_test[sorted_indices]

            binned_probabilities = np.array_split(sorted_probabilities, m)
            binned_y_test = np.array_split(sorted_y_test, m)

            creation_dict['mean_pred_proba'].append([bin.sum()/len(bin) for bin in binned_probabilities])
            creation_dict['mean_freq'].append([bin.sum()/len(bin) for bin in binned_y_test])
            creation_dict['name'].append([names_dict[(i,j,k)] for _ in range(m)])

        
        #print(creation_dict['name'])
        df_creation_dict = {
            'Predicted Prob.': np.concatenate(creation_dict['mean_pred_proba']),
            'Mean Frequency': np.concatenate(creation_dict['mean_freq']),
            'Name': sum(creation_dict['name'], [])
        }

        df = pd.DataFrame(df_creation_dict)
        print(df)

        fig = px.line(df, 
                      x = 'Predicted Prob.', 
                      y = 'Mean Frequency', 
                      color = 'Name', 
                      title = title +f'({m} bins)', 
                      markers = True
                      )
        
        if save:
            fig.write_image(f"Figures/'{title}'.png", 
                            width=1920, 
                            height=1080, 
                            scale=3
                            )
            
        fig.show()

    

    def calibration_curve(self, k = 10):

        predicted_probabilities = [pred_dict['predicted_proba'] for pred_dict in self.predictions_dict_list]
        names = [pred_dict['name'] for pred_dict in self.predictions_dict_list]
        
        creation_dict ={
        'mean_pred_proba': [np.arange(0,1, 1/k)],
        'mean_freq': [np.arange(0,1, 1/k)],
        'name': [['Optimum' for _ in range(k)]]
        }
        for i, pred_proba in enumerate(predicted_probabilities):

            sorted_indices = np.argsort(pred_proba)
            sorted_probabilities = pred_proba[sorted_indices]
            sorted_y_test = self.y_test[sorted_indices]

            binned_probabilities = np.array_split(sorted_probabilities, k)
            binned_y_test = np.array_split(sorted_y_test, k)

            creation_dict['mean_pred_proba'].append([bin.sum()/len(bin) for bin in binned_probabilities])
            creation_dict['mean_freq'].append([bin.sum()/len(bin) for bin in binned_y_test])
            creation_dict['name'].append([names[i] for _ in range(k)])

        
        #print(creation_dict['name'])
        df_creation_dict = {
            'Predicted Prob.': np.concatenate(creation_dict['mean_pred_proba']),
            'Mean Frequency': np.concatenate(creation_dict['mean_freq']),
            'Name': sum(creation_dict['name'], [])
        }

        df = pd.DataFrame(df_creation_dict)

        fig = px.line(df, 
                      x = 'Predicted Prob.', 
                      y = 'Mean Frequency', 
                      color = 'Name', 
                      title = f'Calibration Curve ({k} bins)', 
                      markers = True
                      )
        fig.show()
    


    def calibration_curve_reg(self, k = 10):

        predicted_probabilities = [pred_dict['predicted_proba'] for pred_dict in self.predictions_dict_list]
        names = [pred_dict['name'] for pred_dict in self.predictions_dict_list]
        
        creation_dict ={
        'mean_pred_proba': [np.arange(0,1, 1/k)],
        'mean_freq': [np.arange(0,1, 1/k)],
        'name': [['Optimum' for _ in range(k)]]
        }
        for i, pred_proba in enumerate(predicted_probabilities):

            sorted_indices = np.argsort(pred_proba)
            sorted_probabilities = pred_proba[sorted_indices]
            sorted_y_test = self.y_test[sorted_indices]

            binned_probabilities = np.array_split(sorted_probabilities, k)
            binned_y_test = np.array_split(sorted_y_test, k)

            mean_predicted_proba = [bin.sum()/len(bin) for bin in binned_probabilities]
            res = linregress(
                x = mean_predicted_proba,
                y = [bin.sum()/len(bin) for bin in binned_y_test]
            )

            #creation_dict['mean_pred_proba'].append(mean_predicted_proba)
            creation_dict['mean_pred_proba'].append(np.arange(0,1, 1/k))
            #creation_dict['mean_freq'].append([res.intercept + res.slope * x for x in mean_predicted_proba])
            creation_dict['mean_freq'].append([res.intercept + res.slope * x for x in np.arange(0,1, 1/k)])
            creation_dict['name'].append([names[i] for _ in range(k)])

        
        #print(creation_dict['name'])
        df_creation_dict = {
            'Predicted Prob.': np.concatenate(creation_dict['mean_pred_proba']),
            'Mean Frequency': np.concatenate(creation_dict['mean_freq']),
            'Name': sum(creation_dict['name'], [])
        }

        df = pd.DataFrame(df_creation_dict)

        fig = px.line(df, x = 'Predicted Prob.', y = 'Mean Frequency', color = 'Name', title = f'Calibration Curve with Regression ({k} bins)')
        fig.show()





if __name__=="__main__":

    import pandas as pd
    from loguru import logger
    from imblearn.over_sampling import ADASYN,RandomOverSampler,KMeansSMOTE,SMOTE,BorderlineSMOTE,SVMSMOTE,SMOTENC, RandomOverSampler
    from Visualiser import RawVisualiser
    from gen_parameters import mixed_3d_test_dict, mixed_test_dict
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

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

    visualiser = RawVisualiser()

    """
    Generator Test Case
    -------------------------------------------------------------------------------------------------------------------------------------------
    
    data_generator = Generator(**mixed_test_dict)
    X_train, X_test, y_train, y_test = data_generator.prepare_data(0.2)

    print('X original shape:', np.shape(X_train))
    #print('X:', X_train)
    print('y original shape:', np.shape(y_train))
    """


    

    """
    Assessor test
    -------------------------------------------------------------------------------------------------------------------------------------------
    """
    
    assessor = Assessor(0.2, [mixed_3d_test_dict, mixed_test_dict], balancing_methods, classifiers_dict)

    assessor.generate()
    assessor.balance()
    assessor.clsf_pred()

    #calc_std_metrics() test
    #-------------------------------------------------------------------------------------------------------------------------------------------
    results_df = pd.DataFrame()
    new_results_df = assessor.calc_std_metrics()

    #print(new_results_df)
    results_df = pd.concat([results_df, new_results_df],
                           ignore_index=True,
                           axis = 0).reset_index(drop=True)
    
    print(results_df)
   
    #calibration curves test
    #-------------------------------------------------------------------------------------------------------------------------------------------
    
    assessor.create_calibration_curves()
    
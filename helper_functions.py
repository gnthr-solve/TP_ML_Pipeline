import numpy as np
import scipy.stats as st
import re
from itertools import product

"""
Helper functions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def is_cov_matrix(matrix):

    symmetric = np.allclose(matrix, matrix.T)

    pos_semi_definite = all(np.linalg.eigvals(matrix) > 0)

    return (symmetric and pos_semi_definite)



def extract_table_info(use_param_dict):
    params_dict_list = use_param_dict['params_dict_list']
    
    n_features_list = []
    for params_dict in params_dict_list:
        n_features = 0
        if params_dict['modes_c0'] ==1:
            for param in next(iter(params_dict['params_c0'].values())):
                if isinstance(param, np.ndarray):
                    n_features += np.shape(param)[0]
                else:
                    n_features += 1
            
        else:
            param = next(iter(params_dict['params_c0'].values()))[0]
            #print(param, type(param))
            if isinstance(param, np.ndarray):
                n_features += np.shape(param)[0]
            else:
                n_features += 1
        
        n_features_list.append(n_features)
    
    n_features = sum(n_features_list)
    
    dist_strings = extract_dist_substrings([str(dist) for dist in use_param_dict['distributions']])
    dist_doc_string_list = [dist_string + f' {n_features} feat.'
                        for dist_string, n_features in zip(dist_strings, n_features_list)]
    
    dist_doc_string = ', '.join(dist_doc_string_list)

    sizes = use_param_dict['sizes']
    n_samples = sum(sizes)
    class_ratio = sizes[1]/n_samples

    return (n_features, n_samples, class_ratio, dist_doc_string)



def extract_dist_substrings(strings):
    extracted_substrings = []
    for string in strings:
        match = re.search(r'\.([^\.]+)_gen', string)
        if match:
            extracted_substring = match.group(1)
            extracted_substrings.append(extracted_substring)
    return extracted_substrings



def format_dict_as_string(input_dict):
    # Iterate through the key-value pairs and format them as 'key : value' strings
    formatted_pairs = [f'{key} : {value}' for key, value in input_dict.items()]

    # Join the formatted pairs using ', ' as the separator
    formatted_string = ', '.join(formatted_pairs)

    return formatted_string



def create_simple_normal_dict_list(n_samples_list, n_features_list, class_ratio_list, class_distance_list):
    gen_dict_list = []
    for n_samples, n_features, class_ratio, distance in product(n_samples_list, n_features_list, class_ratio_list, class_distance_list):
        
        gen_dict = {'distributions': [st.multivariate_normal]}
        gen_dict['sizes'] = [int(n_samples * (1-class_ratio)), int(n_samples * class_ratio)]

        mu_c0 = np.zeros(shape = (n_features,))
        mu_c1 = np.sqrt(((distance**2) / n_features) * np.ones(shape = (n_features,)))
        #print(distance, np.linalg.norm(mu_c1))

        sigma_c0 = np.diag(np.ones(shape = (n_features)))
        sigma_c1 = np.diag(np.ones(shape = (n_features)))

        dist_parameter_dict = [{'modes_c0': 1,
                                'modes_c1': 1,
                                'params_c0': {'mean': [mu_c0], 'cov': [sigma_c0]},
                                'params_c1': {'mean': [mu_c1], 'cov': [sigma_c1]}
                                }
        ]

        gen_dict['params_dict_list'] = dist_parameter_dict
        gen_dict_list.append(gen_dict)

    return gen_dict_list






def calculate_no_samples(y, sampling_strategy):
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    if isinstance(sampling_strategy, dict):
        # When sampling_strategy is a dict, the keys correspond to the targeted
        # classes and the values correspond to the desired number of samples for each class
        result = sampling_strategy
    elif isinstance(sampling_strategy, str):
        # When sampling_strategy is a string
        if sampling_strategy == 'auto':
            # When 'auto', resample the minority class until the number of samples 
            # in the minority class is equal to the number of samples in the majority class
            majority_class = max(class_counts, key=class_counts.get)
            result = {k: class_counts[majority_class] for k in class_counts.keys()}
        elif sampling_strategy == 'not minority':
            # Resample all classes but the minority class
            minority_class = min(class_counts, key=class_counts.get)
            result = {k: len(y) - v for k, v in class_counts.items() if k != minority_class}
        elif sampling_strategy == 'not majority':
            # Resample all classes but the majority class
            majority_class = max(class_counts, key=class_counts.get)
            result = {k: len(y) - v for k, v in class_counts.items() if k != majority_class}
        elif sampling_strategy == 'all':
            # Resample all classes
            result = {k: len(y) - v for k, v in class_counts.items()}
        elif sampling_strategy == 'majority':
            # Resample only the majority class
            majority_class = max(class_counts, key=class_counts.get)
            result = {majority_class: len(y) - class_counts[majority_class]}
        elif sampling_strategy == 'minority':
            # Resample only the minority class
            minority_class = min(class_counts, key=class_counts.get)
            result = {minority_class: len(y) - class_counts[minority_class]}
        else:
            raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")
    elif isinstance(sampling_strategy, float):
        # When sampling_strategy is a float, it corresponds to the ratio defined by 
        # N_{rM} = alpha_{us} times N_{m} for undersampling and
        # N_{rm} = alpha_{os} times N_{M} for oversampling
        if sampling_strategy <= 0 or sampling_strategy > 1:
            raise ValueError("When sampling_strategy is a float, it should be in the (0, 1] range.")
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        if len(class_counts) > 2:
            raise ValueError("Float sampling_strategy is only available for binary classification.")
        result = {
            majority_class: int(sampling_strategy * class_counts[majority_class]),
            minority_class: int(sampling_strategy * class_counts[minority_class])
        }
    else:
        raise TypeError(f"Unknown type for sampling_strategy: {type(sampling_strategy).__name__}")
    
    return result









if __name__=="__main__":


    from imblearn.over_sampling import ADASYN,RandomOverSampler,KMeansSMOTE,SMOTE,BorderlineSMOTE,SVMSMOTE,SMOTENC, RandomOverSampler
    from Data_Generator import Multi_Modal_Dist_Generator
    from Visualiser import RawVisualiser
    from gen_parameters import mixed_3d_test_dict
    from Data_Balancer import DictIterDataBalancer

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

    visualiser = RawVisualiser()

    data_generator = Multi_Modal_Dist_Generator(**mixed_3d_test_dict)
    X_train, X_test, y_train, y_test = data_generator.prepare_data(0.2)


    """
    calculate_samples test case
    -------------------------------------------------------------------------------------------------------------------------------------------
    """
    sampling_strategy = 'auto'
    calc_samples = calculate_no_samples(y_train, sampling_strategy)
    print('Calculated sample numbers: \n', sum(calc_samples.values()))
    iter_data_balancer = DictIterDataBalancer(balancers_dict = balancing_methods)
    
    balanced_data = iter_data_balancer.balance_data(X_train, y_train)

    #print(balanced_data)

    for name, X_bal, y_bal in balanced_data:

        print(f'{name}-balanced no 0: \n', np.sum(y_bal == 0), '\n',
              f'{name}-balanced no 1: \n', np.sum(y_bal == 1), '\n',
              f'{name}-balanced size: \n', len(y_bal), '\n'
              )
        print(#f'{name} balanced x-data: \n', X_bal[-20:], '\n',
              #f'{name} balanced y-data: \n', y_bal[-20:], '\n',
              #f'{name} balanced size: \n', len(X_bal), '\n'
              )
        
        common_data = np.isin(X_bal, X_train)
        row_mask = np.all(common_data, axis = 1)
        
        only_balance_X = X_bal[~row_mask]
        only_balance_y = y_bal[~row_mask]

        datasets = [
            (X_train, y_train, 'Train'),
            (only_balance_X, only_balance_y, 'Balanced Train')
        ]

        continue

        #visualiser.plot_2d_scatter((data[0], data[1]),0, 1)
        visualiser.plot_2d_scatter_multiple_datasets_px(datasets, 
                                                        feature1 = 0, 
                                                        feature2 = 1, 
                                                        title = f'Scatter of {name}-balanced Data')
        
        visualiser.plot_3d_scatter_multiple_datasets_px(datasets,
                                                        feature1 = 0, 
                                                        feature2 = 1,
                                                        feature3 = 2,
                                                        title = f'3d Scatter of {name}-balanced Data')

    
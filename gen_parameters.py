import numpy as np
import scipy.stats as st

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
    
    n_features = 0
    for params_dict in params_dict_list:

        if params_dict['modes_c0'] ==1:
            n_features += len(next(iter(params_dict['params_c0'].values()))) 

        else:
            param = next(iter(params_dict['params_c0'].values()))[0]
            #print(param, type(param))
            if isinstance(param, np.ndarray):
                n_features += np.shape(param)[0]
            else:
                n_features += 1
    

    sizes = use_param_dict['sizes']

    n_samples = sum(sizes)
    class_ratio = sizes[1]/n_samples

    return (n_features, n_samples, class_ratio)




def format_dict_as_string(input_dict):
    # Iterate through the key-value pairs and format them as 'key : value' strings
    formatted_pairs = [f'{key} : {value}' for key, value in input_dict.items()]

    # Join the formatted pairs using ', ' as the separator
    formatted_string = ', '.join(formatted_pairs)

    return formatted_string


"""
Bimodal Multinormal + Beta
-------------------------------------------------------------------------------------------------------------------------------------------
"""
#set the parameter dictionary for the MV normal. sigma is the standard deviation
mu_c0_1 = np.array([0,0])
mu_c0_2 = np.array([2,0])
mu_c1_1 = np.array([3,3])
mu_c1_2 = np.array([1,3])
sigma_c0_1 = np.array([[1,0],
                       [0,3]])
sigma_c0_2 = np.array([[1,0],
                       [0,1]])
sigma_c1_1 = np.array([[2,1],
                       [1,1]])
sigma_c1_2 = np.array([[2,1],
                       [1,2]])


distributions = [st.multivariate_normal, st.beta]

#set the parameter dictionaries as a list of dictionaries with parameter dictionaries for classes individually.
dist_parameter_dicts = [{'modes_c0': 2,
                         'modes_c1': 1,
                         'mixing_weights_c0': [0.3, 0.7],
                         'mixing_weights_c1': [0.3, 0.7],
                         'params_c0': {'mean': [mu_c0_1, mu_c0_2], 'cov': [sigma_c0_1, sigma_c0_2]},
                         'params_c1': {'mean': [mu_c1_1], 'cov': [sigma_c1_1]}
                        },
                        {'modes_c0': 1,
                         'modes_c1': 1,
                        #'mixing_weights_c0': [],
                         'params_c0': {'a': [2], 'b': [4]},
                         'params_c1': {'a': [1], 'b': [7]}
                        }
]

sizes = [1900, 100]

mixed_3d_test_dict = {'distributions': distributions,
                      'params_dict_list': dist_parameter_dicts,
                      'sizes': sizes,
                      }



"""
Multiple and Multimodal Distributions: Multinormal + Beta + Exponential Example
-------------------------------------------------------------------------------------------------------------------------------------------
"""
#set the parameter dictionary for the MV normal. sigma is the standard deviation
mu_c0_1 = np.array([0,0])
mu_c0_2 = np.array([2,0])
mu_c1_1 = np.array([3,3])
mu_c1_2 = np.array([1,3])
sigma_c0_1 = np.array([[1,0],
                       [0,3]])
sigma_c0_2 = np.array([[1,0],
                       [0,1]])
sigma_c1_1 = np.array([[2,1],
                       [1,1]])
sigma_c1_2 = np.array([[2,1],
                       [1,2]])


distributions = [st.multivariate_normal, st.beta, st.expon]

#set the parameter dictionaries as a list of dictionaries with parameter dictionaries for classes individually.
dist_parameter_dicts = [{'modes_c0': 2,
                         'modes_c1': 1,
                         'mixing_weights_c0': [0.3, 0.7],
                         'mixing_weights_c1': [0.3, 0.7],
                         'params_c0': {'mean': [mu_c0_1, mu_c0_2], 'cov': [sigma_c0_1, sigma_c0_2]},
                         'params_c1': {'mean': [mu_c1_1], 'cov': [sigma_c1_1]}
                        },
                        {'modes_c0': 1,
                         'modes_c1': 1,
                        #'mixing_weights_c0': [],
                         'params_c0': {'a': [2,3], 'b': [4,5]},
                         'params_c1': {'a': [1,2], 'b': [7,8]}
                        },
                        {'modes_c0': 1,
                         'modes_c1': 1,
                        #'mixing_weights_c0': [],
                         'params_c0': {'loc': [0], 'scale': [1]},
                         'params_c1': {'loc': [0], 'scale': [3]}
                        }
]

sizes = [900, 100]

mixed_test_dict = {'distributions': distributions,
                   'params_dict_list': dist_parameter_dicts,
                   'sizes': sizes,
                   }



"""
Simple Default
-------------------------------------------------------------------------------------------------------------------------------------------
"""
n = 5
mu_c0_1 = np.zeros(shape = (n))
mu_c0_2 = 4*np.ones(shape = (n))
mu_c1_1 = -2 * np.ones(shape = (n))
mu_c1_2 =  2 * np.ones(shape = (n))
sigma_c0_1 = np.array([[1,0,0,0,0],
                       [0,2,0,0,0],
                       [0,0,3,0,0],
                       [0,0,0,3,0],
                       [0,0,0,0,3],])
sigma_c0_2 = np.array([[1,0,0,0,0],
                       [0,1,0,0,0],
                       [0,0,1,0,0],
                       [0,0,0,1,0],
                       [0,0,0,0,1],])

sigma_c1_1 = np.ones(shape=(n,n)) + 2* np.diag(np.ones(shape = (n)))

sigma_c1_2 = np.zeros(shape=(n,n)) + np.diag(np.ones(shape = (n)))


distributions = [st.multivariate_normal, st.beta]

#set the parameter dictionaries as a list of dictionaries with parameter dictionaries for classes individually.
dist_parameter_dicts = [{'modes_c0': 2,
                         'modes_c1': 2,
                         'mixing_weights_c0': [0.3, 0.7],
                         'mixing_weights_c1': [0.6, 0.4],
                         'params_c0': {'mean': [mu_c0_1, mu_c0_2], 'cov': [sigma_c0_1, sigma_c0_2]},
                         'params_c1': {'mean': [mu_c1_1, mu_c1_2], 'cov': [sigma_c1_1, sigma_c1_2]}
                        },
                        {'modes_c0': 1,
                         'modes_c1': 1,
                        #'mixing_weights_c0': [],
                         'params_c0': {'a': [2,2], 'b': [4,2]},
                         'params_c1': {'a': [1,1], 'b': [7,9]}
                        }
]

sizes = [9500, 500]

default_test_dict = {'distributions': distributions,
                     'params_dict_list': dist_parameter_dicts,
                     'sizes': sizes,
                     }



"""
Large Normal
-------------------------------------------------------------------------------------------------------------------------------------------
"""
k = 10
large_normal_test_dict_list = []
for n in range(1, k):
    #set the parameter dictionary for the MV normal. sigma is the standard deviation
    mu_c0_1 = np.zeros(shape = (n))
    mu_c0_2 = 3 * np.ones(shape = (n))
    mu_c1 = [2, -2, 2, -2, 2, -2, 2, -2, 2, -2]

    sigma_c0_1 = np.ones(shape=(n,n)) + np.diag(np.ones(shape = (n)))
    sigma_c0_2 = np.zeros(shape=(n,n)) + np.diag(np.ones(shape = (n)))
    sigma_c1 = np.array([[3, 0, 1, 0, 1, 0, 0, 0, -1, -1],
                        [0, 3, 1, 0, 0, 0, 1, 0, 0, -1],
                        [1, 1, 3, 0, 1, 0, 0, 0, 1, 0],
                        [0, 0, 0, 3, 0, 0, 1, 0, 0, 0],
                        [1, 0, 1, 0, 3, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 3, 1, 0, 0, 0],
                        [0, 1, 0, 1, 0, 1, 3, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 3, 0, 1],
                        [-1, 0, 1, 0, 1, 0, 1, 0, 3, 1],
                        [-1, -1, 0, 0, 0, 0, 0, 1, 1, 3]])



    mu_c1 = mu_c1[:n]
    sigma_c1 = sigma_c1[ :n, :n]

    distributions = [st.multivariate_normal]

    #set the parameter dictionaries as a list of dictionaries with parameter dictionaries for classes individually.
    dist_parameter_dicts = [{'modes_c0': 2,
                            'modes_c1': 1,
                            'mixing_weights_c0': [0.3, 0.7],
                            'mixing_weights_c1': [0.3, 0.7],
                            'params_c0': {'mean': [mu_c0_1, mu_c0_2], 'cov': [sigma_c0_1, sigma_c0_2]},
                            'params_c1': {'mean': [mu_c1], 'cov': [sigma_c1]}
                            },
    ]

    sizes = [9000, 1000]

    large_normal_test_dict_list.append({'distributions': distributions,
                                        'params_dict_list': dist_parameter_dicts,
                                        'sizes': sizes,
                                        })





if __name__=="__main__":

    table_info = extract_table_info(mixed_3d_test_dict)

    #print(table_info)
    #print(extract_table_info(large_normal_test_dict))
    #print(extract_table_info(default_test_dict))

    print(format_dict_as_string(mixed_3d_test_dict))
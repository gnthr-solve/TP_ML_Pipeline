import numpy as np
import scipy.stats as st


"""
Alternative Bimodal Multinormal Experiment for Report
-------------------------------------------------------------------------------------------------------------------------------------------
"""
#set the parameter dictionary for the MV normal. sigma is the standard deviation
n = 15
k = 1
mu_c0_1 = 3*np.ones(shape = (n))
mu_c0_2 = -3*np.ones(shape = (n))
mu_c1 = np.zeros(shape = (n))

sigma_c0_1 =  2* np.diag(np.ones(shape = (n)))
sigma_c0_2 =  0.7* np.diag(np.ones(shape = (n)))

sigma_c1 = np.ones(shape=(n,n)) + np.diag(np.ones(shape = (n)))
#sigma_c1 = -1* np.ones(shape=(n,n)) + 3* np.diag(np.ones(shape = (n)))

distributions = [st.multivariate_normal, st.beta]

#set the parameter dictionaries as a list of dictionaries with parameter dictionaries for classes individually.
dist_parameter_dicts = [{'modes_c0': 2,
                         'modes_c1': 1,
                         'mixing_weights_c0': [0.4, 0.6],
                         'params_c0': {'mean': [mu_c0_1, mu_c0_2], 'cov': [sigma_c0_1, sigma_c0_2]},
                         'params_c1': {'mean': [mu_c1], 'cov': [sigma_c1]}
                        },
]

sizes = [99000, 1000]

alt_experiment_dict = {'distributions': distributions,
                       'params_dict_list': dist_parameter_dicts,
                       'sizes': sizes,
                    }




"""
Presentation Experiment: Bimodal Normal, Beta, Poisson, Gamma
-------------------------------------------------------------------------------------------------------------------------------------------
"""
#set the parameter dictionary for the MV normal. sigma is the standard deviation
mu_c0 = np.array([1.5,1.5])

mu_c1_1 = np.array([0,0])
mu_c1_2 = np.array([3,3])

sigma_c0 = np.array([[1,0],
                     [0,1]])

sigma_c1_1 = np.array([[1,0],
                       [0,1]])
sigma_c1_2 = np.array([[2,-1],
                       [-1,2]])


distributions = [st.multivariate_normal, st.beta, st.poisson, st.gamma]

#set the parameter dictionaries as a list of dictionaries with parameter dictionaries for classes individually.
dist_parameter_dicts = [{'modes_c0': 1,
                         'modes_c1': 2,
                         'mixing_weights_c1': [0.3, 0.7],
                         'params_c0': {'mean': [mu_c0], 'cov': [sigma_c0]},
                         'params_c1': {'mean': [mu_c1_1, mu_c1_2], 'cov': [sigma_c1_1, sigma_c1_2]}
                        },
                        {'modes_c0': 1,
                         'modes_c1': 1,
                         'params_c0': {'a': [7], 'b': [4]},
                         'params_c1': {'a': [2], 'b': [7]}
                        },
                        {'modes_c0': 1,
                         'modes_c1': 1,
                         'params_c0': {'mu': [1]},
                         'params_c1': {'mu': [7]}
                        },
                        {'modes_c0': 1,
                         'modes_c1': 1,
                         'params_c0': {'a': [2], 'scale': [2]},
                         'params_c1': {'a': [3], 'scale': [4]}
                        }
]

sizes = [19000, 1000]

presentation_experiment_dict = {'distributions': distributions,
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

    from helper_tools import extract_table_info, extract_dist_substrings, format_dict_as_string, create_simple_normal_dict_list, is_cov_matrix

    #print(table_info)
    #print(extract_table_info(large_normal_test_dict))
    #print(extract_table_info(default_test_dict))
    #print(mixed_3d_test_dict['distributions'])
    #print(format_dict_as_string(mixed_3d_test_dict))


    """
    String representation test
    -------------------------------------------------------------------------------------------------------------------------------------------
    
    distribution_name_map = {
    'Normal': st.norm,
    'Beta': st.beta,
    'Uniform': st.uniform,
    'Exponential': st.expon,
    'Chi-Square': st.chi2,
    # Add more distributions as needed
    }


    # Get the name of the distribution from the dictionary
    distribution_strings = [str(dist) for dist in mixed_3d_test_dict['distributions']]
    print(distribution_strings)
    distribution_names = []
    for string in distribution_strings:
        #match = re.search(r'\.(.*?)(?=_gen|$)', string)
        #match = re.search(r'[^.]+_([^_]+)_gen', string)
        match = re.search(r'\.([^\.]+)_gen', string)
        if match:
            extracted_substring = match.group(1)
            distribution_names.append(extracted_substring)
    
    #distribution_names = extract_dist_substrings(distribution_strings)
    
    print(distribution_names)
    #print(f"The distribution is: {distribution_name}")
    """

    """
    Distribution names + parameters
    -------------------------------------------------------------------------------------------------------------------------------------------
   
    dist_names = st.distributions.__all__
    #print(dist_names)
    
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        #print(dist)
    
    """


    """
    Bimodal Multinormal Experiment for Report - Parameter Checks
    -------------------------------------------------------------------------------------------------------------------------------------------
    """
    #set the parameter dictionary for the MV normal. sigma is the standard deviation
    n = 15
    k = 1
    mu_c0_1 = np.zeros(shape = (n))
    mu_c0_2 = np.concatenate([4*np.ones(shape = (k)),np.zeros(shape = (n-k))])
    mu_c1 = 4*np.ones(shape = (n))

    sigma_c0_1 =  2* np.diag(np.ones(shape = (n)))
    sigma_c0_2 =  0.7* np.diag(np.ones(shape = (n)))

    sigma_c1 = np.ones(shape=(n,n)) + np.diag(np.ones(shape = (n)))
    #sigma_c1 = -1* np.ones(shape=(n,n)) + 3* np.diag(np.ones(shape = (n)))
    
    print(is_cov_matrix(sigma_c0_1))
    print(is_cov_matrix(sigma_c0_2))
    print(is_cov_matrix(sigma_c1))
    
    
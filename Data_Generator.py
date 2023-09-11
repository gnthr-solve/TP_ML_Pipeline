import pandas as pd
import numpy as np
import scipy.stats as st
import scipy.special as sp
import plotly.express as px
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class ImbalancedDataGenerator:
    def __init__(self, 
                 class_ratio=0.5, 
                 n_samples=1000, 
                 n_features=2, 
                 distance=1.0, 
                 flip_y=0,
                 test_size = 0.2
                ):
        """
        Initialize the ImbalancedDataGenerator.

        Parameters:
        - class_ratio (float): The ratio of the minority class samples to the total samples.
        - n_samples (int): The total number of samples to be generated.
        - n_features (int): The number of features (dimensions) in the dataset.
        - distance (float): The distance between the centers of the two classes. A higher value indicates more separability.
        - random_state (int or RandomState, optional): Seed for random number generation.
        """
        self.class_ratio = class_ratio
        self.n_samples = n_samples
        self.n_features = n_features
        self.distance = distance
        self.random_state = 123
        self.flip_y = flip_y
        self.test_size = test_size

        self.n_minority_samples = int(self.class_ratio * self.n_samples)
        self.n_majority_samples = self.n_samples - self.n_minority_samples

        self.X = None
        self.y = None

    def generate_data(self):
        """
        Generate imbalanced data using sklearn's make_classification method.

        Returns:
        - X (numpy array): Array of shape (n_samples, n_features) containing feature values.
        - y (numpy array): Array of shape (n_samples,) containing class labels.
        """
        self.X, self.y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=int(self.n_features / 2),
            n_redundant=int(self.n_features / 2),
            n_clusters_per_class=1,
            weights=[self.class_ratio, 1 - self.class_ratio],
            flip_y=self.flip_y,
            class_sep=self.distance,
            random_state=self.random_state
        )
        #print(self.X, self.y)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, 
            self.y, 
            test_size=self.test_size, 
            random_state=self.random_state
            )
        return X_train, X_test, y_train, y_test

    def plot_2d(self,feature1: int, feature2: int):

        if self.X is None or self.y is None:
            raise ValueError("Data has not been generated yet. Call generate_data() method first.")

        # Select the first two features for plotting
        x1 = self.X[:, feature1]
        x2 = self.X[:, feature2]

        # Create a DataFrame for plotting
        df = pd.DataFrame({'Feature 1': x1, 'Feature 2': x2, 'Class': self.y})

        # Map class labels to more descriptive names
        class_labels = {0: 'Minority Class', 1: 'Majority Class'}
        df['Class'] = df['Class'].map(class_labels)

        # Create the scatter plot using Plotly
        fig = px.scatter(
            df,
            x='Feature 1',
            y='Feature 2',
            color='Class',
            title='2D Scatter Plot of Imbalanced Data',
            labels={'Feature 1': f'Feature {feature1}', 'Feature 2': f'Feature {feature2}'}
        )
        return fig





class Cont_Dist_Generator:
    
    def __init__(self, distribution, distribution_name, size, random_state = 1234, **parameters):
        self.size = size
        self.dist_name = distribution_name
        self.dist = distribution
        self.params = parameters
        self.random_state = random_state

        self.freeze_distributions()


    def freeze_distributions(self):
        parameters_dict = self.params
        
        dist_list = []

        #n = len(next(iter(parameters_dict.values())))

        for i in range(2):

            param_set = []

            for name, parameters in parameters_dict.items():
                param_set.append(parameters[i])
            

            #create the i-th density
            dist_list.append(self.dist(*param_set))

        self.dist_list = dist_list

    
    def create_data(self):
        
        dist_list = self.dist_list
        parameters_dict = self.params

        self.samples_dict = {}

        for i in range(2):
    
            #go by minority as class 1 (positive) and majority as class 0
            self.samples_dict[f'X_{i}'] = dist_list[i].rvs(size = self.size[i])
            self.samples_dict[f'y_{i}'] = i * np.ones(self.size[i])

            for name, parameters in parameters_dict.items():
                
                self.samples_dict[name + f'_{i}'] = parameters[i]
                
        #return self.samples_dict
    

    def prepare_data(self, test_size):
        
        self.create_data()
        X_0 = self.samples_dict[f'X_0']
        X_1 = self.samples_dict[f'X_1']
        y_0 = self.samples_dict[f'y_0']
        y_1 = self.samples_dict[f'y_1']
        
        X = np.concatenate((X_0, X_1), axis=0)
        y = np.concatenate((y_0, y_1), axis=0)
        
        # Generate a random permutation of indices
        permuted_indices = np.random.permutation(len(X))

        self.X = X[permuted_indices]
        self.y = y[permuted_indices]

        return train_test_split(
            self.X, 
            self.y, 
            test_size = test_size, 
            random_state=self.random_state
            )





class Multi_Dist_Generator:
    
    def __init__(self, distributions, params_dict_list, size, random_state = 1234):
        self.size = size
        self.dists = distributions
        self.params_dict_list = params_dict_list
        self.random_state = random_state

        self.freeze_distributions()

    
    def freeze_distributions(self):
        
        frozen_dists = {'c0': [], 'c1': []}
        
        for l, parameters_dict in enumerate(self.params_dict_list):
            #print(parameters_dict.values())
            k = len(next(iter(parameters_dict['params_c0'].values())))

            for i in range(k):
                params_c0 = {key: value[i] for key, value in parameters_dict['params_c0'].items()}
                params_c1 = {key: value[i] for key, value in parameters_dict['params_c1'].items()}

                frozen_dists['c0'].append(self.dists[l](**params_c0))
                frozen_dists['c1'].append(self.dists[l](**params_c1))

                
        self.frozen_dists = frozen_dists

    

    def create_data(self):

        frozen_dists = self.frozen_dists
        dist_list_c0 = frozen_dists['c0']
        dist_list_c1 = frozen_dists['c1']

        k = len(next(iter(frozen_dists.values())))
        
        self.samples_dict = {}
        sample_features_c0 = []
        sample_features_c1 = []

        for i in range(k):
            
            #go by minority as class 1 (positive) and majority as class 0
            sample_features_c0.append(dist_list_c0[i].rvs(size = self.size[0]))
            sample_features_c1.append(dist_list_c1[i].rvs(size = self.size[1]))
            

        sample_features_c0 = [array if len(array.shape)==2 else array.reshape(-1, 1) for array in sample_features_c0]
        sample_features_c1 = [array if len(array.shape)==2 else array.reshape(-1, 1) for array in sample_features_c1]
        #print(sample_features_c1)

        X_c0 = np.concatenate(sample_features_c0, axis = 1)
        X_c1 = np.concatenate(sample_features_c1, axis = 1)
        #print(X_c1)

        y_c0 = np.zeros(self.size[0])
        y_c1 = np.ones(self.size[1])

        self.X = np.concatenate( (X_c0, X_c1), axis = 0)
        self.y = np.concatenate( (y_c0, y_c1), axis = 0)
        
    

    def prepare_data(self, test_size):
        
        self.create_data()
        
        # Generate a random permutation of indices
        permuted_indices = np.random.permutation(len(self.X))

        X = self.X[permuted_indices]
        y = self.y[permuted_indices]

        return train_test_split(
            X, 
            y, 
            test_size = test_size, 
            random_state=self.random_state
            )








class Multi_Modal_Dist_Generator:
    
    def __init__(self, distributions, params_dict_list, sizes, random_state = 1234):
        self.sizes = sizes
        self.dists = distributions
        self.params_dict_list = params_dict_list
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
        

        self.dists_sample_lists = {key: [array if len(array.shape)==2 else array.reshape(-1, 1) for array in sample_features_list]
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

        self.dists_sample_lists[f'c{c_id}'].append(feature_samples)
        


    def create_unimodal_features(self, c_id, dist, params_dict):

        size = self.sizes[c_id]

        k = len(next(iter(params_dict.values())))

        sample_features_list = []

        for i in range(k):

            params = {key: value[i] for key, value in params_dict.items()}

            frozen_dist = dist(**params)

            sample_features_list.append(frozen_dist.rvs(size = size))

        self.dists_sample_lists[f'c{c_id}'].extend(sample_features_list)

            
    
    def prepare_data(self, test_size = 0.2):
        
        self.create_data()

        return train_test_split(
            self.X, 
            self.y, 
            test_size = test_size, 
            random_state=self.random_state
            )








if __name__ == "__main__":
    
    """
    Single Distribution: Multi-Normal Example
    -------------------------------------------------------------------------------------------------------------------------------------------

    #set the parameter dictionary. sigma is the standard deviation
    mu_c1 = [0,0]
    mu_c2 = [3,3]
    sigma_c1 = np.array([[1,0],
                        [0,3]])
    sigma_c2 = np.array([[2,1],
                        [1,1]])
    parameters = {'mu': [mu_c1, mu_c2], 'sigma': [sigma_c1, sigma_c2]}

    #set the distribution
    distribution = st.multivariate_normal

    #set the distribution name on the graph
    distribution_name = 'Normal-Distribution Family'

    size = [90, 10]
    """

    """
    Test and Comparison Cont_Dist_Generator
    -------------------------------------------------------------------------------------------------------------------------------------------

    dist_gen_spec = Cont_Dist_Generator(distributions, distribution_name, size, **parameters)

    #dist_gen_spec.freeze_distributions()
    spec_samples = dist_gen_spec.prepare_data(0.2)

    #print(dist_fam.dist_list)
    print(spec_samples, '\n')



    dist_gen_sklearn = ImbalancedDataGenerator(class_ratio= 0.1, n_samples= 100, n_features=2)

    sklearn_samples = dist_gen_sklearn.generate_data()

    print(sklearn_samples)
    """



    """
    Multiple Distributions: Multinormal + Beta + Exponential Example
    -------------------------------------------------------------------------------------------------------------------------------------------

    #set the parameter dictionary for the MV normal. sigma is the standard deviation
    mu_c0 = [0,0]
    mu_c1 = [3,3]
    sigma_c0 = np.array([[1,0],
                        [0,3]])
    sigma_c1 = np.array([[2,1],
                        [1,1]])


    distributions = [st.multivariate_normal, st.beta, st.expon]

    #set the parameter dictionaries as a list of dictionaries with parameter dictionaries for classes individually.
    dist_parameter_dicts = [{'params_c0': {'mean': [mu_c0], 'cov': [sigma_c0]},
                            'params_c1': {'mean': [mu_c1], 'cov': [sigma_c1]}
                            },
                            {'params_c0': {'a': [2,3], 'b': [4,5]},
                            'params_c1': {'a': [1,2], 'b': [7,8]}
                            },
                            {'params_c0': {'loc': [0], 'scale': [1]},
                            'params_c1': {'loc': [0], 'scale': [3]}
                            }
    ]

    size = [90, 10]
    """

    """
    Test and Comparison Mult_Dist_Generator
    -------------------------------------------------------------------------------------------------------------------------------------------

    dist_gen_spec = Multi_Dist_Generator(distributions, dist_parameter_dicts, size)

    #dist_gen_spec.create_data()
    #spec_samples = (dist_gen_spec.X, dist_gen_spec.y)

    spec_samples = dist_gen_spec.prepare_data(0.2)

    #print(dist_fam.dist_list)
    print(spec_samples, '\n')



    dist_gen_sklearn = ImbalancedDataGenerator(class_ratio= 0.1, n_samples= 100, n_features=5)

    sklearn_samples = dist_gen_sklearn.generate_data()

    #print(sklearn_samples)
    """



    """
    Multiple and Multimodal Distributions: Multinormal + Beta + Exponential Example
    -------------------------------------------------------------------------------------------------------------------------------------------
    """
    #set the parameter dictionary for the MV normal. sigma is the standard deviation
    mu_c0_1 = [0,0]
    mu_c0_2 = [2,0]
    mu_c1_1 = [3,3]
    mu_c1_2 = [1,3]
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

    size = [90, 10]



    """
    Test and Comparison Multi_Modal_Dist_Generator
    -------------------------------------------------------------------------------------------------------------------------------------------
    """
    dist_gen_spec = Multi_Modal_Dist_Generator(distributions, dist_parameter_dicts, size)

    #dist_gen_spec.create_data()
    #spec_samples = (dist_gen_spec.X, dist_gen_spec.y)

    spec_samples = dist_gen_spec.prepare_data(0.2)

    print(spec_samples, '\n')



    #dist_gen_sklearn = ImbalancedDataGenerator(class_ratio= 1-0.001, n_samples= 1000, n_features=5, flip_y=0.1)
    dist_gen_sklearn = ImbalancedDataGenerator(class_ratio= 1-0.001, n_samples= 1000, n_features=5)

    X_train, X_test, y_train, y_test = dist_gen_sklearn.generate_data()
    

    # If one uses flip_y = 0.1 then many more samples than expected exist of the minority class (44 if one uses 1 as minority with random state 123).
    # If one leaves the argument out exactly 1 exists (as expected). However then we get either a training or
    # a testset without minority class ---> Error in balancers (if no min. in training) or Error in Metrics (if no min. in test)
    y_test_1_mask = y_test == 1
    y_train_1_mask = y_train == 1
    
    n_1_test = np.sum(y_test_1_mask)
    n_1_train = np.sum(y_train_1_mask)
    print('Number of C1 in Train: \n', n_1_train)
    print('Number of C1 in Test: \n', n_1_test)


import pandas as pd
import numpy as np
import scipy.stats as st
import scipy.special as sp
import numpy.random as rnd
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




#Beta-Distribution
#-------------------------------------------------------------------------------------------------------------------------------------------
'''
#set the parameter dictionary.
parameters = {'$\\alpha$': [2,3,4,5], '$\\beta$': [2,3,4,5]}

#set the distribution
distribution = st.beta

#set the distribution name on the graph
distribution_name = 'Beta-Distribution Family'

#set the domain
r = 1
domain = np.linspace(0,r,1000)
'''




#Multi_normal
#-------------------------------------------------------------------------------------------------------------------------------------------
''''''
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



#Test and Comparison
#-------------------------------------------------------------------------------------------------------------------------------------------
''''''
dist_gen_spec = Cont_Dist_Generator(distribution, distribution_name, size, **parameters)

#dist_gen_spec.freeze_distributions()
spec_samples = dist_gen_spec.prepare_data(0.2)

#print(dist_fam.dist_list)
print(spec_samples, '\n')



dist_gen_sklearn = ImbalancedDataGenerator(class_ratio= 0.1, n_samples= 100, n_features=2)

sklearn_samples = dist_gen_sklearn.generate_data()

print(sklearn_samples)






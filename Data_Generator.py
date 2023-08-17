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
    
    def __init__(self, distribution, distribution_name, size, **parameters):
        self.size = size
        self.dist_name = distribution_name
        self.dist = distribution
        self.params = parameters


    def create_densities(self):
        parameters_dict = self.params
        
        dist_list = []

        n = len(next(iter(parameters_dict.values())))

        for i in range(n):

            param_set = []

            for name, parameters in parameters_dict.items():
                param_set.append(parameters[i])
            

            #create the i-th density
            dist_list.append(self.dist(*param_set))

        self.dist_list = dist_list

    
    def create_data(self):
        
        dist_list = self.dist_list
        parameters_dict = self.params

        k = len(next(iter(parameters_dict.values())))

        self.samples_dict = {}

        for i in range(k):

            legend_string = ''
        
            for name, parameters in parameters_dict.items():
                
                legend_string = legend_string + '{name} = {parameter}, '.format(name = name, parameter = parameters[i])
            
            self.samples_dict[legend_string] = dist_list[i].rvs(size = self.size)
            
            






#Normal-Verteilung
#-------------------------------------------------------------------------------------------------------------------------------------------
'''
#set the parameter dictionary. sigma is the standard deviation
parameters = {'$\mu$': [0,0,0,0], '$\sigma$': [0.5, 1, 1.5, 2]}

#set the distribution
distribution = st.norm

#set the distribution name on the graph
distribution_name = 'Normal-Distribution Family'

#set the domain
r = 10
domain = np.linspace(-r,r,1000)
'''


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


#Exponential-Distribution
#-------------------------------------------------------------------------------------------------------------------------------------------
'''
#set the parameter dictionary.
#Note : scale is 1/lambda
#   loc represents a shift on the x axis
parameters = {'loc': [0, 0, 0, 0], 'scale ($\lambda^{-1}$)': [0.5, 0.75, 1, 1.25]}

#set the distribution
distribution = st.expon

#set the distribution name on the graph
distribution_name = 'Exponential-Distribution Family'

#set the domain
r = 10
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
                     [0,1]])
parameters = {'$\mu$': [mu_c1, mu_c2], '$\sigma$': [sigma_c1,sigma_c2]}

#set the distribution
distribution = st.multivariate_normal

#set the distribution name on the graph
distribution_name = 'Normal-Distribution Family'

size = 20



#Initialisation and execution
#-------------------------------------------------------------------------------------------------------------------------------------------
''''''
dist_fam = Cont_Dist_Generator(distribution, distribution_name, size, **parameters)

dist_fam.create_densities()
dist_fam.create_data()

#print(dist_fam.dist_list)
print(dist_fam.samples_dict)









import pandas as pd
import numpy as np
import scipy.stats as st
import scipy.special as sp
import numpy.random as rnd


class cont_dist_fam_plt:
    """
    
    """
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
dist_fam = cont_dist_fam_plt(distribution, distribution_name, size, **parameters)

dist_fam.create_densities()
dist_fam.create_data()

#print(dist_fam.dist_list)
print(dist_fam.samples_dict)









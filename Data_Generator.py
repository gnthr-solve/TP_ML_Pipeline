import pandas as pd
import numpy as np
import scipy.stats as st
import scipy.special as sp
import numpy.random as rnd


class cont_dist_fam_plt:
    """
    
    """
    def __init__(self, distribution, distribution_name, domain, **parameters):
        self.domain = domain
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
            


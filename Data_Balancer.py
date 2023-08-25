


class DataBalancer:
    def __init__(self, balancer=None, random_state=42):
        self.balancer = balancer
        self.random_state = random_state

    def balance_data(self, X, y):
        if self.balancer is None:
            raise ValueError("Balancer object is not provided. Please initialize the balancer.")
        
        X_resampled, y_resampled = self.balancer.fit_resample(X, y)
        return X_resampled, y_resampled
    


class IterDataBalancer:
    def __init__(self, balancers = [], random_state=42):
        self.balancers = balancers
        self.random_state = random_state

    def balance_data(self, X, y):
        
        balanced_data = []

        for balancer in self.balancers:
            
            if balancer == None:
                balanced_data.append((X,y))
            else:
                balanced_data.append(balancer.fit_resample(X, y))
        
        return balanced_data



if __name__=="__main__":

    import numpy as np
    import pandas as pd
    from loguru import logger
    from imblearn.over_sampling import ADASYN,RandomOverSampler,KMeansSMOTE,SMOTE,BorderlineSMOTE,SVMSMOTE,SMOTENC, RandomOverSampler
    from Data_Generator import Multi_Modal_Dist_Generator
    from Visualiser import Visualiser
    from parameters import mixed_3d_test_dict


    balancing_methods = {
    "Unbalanced": None,
    "ADASYN": ADASYN,
    #"RandomOverSampler": RandomOverSampler,
    "KMeansSMOTE": KMeansSMOTE,
    #"SMOTE": SMOTE,
    "BorderlineSMOTE": BorderlineSMOTE,
    #"SVMSMOTE": SVMSMOTE,
    #"SMOTENC": SMOTENC,
    }

    data_generator = Multi_Modal_Dist_Generator(**mixed_3d_test_dict)
    X_train, X_test, y_train, y_test = data_generator.prepare_data(0.2)

    visualiser = Visualiser()


    """
    DataBalancer Test Case
    -------------------------------------------------------------------------------------------------------------------------------------------
    
    method = "SMOTE"
    balancing_method = balancing_methods[method](sampling_strategy='auto', random_state=123)
    data_balancer = DataBalancer(balancer=balancing_method)

    X_train_balanced, y_train_balanced = data_balancer.balance_data(X_train, y_train)


    print('Original x-data: \n', X_train[-20:], '\n',
          'Original y-data: \n', y_train[-20:], '\n',
          'Original size: \n', len(X_train), '\n')
    
    print('Balanced x-data: \n', X_train_balanced[-20:], '\n',
          'Balanced y-data: \n', y_train_balanced[-20:], '\n',
          'Balanced size: \n', len(X_train_balanced), '\n')

    visualiser.plot_2d_scatter((X_train, y_train),0, 1)
    visualiser.plot_3d_scatter((X_train, y_train),0,1,2)
    visualiser.plot_2d_scatter((X_train_balanced, y_train_balanced),0, 1)
    visualiser.plot_3d_scatter((X_train_balanced, y_train_balanced),0,1,2)
    """

    """
    IterDataBalancer Test Case
    -------------------------------------------------------------------------------------------------------------------------------------------
    """
    balancers = [method(sampling_strategy='auto', random_state=123) 
                 if method != None else method
                 for name, method in balancing_methods.items()]
    
    iter_data_balancer = IterDataBalancer(balancers = balancers)
    
    balanced_data = iter_data_balancer.balance_data(X_train, y_train)


    print(#'Original x-data: \n', X_train[-20:], '\n',
          #'Original y-data: \n', y_train[-20:], '\n',
          'Original size: \n', len(X_train), '\n')
    
    #visualiser.plot_2d_scatter((X_train, y_train),0, 1)

    for ind, data in enumerate(balanced_data):

        print(#'Balanced x-data: \n', data[0][-20:], '\n',
              #'Balanced y-data: \n', data[1][-20:], '\n',
              #'Balanced size: \n', len(data[0]), '\n'
              )
        
        common_data = np.isin(data[0], X_train)
        row_mask = np.all(common_data, axis = 1)
        
        only_balance_X = data[0][~row_mask]
        only_balance_y = data[1][~row_mask]

        datasets = [
            (X_train, y_train, 'Train'),
            (only_balance_X, only_balance_y, 'Balanced Train')
        ]

        #visualiser.plot_2d_scatter((data[0], data[1]),0, 1)
        visualiser.plot_2d_scatter_multiple_datasets_px(datasets, 
                                                     feature1 = 0, 
                                                     feature2 = 1, 
                                                     title = f'Scatter of {balancers[ind]}-balanced Data')
from helper_tools import Data
import numpy as np

class DataBalancer:
    def __init__(self, balancer=None, random_state=42):
        self.balancer = balancer
        self.random_state = random_state

    def balance_data(self, X, y):
        if self.balancer is None:
            return X, y
        
        X_resampled, y_resampled = self.balancer.fit_resample(X, y)
        return X_resampled, y_resampled
    




class DictIterDataBalancer:

    def __init__(self, balancers_dict = {}, balancer_params = {'sampling_strategy': 'auto', 'random_state': 42}):

        self.balancer_list = [(name, balancer) for name, balancer in balancers_dict.items()]
        
        if not isinstance(balancer_params, list):
            self.balancer_params = [balancer_params for _ in range(len(self.balancer_list))]
        else:
            self.balancer_params = balancer_params


    def balance_data(self, X, y):
        
        balanced_data = []

        for ind, (name, balancer) in enumerate(self.balancer_list):
            
            if balancer == None:
                balanced_data.append((name, X, y))

            else:
                balancer = balancer(**self.balancer_params[ind])
                try:
                    balanced_data.append((name, *balancer.fit_resample(X, y)))
                except Exception as e:
                    print(f'Balancer {name} produced error: \n',  e)
        
        return balanced_data






class FMPL_DataBalancer(Data):

    def __init__(self, bal_params_dict = {}):

        self.balancer_dict = {(i,j): assign_list[1] for (i,j,k), assign_list in self.data_dict['assignment_dict'].items()}
        
        default_dict = {'sampling_strategy': 'auto', 'random_state': 42}
        self.bal_params_dict = {name: bal_params_dict[name]
                                if name in bal_params_dict else default_dict
                                for (name, bal) in self.balancer_dict.values()}
        
        

    def balance_data(self, data_ind):

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

            





if __name__=="__main__":

    import numpy as np
    import pandas as pd
    from loguru import logger
    from imblearn.over_sampling import ADASYN,RandomOverSampler,KMeansSMOTE,SMOTE,BorderlineSMOTE,SVMSMOTE,SMOTENC, RandomOverSampler
    from Data_Generator import Multi_Modal_Dist_Generator
    from Visualiser import RawVisualiser
    from gen_parameters import mixed_3d_test_dict


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


    print(#f'Unbalanced train no 0: \n', np.sum(y_train == 0), '\n',
          #f'Unbalanced train no 1: \n', np.sum(y_train == 1), '\n',
          #f'Unbalanced train size: \n', len(y_train), '\n'
        )
    


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
    DictIterDataBalancer Test Case
    -------------------------------------------------------------------------------------------------------------------------------------------
    """
    
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

    
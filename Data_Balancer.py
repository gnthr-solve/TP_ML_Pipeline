from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import numpy as np

class DataBalancer:
    def __init__(self, balancer=None, random_state=42):
        self.balancer = balancer
        self.random_state = random_state

    def balance_data(self, X, y):
        if self.balancer is None:
            raise ValueError("Balancer object is not provided. Please initialize the balancer.")
        
        X_resampled, y_resampled = self.balancer.fit_resample(X, y)
        return X_resampled, y_resampled
    




if __name__=="__main__":

    from loguru import logger
    from imblearn.over_sampling import ADASYN,RandomOverSampler,KMeansSMOTE,SMOTE,BorderlineSMOTE,SVMSMOTE,SMOTENC, RandomOverSampler
    from Data_Generator import Multi_Modal_Dist_Generator
    from Visualiser import Visualiser
    from parameters import mixed_3d_test_dict


    balancing_methods = {
    "ADASYN": ADASYN,
    "RandomOverSampler": RandomOverSampler,
    "KMeansSMOTE": KMeansSMOTE,
    "SMOTE": SMOTE,
    "BorderlineSMOTE": BorderlineSMOTE,
    "SVMSMOTE": SVMSMOTE,
    "SMOTENC": SMOTENC,
    }

    data_generator = Multi_Modal_Dist_Generator(**mixed_3d_test_dict)
    X_train, X_test, y_train, y_test = data_generator.prepare_data(0.2)

    visualiser = Visualiser()

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

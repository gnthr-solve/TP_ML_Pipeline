from sklearn.base import BaseEstimator, ClassifierMixin
from helper_tools import Data
import numpy as np

class Classifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self, classifier=None):
        self.classifier = classifier

    def fit(self, X, y):

        if self.classifier is None:
            raise ValueError("Classifier is not provided. Please initialize the classifier.")
        
        self.classifier.fit(X, y)
        return self

    def predict(self, X):

        if self.classifier is None:
            raise ValueError("Classifier is not provided. Please initialize the classifier.")
        
        return self.classifier.predict(X)

    def predict_proba(self, X):

        if self.classifier is None:
            raise ValueError("Classifier is not provided. Please initialize the classifier.")
        
        return (self.classifier.classes_, self.classifier.predict_proba(X))






class DictIterClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, classifiers_dict = {}, classifier_params = {'random_state': 42}):

        classifier_list = [(name, classifier) for name, classifier in classifiers_dict.items()]

        if not isinstance(classifier_params, list):
            self.classifier_params = [classifier_params for _ in range(len(classifier_list))]
        else:
            self.classifier_params = classifier_params

        self.classifier_dict_list = [{'name': name, 'classifier': classifier(**params)}
                                      for (name, classifier), params in zip(classifier_list, self.classifier_params)]


    def fit(self, X, y):

        if self.classifier_dict_list is []:
            raise ValueError("Classifier is not provided. Please initialize the classifier.")
        
        self.classifiers_dict_list = [
            {**dict, 'classifier': dict['classifier'].fit(X,y)} 
            for dict in self.classifier_dict_list
        ]
        return self
    

    def predict(self, X):

        if self.classifier_dict_list is []:
            raise ValueError("Classifier is not provided. Please initialize the classifier.")
        
        predictions_dict_list = [
            dict(
                 {key: val for key,val in clsf_dict.items() if key != 'classifier'},
                 **{
                     'predicted_proba': clsf_dict['classifier'].predict_proba(X),
                     'predicted_y': clsf_dict['classifier'].predict(X),
                     'classes': clsf_dict['classifier'].classes_
                    }
            )
            for clsf_dict in self.classifier_dict_list
        ]
        
        return predictions_dict_list
    





class FMLP_DataClassifier(Data):

    def __init__(self, clsf_params_dict = {}):

        default_dict = {'random_state': 42}
        classifier_dict = {key: assign_list[2] for key, assign_list in self.data_dict['assignment_dict'].items()}
        
        classifier_dict = {key: (name, clsf(**clsf_params_dict[name]))
                           if name in clsf_params_dict else (name, clsf(**default_dict))
                           for key, (name, clsf) in classifier_dict.items()}
        
        self.classifier_dict = classifier_dict



    def fit(self):

        X = self.data_dict['bal_X_train']
        y = self.data_dict['bal_y_train']
        
        for (i,j,k), (name, clsf) in self.classifier_dict.items():
            
            X_fit = X[i, j, :, :]
            y_fit = y[i, j, :]

            # Drop rows with NaN values
            X_fit = X_fit[~np.isnan(X_fit).all(axis = 1)]
            # Drop columns with NaN values
            X_fit = X_fit[: , ~np.isnan(X_fit).all(axis = 0)]
            
            y_fit = y_fit[~np.isnan(y_fit)]
            
            self.classifier_dict[(i,j,k)] = (name, clsf.fit(X_fit, y_fit))


        return self
    

    def predict(self):

        X = self.data_dict['org_X_test']

        for (i,j,k), (name, clsf) in self.classifier_dict.items():
            
            X_test = X[i, :, :]

            # Drop rows with NaN values
            X_test = X_test[~np.isnan(X_test).all(axis = 1)]
            # Drop columns with NaN values
            X_test = X_test[: , ~np.isnan(X_test).all(axis = 0)]

            n_i = len(X_test)

            self.data_dict['clsf_predictions_y'][i, j, k, : n_i] = clsf.predict(X_test)
            self.data_dict['clsf_predictions_proba'][i, j, k, : n_i, :] = clsf.predict_proba(X_test) 
            self.data_dict['classes_order'][i, j, k, :] = clsf.classes_

    





if __name__=="__main__":

    import pandas as pd
    from loguru import logger
    #from imblearn.over_sampling import ADASYN,RandomOverSampler,KMeansSMOTE,SMOTE,BorderlineSMOTE,SVMSMOTE,SMOTENC, RandomOverSampler
    from Data_Generator import Multi_Modal_Dist_Generator
    from Visualiser import RawVisualiser, CLSFVisualiser
    from gen_parameters import presentation_experiment_dict
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from xgboost import XGBClassifier
    #from lightgbm import LGBMClassifier


    data_generator = Multi_Modal_Dist_Generator(**presentation_experiment_dict)
    X_train, X_test, y_train, y_test = data_generator.prepare_data(0.2)

    visualiser = RawVisualiser()
    clsf_visualiser = CLSFVisualiser()

    classifiers_dict = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    #"SVC": SVC,
    #"Naive Bayes": GaussianNB,
    #"XGboost": XGBClassifier,
    #"Lightgbm": LGBMClassifier
    }

    """
    Classifier Test Case
    -------------------------------------------------------------------------------------------------------------------------------------------
    
    # Initialize the classifier, e.g., Support Vector Machine (SVC)
    classifier = LogisticRegression(random_state=42)
    model = Classifier(classifier=classifier)

    # Fit the model on the data
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    correct_mask = (predictions == y_test)

    correct_predictions = predictions[correct_mask]
    incorrect_predictions = predictions[~correct_mask]
    #print(correct_predictions)
    #print(incorrect_predictions)

    datasets = [
            (X_test[correct_mask], correct_predictions, 'Correctly assigned'),
            (X_test[~correct_mask], incorrect_predictions, 'Incorrectly assigned')
        ]
    
    visualiser.plot_2d_scatter_multiple_datasets_px(datasets, 
                                                     feature1 = 0, 
                                                     feature2 = 2, 
                                                     title = f'Scatter of Predictions')

    visualiser.plot_3d_scatter_multiple_datasets_px(datasets,
                                                        feature1 = 0, 
                                                        feature2 = 1,
                                                        feature3 = 2,
                                                        title = f'3d Scatter of Predictions')
    #print(predictions)
    """


    """
    DictIterClassifier Test Case
    -------------------------------------------------------------------------------------------------------------------------------------------
    
    
    dict_iter_classifier = DictIterClassifier(classifiers_dict = classifiers_dict)
    # Fit the model on the data
    dict_iter_classifier.fit(X_train, y_train)

    # Make predictions
    predictions_dict_list = dict_iter_classifier.predict(X_test)
    #print(predictions_dict_list)


    for predictions_dict in predictions_dict_list:

        name = predictions_dict['name']
        predictions = predictions_dict['predicted_y']

        correct_mask = (predictions == y_test)

        correct_predictions = predictions[correct_mask]
        incorrect_predictions = predictions[~correct_mask]
        #print(correct_predictions)
        #print(incorrect_predictions)

        datasets = [
                (X_test[correct_mask], correct_predictions, 'Correctly assigned'),
                (X_test[~correct_mask], incorrect_predictions, 'Incorrectly assigned')
            ]
        
        visualiser.plot_2d_scatter_multiple_datasets_px(datasets, 
                                                        feature1 = 0, 
                                                        feature2 = 2, 
                                                        title = f'Scatter of {name} Predictions')
        
        visualiser.plot_3d_scatter_multiple_datasets_px(datasets,
                                                        feature1 = 0, 
                                                        feature2 = 1,
                                                        feature3 = 2,
                                                        title = f'3d Scatter of {name} Predictions')
        #print(predictions)

    """

    
    """
    DictIterClassifier used to test CLSFVisualiser Plots
    -------------------------------------------------------------------------------------------------------------------------------------------
    """
    
    dict_iter_classifier = DictIterClassifier(classifiers_dict = classifiers_dict)
    # Fit the model on the data
    dict_iter_classifier.fit(X_train, y_train)

    # Make predictions
    predictions_dict_list = dict_iter_classifier.predict(X_test)
    #print(predictions_dict_list)

    feature1 = 2
    feature2 = 4

    for predictions_dict in predictions_dict_list:

        name = predictions_dict['name']
        predictions = predictions_dict['predicted_y']
        classes = predictions_dict['classes']
        predicted_probabilities = predictions_dict['predicted_proba']

        clsf_visualiser.confusion_scatterplot(X_test,
                                              y_test,
                                              predictions,
                                              feature1 = feature1, 
                                              feature2 = feature2,
                                              #feature_map= feature_map,
                                              title = f'{name} Confusion Scatter',
                                              save = False
                                        )
        
        clsf_visualiser.pred_proba_scatterplot(X_test,
                                               predicted_probabilities,
                                               classes,
                                               feature1 = feature1, 
                                               feature2 = feature2,
                                               #feature_map= feature_map,
                                               title = f'Scatter of {name} Probability Predictions',
                                               save = False
                                            )

        
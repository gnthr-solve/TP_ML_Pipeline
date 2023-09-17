from sklearn.base import BaseEstimator, ClassifierMixin

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
    

    





if __name__=="__main__":

    import pandas as pd
    from loguru import logger
    #from imblearn.over_sampling import ADASYN,RandomOverSampler,KMeansSMOTE,SMOTE,BorderlineSMOTE,SVMSMOTE,SMOTENC, RandomOverSampler
    from Data_Generator import Multi_Modal_Dist_Generator
    from Visualiser import RawVisualiser
    from gen_parameters import mixed_3d_test_dict
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from xgboost import XGBClassifier
    #from lightgbm import LGBMClassifier


    data_generator = Multi_Modal_Dist_Generator(**mixed_3d_test_dict)
    X_train, X_test, y_train, y_test = data_generator.prepare_data(0.2)

    visualiser = RawVisualiser()

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
    """
    
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
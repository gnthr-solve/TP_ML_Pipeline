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




class IterClassifier():
    
    def __init__(self, classifiers = []):
        self.classifiers = classifiers

    def fit(self, X, y):

        if self.classifiers is []:
            raise ValueError("Classifier is not provided. Please initialize the classifier.")
        
        self.classifiers = [classifier.fit(X, y) for classifier in self.classifiers]
        return self

    def predict(self, X):

        if self.classifiers is []:
            raise ValueError("Classifier is not provided. Please initialize the classifier.")
        
        return [classifier.predict(X) for classifier in self.classifiers]






if __name__=="__main__":

    import pandas as pd
    from loguru import logger
    #from imblearn.over_sampling import ADASYN,RandomOverSampler,KMeansSMOTE,SMOTE,BorderlineSMOTE,SVMSMOTE,SMOTENC, RandomOverSampler
    from Data_Generator import Multi_Modal_Dist_Generator
    from Visualiser import Visualiser
    from parameters import mixed_3d_test_dict
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from xgboost import XGBClassifier
    #from lightgbm import LGBMClassifier


    data_generator = Multi_Modal_Dist_Generator(**mixed_3d_test_dict)
    X_train, X_test, y_train, y_test = data_generator.prepare_data(0.2)

    visualiser = Visualiser()

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
    IterClassifier Test Case
    -------------------------------------------------------------------------------------------------------------------------------------------
    """
    # Initialize the classifiers, e.g., Support Vector Machine (SVC)
    classifiers = [(name, classifier(random_state = 42))
                   for name, classifier in classifiers_dict.items()]
    
    iter_classifier = IterClassifier(classifiers = [entry[1] for entry in classifiers])
    # Fit the model on the data
    iter_classifier.fit(X_train, y_train)

    # Make predictions
    predictions_list = iter_classifier.predict(X_test)
    #print(predictions_list)


    for i, predictions in enumerate(predictions_list):
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
                                                        title = f'Scatter of {classifiers[i][0]} Predictions')
        
        visualiser.plot_3d_scatter_multiple_datasets_px(datasets,
                                                        feature1 = 0, 
                                                        feature2 = 1,
                                                        feature3 = 2,
                                                        title = f'3d Scatter of {classifiers[i][0]} Predictions')
        #print(predictions)
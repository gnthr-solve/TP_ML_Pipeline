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








if __name__=="__main__":

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
    from lightgbm import LGBMClassifier


    data_generator = Multi_Modal_Dist_Generator(**mixed_3d_test_dict)
    X_train, X_test, y_train, y_test = data_generator.prepare_data(0.2)

    visualiser = Visualiser()


    """
    Classifier Test Case
    -------------------------------------------------------------------------------------------------------------------------------------------
    """

    # Initialize the classifier, e.g., Support Vector Machine (SVC)
    classifier = SVC(random_state=42)
    model = Classifier(classifier=classifier)

    # Fit the model on the data
    #model.fit(X, y)

    # Make predictions
    #predictions = model.predict(X_test)

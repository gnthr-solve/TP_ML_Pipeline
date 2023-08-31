from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from parameters import default_test_dict
import numpy as np
import plotly.express as px
from scipy.stats import linregress
import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds")
        return result
    return wrapper




class Study():
    def __init__(self, 
                 data_generator=None, 
                 data_balancer=None,
                 data_classifier=None
                 ):
        self.data_generator = data_generator
        self.data_balancer = data_balancer
        self.data_classifier = data_classifier
        self.results = {
            "no balancing": None,
            "balanced": None
        }
        
    def run(self):
        X_train, X_test, y_train, y_test = self.data_generator.generate_data()
        self.y_test = y_test
        self.data_classifier.fit(X_train, y_train)
        self.y_predicted_no_balancing = self.data_classifier.predict(X_test)
        self.X_train_balanced, self.y_train_balanced = self.data_balancer.balance_data(X_train, y_train)
        self.data_classifier.fit(self.X_train_balanced, self.y_train_balanced)
        self.y_predicted_balanced = self.data_classifier.predict(X_test)
        
    def calculate_metrics(self):
        imbalanced_results = {
            "accuracy": accuracy_score(self.y_test,self.y_predicted_no_balancing),
            "precision": precision_score(self.y_test,self.y_predicted_no_balancing), 
            "recall":  recall_score(self.y_test,self.y_predicted_no_balancing),
            "F1 score": f1_score(self.y_test,self.y_predicted_no_balancing),
            "ROC AUC Score": roc_auc_score(self.y_test,self.y_predicted_no_balancing), 
            #"Confusion Matrix": confusion_matrix(self.y_test,self.y_predicted_no_balancing)
        }
        balanced_results = {
            "accuracy": accuracy_score(self.y_test,self.y_predicted_balanced),
            "precision": precision_score(self.y_test,self.y_predicted_balanced), 
            "recall":  recall_score(self.y_test,self.y_predicted_balanced),
            "F1 score": f1_score(self.y_test,self.y_predicted_balanced),
            "ROC AUC Score": roc_auc_score(self.y_test,self.y_predicted_balanced), 
            #"Confusion Matrix": confusion_matrix(self.y_test,self.y_predicted_balanced)
        }
        
        return {
            "imbalanced_results": imbalanced_results,
            "balanced_results": balanced_results
        }








class OwnMetrics():

    def __init__(self, 
                 X_test, 
                 y_test,
                 #test_data, 
                 predictions_dict, 
                 #metrics_dict, 
                 #balanced = False
                 ):
        """
        Input: dictionaries with the parameters of generation, balancing and classification as keys or
        other key value pairs
        """

        self.X_test = X_test
        self.y_test = y_test
        self.clsf_name = predictions_dict['name']
        self.classes = predictions_dict['classes']
        #print(self.classes)
        #print(np.where(self.classes == 1))
        #print(np.shape(np.where(self.classes == 1)))
        #print(np.where(self.classes == 1)[0])

        all_classes_proba_array = predictions_dict['predicted_proba']
        minority_proba_2darray = all_classes_proba_array[:, np.where(self.classes == 1)[0]]
        self.test_probabilities = minority_proba_2darray.flatten()
        #print(minority_proba_2darray[:20])
        #print(self.test_probabilities[:20])
        #print(np.sum(self.test_probabilities[:20], axis = 1))

        self.y_predicted = predictions_dict['predicted_y']
        #print(self.y_predicted)
        

    @timing_decorator
    def confusion_values(self):
        y_test = self.y_test
        y_predicted = self.y_predicted
        y_test_1_mask = y_test == 1
        y_pred_1_mask = y_predicted == 1

        self.TP = np.sum((y_test_1_mask) & (y_pred_1_mask))
        self.FP = np.sum((~y_test_1_mask) & (y_pred_1_mask))
        self.TN = np.sum((~y_test_1_mask) & (~y_pred_1_mask))
        self.FN = np.sum((y_test_1_mask) & (~y_pred_1_mask))

        #self.TP = np.sum((y_test == 1) & (y_predicted == 1))
        #self.FP = np.sum((y_test == 0) & (y_predicted == 1))
        #self.TN = np.sum((y_test == 0) & (y_predicted == 0))
        #self.FN = np.sum((y_test == 1) & (y_predicted == 0))

        return (self.TP, self.FP, self.TN, self.FN)

    def net_benefit(self, harm_to_benefit):

        self.NB = self.TP - harm_to_benefit * self.FP


    def calibration_curve(self, k):

        sorted_indices = np.argsort(self.test_probabilities)
        #print(sorted_indices)

        sorted_probabilities = self.test_probabilities[sorted_indices]
        #print(sorted_probabilities[:20])

        sorted_y_test = self.y_test[sorted_indices]
        #sorted_y_predicted = self.y_predicted[sorted_indices]

        binned_probabilities = np.array_split(sorted_probabilities, k)
        binned_y_test = np.array_split(sorted_y_test, k)

        #print(len(binned_probabilities))
        #print(np.shape(binned_probabilities))
        #print(binned_probabilities[:1])
        #print(binned_y_test[0])

        mean_pred_proba = [bin.sum()/len(bin) for bin in binned_probabilities]
        mean_freq = [bin.sum()/len(bin) for bin in binned_y_test]

        #print(mean_pred_proba)
        #print(mean_freq)

        fig = px.line(x = mean_pred_proba, y = mean_freq, title = 'Calibration Curve')
        fig.show()


    def report(self):
        report = classification_report(self.y_test, self.y_predicted, output_dict = True)
        return pd.DataFrame(report)
    
    @timing_decorator
    def multilabel_confusion_matrix(self, sample_weight=None, labels=None, samplewise=False):
    
        y_true = np.array(self.y_test, dtype='int64')
        y_pred = np.array(self.y_predicted, dtype='int64')
    
        labels = sorted_labels = self.classes
        n_labels = len(sorted_labels)

        # labels are now from 0 to len(labels) - 1 -> use bincount
        tp = y_true == y_pred
        tp_bins = y_true[tp]
        #print(tp_bins)
        if sample_weight is not None:
            tp_bins_weights = np.asarray(sample_weight)[tp]
        else:
            tp_bins_weights = None

        if len(tp_bins):
            tp_sum = np.bincount(
                tp_bins, weights=tp_bins_weights, minlength=len(labels)
            )
            #print(tp_sum)
        else:
            # Pathological case
            true_sum = pred_sum = tp_sum = np.zeros(len(labels))
        if len(y_pred):
            pred_sum = np.bincount(y_pred, weights=sample_weight, minlength=len(labels))
        if len(y_true):
            true_sum = np.bincount(y_true, weights=sample_weight, minlength=len(labels))

        # Retain only selected labels
        indices = np.searchsorted(sorted_labels, labels[:n_labels])
        #print(indices)
        tp_sum = tp_sum[indices]
        #print('True positive sum:',tp_sum)
        true_sum = true_sum[indices]
        #print('True test sum:',true_sum)
        pred_sum = pred_sum[indices]
        #print('True predicted sum:',pred_sum)

        fp = pred_sum - tp_sum
        #print('False positive:',fp)
        fn = true_sum - tp_sum
        #print('False negative:',fn)
        tp = tp_sum
        #print('True positive:', tp)
        tn = y_true.shape[0] - tp - fp - fn
        #print('True negative:', tn)

        return np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)









class SimpleIterMetrics():

    def __init__(self, 
                 X_test, 
                 y_test,
                 predictions_dict_list, 
                 ):

        self.X_test = X_test
        self.y_test = y_test
        self.y_predicted_list = [pred_dict['y_predicted'] for pred_dict in predictions_dict_list]
        
        self.predictions_dict_list = [
            {
                **pred_dict,
                'predicted_proba': (pred_dict['predicted_proba']
                                    [:, np.where(pred_dict['classes'] == 1)[0]]
                                    .flatten()
                                    )
            }
            for pred_dict in predictions_dict_list
        ]
        

        #print(self.predictions_dict_list[0])
        #print(predictions_dict_list[0]['predicted_proba'][:20])
        #print(minority_proba_2darray[:20])
        #print(self.test_probabilities[:20])
        #print(np.sum(self.test_probabilities[:20], axis = 1))
        #print(len(self.test_probabilities))
        #print(self.y_predicted)
        
    def calculate_metrics(self):

        results = {
            "accuracy": [],
            "precision": [], 
            "recall":  [],
            "F1 score": [],
            "ROC AUC Score": [], 
            #"Confusion Matrix": []
        }

        for y_pred in self.y_predicted_list:
            results['accuracy'].append(accuracy_score(self.y_test, y_pred))
            results['precision'].append(precision_score(self.y_test, y_pred))
            results['recall'].append(recall_score(self.y_test, y_pred))
            results['F1 score'].append(f1_score(self.y_test, y_pred))
            results['ROC AUC Score'].append(roc_auc_score(self.y_test, y_pred))
            #results['Confusion Matrix'].append(confusion_matrix(self.y_test, y_pred))

        return results



    
    








class IterMetrics():

    def __init__(self, 
                 X_test, 
                 y_test,
                 predictions_dict_list, 
                 ):

        self.X_test = X_test
        self.y_test = y_test

        
        self.predictions_dict_list = [
            {
                **pred_dict,
                'predicted_proba': (pred_dict['predicted_proba']
                                    [:, np.where(pred_dict['classes'] == 1)[0]]
                                    .flatten()
                                    )
            }
            for pred_dict in predictions_dict_list
        ]
        

        #print(self.predictions_dict_list[0])
        #print(predictions_dict_list[0]['predicted_proba'][:20])
        #print(minority_proba_2darray[:20])
        #print(self.test_probabilities[:20])
        #print(np.sum(self.test_probabilities[:20], axis = 1))
        #print(len(self.test_probabilities))
        #print(self.y_predicted)
        
    def confusion_metrics(self):

        y_predicted_list = [pred_dict['y_predicted'] for pred_dict in self.predictions_dict_list]

        results = {
            "accuracy": [],
            "precision": [], 
            "recall":  [],
            "F1 score": [],
            "ROC AUC Score": [], 
            #"Confusion Matrix": []
        }

        for y_pred in y_predicted_list:
            results['accuracy'].append(accuracy_score(self.y_test, y_pred))
            results['precision'].append(precision_score(self.y_test, y_pred))
            results['recall'].append(recall_score(self.y_test, y_pred))
            results['F1 score'].append(f1_score(self.y_test, y_pred))
            results['ROC AUC Score'].append(roc_auc_score(self.y_test, y_pred))
            #results['Confusion Matrix'].append(confusion_matrix(self.y_test, y_pred))

        return results



    def net_benefit(self, harm_to_benefit):

        self.NB = self.TP - harm_to_benefit * self.FP


    
    def calibration_curve(self, k = 10):

        predicted_probabilities = [pred_dict['predicted_proba'] for pred_dict in self.predictions_dict_list]
        names = [pred_dict['name'] for pred_dict in self.predictions_dict_list]
        
        creation_dict ={
        'mean_pred_proba': [np.arange(0,1, 1/k)],
        'mean_freq': [np.arange(0,1, 1/k)],
        'name': [['Optimum' for _ in range(k)]]
        }
        for i, pred_proba in enumerate(predicted_probabilities):

            sorted_indices = np.argsort(pred_proba)
            sorted_probabilities = pred_proba[sorted_indices]
            sorted_y_test = self.y_test[sorted_indices]

            binned_probabilities = np.array_split(sorted_probabilities, k)
            binned_y_test = np.array_split(sorted_y_test, k)

            creation_dict['mean_pred_proba'].append([bin.sum()/len(bin) for bin in binned_probabilities])
            creation_dict['mean_freq'].append([bin.sum()/len(bin) for bin in binned_y_test])
            creation_dict['name'].append([names[i] for _ in range(k)])

        
        #print(creation_dict['name'])
        df_creation_dict = {
            'Predicted Prob.': np.concatenate(creation_dict['mean_pred_proba']),
            'Mean Frequency': np.concatenate(creation_dict['mean_freq']),
            'Name': sum(creation_dict['name'], [])
        }

        df = pd.DataFrame(df_creation_dict)

        fig = px.line(df, 
                      x = 'Predicted Prob.', 
                      y = 'Mean Frequency', 
                      color = 'Name', 
                      title = f'Calibration Curve ({k} bins)', 
                      markers = True
                      )
        fig.show()
    


    def calibration_curve_reg(self, k = 10):

        predicted_probabilities = [pred_dict['predicted_proba'] for pred_dict in self.predictions_dict_list]
        names = [pred_dict['name'] for pred_dict in self.predictions_dict_list]
        
        creation_dict ={
        'mean_pred_proba': [np.arange(0,1, 1/k)],
        'mean_freq': [np.arange(0,1, 1/k)],
        'name': [['Optimum' for _ in range(k)]]
        }
        for i, pred_proba in enumerate(predicted_probabilities):

            sorted_indices = np.argsort(pred_proba)
            sorted_probabilities = pred_proba[sorted_indices]
            sorted_y_test = self.y_test[sorted_indices]

            binned_probabilities = np.array_split(sorted_probabilities, k)
            binned_y_test = np.array_split(sorted_y_test, k)

            mean_predicted_proba = [bin.sum()/len(bin) for bin in binned_probabilities]
            res = linregress(
                x = mean_predicted_proba,
                y = [bin.sum()/len(bin) for bin in binned_y_test]
            )

            #creation_dict['mean_pred_proba'].append(mean_predicted_proba)
            creation_dict['mean_pred_proba'].append(np.arange(0,1, 1/k))
            #creation_dict['mean_freq'].append([res.intercept + res.slope * x for x in mean_predicted_proba])
            creation_dict['mean_freq'].append([res.intercept + res.slope * x for x in np.arange(0,1, 1/k)])
            creation_dict['name'].append([names[i] for _ in range(k)])

        
        #print(creation_dict['name'])
        df_creation_dict = {
            'Predicted Prob.': np.concatenate(creation_dict['mean_pred_proba']),
            'Mean Frequency': np.concatenate(creation_dict['mean_freq']),
            'Name': sum(creation_dict['name'], [])
        }

        df = pd.DataFrame(df_creation_dict)

        fig = px.line(df, x = 'Predicted Prob.', y = 'Mean Frequency', color = 'Name', title = f'Calibration Curve with Regression ({k} bins)')
        fig.show()


        






if __name__=="__main__":

    import pandas as pd
    from loguru import logger
    from Classifier import Classifier, DictIterClassifier
    from Data_Generator import Multi_Modal_Dist_Generator
    from Visualiser import RawVisualiser
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

    visualiser = RawVisualiser()

    classifiers_dict = {
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    #"SVC": SVC,
    #"Naive Bayes": GaussianNB,
    "XGboost": XGBClassifier,
    #"Lightgbm": LGBMClassifier
    }

    """
    OwnMetrics Test Case
    -------------------------------------------------------------------------------------------------------------------------------------------
    
    dict_iter_classifier = DictIterClassifier(classifiers_dict = classifiers_dict)
    
    dict_iter_classifier.fit(X_train, y_train)
    #Make predictions
    predictions_dict_list = dict_iter_classifier.predict(X_test)
    #print(predictions_dict_list[0])
    
    predictions_dict = predictions_dict_list[0]
    metr = OwnMetrics(X_test, y_test, predictions_dict)

    #metr.calibration_curve(10)

    #report = metr.report()
    #print(report)

    test_confusion_matrix = metr.multilabel_confusion_matrix()
    test_own_confusion_matrix = metr.confusion_values()
    print(test_confusion_matrix)
    """

    """
    IterMetrics Test Case
    -------------------------------------------------------------------------------------------------------------------------------------------
    """
    dict_iter_classifier = DictIterClassifier(classifiers_dict = classifiers_dict)
    
    dict_iter_classifier.fit(X_train, y_train)
    #Make predictions
    predictions_dict_list = dict_iter_classifier.predict(X_test)
    
    metr = IterMetrics(X_test, y_test, predictions_dict_list)

    metr.calibration_curve(20)
    metr.calibration_curve(40)
    metr.calibration_curve_reg(20)
    metr.calibration_curve_reg(40)

    

    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
import category_encoders as ce
from mlxtend.feature_selection import SequentialFeatureSelector
from scipy.stats import ttest_ind

# y variable is the AUC
# the regressors are class ratio, number of samples, number of features,
# distance, balancing method, classifier

dataset= pd.read_csv('results_balanced_linreg.csv')
X_cont = dataset.drop(columns=['balancing_method','classifier','accuracy', 'precision', 'recall', 'F1 score', 'ROC AUC Score'])
X_cat=dataset.drop(columns=['class_ratio','n_samples','n_features','accuracy', 'precision', 'recall', 'F1 score', 'ROC AUC Score'])

encoder = ce.BinaryEncoder(cols=['balancing_method','classifier'])
X_dummies = encoder.fit_transform(X_cat)
X = pd.concat([X_cont, X_dummies], axis=1)

target_list={'accuracy','precision','F1 score','ROC AUC Score'}

for target in target_list:
    y=dataset[target]

    # creating train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101)

    model=LinearRegression()

    model_fit=model.fit(X_train,y_train)
    # making predictions
    predictions = model.predict(X_test)

    # model evaluation
    print('y:', target)
    print('LINEAR REGRESSION\n')
    print(
    'mean_squared_error : ', mean_squared_error(y_test, predictions))
    print(
    'mean_absolute_error : ', mean_absolute_error(y_test, predictions))
    print(f"intercept: {model_fit.intercept_}")
    print(f"coefficients: {model_fit.coef_}")
    r_sq = model_fit.score(X_test, y_test)
    print(f"coefficient of determination: {r_sq}")
    y_pred = model_fit.predict(X_test)
    print(f'predicted response with formula:\n{y_pred}')

    #STEPWISE REGRESSION
    # Perform stepwise regression
    print('STEPWISE REGRESSION\n')
    sfs = SequentialFeatureSelector(LinearRegression(),
                                    k_features=5,
                                    forward=False,
                                    scoring='accuracy',
                                    cv=None)
    selected_features = sfs.fit(X, y)
    print('Selected features:\n')
    print(selected_features)

#HYPOTHESIS TESTING
#t-test to confront accuracy, AUC, precision and F1 score differences
# between the balanced and unbalanced files

results_imbalanced_linreg = pd.read_csv('results_imbalanced_linreg.csv')
results_balanced_linreg = pd.read_csv('results_balanced_linreg.csv')

alpha = 0.05

for target in target_list:
    metric_balanced = [result_balanced[target] for result_balanced in results_balanced_linreg]
    metric_imbalanced = [result_imbalanced[target] for result_imbalanced in results_imbalanced_linreg]
    t_statistic, p_value = ttest_ind(metric_balanced, metric_imbalanced)
    print(f'T-TEST on balanced vs imbalanced for {target}')
    print(f'T-statistic: {t_statistic:.4f}')
    print(f'p_value: {p_value:.4f}')
    if p_value < alpha:
        print('Significant difference')
    else:
        print('No significant difference')


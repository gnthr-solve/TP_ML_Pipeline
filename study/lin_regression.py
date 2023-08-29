import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
import category_encoders as ce
# from mlxtend.feature_selection import SequentialFeatureSelector
from scipy.stats import ttest_ind
from scipy.stats import f

# import rfpimp

# y variable is the AUC
# the regressors are class ratio, number of samples, number of features,
# distance, balancing method, classifier

dataset = pd.read_csv('results_balanced_linreg.csv')
X_cont = dataset.drop(
    columns=['balancing_method', 'classifier', 'accuracy', 'precision', 'recall', 'F1 score', 'ROC AUC Score'])
X_cat = dataset.drop(
    columns=['class_ratio', 'n_samples', 'n_features', 'accuracy', 'precision', 'recall', 'F1 score', 'ROC AUC Score'])

encoder = ce.BinaryEncoder(cols=['balancing_method', 'classifier'])
X_dummies = encoder.fit_transform(X_cat)
X = pd.concat([X_cont, X_dummies], axis=1)

target_list = ['accuracy', 'precision', 'F1 score', 'ROC AUC Score']

for target in target_list:
    y = dataset[target]

    # creating train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=101)

    model = LinearRegression()

    model_fit = model.fit(X_train, y_train)
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
    SS_res = np.sum(np.square(y_test - predictions))
    SS_tot = np.sum(np.square(y_test - np.mean(y_test)))
    df_mod = X_test.shape[1]  # Degrees of freedom for the model
    df_res = X_test.shape[0] - X_test.shape[1]  # Degrees of freedom for residuals
    MS_mod = SS_res / df_mod
    MS_res = SS_tot / df_res
    F = MS_mod / MS_res
    p_value = f.sf(F, df_mod, df_res)  # Survival function (1 - cdf)
    print(f'predicted p_value with formula:\n{p_value}')
    print(f'predicted F with formula:\n{F}')

# Permutation feature score to visualize the features in order of influence on the target metric

# imp = rfpimp.importances(model_fit, X_test, y_test)

# fig, ax = plt.subplots(figsize=(6, 3))

# ax.barh(imp.index, imp['Importance'], height=0.8, facecolor='grey', alpha=0.8, edgecolor='k')
# ax.set_xlabel('Importance score')
# ax.set_title('Permutation feature importance')
# ax.text(0.8, 0.15, 'aegis4048.github.io', fontsize=12, ha='center', va='center',
# transform=ax.transAxes, color='grey', alpha=0.5)
# plt.gca().invert_yaxis()

# fig.tight_layout()


# STEPWISE REGRESSION
# Perform stepwise regression
print('STEPWISE REGRESSION\n')
# sfs = SequentialFeatureSelector(LinearRegression(),
# k_features=5,
# forward=False,
# scoring='accuracy',
# cv=None)
# selected_features = sfs.fit(X, y)
# print('Selected features:\n')
# print(selected_features)


# HYPOTHESIS TESTING
# t-test to confront accuracy, AUC, precision and F1 score
# between the balanced and unbalanced files

results_imbalanced_linreg = pd.read_csv('results_imbalanced_linreg.csv')
results_balanced_linreg = pd.read_csv('results_balanced_linreg.csv')

alpha = 0.05

for target in target_list:
    metric_balanced = results_balanced_linreg[target]
    metric_imbalanced = results_imbalanced_linreg[target]
    t_statistic, p_value = ttest_ind(metric_balanced, metric_imbalanced)
    print(f'T-TEST on balanced vs imbalanced for {target}')
    print(f'T-statistic: {t_statistic:.6f}')
    print(f'p_value: {p_value:.10f}')
    if p_value < alpha:
        print('Significant difference')
    else:
        print('No significant difference')

######################################################
# 3D PLOTTING TARGET METRIC VS 2 REGRESSORS
for target in target_list:
    z = dataset[target]

    for regressor in regressors_list:
        x = X[regressor].values.reshape(-1, 1)
        regressor_2 = regressor(index(regressor) + 1)
        for regressor_2 in regressor_list:
            y = X[regressor_2].values.reshape(-1, 1)

            regressors_1_2 = [x, y]
            model = LinearRegression()
            model_fit = model.fit(regressors_1_2, z)
            predicted = model.predict()

            x_min = min(x)
            x_max = max(x)
            step_x = floor((max(x) - min(x)) / size(x))

            x_pred = np.linspace(x_min, x_max, step)

            y_min = min(y)
            y_max = max(y)
            step_y = np.linspace(y_min, y_max, step_y)

            xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
            model_visualization = np.array([xx_pred.flatten(), y_pred.flatten()]).T

            plt.style.use('default')

            fig = plt.figure(figsize=(12, 4))

            ax1 = fig.add_subplot(131, projection='3d')
            ax2 = fig.add_subplot(132, projection='3d')
            ax3 = fig.add_subplot(133, projection='3d')

            axes = [ax1, ax2, ax3]

            for ax in axes:
                ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
                ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0, 0, 0, 0), s=20,
                           edgecolor='#70b3f0')
                ax.set_xlabel(regressor, fontsize=12)
                ax.set_ylabel(regressor_2, fontsize=12)
                ax.set_zlabel(target, fontsize=12)
                ax.locator_params(nbins=4, axis='x')
                ax.locator_params(nbins=5, axis='x')

            # ax1.text2D(0.2, 0.32, fontsize=13, ha='center', va='center',
            #           transform=ax1.transAxes, color='grey', alpha=0.5)

            ax1.view_init(elev=28, azim=120)
            ax2.view_init(elev=4, azim=114)
            ax3.view_init(elev=60, azim=165)

            fig_regression_2D.suptitle('$R^2 = %.2f$' % r2, fontsize=20)
            fig_regression_2D.tight_layout()

# TO DO: sistema i plot 2D e 3D
# csv and barplot for the measures
################################################################

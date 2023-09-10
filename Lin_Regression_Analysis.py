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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class LinearRegressionAnalysis:
    def __init__(self,
                 dataset_path,
                 ):
        self.dataset=pd.read_csv(dataset_path)
        self.encoder=ce.BinaryEncoder(cols=['balancing_method', 'classifier'])

    def prepare_data(self, X_cat, X_cont):        #give as input the categorical and the continuous regressor columns
        X_dummies=self.encoder.fit_transform(X_cat)
        X=pd.concat([X_cont, X_dummies], axis=1)
        return X

    def perform_linear_regression(self, target, regressors):
        X_cont=self.dataset[regressors]
        X_cat=self.dataset[['balancing_method','classifier']]

        X= self.prepare_data(X_cat, X_cont)
        y= self.dataset[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

        model = LinearRegression()
        model_fit = model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        MSE = mean_squared_error(y_test, predictions)
        mean_abs_error = mean_absolute_error(y_test, predictions)
        intercept = model_fit.intercept_
        coefficients = model_fit.coef_
        coefficient_of_determination = model_fit.score(X_test, y_test)
        SS_res = np.sum(np.square(y_test - predictions))
        SS_tot = np.sum(np.square(y_test - np.mean(y_test)))
        df_mod = X.shape[1]  # Degrees of freedom for the model
        df_res = y_test.shape[0] - df_mod  # Degrees of freedom for residuals
        MS_mod = SS_res / df_mod
        MS_res = SS_tot / df_res
        F = MS_mod / MS_res
        p_value = f.sf(F, df_mod, df_res)

        results = {
            'TARGET METRIC': target,
            'Mean squared error': MSE,
            'Mean absolute error': mean_abs_error,
            'Intercept': intercept,
            'Coefficients': coefficients,
            'Coefficient of determination': coefficient_of_determination,
            'p value': p_value,
            'F value': F
        }
        results_dataframe = pd.DataFrame(results)

        return results_dataframe

    def plot_target_vs_regressor(self, target, regressors):
        X_cont = self.dataset[regressors]
        X_cat = self.dataset[['balancing_method', 'classifier']]

        X = self.prepare_data(X_cat, X_cont)
        y = self.dataset[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

        model = LinearRegression()
        model_fit = model.fit(X_train, y_train)

        plt.figure(figsize=(12, 6))

        for regressor in regressors:
            x_values = X_cont[regressor]
            y_values = model_fit.intercept_ + model_fit.coef_[X.columns.get_loc(regressor)] * x_values
            plt.scatter(x_values, y_values, label=f'{regressor} - slope: {model_fit.coef_[X.columns.get_loc(regressor)]:.2f}',
                        alpha=0.5)

        plt.xlabel(', '.join(regressors))
        plt.ylabel(target)
        plt.title(f'{target} vs {", ".join(regressors)}')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    analyzer = LinearRegressionAnalysis('n_samples_experiment.csv_experiment.csv')
    target_metric = 'F1 score'
    regressors_list = ['class_ratio', 'n_samples', 'n_features']
    regressors_to_plot = ['class_ratio']

    results = analyzer.perform_linear_regression(target_metric, regressors_list)
    print(f"Linear Regression Results for {target_metric}:\n{results}")

    analyzer.plot_target_vs_regressor(target_metric, regressors_to_plot)
'''
import pandas as pd
from scipy.stats import ttest_ind


class Hypothesis_T_Test:
    def __init__(self, target, column, test_comb):
        self.target = target
        self.column = column
        self.test_comb = test_comb
        self.alpha = 0.05

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)

    def perform_t_test(self):
        # Filter the data based on the specified column and test combination
        filter_condition = (self.data[self.column] == self.test_comb[0]) | (self.data[self.column] == self.test_comb[1])
        filtered_data = self.data[filter_condition]

        # Split the data into two groups based on the test combination
        group1 = filtered_data[filtered_data[self.column] == self.test_comb[0]][self.target]
        group2 = filtered_data[filtered_data[self.column] == self.test_comb[1]][self.target]

        # Perform a t-test
        t_statistic, p_value = ttest_ind(group1, group2)

        print(f'T-TEST on {self.target} for {self.test_comb[0]} vs {self.test_comb[1]}')
        print(f'T-statistic: {t_statistic:.6f}')
        print(f'p_value: {p_value:.15f}')

        if p_value < self.alpha:
            print('Significant difference')
        else:
            print('No significant difference')


# Example usage:
if __name__ == "__main__":
    test = Hypothesis_T_Test(target='accuracy', column='classifier', test_comb=('Random Forest', 'XGboost'))
    test.load_data('results_balanced.csv')
    test.perform_t_test()
'''
""" 
import pandas as pd
from scipy.stats import ttest_ind

class Hypothesis_T_Test:
    def __init__(self, target, column, test_combinations):
        self.target = target
        self.column = column
        self.test_combinations = test_combinations
        self.alpha = 0.05

    def load_data(self, file_path):
        # Load the data from the CSV file
        self.data = pd.read_csv(file_path)

    def perform_t_tests(self):
        for test_comb in self.test_combinations:
            # Filter the data based on the specified column and test combination
            filter_condition = (
                    (self.data[self.column] == test_comb[0]) | (self.data[self.column] == test_comb[1])
            )
            filtered_data = self.data[filter_condition]

            # Split the data into two groups based on the test combination
            group1 = filtered_data[filtered_data[self.column] == test_comb[0]][self.target]
            group2 = filtered_data[filtered_data[self.column] == test_comb[1]][self.target]

            # Perform a t-test
            t_statistic, p_value = ttest_ind(group1, group2)

            print(f'T-TEST on {self.target} for {test_comb[0]} vs {test_comb[1]}')
            print(f'T-statistic: {t_statistic:.6f}')
            print(f'p_value: {p_value:.15f}')

            mean_group1 = group1.mean()
            mean_group2 = group2.mean()
            print(f'{test_comb[0]} has mean {self.target}={mean_group1}')
            print(f'{test_comb[1]} has mean {self.target}= {mean_group2}')

            if p_value < self.alpha:
                print('Significant difference')
                if mean_group1 > mean_group2:
                    print(f'{test_comb[0]} has a better {self.target}')
                else:
                    print(f'{test_comb[1]} has a better {self.target}')
            else:
                print('No significant difference')




# Example usage:
if __name__ == "__main__":
    test_balancers = [('SMOTE', 'ADASYN'), ('RandomOverSampler', 'SMOTE'), ('RandomOverSampler', 'ADASYN'),
                      ('BorderlineSMOTE', 'SMOTE'), ('BorderlineSMOTE', 'ADASYN'), ('BorderlineSMOTE', 'RandomOverSampler'),
                      ('BorderlineSMOTE', 'SVMSMOTE'), ('SVMSMOTE', 'ADASYN'), ('SVMSMOTE', 'SMOTE'), ('SVMSMOTE', 'RandomOverSampler')
                      ]
    test_classifiers=[('Logistic Regression', 'Decision Tree'), ('Logistic Regression', 'Random Forest'), ('Logistic Regression', 'XGboost'),
                      ('Logistic Regression', 'Lightgbm'), ('Decision Tree', 'Random Forest'), ('Decision Tree', 'XGboost'),
                      ('Decision Tree', 'Lightgbm'), ('Random Forest', 'XGboost'), ('Random Forest', 'Lightgbm'),
                      ('Lightgbm', 'XGboost')]

    target_list=['accuracy', 'precision', 'recall', 'F1 score', 'ROC AUC Score']

    hp_test_results=pd.dataframe()
    test = Hypothesis_T_Test(target='accuracy', column='balancing_method', test_combinations=test_balancers)
    test.load_data('results_balanced.csv')
    test.perform_t_tests()
"""


import pandas as pd
from scipy.stats import ttest_ind

class Hypothesis_T_Test:
    def __init__(self, target_list, column_bal, column_clas, test_combination_balancers, test_combination_classifiers):
        self.target_list = target_list
        self.column_bal = column_bal
        self.column_clas = column_clas
        self.test_combination_balancers = test_combination_balancers
        self.test_combination_classifiers = test_combination_classifiers
        self.alpha = 0.05

    def load_data(self, file_path):
        # Load the data from the CSV file
        self.data = pd.read_csv(file_path)

    def perform_t_tests(self):
        results = []

        for target in self.target_list:
            for test_comb_bal in self.test_combination_balancers:
                # Filter the data based on the specified column and test combination
                filter_condition = (
                    (self.data[self.column_bal] == test_comb_bal[0]) | (self.data[self.column_bal] == test_comb_bal[1])
                )
                filtered_data = self.data[filter_condition]

                # Split the data into two groups based on the test combination
                group1 = filtered_data[filtered_data[self.column_bal] == test_comb_bal[0]][target]
                group2 = filtered_data[filtered_data[self.column_bal] == test_comb_bal[1]][target]

                # Perform a t-test
                t_statistic, p_value = ttest_ind(group1, group2)

                mean_group1 = group1.mean()
                mean_group2 = group2.mean()

                result_row_bal = {
                    'Target measure': target,
                    'First method': test_comb_bal[0],
                    'Second method': test_comb_bal[1],
                    'Target mean value 1': mean_group1,
                    'Target mean value 2': mean_group2,
                    'T statistic': t_statistic,
                    'p value': p_value
                }
                results.append(result_row_bal)

            for test_comb_clas in self.test_combination_classifiers:
                # Filter the data based on the specified column and test combination
                filter_condition = (
                        (self.data[self.column_clas] == test_comb_clas[0]) | (
                            self.data[self.column_clas] == test_comb_clas[1])
                )
                filtered_data = self.data[filter_condition]

                # Split the data into two groups based on the test combination
                group1 = filtered_data[filtered_data[self.column_clas] == test_comb_clas[0]][target]
                group2 = filtered_data[filtered_data[self.column_clas] == test_comb_clas[1]][target]

                # Perform a t-test
                t_statistic, p_value = ttest_ind(group1, group2)

                mean_group1 = group1.mean()
                mean_group2 = group2.mean()

                result_row_clas = {
                    'Target measure': target,
                    'First method': test_comb_clas[0],
                    'Second method': test_comb_clas[1],
                    'Target mean value 1': mean_group1,
                    'Target mean value 2': mean_group2,
                    'T statistic': t_statistic,
                    'p value': p_value
                }
                results.append(result_row_clas)

        # Create a DataFrame from the results and save it to a CSV file
        results_df = pd.DataFrame(results)
        results_df.to_csv('hp_test_results.csv', index=False)

# Example usage:
if __name__ == "__main__":
    test_balancers = [('SMOTE', 'ADASYN'), ('RandomOverSampler', 'SMOTE'), ('RandomOverSampler', 'ADASYN'),
                      ('BorderlineSMOTE', 'SMOTE'), ('BorderlineSMOTE', 'ADASYN'), ('BorderlineSMOTE', 'RandomOverSampler'),
                      ('BorderlineSMOTE', 'SVMSMOTE'), ('SVMSMOTE', 'ADASYN'), ('SVMSMOTE', 'SMOTE'), ('SVMSMOTE', 'RandomOverSampler')
                      ]
    test_classifiers = [('Logistic Regression', 'Decision Tree'), ('Logistic Regression', 'Random Forest'), ('Logistic Regression', 'XGboost'),
                      ('Logistic Regression', 'Lightgbm'), ('Decision Tree', 'Random Forest'), ('Decision Tree', 'XGboost'),
                      ('Decision Tree', 'Lightgbm'), ('Random Forest', 'XGboost'), ('Random Forest', 'Lightgbm'),
                      ('Lightgbm', 'XGboost')]

    test_target_list = ['accuracy', 'precision', 'recall', 'F1 score', 'ROC AUC Score']

    test = Hypothesis_T_Test(target_list=test_target_list, column_bal='balancing_method', column_clas='classifier', test_combination_balancers=test_balancers, test_combination_classifiers=test_classifiers)
    test.load_data('results_balanced.csv')
    test.perform_t_tests()

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

    test = Hypothesis_T_Test(target='accuracy', column='balancing_method', test_combinations=test_balancers)
    test.load_data('results_balanced.csv')
    test.perform_t_tests()

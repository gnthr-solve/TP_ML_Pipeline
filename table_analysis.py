import pandas as pd
import numpy as np

"""
Distance Experiment
-------------------------------------------------------------------------------------------------------------------------------------------

dist_results_df = pd.read_csv('Experiments/cls_dist_std_mv_normal.csv', index_col=0)


eval_columns = ['accuracy', 'precision', 'recall', 'F1 score', 'ROC AUC Score']

#Using groupby on distances
#-------------------------------------------------------------------------------------------------------------------------------------------

grouped_dist_results = dist_results_df.groupby('cluster distance')

distance_mean_values = grouped_dist_results[eval_columns].mean()
distance_mean_values = distance_mean_values.round(3)

print(distance_mean_values)
#print(type(distance_mean_values))

distance_mean_values.to_csv('Analysed_Experiments/distance_mean_values.csv')


#Using groupby on features and distances
#-------------------------------------------------------------------------------------------------------------------------------------------

grouped_dist_results = dist_results_df.groupby(['n_features', 'cluster distance'])
feat_dist_mean_values = grouped_dist_results[eval_columns].mean()

feat_dist_mean_values = feat_dist_mean_values.round(3)
print(feat_dist_mean_values)

feat_dist_mean_values.to_csv('Analysed_Experiments/feat_dist_mean_values.csv')


#Using groupby on classifiers
#-------------------------------------------------------------------------------------------------------------------------------------------

grouped_dist_results = dist_results_df.groupby('classifier')

distance_mean_values = grouped_dist_results[eval_columns].mean()
distance_mean_values = distance_mean_values.round(3)

print(distance_mean_values)
#print(type(distance_mean_values))

distance_mean_values.to_csv('Analysed_Experiments/dist_clsf_mean_values.csv')

#Using groupby on classifiers
#-------------------------------------------------------------------------------------------------------------------------------------------

grouped_dist_results = dist_results_df.groupby('balancer')

distance_mean_values = grouped_dist_results[eval_columns].mean()
distance_mean_values = distance_mean_values.round(3)

print(distance_mean_values)
#print(type(distance_mean_values))

distance_mean_values.to_csv('Analysed_Experiments/dist_bal_mean_values.csv')
"""



"""
Alternative Bimodal Multinormal Experiment means at 5, -5
-------------------------------------------------------------------------------------------------------------------------------------------

def geometric_mean(x):
    return np.power(np.prod(x), 1/len(x))

bimodal_results_df = pd.read_csv('Experiments/bimodal_maj_experiment.csv', index_col=0)

bimodal_results_df = bimodal_results_df[['balancer','classifier','accuracy','precision','recall','F1 score','ROC AUC Score']]

#bimodal_results_df = bimodal_results_df.round(4)
print(bimodal_results_df)
#bimodal_results_df.to_csv('Analysed_Experiments/bimodal_full_table.csv')


eval_columns = ['accuracy', 'precision', 'recall', 'F1 score', 'ROC AUC Score']



#Using groupby on classifiers bimodals with geometric mean
#-------------------------------------------------------------------------------------------------------------------------------------------

grouped_bimodal_results = bimodal_results_df.groupby('classifier')

bimodal_geom_mean_values = grouped_bimodal_results[eval_columns].agg(geometric_mean).reset_index()
bimodal_geom_mean_values = bimodal_geom_mean_values.round(3)

#print(bimodal_geom_mean_values)
#print(type(bimodal_geom_mean_values))

#bimodal_geom_mean_values.to_csv('Analysed_Experiments/bimodal_geom_mean_values.csv')


#Using groupby on classifiers with arithmetic mean
#-------------------------------------------------------------------------------------------------------------------------------------------

grouped_bimodal_results = bimodal_results_df.groupby('classifier')

bimodal_mean_values = grouped_bimodal_results[eval_columns].mean()
bimodal_mean_values = bimodal_mean_values.round(3)

#print(bimodal_mean_values)
#print(type(bimodal_mean_values))

bimodal_mean_values.to_csv('Analysed_Experiments/bimodal_clsf_mean_values.csv')



#Using groupby on balancer with geometric mean
#-------------------------------------------------------------------------------------------------------------------------------------------

grouped_bimodal_results = bimodal_results_df.groupby('balancer')

bimodal_geom_mean_values = grouped_bimodal_results[eval_columns].agg(geometric_mean).reset_index()
bimodal_geom_mean_values = bimodal_geom_mean_values.round(3)

print(bimodal_geom_mean_values)
#print(type(bimodal_geom_mean_values))

#bimodal_geom_mean_values.to_csv('Analysed_Experiments/bimodal_geom_mean_values.csv')

#Using groupby on balancers with arithmetic mean
#-------------------------------------------------------------------------------------------------------------------------------------------

grouped_bimodal_results = bimodal_results_df.groupby('balancer')

bimodal_mean_values = grouped_bimodal_results[eval_columns].mean()
bimodal_mean_values = bimodal_mean_values.round(3)

print(bimodal_mean_values)
#print(type(bimodal_mean_values))

bimodal_mean_values.to_csv('Analysed_Experiments/bimodal_bal_mean_values.csv')
"""


"""
Alternative Bimodal Multinormal Experiment means at 3, -3
-------------------------------------------------------------------------------------------------------------------------------------------
"""
def geometric_mean(x):
    return np.power(np.prod(x), 1/len(x))

bimodal_results_df = pd.read_csv('Experiments/bimodal_maj_lower_dist.csv', index_col=0)

bimodal_results_df = bimodal_results_df[['balancer','classifier','accuracy','precision','recall','F1 score','ROC AUC Score']]

bimodal_results_df = bimodal_results_df.round(4)
print(bimodal_results_df)
bimodal_results_df.to_csv('Analysed_Experiments/bimodal_lower_dist_full_table.csv')


eval_columns = ['accuracy', 'precision', 'recall', 'F1 score', 'ROC AUC Score']



#Using groupby on classifiers bimodals with geometric mean
#-------------------------------------------------------------------------------------------------------------------------------------------

grouped_bimodal_results = bimodal_results_df.groupby('classifier')

bimodal_geom_mean_values = grouped_bimodal_results[eval_columns].agg(geometric_mean).reset_index()
bimodal_geom_mean_values = bimodal_geom_mean_values.round(3)

#print(bimodal_geom_mean_values)
#print(type(bimodal_geom_mean_values))

#bimodal_geom_mean_values.to_csv('Analysed_Experiments/bimodal_lower_dist_geom_mean_values.csv')


#Using groupby on classifiers with arithmetic mean
#-------------------------------------------------------------------------------------------------------------------------------------------

grouped_bimodal_results = bimodal_results_df.groupby('classifier')

bimodal_mean_values = grouped_bimodal_results[eval_columns].mean()
bimodal_mean_values = bimodal_mean_values.round(3)

#print(bimodal_mean_values)
#print(type(bimodal_mean_values))

bimodal_mean_values.to_csv('Analysed_Experiments/bimodal_lower_dist_clsf_mean_values.csv')



#Using groupby on balancer with geometric mean
#-------------------------------------------------------------------------------------------------------------------------------------------

grouped_bimodal_results = bimodal_results_df.groupby('balancer')

bimodal_geom_mean_values = grouped_bimodal_results[eval_columns].agg(geometric_mean).reset_index()
bimodal_geom_mean_values = bimodal_geom_mean_values.round(3)

print(bimodal_geom_mean_values)
#print(type(bimodal_geom_mean_values))

#bimodal_geom_mean_values.to_csv('Analysed_Experiments/bimodal_lower_dist_geom_mean_values.csv')

#Using groupby on balancers with arithmetic mean
#-------------------------------------------------------------------------------------------------------------------------------------------

grouped_bimodal_results = bimodal_results_df.groupby('balancer')

bimodal_mean_values = grouped_bimodal_results[eval_columns].mean()
bimodal_mean_values = bimodal_mean_values.round(3)

print(bimodal_mean_values)
#print(type(bimodal_mean_values))

bimodal_mean_values.to_csv('Analysed_Experiments/bimodal_lower_dist_bal_mean_values.csv')

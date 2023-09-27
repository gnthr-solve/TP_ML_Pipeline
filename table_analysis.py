import pandas as pd


"""
Distance Experiment
-------------------------------------------------------------------------------------------------------------------------------------------
"""
dist_results_df = pd.read_csv('Experiments/cls_dist_std_mv_normal.csv', index_col=0)


eval_columns = ['accuracy', 'precision', 'recall', 'F1 score', 'ROC AUC Score']
#Using groupby on distances
#-------------------------------------------------------------------------------------------------------------------------------------------


grouped_dist_results = dist_results_df.groupby('cluster distance')

distance_mean_values = grouped_dist_results[eval_columns].mean()
distance_mean_values = distance_mean_values.round(3)

print(distance_mean_values)
#print(type(distance_mean_values))

#distance_mean_values.to_csv('Analysed_Experiments/distance_mean_values.csv')


#Using groupby on features and distances
#-------------------------------------------------------------------------------------------------------------------------------------------

grouped_dist_results = dist_results_df.groupby(['n_features', 'cluster distance'])
feat_dist_mean_values = grouped_dist_results[eval_columns].mean()

feat_dist_mean_values = feat_dist_mean_values.round(3)
print(feat_dist_mean_values)

feat_dist_mean_values.to_csv('Analysed_Experiments/feat_dist_mean_values.csv')
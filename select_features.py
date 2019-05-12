from data import game_avg_features, game_avg_labels
from feature_selector import FeatureSelector
import pandas as pd

# Todo load all features instead of just offensive stats
features = pd.DataFrame(data=game_avg_features())
labels = pd.DataFrame(data=game_avg_labels())
labels = labels.drop(labels.columns[0], axis=1)  # Can only predict one value at a time

# Set features header
columns = ['A_FG', 'A_FGA', 'A_3P', 'A_3PA', 'A_FT', 'A_FTA', 'A_ORB', 'A_DRB', 'A_TRB', 'A_AST', 'A_STL', 'A_BLK', 'A_TOB', 'A_PF', 'A_AVG_PTS',
                'H_FG', 'H_FGA', 'H_3P', 'H_3PA', 'H_FT', 'H_FTA', 'H_ORB', 'H_DRB', 'H_TRB', 'H_AST', 'H_STL', 'H_BLK', 'H_TOB', 'H_PF', 'H_AVG_PTS']

features.columns = columns

fs = FeatureSelector(data=features, labels=labels)
fs.identify_missing(missing_threshold=0.9)
fs.identify_collinear(correlation_threshold=0.6)
# fs.identify_zero_importance(eval_metric='l2', task='regression')

print(fs.record_collinear.head())

# fs.plot_feature_importances()

# Results often say to drop: FGA, 3PA, FTA, TRB, AVG_PTS

from feature_selector import FeatureSelector
import pandas as pd
import numpy as np


def features(file='data/game_features_2016-19.csv', usecols=None):
    if usecols is None:
        usecols = ['GAME_ID', 'A_FG', 'A_3P', 'A_FT', 'H_FG', 'H_3P', 'H_FT']
    else:
        usecols = ['GAME_ID'] + usecols

    data = np.array(pd.read_csv(file, header=0, usecols=usecols)).tolist()
    data.sort(key=lambda x: x[0])
    data = list(map(lambda x: x[1:], data))

    return np.array(data)


def labels(file='data/game_features_2016-19.csv'):
    use_cols = ['GAME_ID', 'AWAY_POINTS', 'HOME_POINTS']

    data = np.array(pd.read_csv(file, header=0, usecols=use_cols)).tolist()
    data.sort(key=lambda x: x[0])
    data = list(map(lambda x: x[1:], data))

    return np.array(data)


# Set features header
columns = ['A_FG', 'A_FGA', 'A_3P', 'A_3PA', 'A_FT', 'A_FTA', 'A_ORB', 'A_DRB', 'A_TRB', 'A_AST', 'A_STL', 'A_BLK', 'A_TOB', 'A_PF', 'A_AVG_PTS',
                'H_FG', 'H_FGA', 'H_3P', 'H_3PA', 'H_FT', 'H_FTA', 'H_ORB', 'H_DRB', 'H_TRB', 'H_AST', 'H_STL', 'H_BLK', 'H_TOB', 'H_PF', 'H_AVG_PTS']

# Todo load all features instead of just offensive stats
features = pd.DataFrame(data=features(usecols=columns))
labels = pd.DataFrame(data=labels())
# labels = labels.drop(labels.columns[0], axis=1)  # Can only predict one value at a time

features.columns = columns
print(len(features.columns), len(columns))

fs = FeatureSelector(data=features, labels=labels)
fs.identify_missing(missing_threshold=0.9)
fs.identify_collinear(correlation_threshold=0.5)
fs.plot_collinear()
fs.identify_zero_importance(eval_metric='l2', task='regression')

# Drop:
# FGA, 3PA, FTA
# TRB
# AVG_PTS

print(fs.record_collinear.head())

# fs.plot_feature_importances()

# Results often say to drop: FGA, 3PA, FTA, TRB, AVG_PTS

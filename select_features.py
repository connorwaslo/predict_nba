from data import game_avg_features, game_avg_labels
from feature_selector import FeatureSelector
import pandas as pd

# Todo load all features instead of just offensive stats
features = pd.DataFrame(data=game_avg_features())
labels = pd.DataFrame(data=game_avg_labels())
labels = labels.drop(labels.columns[1], axis=1)  # Can only predict one value at a time

print(labels)

print(features.shape, labels.shape)

fs = FeatureSelector(data=features, labels=labels)
fs.identify_missing(missing_threshold=0.9)
fs.identify_collinear(correlation_threshold=0.6)
fs.identify_zero_importance(eval_metric='l2', task='regression')

from feature_selector import FeatureSelector
from data import adv_features, adv_labels
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
import numpy as np

columns = ['A_TS%', 'A_eFG%', 'A_3PAr', 'A_FTr', 'A_ORB%', 'A_DRB%',
                             'A_TRB%', 'A_AST%', 'A_STL%', 'A_BLK%', 'A_TOV%', 'A_ORtg', 'A_DRtg',
                             'H_TS%', 'H_eFG%', 'H_3PAr', 'H_FTr', 'H_ORB%', 'H_DRB%',
                             'H_TRB%', 'H_AST%', 'H_STL%', 'H_BLK%', 'H_TOV%', 'H_ORtg', 'H_DRtg'
                             ]

features = pd.DataFrame(adv_features())
labels = pd.DataFrame(adv_labels())

features.columns = columns
labels.columns = ['AWAY_POINTS', 'HOME_POINTS']

print(len(features), len(labels))

fs = FeatureSelector(data=features, labels=labels)
fs.identify_missing(missing_threshold=0.9)
fs.identify_collinear(correlation_threshold=0.5)
fs.plot_collinear()

fs2 = FeatureSelector(data=features, labels=labels[:,])
fs2.identify_zero_importance(eval_metric='l2', task='regression')
# fs2.identify_low_importance()

print(fs.record_collinear.head())

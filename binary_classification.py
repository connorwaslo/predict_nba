import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from data import adv_diff_features, adv_diff_labels

feature_cols = ['TS%', 'eFG%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%',
                             'AST%', 'STL%', 'BLK%', 'TOV%', 'ORtg', 'DRtg']
all_features = pd.DataFrame(adv_diff_features(None))
all_features.columns = feature_cols
all_features.pop('BLK%')
all_features.pop('STL%')
all_features.pop('TRB%')
all_features.pop('eFG%')
all_features.pop('3PAr')
all_features.pop('FTr')

all_labels = pd.DataFrame(adv_diff_labels())
all_labels.columns = ['POINT_DIFF']

# Convert labels to binary classes
all_labels.loc[all_labels.POINT_DIFF > 0] = 0
all_labels.loc[all_labels.POINT_DIFF < 0] = 1
# print(all_labels)

all_features['WINNER'] = all_labels['POINT_DIFF']

away_winners = all_features.loc[all_features.WINNER == 0]
# print(away_winners)

home_winners = all_features.loc[all_features.WINNER == 1]
# print(home_winners)

remove_count = len(home_winners) - len(away_winners)
# print(remove_count)

# Randomly remove rows in home_winners in order to undersample and even out the classes
np.random.seed(10)
drop_games = np.random.choice(home_winners.index, remove_count, replace=False)
home_winners = home_winners.drop(drop_games)
# print(home_winners)

features = pd.concat([away_winners, home_winners], axis=0)
print(features.columns)
labels = features.pop('WINNER')

train_x, val_x, train_y, val_y = train_test_split(features, labels, shuffle=True)
log = LogisticRegression()
log.fit(train_x, train_y)
log_pred = log.predict(val_x)
print('Log Reg Accuracy:', accuracy_score(val_y, log_pred))

sgd = SGDClassifier()
sgd.fit(train_x, train_y)
sgd_pred = sgd.predict(val_x)
print('SGD Accuracy:', accuracy_score(val_y, sgd_pred))

gau = GaussianNB()
gau.fit(train_x, train_y)
gau_pred = gau.predict(val_x)
print('GaussianNB Accuracy:', accuracy_score(val_y, gau_pred))

ber = BernoulliNB()
ber.fit(train_x, train_y)
ber_pred = ber.predict(val_x)
print('BernoulliNB Accuracy:', accuracy_score(val_y, ber_pred))

forest = RandomForestClassifier(n_estimators=500)
forest.fit(train_x, train_y)
forest_pred = forest.predict(val_x)
print('RF Accuracy:', accuracy_score(val_y, forest_pred))

vote = VotingClassifier([('lg', log), ('gau', gau), ('ber', ber), ('rf', forest)])
vote.fit(train_x, train_y)
vote_pred = vote.predict(val_x)
print('Vote Accuracy:', accuracy_score(val_y, vote_pred))

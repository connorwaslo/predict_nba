from data import features, labels
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd


feats = features()
labs = labels()

train_split = int(len(feats) * 0.9)

train_x = feats[:train_split]
train_y = labs[:train_split]

val_x = feats[train_split:]
val_y = labs[train_split:]

clf = RandomForestRegressor(n_estimators=500)

clf.fit(train_x, train_y)

preds = clf.predict(val_x)

wins = 0
games = 0
a_wins = 0
h_wins = 0
for pred, outcome in zip(preds, val_y):
    if outcome[0] > outcome[1]:
        a_wins += 1
    else:
        h_wins += 1

    if pred[0] > pred[1] and outcome[0] > outcome[1]:
        wins += 1
    elif pred[1] > pred[0] and outcome[1] > outcome[0]:
        wins += 1

    games += 1

print('Wins:', wins)
print('Losses:', int(games - wins))
print('%:', float(wins / games))

print('Away Wins:', a_wins)
print('Home Wins:', h_wins)

    # diff = [outcome[0] - pred[0], outcome[1] - pred[1]]
    # print(pred, outcome, diff)

print('Accuracy:', metrics.explained_variance_score(val_y, preds))

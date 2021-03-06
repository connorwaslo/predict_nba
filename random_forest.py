from data import features, classifier_labels
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import csv
import numpy as np


def k_folds_split(folds=10, iter=0, features=[], labels=[]):
    val_set_size = int(len(features) / folds)
    val_start = iter * val_set_size

    train_x = features[:val_start] + features[val_start + val_set_size:]
    train_y = labels[:val_start] + labels[val_start + val_set_size:]

    val_x = features[val_start:val_start + val_set_size]
    val_y = labels[val_start:val_start + val_set_size]

    return train_x, train_y, val_x, val_y


feats = features()
labs = classifier_labels()

train_split = int(len(feats) * 0.9)

folds = 10

for i in range(folds):
    train_x, train_y, val_x, val_y = k_folds_split(iter=i, features=feats, labels=labs)

    clf = RandomForestClassifier(n_estimators=500)

    clf.fit(train_x, train_y)

    preds = clf.predict(val_x)

    wins = 0
    games = 0
    a_wins = 0
    h_wins = 0
    for pred, outcome in zip(preds, val_y):
        outcome = outcome[0]

        if pred == outcome:
            wins += 1
        if outcome == 0:
            a_wins += 1
        else:
            h_wins += 1

        games += 1

    print('** ITERATION', i, '**')
    print('Wins:', wins)
    print('Losses:', int(games - wins))
    print('%:', float(wins / games))

    print('Away Wins:', a_wins)
    print('Home Wins:', h_wins)

    # diff = [outcome[0] - pred[0], outcome[1] - pred[1]]
    # print(pred, outcome, diff)

    accuracy = metrics.accuracy_score(val_y, preds)
    print('Accuracy:', accuracy)
    print('****************')

    file = 'Random Forest 10 Folds No Rebounds.csv'
    with open('result_tracking/Random Forest/' + file, 'a', newline='') as f:
        writer = csv.writer(f)

        writer.writerow([i, wins, int(games - wins), float(wins / games), a_wins, h_wins, accuracy])


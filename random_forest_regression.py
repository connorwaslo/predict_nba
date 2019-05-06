from data import features, labels, odds, test_features, test_labels
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from betting import moneyline_profit
import csv


def k_folds_split(folds=10, iter=0, features=[], labels=[], odds=[]):
    val_set_size = int(len(features) / folds)
    val_start = iter * val_set_size

    train_x = features[:val_start] + features[val_start + val_set_size:]
    train_y = labels[:val_start] + labels[val_start + val_set_size:]

    val_x = features[val_start:val_start + val_set_size]
    val_y = labels[val_start:val_start + val_set_size]

    odds = odds[val_start:val_start + val_set_size]

    return train_x, train_y, val_x, val_y, odds


# feats = features()
# labs = labels()
# odds = odds()

# train_split = int(len(feats) * 0.9)

# folds = 10

for i in range(1):
    # train_x, train_y, val_x, val_y, odds = k_folds_split(iter=i, features=feats, labels=labs, odds=odds)
    train_x, train_y = features() + test_features()[:900], labels() + test_labels()[:900]
    val_x, val_y = test_features()[900:], test_labels()[900:]
    odds = odds()

    clf = RandomForestRegressor(n_estimators=500)

    clf.fit(train_x, train_y)

    preds = clf.predict(val_x)

    wins = 0
    games = 0
    a_wins = 0
    h_wins = 0

    ml_profit = 0.0
    ml_bets = 0

    for pred, outcome, game_odds in zip(preds, val_y, odds):
        if outcome[0] > outcome[1]:
            a_wins += 1
        else:
            h_wins += 1

        if pred[0] > pred[1] and outcome[0] > outcome[1]:
            ml_profit += moneyline_profit(pred_winner=1, away_ml=game_odds[0], home_ml=game_odds[1])
            ml_bets += 10
            wins += 1
        elif pred[1] > pred[0] and outcome[1] > outcome[0]:
            ml_profit += moneyline_profit(pred_winner=0, away_ml=game_odds[0], home_ml=game_odds[1])
            ml_bets += 10
            wins += 1
        else:
            ml_profit -= 10

        games += 1

    print('** ITERATION', i, '**')
    print('Wins:', wins)
    print('Losses:', int(games - wins))
    print('%:', float(wins / games))

    print('Moneyline profit:', ml_profit, ml_bets, float(ml_profit / ml_bets))

    print('Away Wins:', a_wins)
    print('Home Wins:', h_wins)

    # diff = [outcome[0] - pred[0], outcome[1] - pred[1]]
    # print(pred, outcome, diff)

    accuracy = metrics.explained_variance_score(val_y, preds)
    print('Accuracy:', accuracy)
    print('****************')

    file = 'Random Forest Regression 10 Folds 4 Major Stats.csv'
    with open('result_tracking/' + file, 'a', newline='') as f:
        writer = csv.writer(f)

        writer.writerow([i, wins, int(games - wins), float(wins / games), a_wins, h_wins, accuracy])


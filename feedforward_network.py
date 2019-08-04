import tensorflow as tf
import numpy as np
import csv
from data import features, labels, odds, classifier_labels, test_features, test_labels, features_2016_19, labels_2016_19, game_avg_features, game_avg_labels
from betting import spread_profit, moneyline_profit, totals_profit


def loss(y_true, y_pred):
    return abs(y_true - y_pred)


def norm_mean_square_error(y_true, y_pred):
    mse = ((y_true - y_pred) ** 2).mean()
    return mse / (max(y_true) - min(y_true))


def k_folds_split(folds=10, iter=0, features=[], labels=[]):
    val_set_size = int(len(features) / folds)
    val_start = iter * val_set_size

    print('Features:', len(features), 'Labels:', len(labels))

    train_x = np.array(features[:val_start] + features[val_start + val_set_size:])
    train_y = np.array(labels[:val_start] + labels[val_start + val_set_size:])

    val_x = np.array(features[val_start:val_start + val_set_size])
    val_y = np.array(labels[val_start:val_start + val_set_size])

    return train_x, train_y, val_x, val_y


def season_split(features=[], labels=[]):
    train_x, train_y = np.array(features[:-1230]), np.array(labels[:-1230])
    test_x, test_y = np.array(features[-1230:]), np.array(labels[-1230:])

    return train_x, train_y, test_x, test_y


accuracy = 0.0

first_layer_sizes = [256]
second_layer_factors = [1] # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1

for i in range(1):
    # train_x, train_y, test_x, test_y = k_folds_split(iter=i, features=game_avg_features(), labels=game_avg_labels())
    train_x, train_y, test_x, test_y = season_split(features=game_avg_features(), labels=game_avg_labels())
    # train_x, train_y = np.array(features_2016_19()[:-369]), np.array(labels_2016_19()[:-369])

    print(train_y)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=train_x[0].shape),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(2)
    ])
    # model.add(tf.keras.layers.Dense(int(first_layer * layer_factor + 0.5), activation=tf.nn.relu))
    # model.add(tf.layers.Dropout(0.1))

    # model.add(tf.keras.layers.Dense(2))

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_squared_error'])

    # Print model summary
    model.summary()

    # Callbacks
    # early_stop = callbacks.EarlyStopping(min_delta=0.01, restore_best_weights=True)
    early_stop = tf.keras.callbacks.EarlyStopping(min_delta=0.01, restore_best_weights=True)
    csv_log = tf.keras.callbacks.CSVLogger('result_tracking/Feed Forward/No Rebounds.csv')

    # test_x, test_y = np.array(test_features()), np.array(test_labels())
    # print(test_x[0].shape, train_x[0].shape)

    history = model.fit(train_x, train_y, epochs=200, verbose=1, callbacks=[early_stop, csv_log])
    # print(model.evaluate(val_x, val_y))

    predictions = model.predict(test_x)

    # predictions = model.predict(val_x)
    # odds_start = i * len(test_x)
    # game_odds = odds()[odds_start:odds_start + len(test_x)]
    game_odds = odds()[-1230:]

    wins = 0
    BET_SIZE = 10.00

    ml_profit = 0
    total_pts_profit = 0
    total_bet = 0
    total_pts_bet = 0

    total_pts_correct = 0

    moneyline_rois = []

    betting_data = [['Predicted Spread', 'Actual Spread', 'Spread Profit', 'Away ML', 'Home ML', 'ML Profit', 'Over/Under', 'Over/Under Profit']]

    count = 0
    for pred, actual, betting_odds in zip(predictions, test_y, game_odds):
        # Logic for predicting scores
        pred_spread = int(pred[1] - pred[0] + 0.5)
        actual_spread = int(actual[1] - actual[0])

        curr_bet = [pred_spread, actual_spread, 0, '', betting_odds[0], betting_odds[1]]

        print(pred, actual, [actual[0] - pred[0], actual[1] - pred[1]], pred_spread, actual_spread, actual_spread - pred_spread)
        total_bet += BET_SIZE
        # If the away team wins by prediction and in reality
        if pred[0] > pred[1] and actual[0] > actual[1]:
            profit = moneyline_profit(bet_size=BET_SIZE, pred_winner=0, away_ml=betting_odds[0], home_ml=betting_odds[1])
            ml_profit += profit
            curr_bet.append(profit)
            print('Win', profit, ' Total:', ml_profit, ' | Odds:', betting_odds[0])
            wins += 1
        # If the home team wins by prediction and in reality
        elif pred[0] < pred[1] and actual[0] < actual[1]:
            profit = moneyline_profit(bet_size=BET_SIZE, pred_winner=1, away_ml=betting_odds[0], home_ml=betting_odds[1])
            ml_profit += profit
            curr_bet.append(profit)
            print('Win', profit, ' Total:', ml_profit, ' | Odds:', betting_odds[1])
            wins += 1
        else:
            ml_profit -= BET_SIZE
            curr_bet.append(-10)
            print('Loss, subtracting monies', ml_profit)

        if betting_odds[3] != 'pk':
            curr_bet.append(betting_odds[3])
            if int(pred[1] + pred[0]) > float(betting_odds[3]) and int(actual[1] + actual[0]) > float(betting_odds[3]):
                total_pts_profit += totals_profit()
                curr_bet.append(totals_profit())
                total_pts_correct += 1
            elif int(pred[1] + pred[0]) < float(betting_odds[3]) and int(actual[1] + actual[0]) < float(betting_odds[3]):
                total_pts_profit += totals_profit()
                curr_bet.append(totals_profit())
                total_pts_correct += 1
            else:
                total_pts_profit -= BET_SIZE

            total_pts_bet += BET_SIZE
        else:
            curr_bet.append('N/A')

        print(count)

        if count > 0 and (count + 1) % 123 == 0:
            print('One tenth')
            moneyline_rois.append([ml_profit, total_bet, total_pts_profit, total_pts_bet])

            ml_profit = 0
            total_pts_profit = 0
            total_bet = 0
            total_pts_bet = 0

        betting_data.append(curr_bet)
        count += 1

        with open('result_tracking/Feed Forward/Profits/2019 Bets.csv', 'a', newline='') as f:
            writer = csv.writer(f)

            writer.writerow([pred[0], pred[1], actual[0], actual[1]] + betting_odds)

    print(wins, int(len(predictions) - wins), float(wins / len(predictions)))
    # print('Moneyline:', ml_profit, total_bet, float(ml_profit / 12300))
    # print('Total Points:', total_pts_profit, total_pts_bet, float(total_pts_profit / total_pts_bet))
    # print('Total Points Predicted Correctly:', total_pts_correct, int(total_pts_bet / 10), float(total_pts_correct / int(total_pts_bet / 10)))

    # file = str(first_layer) + 'x' + str(layer_factor) + ' FFNN Team Avgs FG 3P FT - 10 Folds 2016-19.csv'
    file = 'Predict 2019.csv'
    with open('result_tracking/Feed Forward/Profits/' + file, 'a', newline='') as f:
        writer = csv.writer(f)

        # writer.writerow([i, wins, int(len(predictions) - wins), float(wins / len(predictions)), ml_profit, total_bet, float(ml_profit / total_bet),
        #                  total_pts_profit, total_pts_bet, float(total_pts_profit / total_pts_bet)])

        writer.writerow([wins, int(len(predictions) - wins), float(wins / len(predictions))])
        for tenth in moneyline_rois:
            print(tenth)
            writer.writerow(tenth)

        writer.writerow(['-'])

    # with open('result_tracking/Feed Forward/Profits/2015-19.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #
    #     for row in betting_data:
    #         writer.writerow(row)

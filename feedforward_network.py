import tensorflow as tf
import numpy as np
import csv
from data import features, labels, odds, classifier_labels, test_features, test_labels, features_2016_19, labels_2016_19, game_avg_features, game_avg_labels
from betting import spread_profit, moneyline_profit, totals_profit


def loss(y_true, y_pred):
    return abs(y_true - y_pred)


def k_folds_split(folds=10, iter=0, features=[], labels=[]):
    val_set_size = int(len(features) / folds)
    val_start = iter * val_set_size

    print('Features:', len(features), 'Labels:', len(labels))

    train_x = np.array(features[:val_start] + features[val_start + val_set_size:])
    train_y = np.array(labels[:val_start] + labels[val_start + val_set_size:])

    val_x = np.array(features[val_start:val_start + val_set_size])
    val_y = np.array(labels[val_start:val_start + val_set_size])

    return train_x, train_y, val_x, val_y


accuracy = 0.0

for runs in range(10):
    for i in range(10):
        train_x, train_y, test_x, test_y = k_folds_split(iter=i, features=game_avg_features(), labels=game_avg_labels())
        # train_x, train_y = np.array(features_2016_19()[:-369]), np.array(labels_2016_19()[:-369])

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=train_x[0].shape),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2)
        ])

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
        # game_odds = odds()

        wins = 0
        BET_SIZE = 10.00

        ml_profit = 0
        total_pts_profit = 0
        total_bet = 0
        total_pts_bet = 0

        total_pts_correct = 0

        for pred, actual in zip(predictions, test_y):
            # Logic for predicting scores
            pred_spread = int(pred[1] - pred[0])
            actual_spread = int(actual[1] - actual[0])

            # if odds[3] != 'pk':
            #     if int(pred[1] + pred[0]) > float(odds[3]) and int(actual[1] + actual[0]) > float(odds[3]):
            #         total_pts_profit += totals_profit()
            #         total_pts_correct += 1
            #     elif int(pred[1] + pred[0]) < float(odds[3]) and int(actual[1] + actual[0]) < float(odds[3]):
            #         total_pts_profit += totals_profit()
            #         total_pts_correct += 1
            #     else:
            #         total_pts_profit -= BET_SIZE
            #
            #     total_pts_bet += BET_SIZE

            print(pred, actual, [actual[0] - pred[0], actual[1] - pred[1]], pred_spread, actual_spread, actual_spread - pred_spread)
            # total_bet += BET_SIZE
            if pred[0] > pred[1] and actual[0] > actual[1]:
                # ml_profit += moneyline_profit(bet_size=BET_SIZE, pred_winner=0, actual_winner=0, away_ml=odds[0], home_ml=odds[1])
                wins += 1
            elif pred[0] < pred[1] and actual[0] < actual[1]:
                # ml_profit += moneyline_profit(bet_size=BET_SIZE, pred_winner=0, actual_winner=0, away_ml=odds[0], home_ml=odds[1])
                wins += 1
            # else:
            #     ml_profit -= BET_SIZE

        print(wins, int(len(predictions) - wins), float(wins / len(predictions)))
        # print('Moneyline:', ml_profit, total_bet, float(ml_profit / total_bet))
        # print('Total Points:', total_pts_profit, total_pts_bet, float(total_pts_profit / total_pts_bet))
        # print('Total Points Predicted Correctly:', total_pts_correct, int(total_pts_bet / 10), float(total_pts_correct / int(total_pts_bet / 10)))

        file = 'FFNN Team Avgs FG 3P FT - 10 Folds.csv'
        with open('result_tracking/Feed Forward/' + file, 'a', newline='') as f:
            writer = csv.writer(f)

            writer.writerow([i, wins, int(len(predictions) - wins), float(wins / len(predictions))])

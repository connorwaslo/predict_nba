import tensorflow as tf
from sklearn.model_selection import train_test_split
from data import adv_features, adv_labels, odds
import csv


odds = odds(['18-19'])  # Get the betting odds for 2018-19 season

network_sizes = []

# Test each network size
for size in network_sizes:

    with open('result_tracking/Feed Forward/Advanced Stats/Rtgs TS TOV Correct.csv', 'a', newline='') as f:
        writer = csv.writer(f)

        writer.writerow([])
        writer.writerow([size])

    # Run the network 10 times to calculate an average from that
    for runs in range(10):
        train_x, test_x, train_y, test_y = train_test_split(adv_features(),
                                                            adv_labels(),
                                                            test_size=0.1)

        val_x = adv_features(['2018-19'])
        val_y = adv_labels(['2018-19'])

        print(len(train_x), len(train_y))
        print(len(test_x), len(test_y))
        print(len(val_x), len(val_y))

        print(train_x[0], train_y[0])

        away_wins = 0
        home_wins = 0
        for game in val_y:
            if game[0] > game[1]:
                away_wins += 1
            else:
                home_wins += 1

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=train_x[0].shape),
            tf.keras.layers.Dense(size, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(2)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.0001)
        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mean_squared_error', 'mean_absolute_error'])

        model.summary()

        early_stop = tf.keras.callbacks.EarlyStopping(min_delta=0.01, patience=10, restore_best_weights=True)
        csv_log = tf.keras.callbacks.CSVLogger('result_tracking/Feed Forward/No Rebounds.csv')

        history = model.fit(x=train_x, y=train_y, epochs=10000, callbacks=[early_stop, csv_log], validation_data=(test_x, test_y), shuffle=True)

        predictions = model.predict(val_x)

        ml_wins = 0
        spread_wins = 0
        for pred, actual, betting_odds in zip(predictions, val_y, odds):
            # Moneyline
            if pred[0] > pred[1] and actual[0] > actual[1]:
                ml_wins += 1
            elif pred[1] > pred[0] and actual[1] > actual[0]:
                ml_wins += 1

            # Spread
            pred_spread = pred[0] - pred[1]  # + away win, - home win
            actual_spread = actual[0] - actual[1]
            spread = betting_odds[3]

            # Check for the favorite
            away_favorite = False
            if betting_odds[0] < 0:
                away_favorite = True

            if away_favorite:
                # Did the away team beat the spread?
                if actual[0] - actual[1] > spread:
                    if pred[0] - pred[1] > spread:
                        spread_wins += 1
                elif actual[1] > actual[0]:
                    if pred[1] > pred[0]:
                        spread_wins += 1
            # If the home team is the favorite
            else:
                if actual[1] - actual[0] > spread:
                    if pred[1] - pred[0] > spread:
                        spread_wins += 1
                elif actual[0] > actual[1]:
                    if pred[0] > pred[1]:
                        spread_wins += 1

        print('Moneyline:', ml_wins, len(predictions), float(ml_wins / len(predictions)))
        print('Spread:', spread_wins, len(predictions), float(spread_wins / len(predictions)))
        print(away_wins, home_wins, 'Home%:', float(home_wins / (away_wins + home_wins)))

        with open('result_tracking/Feed Forward/Advanced Stats/Rtgs TS TOV Correct.csv', 'a', newline='') as f:
            writer = csv.writer(f)

            writer.writerow([ml_wins, len(predictions) - ml_wins, len(predictions), float(ml_wins / len(predictions)), '', '', '', '',
                             spread_wins, len(predictions) - spread_wins, len(predictions), float(spread_wins / len(predictions))])

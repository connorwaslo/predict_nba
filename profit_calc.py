import tensorflow as tf
from sklearn.model_selection import train_test_split
from data import adv_features, adv_labels, odds
from betting import moneyline_profit
import matplotlib.pyplot as plt

odds = odds(['18-19'])

train_x, test_x, train_y, test_y = train_test_split(adv_features(), adv_labels(), test_size=0.1)
val_x = adv_features(['2018-19'])
val_y = adv_labels(['2018-19'])

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=train_x[0].shape),
    tf.keras.layers.Dense(130, activation=tf.nn.relu),
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

BET_SIZE = 10
ml_profit = 0
total_bet = 0
wins = 0
totals = []

for pred, actual, betting_odds in zip(predictions, val_y, odds):
    total_bet += BET_SIZE
    # If the away team wins by prediction and in reality
    if pred[0] > pred[1] and actual[0] > actual[1]:
        profit = moneyline_profit(bet_size=BET_SIZE, pred_winner=0, away_ml=betting_odds[0], home_ml=betting_odds[1])
        ml_profit += profit
        wins += 1
        print('Win', profit, ' Total:', ml_profit, ' | Odds:', betting_odds[0])

    # If the home team wins by prediction and in reality
    elif pred[0] < pred[1] and actual[0] < actual[1]:
        profit = moneyline_profit(bet_size=BET_SIZE, pred_winner=1, away_ml=betting_odds[0], home_ml=betting_odds[1])
        ml_profit += profit
        print('Win', profit, ' Total:', ml_profit, ' | Odds:', betting_odds[1])
        wins += 1
    else:
        ml_profit -= BET_SIZE
        print('Loss, subtracting monies', ml_profit)

    totals.append(ml_profit)

print(wins, len(predictions), float(wins / len(predictions)))
print('Profit:', ml_profit, ' / ', total_bet)
print('ROI:', float(ml_profit / total_bet))

plt.plot(totals)
plt.show()

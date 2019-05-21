import tensorflow as tf
from sklearn.model_selection import train_test_split
from data import adv_diff_features, adv_diff_labels, odds
from betting import moneyline_profit, spread_profit
import matplotlib.pyplot as plt

odds = odds(['18-19'])

train_x, test_x, train_y, test_y = train_test_split(adv_diff_features(None), adv_diff_labels(), test_size=0.1)
val_x = adv_diff_features(['2018-19'])
val_y = adv_diff_labels(['2018-19'])

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=train_x[0].shape),
    tf.keras.layers.Dense(160, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)
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

sp_wins = 0
sp_profit = 0
total_spread_bet = 0

wins1 = 0
twins1 = 0
wins2 = 0
twins2 = 0
wins3 = 0
twins3 = 0
wins4 = 0
twins4 = 0

min_line = -250

for pred, actual, betting_odds in zip(predictions, val_y, odds):
    # Calc spread profits
    if betting_odds[2] != 'pk' and betting_odds != '':
        spread = abs(float(betting_odds[2]))

        away_fav = False
        if float(betting_odds[0]) < 0:
            away_fav = True

        # If away is the underdog
        if not away_fav:
            # If away team (underdog) wins
            if pred > 0 and actual > 0:
                sp_wins += 1
                sp_profit += spread_profit()
                pass
            # If Home (fav) wins and covers the spread
            elif pred < 0 and actual < 0 and abs(pred) > spread and abs(actual) > spread:
                # Win
                sp_wins += 1
                sp_profit += spread_profit()
                pass
            # If Home (fav) wins, but doesn't cover the spread and we picked the Away (underdog)
            elif pred > 0 and actual < 0 and abs(pred) < spread and abs(actual) < spread:
                sp_wins += 1
                sp_profit += spread_profit()
                pass
            else:
                sp_profit -= 10
        else:
            # If Home (underdog) wins
            if pred < 0 and actual < 0:
                sp_wins += 1
                sp_profit += spread_profit()
                pass
            # Away (fav) wins and covers the spread
            elif pred > 0 and actual > 0 and pred > spread and actual > spread:
                sp_wins += 1
                sp_profit += spread_profit()
                pass
            elif pred < 0 and actual > 0 and abs(pred) < spread and actual < spread:
                sp_wins += 1
                sp_profit += spread_profit()
            else:
                sp_profit -= 10
        total_spread_bet += 10

    # Todo: test accuracy in each score region to determine which is actually most confident
    # If the away team wins by prediction and in reality
    if pred > 0 and actual > 0:
        # Only bet on game if their lines is better than -300
        if betting_odds[0] >= min_line:
            # if abs(pred) <= 5:
            #     wins1 += 1
            #     twins1 += 1
            #     profit = moneyline_profit(bet_size=BET_SIZE, pred_winner=0, away_ml=betting_odds[0], home_ml=betting_odds[1])
            if 6 <= abs(pred) <= 10:
                wins2 += 1
                twins2 += 1
                profit = moneyline_profit(bet_size=2*BET_SIZE, pred_winner=0, away_ml=betting_odds[0], home_ml=betting_odds[1])
            # elif abs(pred) <= 15:
            #     wins3 += 1
            #     twins3 += 1
            #     profit = moneyline_profit(bet_size=3*BET_SIZE, pred_winner=0, away_ml=betting_odds[0], home_ml=betting_odds[1])
            # else:
            #     wins4 += 1
            #     twins4 += 1
            #     profit = moneyline_profit(bet_size=4*BET_SIZE, pred_winner=0, away_ml=betting_odds[0], home_ml=betting_odds[1])
                ml_profit += profit
                print('Away Win', profit, ' Total:', ml_profit, ' | Odds:', betting_odds[0])

        wins += 1

    # If the home team wins by prediction and in reality
    elif pred < 0 and actual < 0:
        # Only bet on game if their lines is better than -300
        if betting_odds[1] >= min_line:
            # if abs(pred) <= 5:
            #     wins1 += 1
            #     twins1 += 1
            #     total_bet += BET_SIZE
            #     profit = moneyline_profit(bet_size=BET_SIZE, pred_winner=1, away_ml=betting_odds[0], home_ml=betting_odds[1])
            if 6 <= abs(pred) <= 10:
                wins2 += 1
                twins2 += 1
                total_bet += 2 * BET_SIZE
                profit = moneyline_profit(bet_size=2*BET_SIZE, pred_winner=1, away_ml=betting_odds[0], home_ml=betting_odds[1])
            # elif abs(pred) <= 15:
            #     wins3 += 1
            #     twins3 += 1
            #     profit = moneyline_profit(bet_size=3*BET_SIZE, pred_winner=1, away_ml=betting_odds[0], home_ml=betting_odds[1])
            # else:
            #     wins4 += 1
            #     twins4 += 1
            #     total_bet += 3 * BET_SIZE
            #     profit = moneyline_profit(bet_size=4*BET_SIZE, pred_winner=1, away_ml=betting_odds[0], home_ml=betting_odds[1])
            # profit = moneyline_profit(bet_size=BET_SIZE, pred_winner=1, away_ml=betting_odds[0], home_ml=betting_odds[1])
                ml_profit += profit
                print('Home Win', profit, ' Total:', ml_profit, ' | Odds:', betting_odds[1])

        wins += 1
    else:
        if pred > 0 and actual < 0:
            if betting_odds[0] > min_line:
                # if abs(pred) <= 5:
                #     twins1 += 1
                #     loss = BET_SIZE
                #     total_bet += BET_SIZE
                if 6 <= abs(pred) <= 10:
                    twins2 += 1
                    loss = 2 * BET_SIZE
                    total_bet += 2 * BET_SIZE
                # elif abs(pred) <= 15:
                #     twins3 += 1
                #     loss = 3 * BET_SIZE
                #     total_bet += 3 * BET_SIZE
                # else:
                #     twins4 += 1
                #     total_bet += 4 * BET_SIZE
                #     loss = 4 * BET_SIZE

                    ml_profit -= loss
                    print('Loss, subtracting', ml_profit)
        elif pred < 0 and actual > 0:
            if betting_odds[1] > min_line:
                # if abs(pred) <= 5:
                #     twins1 += 1
                #     loss = BET_SIZE
                #     total_bet += BET_SIZE
                if 6 <= abs(pred) <= 10:
                    twins2 += 1
                    loss = 2 * BET_SIZE
                    total_bet += 2 * BET_SIZE
                # elif abs(pred) <= 15:
                #     twins3 += 1
                #     loss = 3 * BET_SIZE
                #     total_bet += 3 * BET_SIZE
                # else:
                #     twins4 += 1
                #     total_bet += 4 * BET_SIZE
                #     loss = 4 * BET_SIZE

                    ml_profit -= loss
                    print('Loss, subtracting', ml_profit)

    totals.append(ml_profit)

print(wins, len(predictions), float(wins / len(predictions)))
print('ML Profit:', ml_profit, ' / ', total_bet)
print('Spread Profit:', sp_profit, ' / ', total_spread_bet)
print('Spread Wins:', sp_wins, ' / ', int(total_spread_bet / 10))
print('ROI:', float(ml_profit / total_bet))

# print('Wins1:', wins1, twins1, float(wins1 / twins1))
print('Wins2:', wins2, twins2, float(wins2 / twins2))
# print('Wins3:', wins3, twins3, float(wins3 / twins3))
# print('Wins4:', wins4, twins4, float(wins4 / twins4))

plt.plot(totals)
plt.show()

import tensorflow as tf
from sklearn.model_selection import train_test_split
from data import adv_diff_features, adv_diff_labels, odds
from betting import moneyline_profit, spread_profit
import matplotlib.pyplot as plt

# Use moneyline odds from most recent season
odds = odds(['18-19'])

# Train test split including shuffle of data up til 2018
train_x, test_x, train_y, test_y = train_test_split(adv_diff_features(None), adv_diff_labels(), test_size=0.1)

# Features and labels for most recent season - should tell us how well the model generalized
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

"""
Here's where we get into the nitty-gritty logic for figuring out profit for both moneyline and spread

Spoiler: This algorithm is awful at predicting the spread correctly -- there are just too many moving
parts for that to work well

On paper moneyline should look fantastic... but Vegas is really good at creating the lines which means
one loss can cancel out the earnings for 5 wins. Unfortunate. This tells me I need to find a way to get
better at predicting underdog wins. This strategy is still less profitable than straight up betting on the
Vegas favorite, but you lose money both ways.

The question may be is it better to get better at predicting all games or figure out which games are
better to bet on...

While there's obviously room for improvement here, it's be interesting and I'll continue to touch back
on this as I learn more about data science.
"""

# True to show profits/losses for each individual game, False to ignore them and just see final results
PRINT_INDIV_RESULTS = True

BET_SIZE = 10
ml_profit = 0
total_bet = 0
wins = 0
totals = []

sp_wins = 0
sp_profit = 0
total_spread_bet = 0

fav_wins = 0
fav_betting = 0
for actual, betting_odds in zip(val_y, odds):
    if betting_odds[0] < 0 and actual > 0:
        profit = moneyline_profit(bet_size=BET_SIZE, pred_winner=0, away_ml=betting_odds[0], home_ml=betting_odds[1])
        fav_betting += profit
        fav_wins += 1
    elif betting_odds[1] < 0 and actual < 0:
        profit = moneyline_profit(bet_size=BET_SIZE, pred_winner=1, away_ml=betting_odds[0], home_ml=betting_odds[1])
        fav_betting += profit
        fav_wins += 1
    else:
        fav_betting -= BET_SIZE


print('Fav wins:', fav_wins, len(val_y), float(fav_wins / len(val_y)), '|', fav_betting, int(BET_SIZE * len(val_y)))

for pred, actual, betting_odds in zip(predictions, val_y, odds):
    # Calc spread profits
    if betting_odds[2] != 'pk' and betting_odds != '':
        spread = abs(float(betting_odds[2]))

        away_fav = False
        if float(betting_odds[0]) < 0:
            away_fav = True

        # Spread depends on both team and score, so this tree defines all possible outcomes
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

    total_bet += BET_SIZE

    # If the away team wins by prediction and in reality
    if pred > 0 and actual > 0:
        profit = moneyline_profit(bet_size=BET_SIZE, pred_winner=0, away_ml=betting_odds[0], home_ml=betting_odds[1])
        ml_profit += profit

        if PRINT_INDIV_RESULTS:
            print('Away Win', profit, ' Total:', ml_profit, ' | Odds:', betting_odds[0])

        wins += 1

    # If the home team wins by prediction and in reality
    elif pred < 0 and actual < 0:
        profit = moneyline_profit(bet_size=BET_SIZE, pred_winner=1, away_ml=betting_odds[0], home_ml=betting_odds[1])
        ml_profit += profit

        if PRINT_INDIV_RESULTS:
            print('Home Win', profit, ' Total:', ml_profit, ' | Odds:', betting_odds[1])

        wins += 1
    else:
        if pred > 0 and actual < 0:
            loss = BET_SIZE

            ml_profit -= loss
            if PRINT_INDIV_RESULTS:
                print('Loss, subtracting', ml_profit)
        elif pred < 0 and actual > 0:
            loss = BET_SIZE

            ml_profit -= loss
            if PRINT_INDIV_RESULTS:
                print('Loss, subtracting', ml_profit)

    totals.append(ml_profit)

print('Wins', wins)
print('Total Games', len(predictions), '|', float(wins / len(predictions)))
print('ML Profit:', ml_profit, ' / ', total_bet)
print('Spread Profit:', sp_profit, ' / ', total_spread_bet)
print('Spread Wins:', sp_wins, ' / ', int(total_spread_bet / 10))
print('ROI:', float(ml_profit / total_bet))

plt.plot(totals)
plt.show()

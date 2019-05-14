import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def features(file):
    use_cols = ['GAME_ID', 'A_FG', 'A_3P', 'A_FT', 'H_FG', 'H_3P', 'H_FT']

    data = np.array(pd.read_csv(file, header=0, usecols=use_cols)).tolist()
    data.sort(key=lambda x: x[0])
    data = list(map(lambda x: x[1:], data))

    return np.array(data)


def labels(file):
    use_cols = ['GAME_ID', 'AWAY_POINTS', 'HOME_POINTS']

    data = np.array(pd.read_csv(file, header=0, usecols=use_cols)).tolist()
    data.sort(key=lambda x: x[0])
    data = list(map(lambda x: x[1:], data))

    return np.array(data)


train_x, test_x, train_y, test_y = train_test_split(features('data/game_features_2016-19.csv'),
                                                    labels('data/game_features_2016-19.csv'),
                                                    test_size=0.1)

val_x = features('data/game_validation_2019.csv')
val_y = labels('data/game_validation_2019.csv')

print(len(train_x), len(train_y))
print(len(test_x), len(test_y))
print(len(val_x), len(val_y))

away_wins = 0
home_wins = 0
for game in train_y:
    if game[0] > game[1]:
        away_wins += 1
    else:
        home_wins += 1

print(away_wins, home_wins)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=train_x[0].shape),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation=tf.nn.softmax)
])

optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mean_squared_error'])

model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(min_delta=0.000001, patience=10, restore_best_weights=True)
csv_log = tf.keras.callbacks.CSVLogger('result_tracking/Feed Forward/No Rebounds.csv')

model.fit(x=train_x, y=train_y, epochs=10000, callbacks=[early_stop, csv_log], validation_data=(test_x, test_y), shuffle=True)

predictions = model.predict(val_x)

wins = 0
for pred, actual in zip(predictions, val_y):

    print(pred, actual)

print(wins, len(predictions), float(wins / len(predictions)))

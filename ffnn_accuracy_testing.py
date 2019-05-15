import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def features(file):
    # use_cols = ['GAME_ID', 'A_FG', 'A_3P', 'A_FT', 'H_FG', 'H_3P', 'H_FT']
    # use_cols = ['GAME_ID', 'A_FG', 'A_FGA', 'A_3P', 'A_3PA', 'A_FT', 'A_FTA', 'A_ORB', 'A_DRB', 'A_TRB', 'A_AST', 'A_STL', 'A_BLK', 'A_TOB', 'A_PF', 'A_AVG_PTS',
    #             'H_FG', 'H_FGA', 'H_3P', 'H_3PA', 'H_FT', 'H_FTA', 'H_ORB', 'H_DRB', 'H_TRB', 'H_AST', 'H_STL', 'H_BLK', 'H_TOB', 'H_PF', 'H_AVG_PTS']

    # 256: val_loss = 138.7081, 60.46%
    # use_cols = ['GAME_ID', 'A_FG', 'A_3P', 'A_FT', 'A_ORB', 'A_DRB', 'A_AST', 'A_STL', 'A_BLK', 'A_TOB', 'A_PF',
    #             'H_FG', 'H_3P', 'H_FT', 'H_ORB', 'H_DRB', 'H_AST', 'H_STL', 'H_BLK', 'H_TOB', 'H_PF']

    # 128: val_loss = 129.4028, 64.45%
    use_cols = ['GAME_ID', 'A_FG', 'A_3P', 'A_FT', 'A_ORB', 'A_DRB', 'A_AST', 'A_STL', 'A_BLK',
                'H_FG', 'H_3P', 'H_FT', 'H_ORB', 'H_DRB', 'H_AST', 'H_STL', 'H_BLK']

    # use_cols = ['GAME_ID', 'A_FG', 'A_3P', 'A_FT', 'A_ORB', 'A_DRB', 'A_STL', 'A_BLK',
    #             'H_FG', 'H_3P', 'H_FT', 'H_ORB', 'H_DRB', 'H_STL', 'H_BLK']

    # use_cols = ['GAME_ID', 'A_FG', 'A_3P', 'A_FT', 'A_ORB', 'A_DRB',
    #             'H_FG', 'H_3P', 'H_FT', 'H_ORB', 'H_DRB']

    # 256: 59%
    # use_cols = ['GAME_ID', 'A_FG', 'A_3P', 'A_FT', 'A_ORB',
    #             'H_FG', 'H_3P', 'H_FT', 'H_ORB']

    # 512: 59.8%
    # use_cols = ['GAME_ID', 'A_FG', 'A_3P', 'A_FT',
    #             'H_FG', 'H_3P', 'H_FT']

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
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
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

wins = 0
for pred, actual in zip(predictions, val_y):
    if pred[0] > pred[1] and actual[0] > actual[1]:
        wins += 1
    elif pred[1] > pred[0] and actual[1] > actual[0]:
        wins += 1
    # print(pred, actual)

print(wins, len(predictions), float(wins / len(predictions)))

plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.show()

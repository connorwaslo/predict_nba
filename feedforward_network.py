import tensorflow as tf
import numpy as np
import csv
from data import features, labels, test_features, test_labels


def loss(y_true, y_pred):
    return abs(y_true - y_pred)


def k_folds_split(folds=10, iter=0, features=[], labels=[]):
    val_set_size = int(len(features) / folds)
    val_start = iter * val_set_size

    train_x = np.array(features[:val_start] + features[val_start + val_set_size:])
    train_y = np.array(labels[:val_start] + labels[val_start + val_set_size:])

    val_x = np.array(features[val_start:val_start + val_set_size])
    val_y = np.array(labels[val_start:val_start + val_set_size])

    return train_x, train_y, val_x, val_y


accuracy = 0.0
for i in range(1):
    train_x, train_y, val_x, val_y = k_folds_split(iter=i, features=features(), labels=labels())

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
                  metrics=['mean_absolute_error', 'mean_squared_error'])

    # Print model summary
    model.summary()

    # Callbacks
    # early_stop = callbacks.EarlyStopping(min_delta=0.01, restore_best_weights=True)
    early_stop = tf.keras.callbacks.EarlyStopping(min_delta=0.01, restore_best_weights=True)
    csv_log = tf.keras.callbacks.CSVLogger('result_tracking/Feed Forward/No Rebounds.csv')

    test_x, test_y = np.array(test_features()), np.array(test_labels())
    print(test_x[0].shape, train_x[0].shape)

    history = model.fit(train_x, train_y, epochs=300, verbose=1, callbacks=[early_stop, csv_log])
    print(model.evaluate(val_x, val_y))

    predictions = model.predict(test_x)

    wins = 0
    for pred, actual in zip(predictions, test_y):
        print(pred, actual, [actual[0] - pred[0], actual[1] - pred[1]])
        if pred[0] > pred[1] and actual[0] > actual[1]:
            wins += 1
        elif pred[0] < pred[1] and actual[0] < actual[1]:
            wins += 1

    print(wins, int(len(predictions) - wins), float(wins / len(predictions)))

    file = 'FFNN No Rebounds 18 Season.csv'
    with open('result_tracking/Feed Forward/' + file, 'a', newline='') as f:
        writer = csv.writer(f)

        writer.writerow([i, wins, int(len(predictions) - wins), float(wins / len(predictions))])

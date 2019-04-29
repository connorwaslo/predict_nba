import tensorflow as tf
import numpy as np
import csv
from data import features, labels


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


for i in range(10):
    train_x, train_y, val_x, val_y = k_folds_split(iter=i, features=features(), labels=labels())

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=train_x[0].shape),
        tf.keras.layers.Dense(768, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])

    # Print model summary
    model.summary()

    history = model.fit(train_x, train_y, epochs=50, verbose=1)
    print(model.evaluate(val_x, val_y))
    predictions = model.predict(val_x)

    wins = 0
    for pred, actual in zip(predictions, val_y):
        print(pred, actual, [actual[0] - pred[0], actual[1] - pred[1]])
        if pred[0] > pred[1] and actual[0] > actual[1]:
            wins += 1
        elif pred[0] < pred[1] and actual[0] < actual[1]:
            wins += 1

    print(wins, int(len(predictions) - wins), float(wins / len(predictions)))

    file = 'FFNN 4 Major Stats.csv'
    with open('result_tracking/Feed Forward/' + file, 'a', newline='') as f:
        writer = csv.writer(f)

        writer.writerow([i, wins, int(len(predictions) - wins), float(wins / len(predictions))])

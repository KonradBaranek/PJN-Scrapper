from __future__ import absolute_import, division, print_function


import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# import Scrapper
# import numpy
#
# print('helllo')
# dicts = []
# for num in range(277, 500):
#     print('page', num)
#     Scrapper.scrapp_filmweb_page(num)
#     # print('directors: ' + str(len(Scrapper.directors)))

print(tf.__version__)

# dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path = "./movies.data"
print(dataset_path)
column_names = ['Score', 'Title', 'Year', 'Want to see', 'Votes',
                'Genre', 'Director', 'Country', 'Cast']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          sep="\t", encoding="utf-8")

dataset = raw_dataset.copy()
dataset.pop('Title')
dataset.isna().sum()
dataset = dataset.dropna()
print(dataset)

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# sns.pairplot(train_dataset[['Score', 'Year', 'Want to see', 'Votes', 'Genre', 'Director', 'Country', 'Cast']],
# diag_kind="kde")

train_stats = train_dataset.describe()
print(format(train_stats))

train_stats.pop('Score')
train_stats = train_stats.transpose()
train_labels = train_dataset.pop('Score')
test_labels = test_dataset.pop('Score')
print("end")

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()

model.summary()


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


EPOCHS = 1000


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [Score]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 10])

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$Score^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 20])
    plt.show()


history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} Score".format(mae))

test_predictions = model.predict(normed_test_data).flatten()

plt.clf()
plt.scatter(test_labels, test_predictions)
plt.xlabel('Rzeczywiste wartości')
plt.ylabel('Predykcje Score')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

plt.clf()
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Błąd przewidywania [Score]")
_ = plt.ylabel("Count")
plt.show()
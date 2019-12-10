import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm
import random

from experiments.npe_bt.model import get_npe_model


def shuffle_together(x, y):
    c = list(zip(x, y))
    random.shuffle(c)
    x, y = zip(*c)

    return x, y


def breakdown(X):
    return [np.array([x[i] for x in X]) for i in range(len(X[0]))]


def start_train(dataset, model_save, epochs=100, batch_size=50):
    with open(dataset, "rb") as f:
        data = pickle.load(f)

    train_x, train_y = data["train"]
    train_x, train_y = shuffle_together(train_x, train_y)

    val_x, val_y = data["val"]
    val_x, val_y = shuffle_together(val_x, val_y)

    b_train_x = breakdown(train_x)
    b_val_x = breakdown(val_x)

    model = get_npe_model()

    opt = tf.keras.optimizers.RMSprop(learning_rate=0.0003)
    model.compile(loss="mse", optimizer=opt)

    history = model.fit(
        b_train_x,
        np.array(train_y),
        validation_data=(b_val_x, np.array(val_y)),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_save, monitor="val_loss", verbose=1
            )
        ],
    )

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.show()

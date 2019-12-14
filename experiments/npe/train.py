import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm
import random

from experiments.npe.model import get_npe_model


def start_train(dataset, model_save, epochs=20, batch_size=50):
    data = np.load(dataset)

    train_x = []
    val_x = []

    # TODO(shreyask): Make this more general.
    for i in range(5):
        train_x.append(data["arr_%i" % (i)])
        val_x.append(data["arr_%i" % (i + 5)])

    train_y = data["train_y"]
    val_y = data["val_y"]

    model = get_npe_model(max_pairs=2)

    opt = tf.keras.optimizers.RMSprop(learning_rate=0.0003)
    model.compile(loss="mse", optimizer=opt)

    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
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

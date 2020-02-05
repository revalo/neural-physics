import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm
import random

from experiments.npe.model import get_npe_model
from experiments.npe_bt.simulate import show_simulation
from memory_profiler import profile


# @profile
def main():
    SEED = 1338
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data = np.load("data/circles_uc1.npz", allow_pickle=True)

    train_x = []
    val_x = []

    for i in range((data["config"].item()["num_objects"] - 1) * 2 + 1):
        train_x.append(data["arr_%i" % (i)])
        val_x.append(
            data["arr_%i" % (i + (data["config"].item()["num_objects"] - 1) * 2 + 1)]
        )

    train_y = data["train_y"]
    val_y = data["val_y"]

    BATCH_SIZE = 50
    ITERATIONS = 1000
    UC_ITERATION = 20
    BIG_BATCH = 131072
    SPLIT = 0.1
    EPOCHS = 30  # Number of epochs.
    SAMPLES = 100  # Number of samples to compute uc inference.
    FILENAME = "losses.p"

    model_vanilla = get_npe_model(max_pairs=2, probabilistic=True)
    model_vanilla.compile(
        loss="msle", optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0003)
    )

    model_uc = get_npe_model(max_pairs=2, probabilistic=True)
    model_uc.compile(
        loss="msle", optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0003)
    )

    initial_weights = model_uc.get_weights()

    def sample_indexes(indexes):
        sampled_x = [x[indexes] for x in train_x]

        sampled_y = train_y[indexes]

        return sampled_x, sampled_y

    def sample_randomly():
        indexes = np.random.choice(len(train_x[0]), BATCH_SIZE)
        return sample_indexes(indexes)

    def get_uncertainty(model, samples=SAMPLES):
        xs = np.zeros((len(train_x[0]), samples))
        ys = np.zeros((len(train_x[0]), samples))

        for s in range(samples):
            p = model.predict(train_x, batch_size=BIG_BATCH)
            xs[:, s] = p[:, 0]
            ys[:, s] = p[:, 1]

        return (xs.std(axis=1) + ys.std(axis=1)) / 2

    def sample_uncertainty(model, samples=SAMPLES):
        var = get_uncertainty(model, samples=samples)
        indexes = np.argsort(var)[::-1][: int(SPLIT * len(train_x[0]))]
        sub_indexes = np.random.choice(len(indexes), BATCH_SIZE)

        indexes = indexes[sub_indexes]

        return sample_indexes(indexes)

    def join_xs(pool, new):
        return [np.concatenate((a, b)) for a, b in zip(pool, new)]

    def join_ys(pool, new):
        return np.concatenate((pool, new))

    pool_vx, pool_vy = sample_randomly()
    pool_ux, pool_uy = sample_randomly()

    vanilla = []
    uc = []

    for iteration in range(ITERATIONS):
        model_vanilla.set_weights(initial_weights)
        model_uc.set_weights(initial_weights)

        val_v = float("inf")
        val_u = float("inf")

        for epoch in range(EPOCHS):
            hv = model_vanilla.fit(
                pool_vx,
                pool_vy,
                epochs=1,
                verbose=0,
                batch_size=BATCH_SIZE,
                shuffle=True,
            )
            hu = model_uc.fit(
                pool_ux,
                pool_uy,
                epochs=1,
                verbose=0,
                batch_size=BATCH_SIZE,
                shuffle=True,
            )

            cv = model_vanilla.evaluate(val_x, val_y, batch_size=BIG_BATCH, verbose=0)
            cu = model_uc.evaluate(val_x, val_y, batch_size=BIG_BATCH, verbose=0)

            if cv < val_v:
                val_v = cv

            if cu < val_u:
                val_u = cu

        print(
            "Iteration %i, Vanilla: %f, UC: %f, (%i)"
            % (iteration, val_v, val_u, pool_ux[0].shape[0])
        )

        vx, vy = sample_randomly()
        pool_vx, pool_vy = join_xs(pool_vx, vx), join_ys(pool_vy, vy)

        if iteration < UC_ITERATION:
            ux, uy = sample_randomly()
        else:
            ux, uy = sample_uncertainty(model_uc)

        pool_ux, pool_uy = join_xs(pool_ux, ux), join_ys(pool_uy, uy)

        vanilla.append(val_v)
        uc.append(val_u)

        import pickle

        with open(FILENAME, "wb") as f:
            pickle.dump({"vanilla": vanilla, "uc": uc,}, f)


if __name__ == "__main__":
    main()

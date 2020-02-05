"""Driver file that trains PNPE models using an active learning approach.

Train probablistic neural physics engine models using uncertainty sampling. Basically,
it produces the minimum validation loss for the number of training examples used.
"""
import tensorflow as tf
import numpy as np

from absl import app
from absl import flags

import pickle
import tqdm
import random
import os

from experiments.npe.model import get_npe_model

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 50, "Training batch size.")
flags.DEFINE_integer("iterations", 1000, "Maximum number of training iterations.")
flags.DEFINE_integer("uc_iteration", 20, "At what iteration to kick-in UC sampling.")
flags.DEFINE_integer(
    "big_batch",
    131072,
    "Maximum batch size your GPU can fit. Used for speedy inference.",
)
flags.DEFINE_float(
    "split", 0.1, "What proportion of the sorted uncertain set to sample from."
)
flags.DEFINE_float("learning_rate", 0.0003, "Learning rate.")
flags.DEFINE_integer("epochs", 30, "Number of epochs to train.")
flags.DEFINE_integer(
    "samples",
    100,
    "How many samples to make to evaluate uncertainty. Bear in mind, you need to the NUM_EXAMPLES x SAMPLES matrix in your memory.",
)
flags.DEFINE_boolean("vanilla", False, "Randomly sample.")
flags.DEFINE_boolean("scratch", True, "Retrain model from scratch every iteration.")
flags.DEFINE_string(
    "name", None, "Name of this training run, useful for generating the log files."
)
flags.DEFINE_string("dataset", "data/circles_uc1.npz", "Path to the dataset.")
flags.DEFINE_string(
    "save_dir", None, "Path to directory to save logs and models.", short_name="s"
)
flags.DEFINE_string("scene", "ThreeCircles", "Name of the scene.")
flags.DEFINE_integer(
    "seed",
    1338,
    "A lucky number that makes things work. JK. Just for deterministic outputs.",
)

flags.mark_flags_as_required(["save_dir", "name"])


def main(argv):
    """Giant script to run the training process. Probably needs a cleanup of sorts.
    TODO(shreyask): ^^^
    """

    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)

    data = np.load(FLAGS.dataset, allow_pickle=True)

    train_x = []
    val_x = []

    for i in range((data["config"].item()["num_objects"] - 1) * 2 + 1):
        train_x.append(data["arr_%i" % (i)])
        val_x.append(
            data["arr_%i" % (i + (data["config"].item()["num_objects"] - 1) * 2 + 1)]
        )

    train_y = data["train_y"]
    val_y = data["val_y"]

    NUM_EXAMPLES = len(train_x[0])

    if FLAGS.scene == "ThreeCircles":
        MAX_PAIRS = 2
    elif FLAGS.scene == "BlockTower":
        MAX_PAIRS = 5
    else:
        print("Unkown scene %s." % (FLAGS.scene))
        return

    SAVE_DIR = os.path.join(FLAGS.save_dir, FLAGS.name)
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    MODEL_SAVE = os.path.join(SAVE_DIR, "model.h5")
    LOSS_SAVE = os.path.join(SAVE_DIR, "losses.p")

    model = get_npe_model(max_pairs=MAX_PAIRS, probabilistic=True)
    model.compile(
        loss="msle",
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=FLAGS.learning_rate),
    )

    initial_weights = model.get_weights()

    def sample_indexes(indexes):
        sampled_x = [x[indexes] for x in train_x]

        sampled_y = train_y[indexes]

        return sampled_x, sampled_y, indexes

    def sample_randomly():
        indexes = np.random.choice(NUM_EXAMPLES, FLAGS.batch_size)
        return sample_indexes(indexes)

    def get_uncertainty(model, samples=FLAGS.samples):
        xs = np.zeros((NUM_EXAMPLES, samples))
        ys = np.zeros((NUM_EXAMPLES, samples))

        for s in range(samples):
            p = model.predict(train_x, batch_size=FLAGS.big_batch)
            xs[:, s] = p[:, 0]
            ys[:, s] = p[:, 1]

        return (xs.std(axis=1) + ys.std(axis=1)) / 2

    def sample_uncertainty(model, samples=FLAGS.samples):
        var = get_uncertainty(model, samples=samples)
        indexes = np.argsort(var)[::-1][: int(FLAGS.split * len(train_x[0]))]
        sub_indexes = np.random.choice(len(indexes), FLAGS.batch_size)

        indexes = indexes[sub_indexes]

        return sample_indexes(indexes)

    def join_xs(pool, new):
        return [np.concatenate((a, b)) for a, b in zip(pool, new)]

    def join_ys(pool, new):
        return np.concatenate((pool, new))

    pool_x, pool_y, idx = sample_randomly()
    pool_indexes = set()
    pool_indexes.update(idx)

    best_losses = []
    examples = []

    for iteration in range(FLAGS.iterations):
        if FLAGS.scratch:
            model.set_weights(initial_weights)

        best = float("inf")

        for epoch in range(FLAGS.epochs):
            model.fit(
                pool_x,
                pool_y,
                epochs=1,
                verbose=0,
                batch_size=FLAGS.batch_size,
                shuffle=True,
            )

            current = model.evaluate(
                val_x, val_y, batch_size=FLAGS.big_batch, verbose=0
            )

            best = min(best, current)

        print(
            "Iteration %i, Best: %f, Examples: %i, Pool: %i"
            % (iteration, best, pool_x[0].shape[0], len(pool_indexes))
        )

        if len(best_losses) == 0 or best < min(best_losses):
            # We have a new winner, let's save this model!
            model.save(MODEL_SAVE)

        best_losses.append(best)
        examples.append(len(pool_indexes))

        if FLAGS.vanilla:
            vx, vy, idx = sample_randomly()
            pool_x, pool_y = join_xs(pool_x, vx), join_ys(pool_y, vy)
            pool_indexes.update(idx)
        else:
            if iteration < FLAGS.uc_iteration:
                ux, uy, idx = sample_randomly()
            else:
                ux, uy, idx = sample_uncertainty(model)

            pool_x, pool_y = join_xs(pool_x, ux), join_ys(pool_y, uy)
            pool_indexes.update(idx)

        with open(LOSS_SAVE, "wb") as f:
            pickle.dump(
                {
                    "losses": best_losses,
                    "examples": examples,
                    "config": FLAGS.flag_values_dict(),
                    "max_iteration_reached": iteration,
                },
                f,
            )


if __name__ == "__main__":
    app.run(main)


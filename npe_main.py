"""Main endpoint for working with the NPE engine."""

import tensorflow as tf
import numpy as np

from absl import app
from absl import flags

# Environments
from renderer.threecircles import ThreeCircles
from renderer.rectangles import Rectangles

# NPE-related
from experiments.npe.train import breakdown
from experiments.npe.datagen import collect_data
from experiments.npe_bt.simulate import show_simulation
from experiments.npe_bt.train import start_train


FLAGS = flags.FLAGS
flags.DEFINE_boolean("gen_data", False, "Generate data.", short_name="g")
flags.DEFINE_boolean("train", False, "Train the model.")
flags.DEFINE_boolean(
    "show_world", False, "Show a couple step of the world.", short_name="s"
)
flags.DEFINE_boolean("model_simulation", False, "Show steps of the model.")

flags.DEFINE_integer("simulation_steps", 1000, "Number of steps to simulate.")
flags.DEFINE_integer("epochs", 20, "Number training epochs.")
flags.DEFINE_integer("batch_size", 50, "Training batch size.")
flags.DEFINE_string("dataset", None, "Path to dataset pickle file.")
flags.DEFINE_string("model", None, "Path to saved model weights.")
flags.DEFINE_integer("num_sequences", 50000, "Number of sequences to generate.")
flags.DEFINE_integer("sequence_len", 60, "Number of frames per sequence.")
flags.DEFINE_float(
    "validation_split", 0.1, "Fraction of the sequences reserved for validation."
)
flags.DEFINE_string(
    "scene", "ThreeCircles", "Scene to use. Options are ThreeCircles, BlockTower."
)

# Scene specific flags
flags.DEFINE_integer("radius", 30, "The radius of the balls.")

flags.mark_bool_flags_as_mutual_exclusive(
    ["gen_data", "show_world", "model_simulation", "train"],
    required=True,
    flag_values=FLAGS,
)


def create_scene():
    if FLAGS.scene == "ThreeCircles":
        return ThreeCircles(headless=False, radius=FLAGS.radius)
    elif FLAGS.scene == "BlockTower":
        return Rectangles(headless=False, rand_height=False, wall_elasticity=1.0)
    else:
        raise ValueError("Please specify a valid scene.")


def generate_data():
    if not FLAGS.dataset:
        print("Please pass in a dataset location.")
        return

    try:
        with open(FLAGS.dataset, "wb"):
            pass
    except IOError:
        print("Please pass in valid dataset location.")
        return

    validation_sequences = int(FLAGS.num_sequences * FLAGS.validation_split)

    train_x, train_y, train_complex = collect_data(
        num_sequences=FLAGS.num_sequences,
        sequence_length=FLAGS.sequence_len,
        radius=FLAGS.radius,
        seed=1337,
    )
    val_x, val_y, _ = collect_data(
        num_sequences=validation_sequences,
        sequence_length=FLAGS.sequence_len,
        radius=FLAGS.radius,
        seed=5332,
    )

    print("Breaking down!")
    btrain_x = breakdown(train_x)
    bval_x = breakdown(val_x)

    print("Saving!")
    np.savez(
        FLAGS.dataset,
        *btrain_x,
        *bval_x,
        train_y=train_y,
        val_y=val_y,
        train_complex=train_complex,
        config={
            "scene": FLAGS.scene,
            "num_sequences": FLAGS.num_sequences,
            "sequence_len": FLAGS.sequence_len,
            "num_objects": (len(btrain_x) - 1) // 2 + 1,
        }
    )


def show_world():
    scene = create_scene()

    for frame in range(FLAGS.simulation_steps):
        print(frame)
        scene.step()
        scene.draw()


def show_model_simulation():
    model = tf.keras.models.load_model(FLAGS.model)
    show_simulation(model, radius=FLAGS.radius)


def train():
    if not FLAGS.dataset:
        print("Pass in path to valid dataset pickle.")
        return

    if not FLAGS.model:
        print("Pass in save location for model.")
        return

    start_train(FLAGS.dataset, FLAGS.model, FLAGS.epochs, FLAGS.batch_size)


def main(argv):
    if FLAGS.gen_data:
        generate_data()
    elif FLAGS.show_world:
        show_world()
    elif FLAGS.model_simulation:
        show_model_simulation()
    elif FLAGS.train:
        train()


if __name__ == "__main__":
    app.run(main)

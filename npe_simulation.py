import tensorflow as tf
from experiments.npe.simulate import show_simulation

if __name__ == "__main__":
    model = tf.keras.models.load_model("model_zoo/npe3.h5")

    show_simulation(model)

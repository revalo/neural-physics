"""Routines to help train the adversary.
"""

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


def get_actor(latent_dim=12, config_dim=3 * 4, hidden_dim=50, num_layers=4):
    """Takes latent dimension and outputs configuration."""

    latent = keras.Input(shape=(latent_dim,))

    x = keras.layers.Dense(hidden_dim, activation="relu")(latent)
    for _ in range(num_layers - 1):
        x = keras.layers.Dense(hidden_dim, activation="relu")(x)

    configuration = keras.layers.Dense(config_dim, activation="linear")(x)

    model = keras.Model(inputs=latent, outputs=configuration)

    return model


def get_critic(config_dim=3 * 4, hidden_dim=50, num_layers=4):
    """Takes configuration and predicts rewards."""

    configuration = keras.Input(shape=(config_dim,))

    x = keras.layers.Dense(hidden_dim, activation="relu")(configuration)
    for _ in range(num_layers - 1):
        x = keras.layers.Dense(hidden_dim, activation="relu")(x)

    reward = keras.layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs=configuration, outputs=reward)

    return model


def get_joint(actor, critic, latent_dim=12):
    latent = keras.Input(shape=(latent_dim,))
    configuration = actor(latent)
    reward = critic(configuration)

    return keras.Model(inputs=latent, outputs=reward)


def get_experience(actor, environment, latent_dim=12):
    latent = np.random.uniform(0.0, 1.0, size=(environment.batch_size, latent_dim))
    configurations = actor.predict(latent)
    rewards, _, _ = environment.get_reward(configurations)

    return latent, configurations, rewards


def sample_non_colliding_config():
    def is_colliding(config):
        for c_i in range(3):
            for c_j in range(c_i + 1, 3):
                xi, yi = config[c_i : c_i + 2]
                xj, yj = config[c_j : c_j + 2]

                if (xi - xj) ** 2 + (yi - yj ** 2) >= (35.0 / 256.0 * 2) ** 2:
                    return True

        return False

    c = np.random.uniform(0.0, 1.0, size=(3 * 4,))
    while is_colliding(c):
        c = np.random.uniform(0.0, 1.0, size=(3 * 4,))

    return c


def get_random_experience(environment):
    configurations = np.random.uniform(0.0, 1.0, size=(environment.batch_size, 3 * 4))
    # confs = []

    # for i in range(environment.batch_size):
    #     confs.append(sample_non_colliding_config())

    # configurations = np.array(confs)

    rewards, actual, predicted = environment.get_reward(configurations)

    return rewards, configurations, actual


def get_naive_adversarial_loss(env, sample_pick=128, target=1024):
    rv = []
    examples = []

    while len(rv) < target:
        reach = min(target - len(rv), sample_pick)
        rewards, configurations, actual = get_random_experience(env)

        worst_configs = configurations[rewards.argsort()][::-1]
        actuals = actual[rewards.argsort()[::-1], :, :, :]

        rv.extend(worst_configs[:reach])
        examples.extend(actuals[:reach, :, :, :])

    return np.array(rv), np.array(examples)

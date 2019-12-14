"""Model implementation of the neural physics engines.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
import numpy as np

tfpl = tfp.layers
tfd = tfp.distributions


def get_npe_model(
    state_dim=4,
    out_dim=2,
    past_timesteps=2,
    max_pairs=3,
    pair_encoding_dim=50,
    hidden_size=50,
    predict_sigmoid=False,
    probabilistic=False,
    activation="relu",
    layers_encoder=1,
):
    """Gets the neural physics engine model.

    state_dim: Length of the vector to represent the state of each object.
    past_timesteps: Number of timesteps in past given to the model.
    max_pairs: Maximum number of context objects given to the model.
    pair_encoding_dim: Length of the pairwise encoding vector.
    """

    num_states = past_timesteps * state_dim

    # Define the state encoder model.
    state_input = keras.layers.Input(shape=(num_states,))
    x = keras.layers.Dense(16, activation="relu")(state_input)
    x = keras.layers.Dense(16, activation=activation)(x)
    encoded_state = keras.layers.Dense(8, activation=activation)(x)
    state_model = keras.Model(state_input, encoded_state)

    # Define pairwise encoder model.
    state_a = keras.layers.Input(shape=(num_states,))
    state_b = keras.layers.Input(shape=(num_states,))
    used = keras.layers.Input(
        shape=(1,)
    )  # Indicates if this pairwise encoder is in use.

    encoded_a = state_model(state_a)
    encoded_b = state_model(state_b)

    concatenated = keras.layers.concatenate([encoded_a, encoded_b])
    x = keras.layers.Dense(hidden_size, activation=activation)(concatenated)
    for _ in range(layers_encoder):
        x = keras.layers.Dense(hidden_size, activation=activation)(x)
    # TODO(shreyask): This could probably use a perf boost.
    encoded_pair = (
        keras.layers.Dense(pair_encoding_dim, activation=activation)(x) * used[0]
    )
    pair_model = keras.Model([state_a, state_b, used], encoded_pair)

    # Define the decoder model.
    key_state_input = keras.layers.Input(shape=(num_states,))
    pair_encodings_sum = keras.layers.Input(shape=(pair_encoding_dim,))
    concatenated = keras.layers.concatenate([key_state_input, pair_encodings_sum])
    x = keras.layers.Dense(hidden_size, activation=activation)(concatenated)
    x = keras.layers.Dense(hidden_size, activation=activation)(x)
    if probabilistic:
        x = keras.layers.Dropout(0.2)(x, training=True)
    x = keras.layers.Dense(hidden_size, activation=activation)(x)
    if probabilistic:
        x = keras.layers.Dropout(0.2)(x, training=True)
    x = keras.layers.Dense(hidden_size, activation=activation)(x)
    x = keras.layers.Dense(hidden_size, activation=activation)(x)

    velocity = keras.layers.Dense(out_dim, activation="linear")(x)

    decoder_model = keras.Model([key_state_input, pair_encodings_sum], velocity)

    # Define the NPE model.
    key_state_input = keras.layers.Input(shape=(num_states,))

    context_state_inputs = [
        keras.layers.Input(shape=(num_states,)) for _ in range(max_pairs)
    ]

    used_inputs = [keras.layers.Input(shape=(1,)) for _ in range(max_pairs)]

    pair_encodings = [
        pair_model([key_state_input, context_state_input, used])
        for context_state_input, used in zip(context_state_inputs, used_inputs)
    ]

    added = keras.layers.add(pair_encodings)
    decoded = decoder_model([key_state_input, added])
    npe_model = keras.Model(
        [key_state_input] + context_state_inputs + used_inputs, decoded
    )

    return npe_model

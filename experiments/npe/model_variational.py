"""A whole host of experiments for stochastic NPE.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
import numpy as np

tfpl = tfp.layers
tfd = tfp.distributions


def get_npe_model(
    state_dim=2,
    past_timesteps=2,
    max_pairs=3,
    pair_encoding_dim=64,
    hidden_size=50,
    num_layers=4,
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
    x = keras.layers.Dense(16, activation="relu")(x)
    encoded_state = keras.layers.Dense(8, activation="relu")(x)
    state_model = keras.Model(state_input, encoded_state)

    # Define pairwise encoder model.
    state_a = keras.layers.Input(shape=(num_states,))
    state_b = keras.layers.Input(shape=(num_states,))
    used = keras.layers.Input(
        shape=(1,)
    )  # Indicates if this pairwise encoder is in use.

    # encoded_a = state_model(state_a)
    # encoded_b = state_model(state_b)

    concatenated = keras.layers.concatenate([state_a, state_b, (state_a - state_b)])
    # concatenated = keras.layers.concatenate([state_a, state_b])
    x = keras.layers.Dense(hidden_size, activation="relu")(concatenated)
    # x = keras.layers.Dense(hidden_size, activation="relu")(x)
    # x = keras.layers.Dense(hidden_size, activation="relu")(x)
    # x = keras.layers.Dense(hidden_size, activation="relu")(x)
    # x = keras.layers.Dense(hidden_size, activation="relu")(x)
    for _ in range(num_layers):
        x = keras.layers.Dense(hidden_size, activation="relu")(x)
    # TODO(shreyask): This could probably use a perf boost.
    encoded_pair = keras.layers.Dense(pair_encoding_dim, activation="relu")(x) * used[0]
    pair_model = keras.Model([state_a, state_b, used], encoded_pair)

    # Define the decoder model.
    key_state_input = keras.layers.Input(shape=(num_states,))
    pair_encodings_sum = keras.layers.Input(shape=(pair_encoding_dim,))
    concatenated = keras.layers.concatenate([key_state_input, pair_encodings_sum])
    x = keras.layers.Dense(hidden_size, activation="relu")(concatenated)
    # x = keras.layers.Dense(hidden_size, activation="relu")(x)
    # x = keras.layers.Dense(hidden_size, activation="relu")(x)
    # x = keras.layers.Dense(hidden_size, activation="relu")(x)
    # x = keras.layers.Dense(hidden_size, activation="relu")(x)

    for _ in range(num_layers):
        x = keras.layers.Dense(hidden_size, activation="relu")(x)

    # num_components = 2
    # event_shape = [2]

    # params_size = tfpl.MixtureNormal.params_size(num_components, event_shape)
    # x = keras.layers.Dense(params_size, activation=None)(x)
    # velocity = tfpl.MixtureNormal(num_components, event_shape)(x)

    # event_shape = [2]

    # params_size = tfpl.IndependentNormal.params_size(event_shape)
    # x = keras.layers.Dense(params_size, activation=None)(x)
    # velocity = tfpl.IndependentNormal(event_shape)(x)

    x = keras.layers.Dense(4, activation=None)(x)

    velocity = tfp.layers.DistributionLambda(
        lambda t: tfd.MultivariateNormalDiag(
            loc=t[..., :2], scale_diag=1e-5 + tf.math.softplus(1.0 * t[..., 2:])
        ),
        convert_to_tensor_fn=lambda s: s.mean(),
    )(x)

    # velocity = tfp.layers.DistributionLambda(
    #     lambda t: tfp.distributions.Independent(
    #         tfp.distributions.Normal(
    #             loc=t[..., :2], scale=1e-5 + tf.math.softplus(1.0 * t[..., 2:]),
    #         ),
    #         reinterpreted_batch_ndims=2,
    #     ),
    #     # convert_to_tensor_fn=lambda s: s.mean(),
    # )(x)
    # velocity = keras.layers.Dense(2, activation=None)(x)

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


def get_lossfunc(z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
    """
    Function to calculate given a 2D distribution over x and y, and target data
    of observed x and y points
    params:
    z_mux : mean of the distribution in x
    z_muy : mean of the distribution in y
    z_sx : std dev of the distribution in x
    z_sy : std dev of the distribution in y
    z_rho : Correlation factor of the distribution
    x_data : target x points
    y_data : target y points
    """
    step = tf.constant(1e-3, dtype=tf.float32, shape=(1, 1))

    # Calculate the PDF of the data w.r.t to the distribution
    result0_1 = self.tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
    result0_2 = self.tf_2d_normal(
        tf.add(x_data, step), y_data, z_mux, z_muy, z_sx, z_sy, z_corr
    )
    result0_3 = self.tf_2d_normal(
        x_data, tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr
    )
    result0_4 = self.tf_2d_normal(
        tf.add(x_data, step), tf.add(y_data, step), z_mux, z_muy, z_sx, z_sy, z_corr
    )

    result0 = tf.div(
        tf.add(tf.add(tf.add(result0_1, result0_2), result0_3), result0_4),
        tf.constant(4.0, dtype=tf.float32, shape=(1, 1)),
    )
    result0 = tf.mul(tf.mul(result0, step), step)

    # For numerical stability purposes
    epsilon = 1e-20

    # Apply the log operation
    result1 = -tf.log(tf.maximum(result0, epsilon))  # Numerical stability

    # Sum up all log probabilities for each data point
    return tf.reduce_sum(result1)


def tf_2d_normal(x, y, mux, muy, sx, sy, rho):
    """
        Function that implements the PDF of a 2D normal distribution
        params:
        x : input x points
        y : input y points
        mux : mean of the distribution in x
        muy : mean of the distribution in y
        sx : std dev of the distribution in x
        sy : std dev of the distribution in y
        rho : Correlation factor of the distribution
        """
    # eq 3 in the paper
    # and eq 24 & 25 in Graves (2013)
    # Calculate (x - mux) and (y-muy)
    normx = tf.subtract(x, mux)
    normy = tf.subtract(y, muy)
    # Calculate sx*sy
    sxsy = tf.multiply(sx, sy)
    # Calculate the exponential factor
    z = (
        tf.square(tf.divide(normx, sx))
        + tf.square(tf.divide(normy, sy))
        - 2 * tf.divide(tf.multiply(rho, tf.multiply(normx, normy)), sxsy)
    )
    negRho = 1 - tf.square(rho)
    # Numerator
    result = tf.exp(tf.divide(-z, 2 * negRho))
    # Normalization constant
    denom = 2 * np.pi * tf.multiply(sxsy, tf.sqrt(negRho))
    # Final PDF calculation
    result = tf.divide(result, denom)
    return result


def get_coef(output):
    # eq 20 -> 22 of Graves (2013)

    z = output
    # Split the output into 5 parts corresponding to means, std devs and corr
    z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(z, 5, 1)

    # The output must be exponentiated for the std devs
    z_sx = tf.exp(z_sx)
    z_sy = tf.exp(z_sy)
    # Tanh applied to keep it in the range [-1, 1]
    z_corr = tf.tanh(z_corr)

    return [z_mux, z_muy, z_sx, z_sy, z_corr]


def sample_gaussian_2d(mux, muy, sx, sy, rho):
    """
    Function to sample a point from a given 2D normal distribution
    params:
    mux : mean of the distribution in x
    muy : mean of the distribution in y
    sx : std dev of the distribution in x
    sy : std dev of the distribution in y
    rho : Correlation factor of the distribution
    """
    # Extract mean
    mean = [mux, muy]
    # Extract covariance matrix
    cov = [[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]]
    # Sample a point from the multivariate normal distribution
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]

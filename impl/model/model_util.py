import tensorflow as tf


def get_learning_rate_decay_schedule(initial_learning_rate, decay_rate,
                                     decay_steps) -> tf.keras.optimizers.schedules.ExponentialDecay:
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True)
    return lr_schedule

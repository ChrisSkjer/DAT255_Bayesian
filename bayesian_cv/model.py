import tensorflow as tf

from bayesian_cv.config import ProjectConfig


class MCDropout(tf.keras.layers.Dropout):
    """Dropout layer that stays active during inference for MC sampling."""

    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


def build_model(config: ProjectConfig) -> tf.keras.Model:
    """
    TODO:
    - Build your CNN in Keras.
    - Use MCDropout where you want dropout active during inference.
    - Compile and return the model.
    """
    _ = config
    raise NotImplementedError("Implement model creation in bayesian_cv/model.py")

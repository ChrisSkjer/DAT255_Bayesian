import numpy as np
import tensorflow as tf


def mc_predict(
    model: tf.keras.Model,
    images: tf.Tensor,
    mc_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = []
    for _ in range(mc_samples):
        predictions.append(model(images, training=False).numpy())

    stacked = np.stack(predictions, axis=0)
    mean_prediction = stacked.mean(axis=0)
    variance = stacked.var(axis=0)
    predictive_entropy = -np.sum(
        mean_prediction * np.log(mean_prediction + 1e-10),
        axis=1,
    )
    return mean_prediction, variance, predictive_entropy

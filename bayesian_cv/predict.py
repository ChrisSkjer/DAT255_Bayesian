import tensorflow as tf

from bayesian_cv.config import get_config
from bayesian_cv.data import load_datasets
from bayesian_cv.uncertainty import mc_predict


def main() -> None:
    config = get_config()
    _, _, test_ds = load_datasets(config)

    model = tf.keras.models.load_model(config.model_path)

    batch = next(iter(test_ds))
    images = batch[0] if isinstance(batch, tuple) else batch

    mean_prediction, variance, predictive_entropy = mc_predict(
        model,
        images,
        config.mc_samples,
    )

    predicted_classes = mean_prediction.argmax(axis=1)
    mean_variance = variance.mean(axis=1)

    for index, (predicted_class, entropy, avg_variance) in enumerate(
        zip(predicted_classes, predictive_entropy, mean_variance),
        start=1,
    ):
        print(
            f"sample={index} predicted_class={predicted_class} "
            f"entropy={entropy:.4f} mean_variance={avg_variance:.6f}"
        )


if __name__ == "__main__":
    main()

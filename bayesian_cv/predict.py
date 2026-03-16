import tensorflow as tf

from bayesian_cv.config import get_config
from bayesian_cv.data import load_datasets
from bayesian_cv.uncertainty import mc_predict


def main() -> None:
    config = get_config()
    _, _, test_ds = load_datasets(config)

    model = tf.keras.models.load_model(config.model_path)

    # TODO:
    # - Fetch a batch or individual samples from test_ds.
    # - Call mc_predict(...) to estimate uncertainty.
    _ = (model, test_ds, mc_predict)
    raise NotImplementedError("Implement inference in bayesian_cv/predict.py")


if __name__ == "__main__":
    main()

import tensorflow as tf

from bayesian_cv.config import ProjectConfig


def load_datasets(
    config: ProjectConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    TODO:
    - Load train/validation/test datasets.
    - Add preprocessing and optional augmentation.
    - Return three tf.data.Dataset objects.
    """
    _ = config
    raise NotImplementedError("Implement dataset loading in bayesian_cv/data.py")

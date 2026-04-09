from pathlib import Path

import tensorflow as tf

from bayesian_cv.config import ProjectConfig


IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_MD5 = "e793b78cc4c9e9a4ccc0c1155377a412"
IMAGENETTE_FOLDER_NAME = "imagenette2-160"
DATASET_SEED = 255


def _find_repo_root(start: Path | None = None) -> Path:
    """Find the project root so the data path works from different entry points."""
    current = (start or Path.cwd()).resolve()

    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() and (candidate / "bayesian_cv").exists():
            return candidate

    raise FileNotFoundError("Could not locate the project root.")


def _find_imagenette_dir(raw_data_dir: Path) -> Path | None:
    """Find the extracted ImageNette folder under data/raw."""
    candidates = [
        raw_data_dir / IMAGENETTE_FOLDER_NAME,
        raw_data_dir / f"{IMAGENETTE_FOLDER_NAME}.tgz" / IMAGENETTE_FOLDER_NAME,
    ]

    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate

    return None


def _ensure_imagenette_dataset(raw_data_dir: Path) -> Path:
    """Reuse existing data if it is already there, otherwise download it once."""
    dataset_dir = _find_imagenette_dir(raw_data_dir)
    if dataset_dir is not None:
        print(f"ImageNette already exists at: {dataset_dir}")
        return dataset_dir

    print("ImageNette not found. Downloading to data/raw...")
    tf.keras.utils.get_file(
        origin=IMAGENETTE_URL,
        file_hash=IMAGENETTE_MD5,
        cache_dir=str(raw_data_dir),
        cache_subdir="",
        extract=True,
    )

    dataset_dir = _find_imagenette_dir(raw_data_dir)
    if dataset_dir is None:
        raise FileNotFoundError("ImageNette was downloaded, but the dataset folder could not be found.")

    print(f"ImageNette downloaded to: {dataset_dir}")
    return dataset_dir


def _load_split(split_dir: Path, config: ProjectConfig, shuffle: bool) -> tf.data.Dataset:
    """Load one dataset split from a folder of class subfolders."""
    return tf.keras.utils.image_dataset_from_directory(
        directory=str(split_dir),
        labels="inferred",
        label_mode="categorical",
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=shuffle,
        seed=DATASET_SEED,
    )


def load_datasets(
    config: ProjectConfig,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Load ImageNette from data/raw.

    The dataset is downloaded automatically the first time.
    Later runs reuse the same local copy.
    """
    repo_root = _find_repo_root()
    raw_data_dir = repo_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    imagenette_dir = _ensure_imagenette_dataset(raw_data_dir)

    train_dir = imagenette_dir / "train"
    val_dir = imagenette_dir / "val"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing train split: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Missing val split: {val_dir}")

    # We use the raw ImageNette train/val folders directly to keep the loader simple.
    train_ds = _load_split(train_dir, config, shuffle=True).cache().prefetch(tf.data.AUTOTUNE)
    val_ds = _load_split(val_dir, config, shuffle=False).cache().prefetch(tf.data.AUTOTUNE)
    test_ds = val_ds

    print("Datasets loaded successfully.")
    print(f"Train split: {train_dir}")
    print(f"Validation split: {val_dir}")

    return train_ds, val_ds, test_ds

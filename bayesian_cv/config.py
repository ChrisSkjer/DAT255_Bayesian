from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    image_size: tuple[int, int] = (128, 128)
    batch_size: int = 32
    num_classes: int = 10
    epochs: int = 10
    learning_rate: float = 1e-3
    dropout_rate: float = 0.3
    mc_samples: int = 30
    conv_filters: tuple[int, ...] = (32, 64, 128)
    dense_units: int = 128

    train_dir: Path = Path("data/train")
    val_dir: Path = Path("data/val")
    test_dir: Path = Path("data/test")
    model_path: Path = Path("artifacts/model.keras")


def get_config() -> ProjectConfig:
    return ProjectConfig()

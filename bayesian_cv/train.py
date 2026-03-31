from bayesian_cv.config import get_config
from bayesian_cv.data import load_datasets
from bayesian_cv.model import build_model 


def main() -> None:
    config = get_config()

    train_ds, val_ds, test_ds = load_datasets(config) #må vurdere om vi skal splitte i 2 eller 3

    model = build_model(config)
    
    # TODO:
    # - Add callbacks if you want checkpointing or early stopping.
    # - Train the model.
    # - Evaluate on the test set.
    _ = (train_ds, val_ds, test_ds, model)
    raise NotImplementedError("Implement training loop in bayesian_cv/train.py")


if __name__ == "__main__":
    main()

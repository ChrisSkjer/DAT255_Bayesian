import tensorflow as tf

from bayesian_cv.config import get_config
from bayesian_cv.data import load_datasets
from bayesian_cv.model import build_model


def main() -> None:
    """Train the model, evaluate it, and save it for later MC dropout inference."""
    config = get_config()

    # Load the datasets first so training uses the same data pipeline as the rest of the project.
    train_ds, val_ds, test_ds = load_datasets(config)

    # Build the CNN with dropout layers that will later be reused for MC dropout prediction.
    model = build_model(config)

    # Create the output folder before saving the trained model.
    config.model_path.parent.mkdir(parents=True, exist_ok=True)

    # Early stopping keeps the first training loop simple while avoiding unnecessary epochs.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
    )

    print("Training finished.")
    print(f"Epochs completed: {len(history.history.get('loss', []))}")

    # Evaluate on the held-out dataset so we have baseline metrics before uncertainty analysis.
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Save the trained model so predict.py can reuse it later.
    model.save(config.model_path)
    print(f"Saved model to: {config.model_path}")


if __name__ == "__main__":
    main()

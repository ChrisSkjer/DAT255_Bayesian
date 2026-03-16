# DAT255_Bayesian
Computer vision project exploring predictive uncertainty with Monte Carlo Dropout. A CNN is trained on an image dataset, and dropout is kept active during inference to perform multiple stochastic forward passes. The resulting prediction distribution is used to estimate model uncertainty.

## Minimal structure

```text
bayesian_cv/
  config.py
  data.py
  model.py
  train.py
  predict.py
  uncertainty.py
requirements.txt
```

## Dataset layout

```text
data/
  train/
    class_a/
    class_b/
  val/
    class_a/
    class_b/
  test/
    class_a/
    class_b/
```

## What is ready

- Shared config in `bayesian_cv/config.py`
- `MCDropout` and `mc_predict(...)` for uncertainty estimation
- Empty starter files for data loading, model, training, and prediction

## Colab

Install dependencies:

```bash
pip install -r requirements.txt
```

Then implement your code and run it in Colab when you want GPU training.

# Colab workflow

This project is structured so training can be run locally or in Google Colab.

Suggested Colab flow:

1. Upload or clone the repository into Colab.
2. Add your dataset under `data/train`, `data/val`, and `data/test`.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Train:

```bash
python -m bayesian_cv.train
```

5. Run Monte Carlo Dropout inference:

```bash
python -m bayesian_cv.predict
```

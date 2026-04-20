# 🚀 Streamlit App for ImageNette Classifier

En interaktiv Streamlit-app for å klassifisere ImageNette-bilder og estimere prediksjonsusikkerhet ved hjelp av MC Dropout.

## 📋 Installasjon

### 1. Installer avhengigheter
```bash
pip install -r requirements.txt
```

### 2. Trening av modellen
Hvis du ikke allerede har en trent modell, må du trene den først. Kjør en av notebook-ene:
- `notebooks/03_training_check.ipynb` - Basic training
- `notebooks/04_improved_CNN_Architecture (2).ipynb` - Improved version med eksperimentering

Eller kjør training via Python:
```bash
python -m bayesian_cv.train
```

Modellen vil bli lagret i `artifacts/models/model.keras` (eller den adressen som er konfigurert).

## 🎯 Kjøring av Streamlit-appen

### Fra hovedmappen:
```bash
streamlit run app/app.py
```

### Fra app-mappen:
```bash
cd app
streamlit run app.py
```

Dette åpner appen i nettleseren på `http://localhost:8501`

## 💡 Bruk

1. **Last opp et bilde**: Klikk på "Upload Image" og velg en JPG/PNG-fil
2. **Se resultater**: 
   - Topprediksjonen vises med konfidensgrad
   - Usikkerhetsmetrikker (entropy og variance) beregnes
   - Top-5 prediksjoner vises i et stolpediagram
   - Alle klassesannsynligheter vises i en tabell

3. **Juster MC Samples**: Bruk slideren i sidebar for å endre antall MC samples (10-100)
   - Flere samples = mer nøyaktig usikkerhet, men tregere
   - Færre samples = raskere, men mindre nøyaktig

## 📊 Tolking av resultater

- **Predictive Entropy**: Høy verdi = modellen er usikker
- **Mean Variance**: Viser variasjon på tvers av MC samples
- **Confidence**: Sannsynligheten for topprediksjonen

## ⚙️ Konfigurering

Endre innstillinger i `bayesian_cv/config.py`:
- `image_size`: Bildestørrelse som modellen forventer
- `num_classes`: Antall klasser (ImageNette = 10)
- `mc_samples`: Standard antall MC samples
- `dropout_rate`: Dropout-rate for usikkerhetsstimasjon

## 🔧 Feilsøking

**"Model not found"**: Sørg for at modellen er trent og lagret på riktig sted
```bash
# Sjekk om modellen finnes:
ls artifacts/models/  # or ls artifacts/models/*.keras
```

**Memory issues**: Reducer antall MC samples eller bildestørrelsen

**Slow predictions**: Dette er normalt ved 100 MC samples. Reduser til 30-50 for raskere prediksjoner.

## 📚 Mer informasjon

Se "About Model" tab i Streamlit-appen for mer informasjon om:
- Modellarkitektur
- MC Dropout-metodikk
- Tolking av usikkerhetsmetrikker


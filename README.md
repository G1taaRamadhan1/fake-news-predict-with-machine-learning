[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fake-news-predict-with-machine-learning-ywh9yfhnyswgrhjrgsagz9.streamlit.app)

# Fake News Detector — Streamlit App (Lite Version)

A web app for predicting whether a news article is Real or Fake, using an ensemble of 3 ML models trained on the ISOT Fake News Dataset.

This is the **lite version** — no TensorFlow needed, much faster to install and deploy. This Project is for submitting Final Project Data Science Bootcamp from Digital Skola Batch 46.

## Features

- **Paste any article** in English — the app cleans, vectorizes, and predicts
- **3 models running together**: Logistic Regression, Naive Bayes, Passive Aggressive
- **Per-model breakdown** + ensemble verdict + confidence score
- **Pre-loaded examples** to test quickly
- **Disagreement warning** when models disagree on a verdict


## Project Structure

```
streamlit_app/
├── app.py                              ← main Streamlit app
├── requirements.txt                    ← Python dependencies (no TF)
├── README.md                           ← this file
└── models/
    ├── lr_model.joblib                 (158 KB)  Logistic Regression
    ├── nb_model.joblib                 (626 KB)  Multinomial Naive Bayes
    ├── pac_model.joblib                (158 KB)  Passive Aggressive
    ├── tfidf_vectorizer.joblib         (776 KB)  TF-IDF (20k features, 1-2 grams)
    └── config.json                              MAX_LEN, model performance metadata
```

Total artifact size: ~1.7 MB.

## How It Works

1. **Input** — user pastes a news article into the text area
2. **Cleaning** — same pipeline as training (lowercase, remove (Reuters), URLs, mentions, non-letters, stop words)
3. **Feature extraction** — TF-IDF vector (20,000 features, 1-2 grams)
4. **Prediction** — 3 ML models output P(Fake)
5. **Ensemble** — average of 3 probabilities; if ≥ 0.5 → FAKE
6. **Display** — verdict, confidence, per-model breakdown, bar chart

## Models Performance (on test set)

| Model | Accuracy | F1 | ROC-AUC |
|---|:---:|:---:|:---:|
| Passive Aggressive | 0.9871 | 0.9860 | 0.9986 |
| Logistic Regression | 0.9791 | 0.9772 | 0.9977 |
| Multinomial Naive Bayes | 0.9496 | 0.9458 | 0.9885 |

## Limitations

The model was trained on a specific dataset (Reuters real news vs Wikipedia-flagged fake news, 2016-2017). It may not generalize to:

- News in languages other than English
- News from sources outside the training distribution (BBC, CNN, etc.)
- News from outside 2016-2017 topics
- Short text (less than ~20 words)

The model also relies partly on writing **style** rather than facts, so well-written misinformation could fool it. Use this as an educational demo, not a production fact-checker.



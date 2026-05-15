"""
Fake News Detection - Streamlit App (Lightweight Version)

Uses only ML models (Logistic Regression, Naive Bayes, Passive Aggressive).
No TensorFlow needed - faster install, lighter deployment.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import joblib
import re
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Load models (cached so it only loads once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load all trained ML models and the TF-IDF vectorizer."""
    artifacts = {}
    artifacts['lr'] = joblib.load('models/lr_model.joblib')
    artifacts['nb'] = joblib.load('models/nb_model.joblib')
    artifacts['pac'] = joblib.load('models/pac_model.joblib')
    artifacts['tfidf'] = joblib.load('models/tfidf_vectorizer.joblib')
    with open('models/config.json') as f:
        artifacts['config'] = json.load(f)
    return artifacts

# ---------------------------------------------------------------------------
# Preprocessing function (must match training pipeline EXACTLY)
# ---------------------------------------------------------------------------
STOP_WORDS = set(ENGLISH_STOP_WORDS)

def clean_text(text):
    """Clean text the same way as during training."""
    text = str(text).lower()
    text = re.sub(r'\(reuters\)\s*-?\s*', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(words)

# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------
def predict_news(text, artifacts):
    """Run all 3 ML models on the given text and return their predictions."""
    cleaned = clean_text(text)

    if len(cleaned.split()) < 5:
        return None

    # Vectorize
    vec = artifacts['tfidf'].transform([cleaned])

    # Get probabilities from each model
    prob_lr = float(artifacts['lr'].predict_proba(vec)[0, 1])
    prob_nb = float(artifacts['nb'].predict_proba(vec)[0, 1])

    # Passive Aggressive doesn't have predict_proba
    # Convert decision_function to a probability using a sigmoid
    score_pac = float(artifacts['pac'].decision_function(vec)[0])
    prob_pac = 1.0 / (1.0 + np.exp(-score_pac))

    # Ensemble (simple average of 3 models)
    prob_ensemble = (prob_lr + prob_nb + prob_pac) / 3

    return {
        'cleaned_text': cleaned,
        'word_count': len(cleaned.split()),
        'lr': prob_lr,
        'nb': prob_nb,
        'pac': prob_pac,
        'ensemble': prob_ensemble,
        'verdict': 'FAKE' if prob_ensemble >= 0.5 else 'REAL',
        'confidence': max(prob_ensemble, 1 - prob_ensemble)
    }

# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # Load models with a spinner
    with st.spinner('Loading trained models...'):
        artifacts = load_artifacts()

    # Title and description
    st.title('Fake News Detector')
    st.markdown(
        'Predict whether a news article is **Real** or **Fake** using an ensemble of '
        '3 ML models trained on the ISOT Fake News Dataset.'
    )

    # Sidebar with info
    with st.sidebar:
        st.header('About this App')
        st.markdown("""
        This app uses 3 ML models trained on **38,823 cleaned articles** from the
        ISOT Fake News Dataset:

        - **Logistic Regression** (TF-IDF features)
        - **Multinomial Naive Bayes** (TF-IDF features)
        - **Passive Aggressive** (TF-IDF features) ← best single model

        The final verdict is the **average** of all 3 predictions.
        """)

        st.divider()
        st.subheader('Best Model Performance')
        cfg = artifacts['config']
        col1, col2 = st.columns(2)
        col1.metric('Test F1', f"{cfg['best_f1']:.4f}")
        col2.metric('Test Accuracy', f"{cfg['best_accuracy']:.4f}")
        st.caption(f"Trained on {cfg['training_size']:,} articles, "
                   f"tested on {cfg['test_size']:,} articles.")

        st.divider()
        st.subheader('Limitations')
        st.warning(
            'This model was trained on Reuters (real) and Wikipedia-flagged sources '
            '(fake) from 2016-2017. Articles from outside this distribution may be '
            'misclassified. The model relies partly on writing style, not just facts.'
        )

    # Main content area
    tab1, tab2, tab3 = st.tabs(['Predict', 'Examples', 'How it works'])

    # ----- TAB 1: PREDICT -----
    with tab1:
        st.subheader('Paste a News Article')

        text_input = st.text_area(
            'Article text:',
            height=250,
            placeholder='Paste an English news article here (at least a few sentences)...'
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            predict_button = st.button('🔮 Predict', type='primary', use_container_width=True)
        with col2:
            if text_input:
                st.caption(f'Input: {len(text_input)} chars, {len(text_input.split())} words')

        if predict_button:
            if not text_input.strip():
                st.error('Please paste an article first!')
            elif len(text_input.split()) < 20:
                st.warning('Article is too short. At least 20 words is recommended for reliable prediction.')
            else:
                with st.spinner('Running 3 models...'):
                    result = predict_news(text_input, artifacts)

                if result is None:
                    st.error('After cleaning, the text is too short. Try a longer article.')
                else:
                    show_prediction(result)

    # ----- TAB 2: EXAMPLES -----
    with tab2:
        st.subheader('Try These Examples')
        st.markdown('Click any example below to load it into the predictor.')

        examples = {
            'Real-style (Federal Reserve)': """WASHINGTON - The Federal Reserve raised its benchmark interest rate by a quarter percentage point on Wednesday, citing continued strength in the labor market and inflation that remains above the central bank's 2 percent target. The decision was unanimous among voting members of the Federal Open Market Committee. In a statement released after the meeting, the Fed said it would continue to monitor incoming economic data and adjust policy as appropriate to achieve its dual mandate of maximum employment and price stability.""",

            'Fake-style (sensational)': """SHOCKING! You won't BELIEVE what this politician just admitted on live television! The mainstream media is hiding this from you, but we have the EXCLUSIVE footage that proves everything they've been saying is a complete LIE. Patriots everywhere are sharing this story before it gets censored. The deep state is going to be exposed once and for all and the establishment is in full panic mode! Share this with everyone you know before it's too late!""",

            'Ambiguous (short and neutral)': """The president met with foreign leaders today to discuss trade and security matters. Officials said the talks were productive, though no formal agreement was reached. Further negotiations are expected next month."""
        }

        for label, text in examples.items():
            with st.expander(f'{label}'):
                st.text(text)
                if st.button(f'Predict this →', key=f'btn_{label}'):
                    with st.spinner('Running 3 models...'):
                        result = predict_news(text, artifacts)
                    show_prediction(result)

    # ----- TAB 3: HOW IT WORKS -----
    with tab3:
        st.subheader('How the Prediction Pipeline Works')

        st.markdown("""
        ### 1. Text Cleaning
        Your input goes through the same cleaning pipeline used during training:
        - Lowercase everything
        - Remove `(Reuters)` boilerplate (this was a data leakage source)
        - Remove URLs and @mentions
        - Keep only letters
        - Remove English stop words
        - Drop very short words

        ### 2. Feature Extraction
        Convert text to TF-IDF vectors (20,000 features, 1-2 grams).

        ### 3. Model Predictions
        Each model gives a probability that the article is fake (0 = real, 1 = fake):
        - Logistic Regression
        - Multinomial Naive Bayes
        - Passive Aggressive (sigmoid of decision function)

        ### 4. Ensemble
        The final score is the **average** of the 3 probabilities. If ≥ 0.5 → FAKE, else REAL.

        ### About the Dataset
        Models were trained on the **ISOT Fake News Dataset** with about 45,000 articles:
        - Real: 21,417 articles from Reuters.com
        - Fake: 23,481 articles from sources flagged by PolitiFact and Wikipedia
        - Mostly from 2016-2017
        """)


def show_prediction(result):
    """Display the prediction nicely."""
    # Big verdict banner
    if result['verdict'] == 'FAKE':
        st.error(f"🔴 **VERDICT: FAKE** (confidence: {result['confidence']*100:.1f}%)")
    else:
        st.success(f"🟢 **VERDICT: REAL** (confidence: {result['confidence']*100:.1f}%)")

    # Ensemble score with progress bar
    st.markdown(f"### Ensemble Score: `{result['ensemble']:.4f}`")
    st.progress(float(result['ensemble']))
    st.caption('0 = Real, 1 = Fake. Threshold for FAKE is 0.5.')

    # Per-model scores
    st.markdown('### Individual Model Scores')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric('Logistic Regression', f"{result['lr']:.4f}",
                  delta=('FAKE' if result['lr'] >= 0.5 else 'REAL'),
                  delta_color='inverse' if result['lr'] >= 0.5 else 'normal')

    with col2:
        st.metric('Naive Bayes', f"{result['nb']:.4f}",
                  delta=('FAKE' if result['nb'] >= 0.5 else 'REAL'),
                  delta_color='inverse' if result['nb'] >= 0.5 else 'normal')

    with col3:
        st.metric('Passive Aggressive', f"{result['pac']:.4f}",
                  delta=('FAKE' if result['pac'] >= 0.5 else 'REAL'),
                  delta_color='inverse' if result['pac'] >= 0.5 else 'normal')

    # Bar chart of all scores
    chart_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Naive Bayes', 'Passive Aggressive', 'Ensemble'],
        'P(Fake)': [result['lr'], result['nb'], result['pac'], result['ensemble']]
    })
    st.bar_chart(chart_df.set_index('Model'), height=300)

    # Cleaned text preview
    with st.expander(f'View cleaned text ({result["word_count"]} words)'):
        st.text(result['cleaned_text'][:1000] + ('...' if len(result['cleaned_text']) > 1000 else ''))

    # Disagreement warning
    preds = [result['lr'] >= 0.5, result['nb'] >= 0.5, result['pac'] >= 0.5]
    if len(set(preds)) > 1:
        st.warning('The 3 models disagree on this article. Treat the ensemble result with caution.')


if __name__ == '__main__':
    main()

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import downloader
import re
import string
from nltk.corpus import stopwords

# Fungsi untuk mengunduh resource NLTK
def download_nltk_resources(resource):
    try:
        nltk.data.find(resource)
    except nltk.downloader.DownloadError:
        st.warning(f"Mengunduh resource NLTK: '{resource}'. Ini mungkin memakan waktu sebentar saat pertama kali dijalankan.")
        nltk.download(resource)
        st.info(f"Resource NLTK '{resource}' berhasil diunduh.")

# Unduh resource 'punkt' untuk tokenisasi
download_nltk_resources('tokenizers/punkt')

# Unduh stopwords jika belum pernah
try:
    stopwords.words('english')
except LookupError:
    download_nltk_resources('corpora/stopwords')
stop_words = set(stopwords.words('english'))

# Fungsi untuk pra-pemrosesan teks (sesuaikan dengan yang Anda gunakan)
def preprocess_text(text):
    if isinstance(text, str):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [w for w in tokens if w not in stop_words and w.isalpha()]
        return ' '.join(filtered_tokens)
    return ''

# Load model dan vectorizer
try:
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('LR_model.pkl')
except FileNotFoundError:
    st.error("File model atau vectorizer tidak ditemukan. Pastikan 'tfidf_vectorizer.pkl' dan 'LR_model.pkl' ada di direktori yang sama.")
    st.stop()

# Judul aplikasi
st.title("Deteksi Berita Palsu")
st.subheader("Masukkan teks berita untuk mendeteksi apakah berita tersebut palsu atau asli.")

# Area input teks
news_text = st.text_area("Teks Berita", height=200)

# Tombol prediksi
if st.button("Deteksi"):
    if news_text:
        # Pra-pemrosesan teks input
        processed_text = preprocess_text(news_text)

        # Vectorize teks input menggunakan vectorizer yang sama
        text_vectorized = tfidf_vectorizer.transform([processed_text])

        # Lakukan prediksi
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0][1] # Probabilitas kelas 'REAL'

        # Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi:")
        if prediction == 1:
            st.success(f"Berita ini kemungkinan **ASLI** (Probabilitas: {probability:.2f})")
        else:
            st.warning(f"Berita ini kemungkinan **PALSU** (Probabilitas: {1 - probability:.2f})")
    else:
        st.warning("Mohon masukkan teks berita.")

# Catatan kaki (opsional)
st.markdown("---")
st.markdown("Developed using Streamlit and a trained Logistic Regression model.")

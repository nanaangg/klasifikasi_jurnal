import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================
# ===== LOAD VECTORIZER ======
# ============================
try:
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = joblib.load(f)
except Exception as e:
    st.error("❌ Gagal memuat TF-IDF Vectorizer.")
    st.stop()

# ============================
# ===== LOAD MODELS ==========
# ============================
try:
    with open("model_logistic_regression.pkl", "rb") as f:
        model_logistic = joblib.load(f)
    with open("model_random_forest.pkl", "rb") as f:
        model_rf = joblib.load(f)
except Exception as e:
    st.error("❌ Gagal memuat salah satu model (.pkl).")
    st.stop()

# ============================
# ===== STREAMLIT UI =========
# ============================

st.set_page_config(page_title="Klasifikasi Jurnal", layout="centered")
st.title("📘 Klasifikasi Kalimat Jurnal Ilmiah")
st.markdown("Prediksi apakah kalimat ditulis oleh **👤 Manusia** atau dihasilkan oleh **🤖 AI**.")
st.info("ℹ️ Catatan: Aplikasi ini hanya diperuntukkan untuk teks pada **latar belakang jurnal ilmiah**.")

# Pilih model
model_choice = st.selectbox("Pilih Model Klasifikasi:", ["Logistic Regression", "Random Forest"])

# Input teks
text_input = st.text_area("Masukkan kalimat jurnal:", height=150)

# Tombol prediksi
if st.button("Prediksi"):
    if text_input.strip() == "":
        st.warning("⚠️ Silakan masukkan kalimat terlebih dahulu.")
    else:
        # Transformasi ke TF-IDF
        tfidf_input = vectorizer.transform([text_input])

        # Pilih model berdasarkan opsi
        if model_choice == "Logistic Regression":
            model = model_logistic
        else:
            model = model_rf

        # Prediksi
        prediction = model.predict(tfidf_input)[0]
        label_output = "👤 Manusia" if prediction == 1 else "🤖 AI"

        st.success(f"**Prediksi: {label_output}**")

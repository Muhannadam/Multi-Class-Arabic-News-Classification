import streamlit as st
import joblib
import re
from nltk.corpus import stopwords

# تحميل الملفات المحفوظة
model = joblib.load('baseline_lr_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# إعداد الكلمات التوقفية
arabic_stopwords = set(stopwords.words('arabic'))

# دالة تنظيف النص العربي
def clean_text(text):
    def remove_tashkeel(text):
        return re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)

    def remove_repeated_chars(text):
        return re.sub(r'(.)\1{2,}', r'\1\1', text)

    text = remove_tashkeel(text)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    text = re.sub(r'[\d\u0660-\u0669]+', ' ', text)
    text = remove_repeated_chars(text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in arabic_stopwords and len(w) > 1]
    return ' '.join(tokens)

# Streamlit UI
st.title("🔎 Arabic News Classifier (Logistic Regression)")

input_text = st.text_area("✍️ أدخل المقال أو النص الإخباري هنا", height=200)

if st.button("🔍 تصنيف المقال"):
    if input_text.strip() == "":
        st.warning("الرجاء إدخال نص.")
    else:
        cleaned = clean_text(input_text)
        tfidf_input = vectorizer.transform([cleaned])
        pred = model.predict(tfidf_input)
        label = label_encoder.inverse_transform(pred)[0]
        st.success(f"✅ الفئة المتوقعة: **{label}**")

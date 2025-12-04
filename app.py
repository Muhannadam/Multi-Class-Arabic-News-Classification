import streamlit as st
import joblib
import re
import os
import requests
import nltk
from nltk.corpus import stopwords

# Groq API Key (from environment variable)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# تحميل الملفات المحفوظة
model = joblib.load('baseline_lr_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# إعداد الكلمات التوقفية العربية
try:
    stopwords.words('arabic')
except LookupError:
    nltk.download('stopwords')

arabic_stopwords = set(stopwords.words('arabic'))

# دالة تنظيف النص العربي
def clean_text(text):
    def remove_tashkeel(t): return re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', t)
    def remove_repeated_chars(t): return re.sub(r'(.)\1{2,}', r'\1\1', t)

    text = remove_tashkeel(text)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    text = re.sub(r'[\d\u0660-\u0669]+', ' ', text)
    text = remove_repeated_chars(text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in arabic_stopwords and len(w) > 1]
    return ' '.join(tokens)

# دالة التلخيص واقتراح العنوان باستخدام Groq API
def summarize_and_suggest_title(text):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "allam-2-7b",
        "messages": [
            {"role": "system", "content": "أنت مساعد ذكي. عندما يصلك نص طويل، قم بتلخيصه بشكل مختصر، ثم اقترح عنوانًا قصيرًا وجذابًا باللغة العربية."},
            {"role": "user", "content": f"هذا هو نص المقال:\n\n{text}\n\nرجاءً: 1- لخص المقال في فقرة قصيرة. 2- اقترح عنوانًا ذكيًا للمقال."}
        ],
        "temperature": 0.5,
        "max_tokens": 500
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"❌ خطأ في الاتصال: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ خطأ أثناء التلخيص: {str(e)}"

# واجهة Streamlit
st.title("🔎 Arabic News Classifier (Logistic Regression + Groq AI)")

input_text = st.text_area("✍️ أدخل المقال أو النص الإخباري هنا", height=200)

if st.button("🔍 تصنيف المقال"):
    if input_text.strip() == "":
        st.warning("الرجاء إدخال نص.")
    else:
        # تنظيف النص وتصنيفه
        cleaned = clean_text(input_text)
        tfidf_input = vectorizer.transform([cleaned])
        pred = model.predict(tfidf_input)
        label = label_encoder.inverse_transform(pred)[0]
        st.success(f"✅ الفئة المتوقعة: **{label}**")

        # التلخيص واقتراح العنوان
        with st.spinner("✍️ جاري تلخيص الخبر واقتراح عنوان..."):
            summary_output = summarize_and_suggest_title(input_text)
            st.subheader("📝 تلخيص وعنوان مقترح:")
            st.markdown(summary_output)

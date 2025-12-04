import streamlit as st
import joblib
import re
import os
import requests
import nltk
from nltk.corpus import stopwords

# Load Groq API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load saved models and encoders
model = joblib.load('baseline_lr_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load Arabic stopwords
try:
    stopwords.words('arabic')
except LookupError:
    nltk.download('stopwords')

arabic_stopwords = set(stopwords.words('arabic'))

# Arabic text preprocessing function
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

# Groq summarization + title suggestion function
def summarize_and_suggest_title(text):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "allam-2-7b",
        "messages": [
            {"role": "system", "content": "أنت مساعد ذكي. عندما يصلك نص طويل، قم باقتراح عنوانًا قصيرًا وجذابًا باللغة العربية ثم بتلخيصه بشكل مختصر"},
            {"role": "user", "content": f"هذا هو نص المقال:\n\n{text}\n\nرجاءً: 1- اقترح عنوانًا ذكيًا للمقال 2- لخص المقال في فقرة قصيرة."}
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

# Set page to RTL and Arabic font using HTML injection
st.markdown(
    """
    <style>
    body {
        direction: RTL;
        text-align: right;
        font-family: 'Arial', sans-serif;
    }
    .stTextArea textarea {
        direction: RTL;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
page = st.sidebar.selectbox("انتقل إلى:", ["📄 الصفحة الرئيسية", "ℹ️ حول المشروع"])

# Main Page: Classification
if page == "📄 الصفحة الرئيسية":
    st.title("🔎 مصنف الأخبار العربية")
    st.markdown("**هذا النموذج يقوم بتصنيف المقالات العربية إلى فئات إخبارية، ويقترح عنوانًا ذكيًا ويقدم تلخيصًا موجزًا باستخدام تقنية Groq AI.**")

    input_text = st.text_area("✍️ أدخل المقال أو النص الإخباري هنا:", height=200)

    if st.button("🔍 تصنيف المقال"):
        if input_text.strip() == "":
            st.warning("⚠️ الرجاء إدخال نص.")
        else:
            # Preprocess + predict
            cleaned = clean_text(input_text)
            tfidf_input = vectorizer.transform([cleaned])
            pred = model.predict(tfidf_input)
            label = label_encoder.inverse_transform(pred)[0]
            st.success(f"✅ الفئة المتوقعة: **{label}**")

            # Summarization + title suggestion
            with st.spinner("✍️ جاري التلخيص واقتراح العنوان..."):
                summary_output = summarize_and_suggest_title(input_text)
                st.subheader("📝 التلخيص والعنوان المقترح:")
                st.markdown(summary_output)

# About Page
elif page == "ℹ️ حول المشروع":
    st.title("ℹ️ معلومات عن المشروع")
    st.markdown("""
    هذا المشروع هو نظام تصنيف ذكي للمقالات الإخبارية العربية، يعتمد على نموذج **Logistic Regression** مدرب باستخدام بيانات **SANAD Dataset**.
    
    المزايا:
    - تصنيف المقالات إلى فئات مثل السياسة، الرياضة، الصحة، الدين، وغيرها.
    - تلخيص المقال تلقائيًا واقتراح عنوان ذكي باستخدام نموذج **Allam-2-7B** من منصة **Groq**.
    - واجهة تفاعلية مبنية باستخدام **Streamlit**.

    التقنية المستخدمة:
    - Python (scikit-learn, joblib, NLTK)
    - Groq API (Allam-2-7B)
    - Streamlit
    - GitHub + Streamlit Cloud

    📌 هذا المشروع يهدف إلى تعزيز معالجة اللغة العربية باستخدام تقنيات حديثة في الذكاء الاصطناعي.
    """)

# 📰 Arabic News Classifier with Smart Summarization (TF‑IDF + Groq LLM)

🚀 A lightweight, intelligent web app that classifies Arabic news articles and generates smart summaries and headlines using a hybrid of classical ML and cutting-edge LLMs.

## 📌 Overview

This Streamlit-based app allows users to paste Arabic news text and instantly:

✅ Classify the article into one of several categories (Politics, Sports, Religion, Tech, etc.)  
✅ Generate a concise **summary** and a **catchy headline** in Arabic using the [Allam‑2‑7B](https://groq.com) model via Groq API

---

## 🎯 Use Case

This project serves as a prototype for **automated content understanding** and **headline generation** in Arabic—ideal for:

- Media monitoring systems  
- Arabic news portals  
- NLP research in Arabic  
- Educational projects in AI & Data Science

---

## 🛠️ Tech Stack

| Component              | Details                               |
|------------------------|----------------------------------------|
| **Model**              | Logistic Regression (Scikit-learn)     |
| **Text Vectorization** | TF-IDF (10k max features)              |
| **LLM API**            | Allam‑2‑7B via Groq API (chat format)  |
| **Interface**          | Streamlit (with RTL layout for Arabic) |
| **Data**               | SANAD Arabic News Dataset              |
| **Language**           | Full Arabic Support 🇸🇦                 |

---

## ✨ Features

- 🧠 **Classical ML + LLM hybrid**: Fast classification using trained Logistic Regression + powerful summarization via Allam‑2‑7B
- 🌍 **Arabic-native NLP pipeline**: Preprocessing, cleaning, and stopword removal in Arabic
- 🧾 **Instant results**: Classification + summarization in real-time
- 🖥️ **Streamlit web UI**: Right-to-left layout, bilingual comments, intuitive UX
- 🔐 **API-ready design**: Easily extendable for production environments

---

## ⚙️ Setup & Deployment

### 🔧 Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/arabic-news-classifier.git
cd arabic-news-classifier

# 2. Create virtual environment (optional)
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py

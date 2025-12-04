# Arabic News Classifier & Summarizer (TF‑IDF + Logistic Regression + Groq LLM)

Welcome to our final project for **EMAI 641: Deep Learning**.

This project showcases an integrated NLP system capable of:

- Classifying Arabic news articles into predefined categories  
- Generating intelligent summaries  
- Suggesting concise and meaningful Arabic headlines  

The project uses **Allam‑2‑7B** through **Groq API** exclusively for generative tasks (summarization + title suggestion), while classification relies on classical ML models.

---

## Project Description

We developed an end-to-end Arabic news classification and summarization system using a hybrid of:

- **Classical Machine Learning** (TF‑IDF + Logistic Regression)
- **Generative AI** (Allam model via Groq API)

The system is fully optimized for **Arabic language processing**, supports **RTL layout**, and provides fast, accurate, and production-ready performance.

This project demonstrates practical applications of NLP in the Arabic domain, including:

- News platform automation  
- Content recommendation systems  
- Media monitoring  
- Journalism tools  

---

## Key Features

### **Classification Features**
- Accurate classification of Arabic news articles into **7 major categories**
- Categories: *Finance, Sports, Medical, Technology, Politics, Religion, Culture*
- Built using TF‑IDF vectorization + Logistic Regression  
- Fast inference suitable for real-time applications  

### **Generative Features (via Allam LLM)**
- Smart summarization of the provided article
- Generating an Arabic headline that is:
  - Short  
  - Informative  
  - Engaging  
- Seamless integration with Groq API

### **User Interface**
- Fully RTL (Right-to-Left) Arabic interface  
- Built with Streamlit  
- Clean, modern, and responsive design  
- Secure API key handling using Streamlit Secrets Management  

---

## Technologies Used

| Category | Technologies |
|---------|--------------|
| Programming Language | Python 3.10 |
| Machine Learning | Logistic Regression (TF‑IDF) |
| Generative AI | Allam‑2‑7B (Groq API) |
| Web Framework | Streamlit |
| Data Processing | Pandas, Regex, NLTK |
| Model Persistence | Joblib |
| Training & EDA | Google Colab |
| API Communication | Requests Library |

---

## Dataset Details

- **Dataset Name:** SANAD Dataset  
- **Language:** Arabic  
- **Samples:** ~45,500 articles  
- **Categories:** Finance, Sports, Medical, Technology, Politics, Religion, Culture  
- **Dataset Link:**  
  https://www.kaggle.com/datasets/haithemhermessi/sanad-dataset  

### Preprocessing Steps

- Removing Arabic diacritics (Tashkeel)  
- Eliminating non-Arabic characters and punctuation  
- Removing digits and symbols  
- Normalizing the text  
- Removing Arabic stopwords using NLTK  
- Cleaning repeated characters  
- Whitespace normalization  
- Splitting data into **80% training / 20% testing**

---

## System Workflow

### 1. User Input  
User pastes or enters an Arabic news article.

### 2. Text Preprocessing  
Clean and normalize the Arabic text using custom preprocessing functions.

### 3. ML Classification  
- Apply TF‑IDF vectorization  
- Predict using Logistic Regression  
- Return top predicted category  

### 4. Generative AI Processing  
Send the original article to **Allam‑2‑7B** for:  
- Article summarization  
- Smart Arabic headline suggestion  

### 5. Display Output  
- Classification result  
- Summary  
- Proposed title  

---

## Live Deployment

| Platform | Link |
|---------|------|
| 📘 Google Colab (Training + EDA) | https://colab.research.google.com/drive/1tRoS5dUuBz7TnJMy-BrrV7b89i2RJCCa?usp=sharing |
| 🌐 Streamlit App | https://dl-news-classification.streamlit.app/ |
| 💻 GitHub Repository | https://github.com/Muhannadam/Multi-Class-Arabic-News-Classification/blob/main/README.md |

---
###Developed By

- Muhannad Almuntashiri
- Gaith Alhumaidi
- Eyad Alzahrani

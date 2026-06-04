# 🇮🇳 BharatFilter AI

## Multilingual Toxic Content Detection System using Machine Learning, Naive Search & Rabin-Karp Algorithm

### 📌 Overview

BharatFilter AI is an intelligent content moderation system designed to detect toxic, hateful, and abusive text in real time. The project combines Machine Learning with classical string matching algorithms to improve detection accuracy and demonstrate algorithm performance.

The system supports both English and Indian language toxic words and provides an interactive dashboard built with Streamlit.

---

## 🚀 Features

### 🤖 AI-Powered Detection

* TF-IDF Vectorization
* Logistic Regression Classifier
* Real-time toxic content prediction

### 🔍 Rule-Based Detection

* Naive String Matching Algorithm
* Rabin-Karp String Matching Algorithm
* Toxic keyword identification and highlighting

### 🌐 Multilingual Support

* English Toxic Words
* Hindi Toxic Words
* Marathi Toxic Words

### 📁 File Analysis

* TXT File Analysis
* CSV Dataset Analysis
* Batch Toxicity Detection

### 📊 Interactive Dashboard

* Streamlit-based User Interface
* Toxicity Metrics
* Performance Visualization
* Algorithm Comparison Charts

---

## 🏗️ System Architecture

User Input
↓
Text Preprocessing
↓
┌─────────────────────────────┐
│ Rule-Based Detection │
│ • Naive Search │
│ • Rabin-Karp │
└─────────────────────────────┘
↓
┌─────────────────────────────┐
│ Machine Learning Model │
│ • TF-IDF │
│ • Logistic Regression │
└─────────────────────────────┘
↓
Result Generation
↓
Dashboard Visualization

---

## 🧠 Algorithms Used

### 1. Naive Search Algorithm

A straightforward pattern matching algorithm that checks every possible position in the text.

**Time Complexity:** O(n × m)

### 2. Rabin-Karp Algorithm

Uses hashing techniques for efficient pattern matching.

**Average Time Complexity:** O(n + m)

### 3. Machine Learning Model

* Feature Extraction: TF-IDF Vectorizer
* Classification: Logistic Regression

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Backend

* Python

### Machine Learning

* Scikit-Learn
* Joblib

### Data Processing

* Pandas
* NumPy

### Algorithms

* Naive Pattern Matching
* Rabin-Karp Algorithm

---

## 📂 Project Structure

BharatFilter-AI/

├── app.py

├── train_model.py

├── model.pkl

├── vectorizer.pkl

├── project.ipynb

├── requirements.txt

└── README.md

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/BharatFilter-AI.git
cd BharatFilter-AI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📈 Dashboard Modules

### 🏠 Home

* Project Overview
* System Metrics

### 🔍 Analyze Text

* Real-Time Toxicity Detection
* Highlighted Toxic Words
* Toxicity Statistics

### 📁 File Analysis

* TXT File Scanning
* CSV Dataset Processing

### ℹ️ About

* Project Information
* Technical Details

---

## 🎯 Use Cases

* Social Media Moderation
* Online Community Management
* Content Filtering
* Educational Platforms
* Public Discussion Forums
* Comment Moderation Systems

---

## 📊 Results

The system combines Machine Learning predictions with rule-based detection to improve reliability and explainability.

Key Benefits:

* Fast Detection
* High Accuracy
* Multilingual Support
* Interactive Visualization
* Scalable Architecture

---

## 🔮 Future Enhancements

* Deep Learning Models (LSTM/BERT)
* More Indian Languages
* Speech Toxicity Detection
* Real-Time API Deployment
* Cloud Integration
* User Authentication

---

## 👨‍💻 Author

Mauli Narwade

### Project

BharatFilter AI – Multilingual Toxic Content Detection System

---

## 📜 License

This project is developed for educational, research, and content moderation purposes.

# 🛒 SmartCartAI — E-Commerce Recommendation Engine

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B)](https://ishaanagnihotri18-smartcartai-app.streamlit.app)

An end-to-end personalized product recommendation engine built with collaborative filtering, TF-IDF content similarity, and a hybrid scoring model — deployed on Streamlit.

## 🚀 Live Demo
👉 [Click here to open the app](https://ishaanagnihotri18-smartcartai-app.streamlit.app)

## ✨ Features
- Personalized recommendations using Hybrid, CF, and Content-Based algorithms
- Clickable product cards with full detail pages
- Amazon & Flipkart search integration
- Similar products finder
- Natural language product search
- Analytics dashboard with interactive charts

## 🛠 Tech Stack
Python · Pandas · NumPy · Scikit-learn · SQLite · Streamlit · Plotly

## 📊 Dataset
- 500 users · 200 products · 8,000 interactions
- 20 product categories
- Implicit feedback signals (view, click, cart, purchase, rating)

## 🧠 ML Models
- **Collaborative Filtering** — user-user & item-item cosine similarity
- **Content-Based** — TF-IDF on product descriptions & tags
- **Hybrid** — weighted blend of CF (60%) + content (40%)

## 🏃 Run Locally
```bash
git clone https://github.com/ishaanagnihotri18/SmartCartAI.git
cd SmartCartAI
pip install -r requirements.txt
streamlit run app.py
```
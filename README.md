<div align="center">
  <img src="https://img.icons8.com/?size=120&id=sM5wYQz8i2m5&format=png&color=00D2FF" alt="Brain AI Icon"/>
  <h1 align="center" style="color: #00d2ff;">AI Student Impact Analysis 🚀</h1>
  <p align="center">
    <strong>An End-to-End Machine Learning & Analytics Dashboard exploring the impact of Generative AI on Student Performance.</strong>
  </p>
  
  [![Python](https://img.shields.io/badge/Python-3.9+-00D2FF?style=for-the-badge&logo=python&logoColor=white)](#)
  [![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF007F?style=for-the-badge&logo=streamlit&logoColor=white)](#)
  [![XGBoost](https://img.shields.io/badge/XGBoost-Models-39FF14?style=for-the-badge&logo=xgboost&logoColor=white)](#)
</div>

<hr>

## 🌌 Overview
Welcome to the **AI Student Impact Analysis** project! This repository contains a fully modular, production-ready Machine Learning pipeline and a visually stunning **Cyberpunk/Glassmorphism-themed Streamlit Dashboard**.

The goal of this project is to analyze, visualize, and predict how the usage of Generative AI tools (like ChatGPT, Claude) affects college students' academic outcomes, including their **GPA**, **Skill Retention**, and **Burnout Risk**.

## ✨ Key Features
- 📊 **Interactive EDA:** Dynamic, semi-transparent glassmorphism tables and Plotly charts exploring univariate and bivariate distributions.
- 📈 **GPA & Skill Prediction:** Advanced `XGBoost` regression models predicting Post-Semester GPA based on AI dependency and study hours.
- 🚨 **Burnout Risk Classification:** A highly accurate classification engine determining if a student is at Low, Medium, or High risk of burnout.
- 🧠 **Student Segmentation (Clustering):** Unsupervised `K-Means` clustering grouped students into 8 unique personas (e.g., *The AI Pioneers*, *The Traditional Scholars*), visualized via PCA.
- 🎨 **World-Class UI/UX:** A bespoke dark-mode interface utilizing custom CSS, animate.css, and neon glowing borders to provide a premium user experience.

## 📁 Project Structure
```text
AI-Student-Impact/
├── data/                   # Raw and processed datasets
├── models/                 # Pickled preprocessors and XGBoost models
├── notebooks/              # Jupyter notebooks for initial EDA and training
├── pages/                  # Streamlit Multi-Page App (MPA) files
├── reports/                # Comprehensive project reports and documentation
├── src/                    # Core Python modules (ui, viz, models, data_loader)
├── assets/                 # CSS and styling files
├── app.py                  # Main Streamlit dashboard entry point
├── train_models.py         # Script to train and export all ML models
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start (Local Deployment)
To run this project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BassemRamdan/AI-Student-Impact.git
   cd AI-Student-Impact
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the models (Optional, pre-trained models are already included):**
   ```bash
   python train_models.py
   ```
4. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run app.py
   ```

## 🧪 Machine Learning Pipeline
- **Preprocessing:** Comprehensive pipeline handling skewed data (`np.log1p`), one-hot encoding for nominals, ordinal encoding for skills, and standard scaling.
- **Modeling:** 
  - `gpa_model`: XGBoost Regressor
  - `skill_model`: XGBoost Regressor
  - `burnout_model`: XGBoost Classifier
- **Evaluation:** Real-time feature importance charts and "Actual vs Predicted" scatter plots directly accessible within the dashboard's "Model Insights" page.

<hr>
<div align="center">
  <p>Built with ❤️ by Bassem Ramdan.</p>
</div>

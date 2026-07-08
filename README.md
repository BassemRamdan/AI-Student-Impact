# AI Student Impact - Streamlit App

This Streamlit application is designed to analyze and predict the impact of Generative AI on student outcomes based on a comprehensive dataset.

## Features
- **Exploratory Data Analysis (EDA)**: Interactive visualizations of the dataset.
- **Predict GPA & Skill Retention**: Use trained XGBoost regression models to predict academic outcomes based on AI usage habits.
- **Burnout Risk**: Use trained XGBoost classification models to predict a student's risk of burnout.
- **Clustering**: Segment students into distinct behavioral clusters using KMeans and PCA.

## Structure
- `app.py`: Main entry point.
- `pages/`: Contains the interactive multi-page structure.
- `src/`: Reusable Python modules for data loading, preprocessing, modeling, and visualization.
- `models/`: Pickled pre-trained models.
- `data/`: The dataset (`ai_student_impact_dataset.csv`).
- `notebooks/`: The original Jupyter Notebook analysis.

## Running Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

import pickle
import pandas as pd
import streamlit as st

@st.cache_resource
def load_all_models():
    """Load and cache the preprocessor and trained models."""
    with open("models/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("models/gpa_model.pkl", "rb") as f:
        gpa_model = pickle.load(f)
    with open("models/skill_model.pkl", "rb") as f:
        skill_model = pickle.load(f)
    with open("models/burnout_model.pkl", "rb") as f:
        burnout_model = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return preprocessor, gpa_model, skill_model, burnout_model, label_encoder

def predict_gpa_and_skill(input_data):
    """Predicts Post_Semester_GPA and Skill_Retention_Score based on input data (dict)."""
    preprocessor, gpa_model, skill_model, _, _ = load_all_models()
    df = pd.DataFrame([input_data])
    X_proc = preprocessor.transform(df)
    
    gpa_pred = gpa_model.predict(X_proc)[0]
    skill_pred = skill_model.predict(X_proc)[0]
    
    return gpa_pred, skill_pred

def predict_burnout(input_data):
    """Predicts Burnout Risk Level based on input data (dict)."""
    preprocessor, _, _, burnout_model, label_encoder = load_all_models()
    df = pd.DataFrame([input_data])
    X_proc = preprocessor.transform(df)
    
    pred_encoded = burnout_model.predict(X_proc)[0]
    pred_proba = burnout_model.predict_proba(X_proc)[0]
    
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    classes = label_encoder.classes_
    
    return pred_label, pred_proba, classes

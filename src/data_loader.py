import pandas as pd
import streamlit as st
import numpy as np

@st.cache_data
def load_data():
    """Load the dataset and return as a pandas DataFrame."""
    df = pd.read_csv("data/ai_student_impact_dataset.csv")
    return df

@st.cache_data
def engineer_features(df):
    """Apply feature engineering transformations from the notebook."""
    eng = df.copy()
    
    eng["GPA_Improvement"] = eng["Post_Semester_GPA"] - eng["Pre_Semester_GPA"]
    
    eng["Study_Efficiency"] = eng["Post_Semester_GPA"] / eng["Traditional_Study_Hours"].replace(0, np.nan)
    eng["Study_Efficiency"] = eng["Study_Efficiency"].fillna(eng["Study_Efficiency"].median())
    
    eng["Total_Study_Hours"] = eng["Weekly_GenAI_Hours"] + eng["Traditional_Study_Hours"]
    eng["AI_Usage_Ratio"] = eng["Weekly_GenAI_Hours"] / eng["Total_Study_Hours"].replace(0, np.nan)
    eng["AI_Usage_Ratio"] = eng["AI_Usage_Ratio"].fillna(0)
    
    eng["AI_vs_Traditional"] = np.log1p(eng["Weekly_GenAI_Hours"]) - np.log1p(eng["Traditional_Study_Hours"])
    
    eng["Dependency_per_Hour"] = eng["Perceived_AI_Dependency"] / (eng["Weekly_GenAI_Hours"] + 1)
    
    skill_map = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
    eng["Prompt_Skill_Numeric"] = eng["Prompt_Engineering_Skill"].map(skill_map)
    
    eng["Tool_Efficiency"] = eng["Skill_Retention_Score"] / eng["Tool_Diversity"]
    
    eng["Burnout_Index"] = (
        0.5 * eng["Perceived_AI_Dependency"] / 10 + 0.5 * eng["Anxiety_Level_During_Exams"] / 10
    ) * 100
    
    eng["Academic_Efficiency"] = eng["GPA_Improvement"] / eng["Total_Study_Hours"].replace(0, np.nan)
    eng["Academic_Efficiency"] = eng["Academic_Efficiency"].fillna(0)
    
    eng["Skill_per_Study_Hour"] = eng["Skill_Retention_Score"] / eng["Total_Study_Hours"].replace(0, np.nan)
    eng["Skill_per_Study_Hour"] = eng["Skill_per_Study_Hour"].fillna(eng["Skill_per_Study_Hour"].median())
    
    eng["Prompt_Weighted_AI_Hours"] = eng["Weekly_GenAI_Hours"] * eng["Prompt_Skill_Numeric"]
    
    return eng

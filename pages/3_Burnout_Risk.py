import streamlit as st
import sys
import os
import plotly.graph_objects as go

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import predict_burnout
from src.ui import load_css, page_header, top_navbar

st.set_page_config(page_title="Burnout Risk", layout="wide")
load_css()
top_navbar("Burnout Risk")

page_header("Burnout Risk Classification", "fa-solid fa-fire-flame-curved", "Predict a student's risk of burnout based on their usage patterns and academic characteristics.")

with st.form("burnout_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('#### <i class="fa-solid fa-hashtag" style="color: #2E5EAA;"></i> Numeric Features', unsafe_allow_html=True)
        pre_gpa = st.number_input("Pre-Semester GPA", min_value=0.0, max_value=4.0, value=3.0, step=0.1)
        genai_hours = st.number_input("Weekly GenAI Hours", min_value=0.0, max_value=40.0, value=15.0, step=1.0)
        trad_hours = st.number_input("Traditional Study Hours", min_value=0.0, max_value=40.0, value=5.0, step=1.0)
        tool_diversity = st.number_input("Tool Diversity (count)", min_value=1, max_value=10, value=5, step=1)
        ai_dependency = st.slider("Perceived AI Dependency (1-10)", 1, 10, 8)
        anxiety = st.slider("Anxiety Level During Exams (1-10)", 1, 10, 7)

    with col2:
        st.markdown('#### <i class="fa-solid fa-list" style="color: #E8743B;"></i> Categorical Features', unsafe_allow_html=True)
        major = st.selectbox("Major Category", ["STEM", "Humanities", "Business", "Social Sciences", "Arts"])
        year = st.selectbox("Year of Study", ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"])
        use_case = st.selectbox("Primary Use Case", ["Writing/Editing", "Coding", "Research", "Brainstorming", "Translation"])
        
    with col3:
        st.markdown('#### <i class="fa-solid fa-gavel" style="color: #3CA370;"></i> Policy & Skills', unsafe_allow_html=True)
        prompt_skill = st.selectbox("Prompt Engineering Skill", ["Beginner", "Intermediate", "Advanced"])
        policy = st.selectbox("Institutional Policy", ["Allowed", "Banned", "Unregulated"])
        paid = st.checkbox("Paid Subscription", value=False)
        
    submit = st.form_submit_button("Predict Burnout Risk")

if submit:
    input_data = {
        "Pre_Semester_GPA": pre_gpa,
        "Weekly_GenAI_Hours": genai_hours,
        "Tool_Diversity": tool_diversity,
        "Traditional_Study_Hours": trad_hours,
        "Perceived_AI_Dependency": ai_dependency,
        "Anxiety_Level_During_Exams": anxiety,
        "Major_Category": major,
        "Year_of_Study": year,
        "Primary_Use_Case": use_case,
        "Prompt_Engineering_Skill": prompt_skill,
        "Institutional_Policy": policy,
        "Paid_Subscription": paid
    }
    
    with st.spinner("Classifying risk profile..."):
        try:
            pred_label, pred_proba, classes = predict_burnout(input_data)
            
            if pred_label == "High":
                color = "#ff007f" # Neon Pink
                icon = "fa-triangle-exclamation"
                message = "CRITICAL: This student exhibits severe signs of academic burnout."
            elif pred_label == "Medium":
                color = "#ffea00" # Neon Yellow
                icon = "fa-bolt"
                message = "WARNING: Elevated risk of burnout detected. Monitor closely."
            else:
                color = "#39ff14" # Neon Green
                icon = "fa-shield-halved"
                message = "SAFE: This student is currently maintaining a healthy academic balance."

            st.markdown(f"""
            <div class="animate__animated animate__zoomIn" style="background: rgba(15, 32, 39, 0.6); padding: 40px; border-radius: 20px; text-align: center; border: 2px solid {color}; box-shadow: 0 0 30px {color}40; margin-top: 30px; backdrop-filter: blur(10px);">
                <i class="fa-solid {icon}" style="font-size: 4rem; color: {color}; margin-bottom: 20px; text-shadow: 0 0 15px {color};"></i>
                <h1 style="color: white; font-size: 3.5rem; margin: 0; text-transform: uppercase; letter-spacing: 3px;">{pred_label} RISK</h1>
                <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem; margin-top: 15px;">{message}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

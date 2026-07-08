import streamlit as st
import sys
import os

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import predict_gpa_and_skill
from src.ui import load_css, page_header, top_navbar

st.set_page_config(page_title="Predict GPA", layout="wide")
load_css()
top_navbar("Predict GPA")

page_header("Predict GPA & Skill Retention", "fa-solid fa-arrow-trend-up", "Enter student characteristics to predict their Post-Semester GPA and Skill Retention Score.")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('#### <i class="fa-solid fa-hashtag" style="color: #2E5EAA;"></i> Numeric Features', unsafe_allow_html=True)
        pre_gpa = st.number_input("Pre-Semester GPA", min_value=0.0, max_value=4.0, value=3.0, step=0.1)
        genai_hours = st.number_input("Weekly GenAI Hours", min_value=0.0, max_value=40.0, value=5.0, step=1.0)
        trad_hours = st.number_input("Traditional Study Hours", min_value=0.0, max_value=40.0, value=10.0, step=1.0)
        tool_diversity = st.number_input("Tool Diversity (count)", min_value=1, max_value=10, value=2, step=1)
        ai_dependency = st.slider("Perceived AI Dependency (1-10)", 1, 10, 5)
        anxiety = st.slider("Anxiety Level During Exams (1-10)", 1, 10, 5)

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
        
    submit = st.form_submit_button("Generate Prediction")

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
    
    with st.spinner("Analyzing profile and generating prediction..."):
        try:
            gpa_pred, skill_pred = predict_gpa_and_skill(input_data)
            st.markdown(f"""
            <div class="animate__animated animate__zoomIn" style="display: flex; gap: 20px; margin-top: 30px;">
                <div style="flex: 1; background: rgba(15, 32, 39, 0.6); padding: 30px; border-radius: 15px; text-align: center; border: 1px solid #00d2ff; box-shadow: 0 0 20px rgba(0, 210, 255, 0.3); backdrop-filter: blur(10px);">
                    <h4 style="color: rgba(255,255,255,0.7); margin-top:0; text-transform: uppercase; letter-spacing: 1px;">Predicted GPA</h4>
                    <h1 style="color: #00d2ff; font-size: 4rem; margin: 10px 0; text-shadow: 0 0 15px #00d2ff;">{gpa_pred:.2f}</h1>
                    <p style="color: #39ff14; font-size: 1.1rem; margin: 0;"><i class="fa-solid fa-arrow-trend-up"></i> Projected Outcome</p>
                </div>
                <div style="flex: 1; background: rgba(15, 32, 39, 0.6); padding: 30px; border-radius: 15px; text-align: center; border: 1px solid #9d00ff; box-shadow: 0 0 20px rgba(157, 0, 255, 0.3); backdrop-filter: blur(10px);">
                    <h4 style="color: rgba(255,255,255,0.7); margin-top:0; text-transform: uppercase; letter-spacing: 1px;">Skill Retention Score</h4>
                    <h1 style="color: #9d00ff; font-size: 4rem; margin: 10px 0; text-shadow: 0 0 15px #9d00ff;">{skill_pred:.1f}</h1>
                    <p style="color: #ffea00; font-size: 1.1rem; margin: 0;"><i class="fa-solid fa-brain"></i> Out of 100</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error making prediction: {e}")

import streamlit as st
import sys
import os

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data
from src.viz import (
    plot_feature_dist, plot_scatter_bubble, plot_violin, 
    plot_kde_overlay, plot_density_heatmap, plot_policy_summary, plot_correlation, plot_3d_scatter
)
from src.ui import load_css, page_header, top_navbar

st.set_page_config(page_title="EDA", layout="wide")
load_css()
top_navbar("EDA")

page_header("Exploratory Data Analysis", "fa-solid fa-chart-pie", "Interactive univariate and bivariate analysis of the dataset.")

df = load_data()

st.markdown("""
<div class="animate__animated animate__fadeInUp" style="background: rgba(15, 32, 39, 0.4); padding: 15px; border-radius: 12px; margin-bottom: 20px; border: 1px solid rgba(0, 210, 255, 0.2); backdrop-filter: blur(10px);">
    <h4 style="margin-top:0; color: #00d2ff;"><i class="fa-solid fa-database"></i> Dataset Overview</h4>
    <p style="margin-bottom:0; color: white;">The dataset contains <strong style="color: #39ff14;">{rows:,}</strong> rows and <strong style="color: #39ff14;">{cols}</strong> columns.</p>
</div>
""".format(rows=df.shape[0], cols=df.shape[1]), unsafe_allow_html=True)

st.markdown('<div class="glass-table-container animate__animated animate__fadeInUp animate__delay-1s">', unsafe_allow_html=True)
st.markdown(df.head().to_html(classes="glass-table", index=False), unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr style='opacity: 0.2;'>", unsafe_allow_html=True)
st.markdown("### <i class='fa-solid fa-chart-bar' style='color:#2E5EAA;'></i> Univariate Distributions", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    feature = st.selectbox("Select a numerical feature:", 
                           ["Pre_Semester_GPA", "Weekly_GenAI_Hours", "Traditional_Study_Hours", "Post_Semester_GPA"])
    fig1 = plot_feature_dist(df, feature)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    cat_feature = st.selectbox("Select a categorical feature:", 
                               ["Major_Category", "Year_of_Study", "Primary_Use_Case", "Burnout_Risk_Level"])
    fig2 = plot_feature_dist(df, cat_feature)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("<hr style='opacity: 0.2;'>", unsafe_allow_html=True)
st.markdown("### <i class='fa-solid fa-layer-group' style='color:#E8743B;'></i> Advanced Distributions", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Pre vs Post GPA (KDE)")
    fig_kde = plot_kde_overlay(df, "Pre_Semester_GPA", "Post_Semester_GPA")
    st.plotly_chart(fig_kde, use_container_width=True)

with c2:
    st.markdown("#### GPA by Prompting Skill")
    fig_violin = plot_violin(df, y_col="Post_Semester_GPA", x_col="Prompt_Engineering_Skill")
    st.plotly_chart(fig_violin, use_container_width=True)

st.markdown("<hr style='opacity: 0.2;'>", unsafe_allow_html=True)
st.markdown("### <i class='fa-solid fa-network-wired' style='color:#3CA370;'></i> Bivariate Relationships", unsafe_allow_html=True)

c3, c4 = st.columns(2)
with c3:
    st.markdown("#### GenAI Hours vs. Skill Retention")
    fig_scatter = plot_scatter_bubble(df, "Weekly_GenAI_Hours", "Skill_Retention_Score", "Burnout_Risk_Level", "Tool_Diversity")
    st.plotly_chart(fig_scatter, use_container_width=True)

with c4:
    st.markdown("#### Study Hours vs. Post-Semester GPA")
    fig_density = plot_density_heatmap(df, "Traditional_Study_Hours", "Post_Semester_GPA")
    st.plotly_chart(fig_density, use_container_width=True)

st.markdown("<hr style='opacity: 0.2;'>", unsafe_allow_html=True)
st.markdown("### <i class='fa-solid fa-building-columns' style='color:#C44E52;'></i> Policy Impact & Correlations", unsafe_allow_html=True)

c5, c6 = st.columns(2)
with c5:
    fig_policy = plot_policy_summary(df)
    st.plotly_chart(fig_policy, use_container_width=True)
    
with c6:
    st.markdown("#### Feature Correlation Matrix")
    corr_cols = ["Pre_Semester_GPA", "Post_Semester_GPA", "Weekly_GenAI_Hours", "Traditional_Study_Hours", "Skill_Retention_Score", "Perceived_AI_Dependency"]
    fig_corr = plot_correlation(df, corr_cols)
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("<hr style='opacity: 0.2;'>", unsafe_allow_html=True)
st.markdown("### <i class='fa-solid fa-cube' style='color:#00d2ff;'></i> Multidimensional Burnout Analysis (3D)", unsafe_allow_html=True)
st.markdown("<p style='color: rgba(255,255,255,0.7);'>This interactive 3D chart illustrates how AI Dependency, Anxiety Level, and GenAI Hours jointly impact Burnout Risk. (Sampled to 1,500 points for smooth performance)</p>", unsafe_allow_html=True)
fig_3d = plot_3d_scatter(df, "Perceived_AI_Dependency", "Anxiety_Level_During_Exams", "Weekly_GenAI_Hours", "Burnout_Risk_Level")
st.plotly_chart(fig_3d, use_container_width=True)

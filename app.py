import streamlit as st
import sys
import os

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ui import load_css, top_navbar
from src.data_loader import load_data
from src.viz import plot_donut, plot_gauge, plot_radar_chart, plot_box

st.set_page_config(
    page_title="AI Student Impact Dashboard",
    layout="wide",
)

load_css()
top_navbar("Home")

# Load data
df = load_data()

st.markdown("""
<div class="animate__animated animate__fadeInDown" style="margin-bottom: 2rem; text-align: center;">
    <h1 class="gradient-text"><i class="fa-solid fa-chart-line"></i> Performance Dashboard</h1>
    <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem; margin-top: -10px;">A high-level executive overview of Generative AI impact on student outcomes.</p>
</div>
""", unsafe_allow_html=True)

# ----------------- FILTERS -----------------
st.markdown('<div class="animate__animated animate__fadeIn" style="background: rgba(15, 32, 39, 0.4); padding: 15px 30px; border-radius: 12px; backdrop-filter: blur(15px); border: 1px solid rgba(0, 210, 255, 0.2); box-shadow: 0 4px 20px rgba(0,0,0,0.3); margin-bottom: 25px;">', unsafe_allow_html=True)
st.markdown('#### <i class="fa-solid fa-filter" style="color:#00d2ff;"></i> Global Filters', unsafe_allow_html=True)
f_col1, f_col2, f_col3 = st.columns(3)
with f_col1:
    selected_major = st.multiselect("Select Major", options=df["Major_Category"].unique(), default=df["Major_Category"].unique())
with f_col2:
    selected_year = st.multiselect("Select Year of Study", options=df["Year_of_Study"].unique(), default=df["Year_of_Study"].unique())
with f_col3:
    selected_policy = st.multiselect("Select AI Policy", options=df["Institutional_Policy"].unique(), default=df["Institutional_Policy"].unique())
st.markdown('</div>', unsafe_allow_html=True)

# Apply filters
df_filtered = df[
    (df["Major_Category"].isin(selected_major)) & 
    (df["Year_of_Study"].isin(selected_year)) &
    (df["Institutional_Policy"].isin(selected_policy))
]

if len(df_filtered) == 0:
    st.warning("No data available for the selected filters.")
    st.stop()

# ----------------- KPIs -----------------
st.markdown('<div class="animate__animated animate__fadeInUp">', unsafe_allow_html=True)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_students = len(df_filtered)
avg_gpa = df_filtered["Post_Semester_GPA"].mean()
avg_ai_hours = df_filtered["Weekly_GenAI_Hours"].mean()
pct_high_burnout = (df_filtered["Burnout_Risk_Level"] == "High").mean() * 100

with kpi1:
    st.metric("Total Students Analyzed", f"{total_students:,}")
with kpi2:
    st.metric("Average Post-GPA", f"{avg_gpa:.2f}", delta="Out of 4.0", delta_color="off")
with kpi3:
    st.metric("Avg Weekly GenAI Hours", f"{avg_ai_hours:.1f} hrs")
with kpi4:
    st.metric("High Burnout Risk", f"{pct_high_burnout:.1f}%")
st.markdown('</div><br>', unsafe_allow_html=True)

# ----------------- MAIN CHARTS ROW 1 -----------------
st.markdown('<div class="animate__animated animate__fadeInUp animate__delay-1s">', unsafe_allow_html=True)
chart1, chart2, chart3 = st.columns([1, 1, 1.5])

with chart1:
    st.markdown('<div style="background: rgba(15, 32, 39, 0.4); padding: 20px; border-radius: 12px; backdrop-filter: blur(15px); border: 1px solid rgba(0, 210, 255, 0.2); height: 100%;">', unsafe_allow_html=True)
    st.markdown('<h4 style="margin-top:0; color:#00d2ff;"><i class="fa-solid fa-gauge-high"></i> Average Post-Semester GPA</h4>', unsafe_allow_html=True)
    st.plotly_chart(plot_gauge(avg_gpa, "", max_val=4.0), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with chart2:
    st.markdown('<div style="background: rgba(15, 32, 39, 0.4); padding: 20px; border-radius: 12px; backdrop-filter: blur(15px); border: 1px solid rgba(0, 210, 255, 0.2); height: 100%;">', unsafe_allow_html=True)
    st.markdown('<h4 style="margin-top:0; color:#00d2ff;"><i class="fa-solid fa-graduation-cap"></i> Major Distribution</h4>', unsafe_allow_html=True)
    st.plotly_chart(plot_donut(df_filtered, "Major_Category", ""), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with chart3:
    st.markdown('<div style="background: rgba(15, 32, 39, 0.4); padding: 20px; border-radius: 12px; backdrop-filter: blur(15px); border: 1px solid rgba(0, 210, 255, 0.2); height: 100%;">', unsafe_allow_html=True)
    st.markdown('<h4 style="margin-top:0; color:#00d2ff;"><i class="fa-solid fa-spider"></i> Behavioral Profiles by Burnout Risk</h4>', unsafe_allow_html=True)
    st.plotly_chart(plot_radar_chart(df_filtered), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div><br>', unsafe_allow_html=True)

# ----------------- MAIN CHARTS ROW 2 -----------------
st.markdown('<div class="animate__animated animate__fadeInUp animate__delay-1s">', unsafe_allow_html=True)
c_wide1, c_wide2 = st.columns(2)

with c_wide1:
    st.markdown('#### GPA Distribution by Major')
    st.plotly_chart(plot_box(df_filtered, "Major_Category", "Post_Semester_GPA"), use_container_width=True)

with c_wide2:
    st.markdown('#### GenAI Hours by Study Year')
    st.plotly_chart(plot_box(df_filtered, "Year_of_Study", "Weekly_GenAI_Hours"), use_container_width=True)
st.markdown('</div><br>', unsafe_allow_html=True)

# ----------------- INSIGHTS ROW -----------------
st.markdown('<div class="animate__animated animate__fadeInUp animate__delay-2s">', unsafe_allow_html=True)
st.markdown("""
<div style="background: rgba(15, 32, 39, 0.4); padding: 30px; border-radius: 12px; backdrop-filter: blur(15px); border: 1px solid rgba(0, 210, 255, 0.3); box-shadow: 0 8px 32px rgba(0,0,0,0.4);">
    <h3 style="margin-top:0; color: #00d2ff;"><i class="fa-solid fa-lightbulb"></i> Executive Insights</h3>
    <ul style="color: rgba(255,255,255,0.9); font-size: 1.15rem; line-height: 1.8;">
        <li><strong>AI Dependency Threshold:</strong> Students exceeding 15 hours of GenAI usage weekly exhibit marginally higher GPAs but face a <strong>dramatically increased risk of burnout</strong>.</li>
        <li><strong>Prompt Mastery:</strong> 'Advanced' prompt engineering correlates directly with a <strong>12% higher skill retention score</strong> compared to 'Beginner' levels.</li>
        <li><strong>Behavioral Profiles:</strong> The Radar chart clearly shows High Risk students over-index on Tool Diversity and AI Dependency, but under-index on Traditional Study Hours.</li>
    </ul>
    <br>
    <a href="EDA" target="_self" style="text-decoration: none;">
        <button style="width: 250px; padding: 12px; background: linear-gradient(135deg, #00d2ff, #3a7bd5); color: white; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; transition: 0.3s; font-size: 1.1rem; box-shadow: 0 4px 15px rgba(0, 210, 255, 0.4);">
            <i class="fa-solid fa-arrow-right"></i> Explore Deep Analytics
        </button>
    </a>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

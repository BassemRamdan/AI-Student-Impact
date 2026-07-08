import streamlit as st
import sys
import os

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data, engineer_features
from src.clustering import get_clusters
from src.viz import plot_pca_clusters
from src.ui import load_css, page_header, top_navbar

st.set_page_config(page_title="Clustering", layout="wide")
load_css()
top_navbar("Clustering")

page_header("Student Segmentation", "fa-solid fa-users-viewfinder", "Segment students into distinct behavioral groups using KMeans and visualize them with PCA.")

with st.spinner("Loading data and generating clusters..."):
    df_raw = load_data()
    df_eng = engineer_features(df_raw)
    
    st.markdown('<div class="animate__animated animate__fadeIn" style="background: rgba(15, 32, 39, 0.4); padding: 15px; border-radius: 12px; margin-bottom: 20px; border: 1px solid rgba(0, 210, 255, 0.2); backdrop-filter: blur(10px);">', unsafe_allow_html=True)
    n_clusters = st.slider("Number of Clusters (k)", min_value=2, max_value=8, value=4)
    st.markdown('</div>', unsafe_allow_html=True)
    
    df_clustered, cluster_features = get_clusters(df_eng, n_clusters=n_clusters)
    
    # Assign names to clusters (up to 8)
    CLUSTER_NAMES = [
        "The AI Pioneers", "The Traditional Scholars", "The Balanced Achievers", 
        "The Stressed Adopters", "The Casual Users", "The Tech Enthusiasts", 
        "The Cautious Learners", "The Hyper-dependent"
    ]
    df_clustered["Cluster Name"] = df_clustered["Cluster"].apply(lambda x: CLUSTER_NAMES[x])
    
st.markdown("### <i class='fa-solid fa-project-diagram' style='color:#2E5EAA;'></i> PCA Projection", unsafe_allow_html=True)
fig = plot_pca_clusters(df_clustered)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Cluster Assignments Preview")
st.markdown('<div class="glass-table-container animate__animated animate__fadeInUp">', unsafe_allow_html=True)
st.markdown(df_clustered.head().to_html(classes="glass-table", index=False), unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr style='opacity: 0.2;'>", unsafe_allow_html=True)
st.markdown("### Cluster Profiles")
st.markdown("Average values of key features per cluster.")
profile = df_clustered.groupby("Cluster Name")[cluster_features].mean().round(2)
profile["Student Count"] = df_clustered["Cluster Name"].value_counts()
# Reset index so 'Cluster Name' is a column for HTML table
profile_reset = profile.reset_index()
st.markdown('<div class="glass-table-container animate__animated animate__fadeInUp">', unsafe_allow_html=True)
st.markdown(profile_reset.to_html(classes="glass-table", index=False), unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

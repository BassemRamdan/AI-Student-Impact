import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

@st.cache_data
def get_clusters(df, n_clusters=4):
    """Cluster students using KMeans and reduce dimensionality via PCA."""
    cluster_features = [
        "Pre_Semester_GPA", "Weekly_GenAI_Hours", "Tool_Diversity",
        "Traditional_Study_Hours", "Perceived_AI_Dependency", "Anxiety_Level_During_Exams",
        "Post_Semester_GPA", "Skill_Retention_Score", "Prompt_Skill_Numeric"
    ]
    
    # We rely on engineer_features being called on df first
    cluster_source = df[cluster_features].copy()
    
    cluster_scaler = StandardScaler()
    X_cluster = cluster_scaler.fit_transform(cluster_source)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster)
    
    pca = PCA(n_components=2, random_state=42)
    proj_2d = pca.fit_transform(X_cluster)
    
    df_clustered = df.copy()
    df_clustered["Cluster"] = clusters
    df_clustered["PCA1"] = proj_2d[:, 0]
    df_clustered["PCA2"] = proj_2d[:, 1]
    
    return df_clustered, cluster_features

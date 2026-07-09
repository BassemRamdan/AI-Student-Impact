import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import pandas as pd

# Neon / Cyberpunk Palette for dark themes
PALETTE = ["#00d2ff", "#ff007f", "#39ff14", "#ffea00", "#9d00ff", "#ff6600"]

def get_layout_kwargs():
    return dict(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=40, b=20)
    )

def plot_feature_dist(df, feature, color_by=None):
    if color_by:
        fig = px.histogram(df, x=feature, color=color_by, barmode="overlay", color_discrete_sequence=PALETTE)
    else:
        # If categorical, color by the feature itself for a vibrant look. If numerical, use a distinct color.
        if df[feature].dtype == 'object' or df[feature].nunique() < 10:
            fig = px.histogram(df, x=feature, color=feature, color_discrete_sequence=PALETTE)
        else:
            fig = px.histogram(df, x=feature, color_discrete_sequence=[PALETTE[1]]) # Neon pink for numericals
    fig.update_layout(showlegend=False, **get_layout_kwargs())
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)')
    return fig

def plot_correlation(df, features):
    corr = df[features].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="Viridis", zmin=-1, zmax=1)
    fig.update_layout(**get_layout_kwargs())
    return fig

def plot_scatter_bubble(df, x_col, y_col, color_col, size_col):
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col, size=size_col, opacity=0.7,
        color_discrete_sequence=[PALETTE[2], PALETTE[3], PALETTE[1]],
        category_orders={"Burnout_Risk_Level": ["Low", "Medium", "High"]}
    )
    fig.update_layout(height=500, **get_layout_kwargs())
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    return fig

def plot_pca_clusters(df):
    fig = px.scatter(
        df, x="PCA1", y="PCA2", color=df["Cluster"].astype(str),
        opacity=0.8, color_discrete_sequence=PALETTE,
        title="KMeans Clusters Projected onto First 2 Principal Components"
    )
    fig.update_layout(height=500, **get_layout_kwargs())
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    return fig

def plot_violin(df, y_col, x_col=None, color_sequence=PALETTE):
    if x_col:
        fig = px.violin(df, y=y_col, x=x_col, box=True, color=x_col, color_discrete_sequence=color_sequence)
    else:
        fig = px.violin(df, y=y_col, box=True, color_discrete_sequence=[color_sequence[0]])
    fig.update_layout(**get_layout_kwargs())
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    return fig

def plot_kde_overlay(df, col1, col2, name1="Pre-GPA", name2="Post-GPA"):
    fig = ff.create_distplot(
        [df[col1].dropna(), df[col2].dropna()],
        [name1, name2],
        show_hist=False, show_rug=False,
        colors=[PALETTE[1], PALETTE[0]]
    )
    fig.update_layout(title_text="KDE Overlay", **get_layout_kwargs())
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    return fig

def plot_density_heatmap(df, x_col, y_col):
    fig = px.density_heatmap(df, x=x_col, y=y_col, nbinsx=30, nbinsy=30, color_continuous_scale="electric")
    fig.update_layout(**get_layout_kwargs())
    return fig

def plot_policy_summary(df):
    policy_summary = df.groupby("Institutional_Policy")[["Post_Semester_GPA", "Skill_Retention_Score"]].mean().reset_index()
    fig = px.bar(
        policy_summary.melt(id_vars="Institutional_Policy", var_name="Metric", value_name="Average"),
        x="Institutional_Policy", y="Average", color="Metric", barmode="group",
        color_discrete_sequence=[PALETTE[0], PALETTE[2]],
        title="Average Outcomes by Institutional AI Policy"
    )
    fig.update_layout(height=400, **get_layout_kwargs())
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    return fig

def plot_donut(df, col, title):
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, 'Count']
    fig = px.pie(counts, values='Count', names=col, hole=0.6, title=title, color_discrete_sequence=PALETTE)
    fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#0f2027', width=2)))
    fig.update_layout(height=350, showlegend=False, **get_layout_kwargs())
    return fig

def plot_gauge(value, title, max_val=4.0):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'color': "white"}},
        number = {'font': {'color': "#00d2ff"}},
        gauge = {
            'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00d2ff"},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 1,
            'bordercolor': "rgba(0, 210, 255, 0.2)",
            'steps': [
                {'range': [0, max_val*0.5], 'color': "rgba(255, 0, 127, 0.3)"},
                {'range': [max_val*0.5, max_val*0.8], 'color': "rgba(255, 234, 0, 0.3)"},
                {'range': [max_val*0.8, max_val], 'color': "rgba(57, 255, 20, 0.3)"}
            ]
        }
    ))
    fig.update_layout(height=300, **get_layout_kwargs())
    return fig

# --- NEW ADVANCED VISUALIZATIONS ---

def plot_radar_chart(df):
    """Compares average numerical features across Burnout Risk Levels."""
    cols_to_compare = ["Weekly_GenAI_Hours", "Traditional_Study_Hours", "Tool_Diversity", "Perceived_AI_Dependency", "Anxiety_Level_During_Exams"]
    
    # Normalize data for radar chart (0-1 scale)
    df_norm = df.copy()
    for col in cols_to_compare:
        df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
    avg_profiles = df_norm.groupby("Burnout_Risk_Level")[cols_to_compare].mean().reset_index()
    
    fig = go.Figure()
    colors = {"Low": "#39ff14", "Medium": "#ffea00", "High": "#ff007f"}
    
    for level in ["Low", "Medium", "High"]:
        values = avg_profiles[avg_profiles["Burnout_Risk_Level"] == level][cols_to_compare].values.flatten().tolist()
        values.append(values[0]) # Close the polygon
        categories = cols_to_compare + [cols_to_compare[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=f'{level} Risk',
            line_color=colors[level],
            opacity=0.6
        ))
        
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")
        ),
        showlegend=True,
        **get_layout_kwargs()
    )
    return fig

def plot_box(df, x_col, y_col):
    fig = px.box(df, x=x_col, y=y_col, color=x_col, color_discrete_sequence=PALETTE)
    fig.update_layout(**get_layout_kwargs())
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    return fig

def plot_feature_importance(importance_dict, title):
    # Sort importances
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1])
    features = [k for k, v in sorted_items]
    scores = [v for k, v in sorted_items]
    
    fig = px.bar(x=scores, y=features, orientation='h', title=title)
    fig.update_traces(marker_color=PALETTE[0], opacity=0.8)
    fig.update_layout(xaxis_title="Importance Score", yaxis_title="Features", height=500, **get_layout_kwargs())
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    return fig

def plot_actual_vs_predicted(y_true, y_pred, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode='markers',
        marker=dict(color=PALETTE[1], size=8, opacity=0.7),
        name="Predictions"
    ))
    
    # Diagonal line for perfect prediction
    min_val, max_val = min(y_true), max(y_true)
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', line=dict(color='white', dash='dash'),
        name="Ideal"
    ))
    
    fig.update_layout(
        title=title, xaxis_title="Actual Values", yaxis_title="Predicted Values",
        height=500, **get_layout_kwargs()
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    return fig

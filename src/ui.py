import streamlit as st
import os

_CSS_CACHE = None

def get_css():
    global _CSS_CACHE
    if _CSS_CACHE is None:
        css_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "style.css")
        if os.path.exists(css_path):
            with open(css_path, "r") as f:
                _CSS_CACHE = f.read()
        else:
            _CSS_CACHE = ""
    return _CSS_CACHE

def load_css():
    """Injects custom CSS and external libraries (FontAwesome, Animate.css) into the Streamlit app."""
    # Load external libraries
    st.markdown("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"/>
    """, unsafe_allow_html=True)
    
    # Load custom style.css
    css_content = get_css()
    if css_content:
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

def top_navbar(active_name):
    """Renders a custom fixed top navigation bar replacing the sidebar."""
    nav_items = [
        {"name": "Home", "icon": "fa-solid fa-home", "url": "."},
        {"name": "EDA", "icon": "fa-solid fa-chart-pie", "url": "EDA"},
        {"name": "Predict GPA", "icon": "fa-solid fa-arrow-trend-up", "url": "Predict_GPA"},
        {"name": "Burnout Risk", "icon": "fa-solid fa-fire-flame-curved", "url": "Burnout_Risk"},
        {"name": "Clustering", "icon": "fa-solid fa-users-viewfinder", "url": "Clustering"},
        {"name": "Model Insights", "icon": "fa-solid fa-brain", "url": "Model_Insights"}
    ]
    
    nav_html = '<div class="custom-navbar">'
    for item in nav_items:
        active_class = 'class="active"' if item['name'] == active_name else ''
        nav_html += f'<a href="{item["url"]}" target="_self" {active_class}><i class="{item["icon"]}"></i> {item["name"]}</a>'
    nav_html += '</div>'
    
    st.markdown(nav_html, unsafe_allow_html=True)

def page_header(title, icon_class, subtitle=None):
    """Renders a styled page header with a FontAwesome icon and a gradient title."""
    html = f"""
    <div class="animate__animated animate__fadeInDown" style="margin-bottom: 2rem;">
        <h1 class="gradient-text"><i class="{icon_class}"></i> {title}</h1>
    """
    if subtitle:
        html += f'<p style="color: rgba(255,255,255,0.7); font-size: 1.1rem; margin-top: -10px;">{subtitle}</p>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

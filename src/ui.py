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
    """Renders an instant SPA navigation bar using Streamlit's native page_link."""
    pages = [
        {"name": "Home", "icon": "🏠", "url": "app.py"},
        {"name": "EDA", "icon": "📊", "url": "pages/1_EDA.py"},
        {"name": "Predict GPA", "icon": "📈", "url": "pages/2_Predict_GPA.py"},
        {"name": "Burnout Risk", "icon": "🔥", "url": "pages/3_Burnout_Risk.py"},
        {"name": "Clustering", "icon": "👥", "url": "pages/4_Clustering.py"},
        {"name": "Model Insights", "icon": "🧠", "url": "pages/5_Model_Insights.py"}
    ]
    
    st.markdown("""
        <style>
            /* Hide the default Streamlit sidebar */
            [data-testid="collapsedControl"] { display: none !important; }
            section[data-testid="stSidebar"] { display: none !important; }
            
            /* Style all st.page_link elements globally (acts as our navbar links) */
            [data-testid="stPageLink-NavLink"] {
                background: rgba(15, 32, 39, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 8px;
                padding: 10px !important;
                display: flex;
                justify-content: center;
                align-items: center;
                color: #ffffff;
                font-family: 'Outfit', sans-serif;
                font-weight: 500;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            }
            [data-testid="stPageLink-NavLink"]:hover {
                background: rgba(0, 210, 255, 0.1);
                color: #00d2ff;
                border-color: #00d2ff;
                transform: translateY(-2px);
            }
            /* Highlight active link if needed */
            [data-testid="stPageLink-NavLink"][data-active="true"] {
                background: rgba(0, 210, 255, 0.2);
                color: #00d2ff;
                border-color: #00d2ff;
                box-shadow: 0 0 15px rgba(0, 210, 255, 0.4);
            }
        </style>
    """, unsafe_allow_html=True)
    
    cols = st.columns(len(pages))
    for i, item in enumerate(pages):
        with cols[i]:
            st.page_link(item["url"], label=item["name"], icon=item["icon"])
    
    # Add a visual separator
    st.markdown("<hr style='border-color: rgba(0, 210, 255, 0.3); margin-top: 5px; margin-bottom: 25px;'>", unsafe_allow_html=True)

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

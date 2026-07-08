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
            header[data-testid="stHeader"] { background: rgba(0,0,0,0) !important; display: none !important; }
            
            /* Target the FIRST st.columns on the page (which is our navbar) and make it the fixed header */
            [data-testid="stHorizontalBlock"]:first-of-type {
                position: fixed !important;
                top: 0 !important;
                left: 0 !important;
                width: 100% !important;
                background: rgba(15, 32, 39, 0.6) !important;
                backdrop-filter: blur(15px) !important;
                -webkit-backdrop-filter: blur(15px) !important;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
                z-index: 999999 !important;
                display: flex !important;
                justify-content: center !important;
                align-items: center !important;
                padding: 15px 0 !important;
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3) !important;
                gap: 0 !important;
            }
            
            /* Prevent columns from stretching, make them fit content tightly */
            [data-testid="stHorizontalBlock"]:first-of-type > div[data-testid="column"] {
                flex: 0 1 auto !important;
                width: auto !important;
                min-width: auto !important;
            }
            
            /* Style the page_link to look EXACTLY like the old <a> tags */
            [data-testid="stPageLink-NavLink"] {
                background: transparent !important;
                border: none !important;
                color: #ffffff !important;
                font-family: 'Outfit', sans-serif !important;
                font-size: 1.1rem !important;
                font-weight: 500 !important;
                margin: 0 20px !important;
                padding: 8px 15px !important;
                border-radius: 8px !important;
                transition: all 0.3s ease !important;
                display: flex !important;
                align-items: center !important;
                gap: 8px !important;
            }
            
            [data-testid="stPageLink-NavLink"]:hover {
                background: rgba(255, 255, 255, 0.1) !important;
                color: #00d2ff !important;
                transform: translateY(-2px) !important;
            }
            
            /* Highlight active link */
            [data-testid="stPageLink-NavLink"][data-active="true"] {
                background: rgba(255, 255, 255, 0.15) !important;
                color: #00d2ff !important;
                border-bottom: 2px solid #00d2ff !important;
                border-bottom-left-radius: 0 !important;
                border-bottom-right-radius: 0 !important;
            }
            
            /* Push main content down to clear fixed navbar */
            .block-container {
                padding-top: 100px !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    cols = st.columns(len(pages))
    for i, item in enumerate(pages):
        with cols[i]:
            st.page_link(item["url"], label=item["name"], icon=item["icon"])

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

import streamlit as st

st.set_page_config(
    page_title="Crypto Forecasting | Team 17",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0a0a0f; }
    [data-testid="stSidebar"] { background-color: #12121a; border-right: 1px solid #2a2a3a; }
    h1 { color: #ffffff !important; font-weight: 600; font-size: 2rem !important; 
         border-bottom: 1px solid #2a2a3a; padding-bottom: 0.75rem; }
    h2 { color: #e0e0e0 !important; font-weight: 500; font-size: 1.4rem !important; }
    h3 { color: #a0a0b0 !important; font-size: 1.1rem !important; }
    p, li, .stMarkdown { color: #c0c0d0; line-height: 1.65; }
    [data-testid="stMetricValue"] { color: #4da6ff !important; font-size: 1.6rem !important; }
    [data-testid="stMetricLabel"] { color: #808090 !important; font-size: 0.8rem; text-transform: uppercase; }
    .stSelectbox label { color: #808090 !important; font-size: 0.8rem; text-transform: uppercase; }
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #2a2a3a; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; color: #808090; }
    .stTabs [aria-selected="true"] { color: #ffffff; border-bottom: 2px solid #4da6ff; }
    hr { border-color: #2a2a3a; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page",
    ["Project Overview", "Market Analysis", "Forecast Tool", "Final Presentation"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Team 17**  
STA 160 | UC Davis  
Fall 2025
""")

if page == "Project Overview":
    from views import overview
    overview.render()
elif page == "Market Analysis":
    from views import analysis
    analysis.render()
elif page == "Forecast Tool":
    from views import forecast
    forecast.render()
elif page == "Final Presentation":
    st.title("Final Presentation")
    st.markdown("---")
    
    # Embed YouTube video
    st.markdown("""
        <div style="display: flex; justify-content: center; margin-top: 2rem;">
            <iframe 
                width="800" 
                height="450" 
                src="https://www.youtube.com/embed/Zlj03oBP26g" 
                title="YouTube video player" 
                frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
                allowfullscreen>
            </iframe>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### About This Presentation")
    st.markdown("""
    This video presentation covers our team's approach to cryptocurrency price forecasting, 
    including our methodology, results, and key insights from the project.
    """)

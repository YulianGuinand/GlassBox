import streamlit as st

def load_custom_css():
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Outfit', sans-serif;
        }

        /* Glassmorphism Card Style */
        .glass-card {
            background: linear-gradient(135deg, rgba(30, 30, 40, 0.7) 0%, rgba(20, 20, 30, 0.9) 100%);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(112, 0, 255, 0.2);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            border: 1px solid rgba(0, 240, 255, 0.5);
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.1);
        }

        /* Gradient Text for Headers */
        h1, h2, h3 {
            background: linear-gradient(90deg, #FFFFFF 0%, #E0E0E0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-bottom: 1rem;
        }
        
        /* Highlight specific words */
        .highlight {
            background: linear-gradient(90deg, #00F0FF 0%, #BD00FF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Custom Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background-color: transparent;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 0 20px;
            border: 1px solid transparent;
            color: #CCCCCC;
            font-weight: 500;
            transition: all 0.2s;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(112, 0, 255, 0.2) !important;
            color: #FFFFFF !important;
            border: 1px solid #7000FF !important;
            box-shadow: 0 0 10px rgba(112, 0, 255, 0.3);
        }

        /* Input Fields Modernization */
        .stTextInput, .stNumberInput, .stSelectbox, .stDateInput {
            border-radius: 8px;
        }
        
        </style>
    """, unsafe_allow_html=True)

def card_begin():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

def card_end():
    st.markdown('</div>', unsafe_allow_html=True)

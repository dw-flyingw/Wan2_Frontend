#!/usr/bin/env python3
"""
Wan2.2 Studio - Multi-model video generation frontend.

Supports:
- T2V-A14B: Text to Video
- I2V-A14B: Image to Video
- S2V-14B: Speech to Video
- Animate-14B: Character Animation

Usage:
    streamlit run app.py
"""

import streamlit as st

from utils.config import init_session_state

# Page configuration
st.set_page_config(
    page_title="Wan2.2 Studio",
    page_icon="🎬",
    layout="wide",
)

# Initialize session state
init_session_state()

# Define pages
t2v_page = st.Page("pages/t2v_a14b.py", title="Text to Video", icon="📝")
i2v_page = st.Page("pages/i2v_a14b.py", title="Image to Video", icon="🖼️")
s2v_page = st.Page("pages/s2v_14b.py", title="Speech to Video", icon="🎤")
animate_page = st.Page("pages/animate_14b.py", title="Animate", icon="🏃")

# Navigation
pg = st.navigation(
    [t2v_page, i2v_page, s2v_page, animate_page],
    position="sidebar",
)

# Run the selected page
pg.run()

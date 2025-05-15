#!/usr/bin/env python3
"""
USAF 1951 Resolution Target Analyzer - Streamlit App

A modular web interface for analyzing USAF 1951 resolution targets.
"""

import os
import sys

# Make sure the package is in the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import the Streamlit app
from usaf_package.ui.streamlit_components import run_streamlit_app

if __name__ == "__main__":
    run_streamlit_app() 
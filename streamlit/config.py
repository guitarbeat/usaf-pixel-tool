"""
Configuration settings for USAF 1951 Resolution Target Analyzer

This module contains all constants, default values, color definitions,
and session state prefixes used throughout the application.
"""

import os

# --- File Paths ---
DEFAULT_IMAGE_PATH = os.path.expanduser(
    "~/Library/CloudStorage/Box-Box/FOIL/Aaron/2025-05-12/airforcetarget_images/AF_2_2_00001.png"
)

# --- ROI Colors ---
ROI_COLORS = [
    "#00FF00",
    "#FF00FF",
    "#00FFFF",
    "#FFFF00",
    "#FF8000",
    "#0080FF",
    "#8000FF",
    "#FF0080",
]
INVALID_ROI_COLOR = "#FF0000"  # Red for invalid ROIs

# --- Session State Prefixes ---
SESSION_STATE_PREFIXES = [
    "group_",
    "element_",
    "analyzed_roi_",
    "analysis_results_",
    "last_group_",
    "last_element_",
    "coordinates_",
    "image_path_",
    "image_name_",
    "roi_valid_",
]

# --- UI Defaults ---
DEFAULT_GROUP = 2
DEFAULT_ELEMENT = 2

# --- Other UI Strings ---
WELCOME_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/d/d6/1951usaf_test_target.jpg"
)
WELCOME_IMAGE_CAPTION = "Example USAF 1951 Target"

# --- Constants ---
# Image dimensions and channels
MIN_IMAGE_DIMS = 2
MAX_IMAGE_CHANNELS = 4
RGB_CHANNELS = 3

# Array dimensions and indices
MIN_ARRAY_DIMS = 2
MIN_ARRAY_LENGTH = 2
MIN_ARRAY_SIZE = 2
MIN_PAIRS = 2
MIN_LINE_PAIRS = 2
MIN_BOUNDARIES = 3
ROI_COORDS_LENGTH = 4

# USAF Target constants
MAX_USAF_ELEMENT = 6

"""
Streamlit UI Components for USAF 1951 Resolution Target Analysis.

This module contains UI components and layouts specific to the Streamlit web application.
"""

import os
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

# Import core modules
from usaf_package.main import run_analysis
from usaf_package.core.usaf_target import USAFTarget
from usaf_package.utils.streamlit_helpers import (
    process_uploaded_file, extract_roi_image, save_analysis_results
)

from usaf_package.processing.image_processor import ImageProcessor
from usaf_package.processing.profile_analyzer import ProfileAnalyzer


def run_analysis(
    image_path: str,
    roi: tuple = None,
    group: int = 2,
    element: int = 3
):
    """
    Run the USAF 1951 target analysis pipeline.

    Args:
        image_path (str): Path to the image file
        roi (tuple): (x, y, width, height) region of interest, or None for full image
        group (int): USAF group number
        element (int): USAF element number

    Returns:
        dict: Analysis results
    """
    # Load image
    img_proc = ImageProcessor()
    img_proc.load_image(image_path)

    # Select ROI if provided
    if roi is not None:
        img_proc.select_roi(roi)

    # Get horizontal profile
    profile = img_proc.get_line_profile()

    # Analyze profile using boundary detection
    analyzer = ProfileAnalyzer()
    results = analyzer.analyze_profile(
        profile=profile,
        group=group,
        element=element
    )

    # Add extra info for UI/consistency
    results["raw_profile"] = profile.tolist() if hasattr(profile, 'tolist') else profile
    results["group"] = group
    results["element"] = element

    return results 

# Default image path
DEFAULT_IMAGE_PATH = "/Users/aaron/Library/CloudStorage/Box-Box/FOIL/Aaron/2025-05-12/airforcetarget_images/AF_2_2_00001.png"

__all__ = [
    "run_streamlit_app"
]

# ----------------------
# Session State Helpers
# ----------------------
def load_default_image():
    """Return the default image path if it exists, else None."""
    if os.path.exists(DEFAULT_IMAGE_PATH):
        return DEFAULT_IMAGE_PATH
    return None

def initialize_session_state():
    """Initialize or reset Streamlit session state variables."""
    for key in [
        'coordinates', 'analysis_results', 'analyzed_roi',
        'last_group', 'last_element'
    ]:
        if key not in st.session_state:
            st.session_state[key] = None

def reset_session_state():
    """Reset session state variables."""
    for key in [
        'coordinates', 'analysis_results', 'analyzed_roi',
        'last_group', 'last_element'
    ]:
        st.session_state[key] = None

def get_image_session_keys(idx):
    """Return all session state keys for a given image index."""
    return {
        'group': f'group_{idx}',
        'element': f'element_{idx}',
        'analyzed_roi': f'analyzed_roi_{idx}',
        'analysis_results': f'analysis_results_{idx}',
        'last_group': f'last_group_{idx}',
        'last_element': f'last_element_{idx}'
    }

# ----------------------
# UI Helper Functions
# ----------------------
def group_element_selectors(idx, default_group=2, default_element=2):
    """Display group and element number inputs for a given image index."""
    keys = get_image_session_keys(idx)
    if keys['group'] not in st.session_state:
        st.session_state[keys['group']] = default_group
    if keys['element'] not in st.session_state:
        st.session_state[keys['element']] = default_element
    col_g, col_e = st.columns(2)
    with col_g:
        group = st.number_input(
            "Group", value=st.session_state[keys['group']], min_value=-2, max_value=9, key=keys['group']
        )
    with col_e:
        element = st.number_input(
            "Element", value=st.session_state[keys['element']], min_value=1, max_value=6, key=keys['element']
        )
    return group, element

def display_metrics_row(results):
    """Display a compact row of metrics for analysis results."""
    group = results.get('group')
    element = results.get('element')
    lp_width_um = USAFTarget.line_pair_width_microns(group, element) if group is not None and element is not None else None
    metric_cols = st.columns(4)
    metrics = [
        ("Line Pairs per mm", f"{results['lp_per_mm']:.2f}", "Line pairs per millimeter (lp/mm)"),
        ("Line Pair Width (μm)", f"{lp_width_um:.2f}" if lp_width_um is not None else "-", "Line pair width in microns (μm)"),
        ("Contrast", f"{results['contrast']:.2f}", "Michelson contrast ratio"),
        ("Line Pairs Detected", f"{results['num_line_pairs']}", "Number of line pairs detected in ROI")
    ]
    for i, (label, value, helptext) in enumerate(metrics):
        with metric_cols[i]:
            st.markdown(f"<div style='font-size:1.1em'><b>{label}:</b> {value}</div>", unsafe_allow_html=True)

def display_roi_info(image):
    """
    Display information about the selected ROI in a clean, useful format.
    Args:
        image: The source image
    Returns:
        tuple: ROI parameters as (x, y, width, height) or None
    """
    if not st.session_state.coordinates:
        st.info("Click and drag on the image to select a region of interest")
        return None
    point1, point2 = st.session_state.coordinates
    roi_x = min(point1[0], point2[0])
    roi_y = min(point1[1], point2[1])
    roi_width = abs(point2[0] - point1[0])
    roi_height = abs(point2[1] - point1[1])
    return (int(roi_x), int(roi_y), int(roi_width), int(roi_height))

def handle_image_selection(image, key="usaf_image"):
    """
    Handle interactive ROI selection on an image.
    Args:
        image: The image to display
        key: Unique key for the streamlit component
    Returns:
        bool: True if ROI coordinates changed, else False
    """
    coords = streamlit_image_coordinates(
        image,
        key=key,
        click_and_drag=True
    )
    if coords is not None and coords.get("x1") is not None:
        point1 = (coords["x1"], coords["y1"])
        point2 = (coords["x2"], coords["y2"])
        if (point1[0] != point2[0] and point1[1] != point2[1] and st.session_state.coordinates != (point1, point2)):
            st.session_state.coordinates = (point1, point2)
            return True
    return False

def display_welcome_screen():
    """Display the welcome screen with instructions when no image is loaded."""
    st.info("Please upload a USAF 1951 target image to begin analysis.")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/USAF-1951.svg/1024px-USAF-1951.svg.png", 
                caption="Example USAF 1951 Target", use_container_width=True)
    with col2:
        st.subheader("MIL-STD-150A Target Format")
        st.latex(r"""
        \text{resolution (lp/mm)} = 2^{\text{group} + (\text{element} - 1)/6}
        """)
        st.latex(r"""
        \text{Line Pair Width (μm)} = \frac{1000}{2 \times \text{resolution (lp/mm)}}
        """)
        st.markdown("""
        **Target Arrangement (MIL-STD-150A):**
        - Six groups in a compact spiral arrangement of three layers
        - Largest groups (first layer) on outer sides
        - Smaller layers progress toward center
        - Each group has six elements (1-6)
        - Odd-numbered groups: 1-6 from upper right
        - Even-numbered groups: First element at lower right, 2-6 at left
        """)

def display_line_pair_expander(results):
    """Display the line pair analysis expander with dynamic formulas and values."""
    group = results.get('group')
    element = results.get('element')
    lp_width_um = USAFTarget.line_pair_width_microns(group, element) if group is not None and element is not None else None
    lp_per_mm = results.get('lp_per_mm')
    widths = results.get('line_pair_widths', [])
    widths_str = ', '.join(str(int(w)) for w in widths)
    st.write(f"**Widths:** {widths_str}")
    st.markdown("""
    <b>A line pair</b> = one black bar + one white bar (one cycle)<br>
    <b>Line pair width</b> = distance from start of one black bar to the next
    """, unsafe_allow_html=True)
    st.markdown("<hr style='margin: 0.5em 0;' />", unsafe_allow_html=True)
    st.latex(rf"Group = {group},\quad Element = {element}")
    st.latex(rf"\text{{Line Pairs per mm}} = 2^{{{group} + ({element} - 1)/6}} = {lp_per_mm:.2f}")
    st.latex(rf"\text{{Line Pair Width ($\mu m$)}} = \frac{{1000}}{{2 \times {lp_per_mm:.2f}}} = {lp_width_um:.2f}")
    st.markdown(f"<b>Line Pairs per mm:</b> {lp_per_mm:.2f}<br>"
                f"<b>Line Pair Width (μm):</b> {lp_width_um:.2f}", unsafe_allow_html=True)

# ----------------------
# Plotting Functions
# ----------------------
def detect_line_pair_boundaries(profile, threshold=20):
    """
    Detect line pair boundaries by finding significant transitions in the intensity profile.
    Args:
        profile: 1D numpy array of intensity values
        threshold: Minimum absolute derivative to consider a transition
    Returns:
        List of x positions (pixel indices) where transitions occur
    """
    derivative = np.diff(profile)
    boundaries = np.where(np.abs(derivative) > threshold)[0]
    min_distance = 3  # pixels
    filtered = []
    last = -min_distance
    for idx in boundaries:
        if idx - last >= min_distance:
            filtered.append(idx)
            last = idx
    return filtered

def create_profile_visualization(roi_image, profile, results):
    import matplotlib.patches as patches
    import matplotlib as mpl
    try:
        import seaborn as sns
        cb_palette = sns.color_palette("colorblind")
        magenta = cb_palette[3]
        blue = cb_palette[0]
    except ImportError:
        magenta = '#CC79A7'
        blue = '#0072B2'
    height, width = roi_image.shape[0], roi_image.shape[1]
    boundaries = results.get('boundaries', [])
    line_pair_widths = results.get('line_pair_widths', [])
    avg_line_pair_width = results.get('avg_line_pair_width', 0.0)
    dpi = 200
    fig, (ax_img, ax_prof) = plt.subplots(2, 1, figsize=(10, 5), dpi=dpi, gridspec_kw={'height_ratios': [1, 2]}, sharex=True)
    if roi_image.ndim == 2:
        ax_img.imshow(roi_image, cmap='gray', aspect='equal')
    else:
        ax_img.imshow(roi_image, aspect='equal')
    for x in boundaries:
        ax_img.axvline(x, color=magenta, linestyle='--', linewidth=2, alpha=0.7)
    bar_len = min(100, width // 5)
    ax_img.plot([width - bar_len - 10, width - 10], [height - 10, height - 10], color='black', lw=4)
    ax_img.text(width - bar_len // 2 - 10, height - 20, f"{bar_len} px", color='black', ha='center', va='top', fontsize=10, bbox=dict(fc='white', ec='none', alpha=0.7))
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_ylabel("")
    ax_img.set_title("ROI with Detected Line Pair Boundaries", fontsize=14, pad=10, loc='center')
    x = np.arange(len(profile))
    ax_prof.plot(x, profile, linewidth=2, color=blue, label='Intensity Profile')
    ax_prof.set_xlim(0, width)
    ax_prof.set_ylim(0, 255)
    ax_prof.set_xlabel("Position (pixels)", fontsize=13)
    ax_prof.set_ylabel("Intensity Value", fontsize=13)
    ax_prof.grid(True, alpha=0.2)
    ax_prof.set_title("Intensity Profile (Column Average)", fontsize=14, pad=10, loc='center')
    for i in range(len(line_pair_widths)):
        if i+2 < len(boundaries):
            x0 = boundaries[i]
            x2 = boundaries[i+2]
            y0 = profile[x0]
            y2 = profile[x2]
            mid_x = (x0 + x2) / 2
            min_y = min(y0, y2)
            y_offset = -10 - (i % 2) * 18
            ax_prof.plot([x0, x2], [min_y-10, min_y-10], color=magenta, linewidth=3, alpha=0.7, label='Line Pair Segments' if i == 0 else "")
            width_val = x2 - x0
            ax_prof.annotate(f"{width_val} px", xy=(mid_x, min_y-15), xytext=(0, y_offset), textcoords='offset points',
                         ha='center', va='top', fontsize=10, color=magenta,
                         bbox=dict(boxstyle="round,pad=0.2", fc='white', ec=magenta, alpha=0.7))
    ax_prof.annotate(f"Average line pair width: {avg_line_pair_width:.2f} px",
                 xy=(0.5, 1.08), xycoords='axes fraction', ha='center', va='bottom', fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='black', alpha=0.8))
    ax_prof.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=11, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig 

def annotated_roi_with_distances_figure(roi_image):
    import matplotlib.pyplot as plt
    from usaf_package.processing.profile_analyzer import detect_line_pair_boundaries
    try:
        import seaborn as sns
        cb_palette = sns.color_palette("colorblind")
        magenta = cb_palette[3]
    except ImportError:
        magenta = '#CC79A7'
    profile = np.mean(roi_image, axis=0) if roi_image.ndim == 2 else np.mean(cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY), axis=0)
    boundaries = detect_line_pair_boundaries(profile)
    fig, ax = plt.subplots(figsize=(5, 2), dpi=150)
    if roi_image.ndim == 2:
        ax.imshow(roi_image, cmap='gray', aspect='auto')
    else:
        ax.imshow(roi_image, aspect='auto')
    for x in boundaries:
        ax.axvline(x, color=magenta, linestyle='--', linewidth=2, alpha=0.7)
    for i in range(len(boundaries)-2):
        x0 = boundaries[i]
        x2 = boundaries[i+2]
        y = roi_image.shape[0] - 10
        ax.plot([x0, x2], [y, y], color=magenta, linewidth=2, alpha=0.8)
        ax.text((x0 + x2) / 2, y - 8, f"{x2 - x0} px", color=magenta, fontsize=10, ha='center', va='top', bbox=dict(fc='white', ec=magenta, alpha=0.7, boxstyle='round,pad=0.2'))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("")
    plt.tight_layout(pad=0.2)
    return fig

def simple_profile_figure(profile, width):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 2), dpi=150)
    x = np.arange(len(profile))
    ax.plot(x, profile, linewidth=2, color='#0072B2')
    ax.set_xlim(0, width)
    ax.set_xlabel("Position (pixels)", fontsize=10)
    ax.set_ylabel("Intensity", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_title("")
    plt.tight_layout(pad=0.2)
    return fig 

def combined_profile_and_annotated_roi_figure(roi_image, profile, results, group, element):
    import matplotlib.patches as patches
    import matplotlib as mpl
    try:
        import seaborn as sns
        cb_palette = sns.color_palette("colorblind")
        magenta = cb_palette[3]
        blue = cb_palette[0]
    except ImportError:
        magenta = '#CC79A7'
        blue = '#0072B2'
    height, width = roi_image.shape[0], roi_image.shape[1]
    boundaries = results.get('boundaries', [])
    line_pair_widths = results.get('line_pair_widths', [])
    dpi = 150
    fig, (ax_img, ax_prof) = plt.subplots(2, 1, figsize=(8, 5), dpi=dpi, gridspec_kw={'height_ratios': [1, 1.2]}, sharex=True)
    # Annotated ROI
    if roi_image.ndim == 2:
        ax_img.imshow(roi_image, cmap='gray', aspect='equal')
    else:
        ax_img.imshow(roi_image, aspect='equal')
    for x in boundaries:
        ax_img.axvline(x, color=magenta, linestyle='--', linewidth=2, alpha=0.7)
    for i in range(len(boundaries)-2):
        x0 = boundaries[i]
        x2 = boundaries[i+2]
        y = roi_image.shape[0] - 10
        ax_img.plot([x0, x2], [y, y], color=magenta, linewidth=2, alpha=0.8)
        ax_img.text((x0 + x2) / 2, y - 8, f"{x2 - x0} px", color=magenta, fontsize=10, ha='center', va='top', bbox=dict(fc='white', ec=magenta, alpha=0.7, boxstyle='round,pad=0.2'))
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_ylabel("")
    ax_img.set_title("")
    # Line Profile
    x = np.arange(len(profile))
    ax_prof.plot(x, profile, linewidth=2, color=blue)
    ax_prof.set_xlim(0, width)
    ax_prof.set_ylim(0, 255)
    ax_prof.set_xlabel("Position (pixels)", fontsize=11)
    ax_prof.set_ylabel("Intensity", fontsize=11)
    ax_prof.grid(True, alpha=0.2)
    ax_prof.set_title("")
    for i in range(len(line_pair_widths)):
        if i+2 < len(boundaries):
            x0 = boundaries[i]
            x2 = boundaries[i+2]
            y0 = profile[x0]
            y2 = profile[x2]
            mid_x = (x0 + x2) / 2
            min_y = min(y0, y2)
            y_offset = -10 - (i % 2) * 18
            ax_prof.plot([x0, x2], [min_y-10, min_y-10], color=magenta, linewidth=3, alpha=0.7)
            width_val = x2 - x0
            ax_prof.annotate(f"{width_val} px", xy=(mid_x, min_y-15), xytext=(0, y_offset), textcoords='offset points',
                         ha='center', va='top', fontsize=10, color=magenta,
                         bbox=dict(boxstyle="round,pad=0.2", fc='white', ec=magenta, alpha=0.7))
    # No legend, no average annotation, no matplotlib title
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig

# ----------------------
# Per-Image Analysis Logic
# ----------------------
def analyze_and_display_image(idx, uploaded_file, image, temp_path):
    """
    Handle all UI and analysis for a single image within an expander.
    All image controls, ROI selection, and analysis are contained within the expander.
    """
    # Get image filename for the expander title
    filename = uploaded_file.name if hasattr(uploaded_file, 'name') else f"Image {idx+1}"
    
    # Create an expander for this image
    with st.expander(f"Image {idx+1}: {filename}", expanded=(idx == 0)):
        # Group and element selectors
        group, element = group_element_selectors(idx)
        
        # ROI selection
        roi_changed = handle_image_selection(image, key=f"usaf_image_{idx}")
        if roi_changed:
            st.rerun()
            
        roi = display_roi_info(image)
        keys = get_image_session_keys(idx)
        
        # Analysis section
        if roi and (roi != st.session_state.get(keys['analyzed_roi']) or 
                   group != st.session_state.get(keys['last_group']) or 
                   element != st.session_state.get(keys['last_element'])):
            with st.spinner("Analyzing image..."):
                from usaf_package.processing.image_processor import ImageProcessor
                img_proc = ImageProcessor()
                img_proc.load_image(temp_path)
                img_proc.select_roi(roi)
                profile = img_proc.get_line_profile()
                from usaf_package.processing.profile_analyzer import ProfileAnalyzer
                analyzer = ProfileAnalyzer()
                results = analyzer.analyze_profile(
                    profile=profile,
                    group=group,
                    element=element
                )
                results["raw_profile"] = profile.tolist() if hasattr(profile, 'tolist') else None
                results["group"] = group
                results["element"] = element
                st.session_state[keys['analyzed_roi']] = roi
                st.session_state[keys['analysis_results']] = results
                st.session_state[keys['last_group']] = group
                st.session_state[keys['last_element']] = element
        
        # Results and visualization
        if st.session_state.get(keys['analysis_results']):
            results = st.session_state[keys['analysis_results']]
            
            # Create two columns for visualization
            viz_col1, viz_col2 = st.columns([1, 1])
            
            with viz_col1:
                if roi:
                    roi_x, roi_y, roi_width, roi_height = roi
                    img_with_roi = image.copy()
                    cv2.rectangle(
                        img_with_roi,
                        (int(roi_x), int(roi_y)),
                        (int(roi_x + roi_width), int(roi_y + roi_height)),
                        (255, 0, 0),
                        2
                    )
                    st.image(img_with_roi, caption="Selected Region", use_container_width=True)
            
            with viz_col2:
                if roi:
                    roi_image = extract_roi_image(image, roi)
                    st.image(roi_image, caption="Zoomed View", use_container_width=True)
            
            # Metrics row below images
            display_metrics_row(results)
            
            # Analysis
            display_line_pair_expander(results)
            
            # Combined plot
            if roi:
                from usaf_package.processing.image_processor import ImageProcessor
                img_proc = ImageProcessor()
                img_proc.load_image(temp_path)
                img_proc.select_roi(roi)
                profile = img_proc.get_line_profile()
                st.pyplot(combined_profile_and_annotated_roi_figure(roi_image, profile, results, group, element))
                st.markdown("<div style='text-align:center; font-size:1.1em; margin-top:0.5em;'><b>ROI with Boundaries and Intensity Profile</b></div>", unsafe_allow_html=True)

# ----------------------
# Main App Logic
# ----------------------
def run_streamlit_app():
    """Main Streamlit application function with a compact, efficient UI and per-image controls/results."""
    try:
        st.set_page_config(
            page_title="USAF 1951 Resolution Target Analyzer",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        initialize_session_state()
        st.markdown("<h2 style='margin-bottom:0.5em'>USAF 1951 Resolution Target Analyzer</h2>", unsafe_allow_html=True)
        left_col, right_col = st.columns([1, 2])
        
        with left_col:
            # File uploader in the left column
            uploaded_files = st.file_uploader(
                "Upload USAF target image(s)",
                type=["jpg", "jpeg", "png", "tif", "tiff"],
                accept_multiple_files=True,
                help="Select one or more images containing a USAF 1951 resolution target"
            )
            
            # Add default image support
            if not uploaded_files and load_default_image():
                default_img_path = load_default_image()
                st.info(f"Using default image: {os.path.basename(default_img_path)}")
                uploaded_files = [default_img_path]
            
            # Welcome screen if no images
            if not uploaded_files:
                display_welcome_screen()
                return
        
        with right_col:
            # Process each uploaded image
            for idx, uploaded_file in enumerate(uploaded_files):
                image, temp_path = process_uploaded_file(uploaded_file)
                if image is not None:
                    # Each image gets its own expander with all controls and analysis
                    analyze_and_display_image(idx, uploaded_file, image, temp_path)
                else:
                    st.error(f"Failed to load image: {getattr(uploaded_file, 'name', f'Image {idx+1}')}. Please check the file format.")
    
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("For detailed error information, set DEBUG=1 in environment variables.") 
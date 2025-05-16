#!/usr/bin/env python3
"""
USAF 1951 Resolution Target Analyzer

A comprehensive tool for analyzing USAF 1951 resolution targets in microscopy and imaging systems.
"""

import os
import sys
import io
import time
import math
import tempfile
import logging
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union
from streamlit_image_coordinates import streamlit_image_coordinates

# Configure logger
logger = logging.getLogger(__name__)

#-----------------------------------------------------------
# USAF Target Class - Core functionality
#-----------------------------------------------------------
class USAFTarget:
    """Represents a USAF 1951 resolution target with methods for calculating line pairs."""
    
    def __init__(self):
        """Initialize the USAF target object."""
        # Precomputed line pairs per mm for group 0, element 1
        self.base_lp_per_mm = 1.0
    
    def lp_per_mm(self, group: int, element: int) -> float:
        """
        Calculate line pairs per mm for a specific group and element.
        """
        # Formula: lp/mm = 2^(group + (element-1)/6) * base_lp_per_mm
        return self.base_lp_per_mm * (2 ** (group + (element - 1) / 6))
    
    def line_pair_width_microns(self, group: int, element: int) -> float:
        """
        Calculate width of a single line pair in microns.
        """
        # Convert lp/mm to microns width (1 lp/mm = 1000 µm width)
        return 1000.0 / self.lp_per_mm(group, element)
    
    def resolution_to_group_element(self, resolution_um: float) -> Dict[str, Union[int, float]]:
        """
        Convert a resolution in microns to the closest USAF group and element.
        """
        if resolution_um <= 0:
            return {"group": 0, "element": 1, "exact_group": 0, "exact_element": 1}
        
        # Convert resolution to lp/mm
        lp_per_mm = 1000.0 / resolution_um
        
        # Calculate log2 result
        log2_result = math.log2(lp_per_mm / self.base_lp_per_mm)
        
        # Calculate exact group and element
        exact_group = int(log2_result)
        exact_element = round(((log2_result - exact_group) * 6) + 1)
        
        # Adjust if element is out of range
        if exact_element > 6:
            exact_group += 1
            exact_element = 1
        elif exact_element < 1:
            exact_group -= 1
            exact_element = 6
        
        return {
            "group": exact_group,
            "element": exact_element,
            "exact_group": log2_result,
            "exact_element": ((log2_result - exact_group) * 6) + 1
        }

#-----------------------------------------------------------
# Image Processing
#-----------------------------------------------------------
def detect_line_pair_boundaries(profile, threshold=20, min_distance=3):
    """
    Detect line pair boundaries by finding significant transitions in the intensity profile.
    """
    # Calculate derivative of the profile
    derivative = np.diff(profile)
    
    # Find all positions where derivative exceeds threshold
    candidate_boundaries = np.where(np.abs(derivative) > threshold)[0]
    
    # Apply minimum distance filter
    filtered = []
    last = -min_distance
    for idx in candidate_boundaries:
        if idx - last >= min_distance:
            filtered.append(idx)
            last = idx
    
    return filtered, derivative

class ImageProcessor:
    """Process images to analyze USAF targets: load, select ROI, extract profile."""
    
    def __init__(self):
        self.image = None
        self.grayscale = None
        self.roi = None
        self.profile = None
        self.individual_profiles = None
    
    def load_image(self, image_path: str) -> bool:
        """Load image from path and convert to RGB and grayscale."""
        try:
            if not os.path.isfile(image_path):
                logger.error(f"Image file not found: {image_path}")
                return False
            # Try loading with OpenCV
            try:
                self.image = cv2.imread(image_path)
                if self.image is None:
                    raise ValueError("OpenCV couldn't load the image")
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.warning(f"OpenCV image loading failed: {e}. Trying with skimage...")
                from skimage import io
                self.image = io.imread(image_path)
            # Convert to grayscale if image has multiple channels
            if len(self.image.shape) > 2:
                self.grayscale = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            else:
                self.grayscale = self.image
            return True
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return False
    
    def select_roi(self, roi: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Select region of interest from grayscale image."""
        if self.grayscale is None:
            return None
        try:
            x, y, width, height = roi
            self.roi = self.grayscale[y:y+height, x:x+width]
            return self.roi
        except Exception as e:
            logger.error(f"Error selecting ROI: {e}")
            return None
    
    def get_line_profile(self) -> Optional[np.ndarray]:
        """Extract horizontal (column average) intensity profile from ROI."""
        if self.roi is None:
            return None
        try:
            use_roi = self.roi
            # Store individual line profiles (one for each row in the ROI)
            self.individual_profiles = use_roi.copy()
            # Calculate the averaged profile
            self.profile = np.mean(use_roi, axis=0)
            return self.profile
        except Exception as e:
            logger.error(f"Error getting line profile: {e}")
            return None
            
    def get_individual_profiles(self) -> Optional[np.ndarray]:
        """Returns the individual line profiles from the ROI."""
        return self.individual_profiles

class ProfileAnalyzer:
    """
    Analyzer for USAF intensity profiles: calculates line pairs and computes resolution.
    """
    def __init__(self, usaf_target: USAFTarget = None, 
                 threshold: int = 20, 
                 min_distance: int = 3):
        """Initialize the profile analyzer with detection parameters."""
        self.usaf_target = usaf_target or USAFTarget()
        self.threshold = threshold
        self.min_distance = min_distance

    def analyze_profile(self, profile: np.ndarray, group: int, element: int) -> Dict:
        """Analyze profile to calculate metrics for lp resolution and pixel size."""
        # Apply boundary detection algorithm
        boundaries, derivative = detect_line_pair_boundaries(
            profile, 
            threshold=self.threshold, 
            min_distance=self.min_distance
        )
        
        # Calculate line pair widths and contrast
        line_pair_widths = []
        contrast = 0.0
        
        if len(boundaries) >= 3:  # Need at least 3 boundaries for 1 complete line pair
            line_pair_widths = [boundaries[i+2] - boundaries[i] for i in range(len(boundaries)-2)]
            
            # Calculate contrast if we have boundaries
            try:
                # Find peaks and valleys in the profile
                peaks = [profile[i] for i in range(1, len(profile)-1) 
                        if profile[i] > profile[i-1] and profile[i] > profile[i+1]]
                valleys = [profile[i] for i in range(1, len(profile)-1) 
                          if profile[i] < profile[i-1] and profile[i] < profile[i+1]]
                
                if peaks and valleys:
                    max_intensity = np.mean(sorted(peaks, reverse=True)[:min(3, len(peaks))])
                    min_intensity = np.mean(sorted(valleys)[:min(3, len(valleys))])
                    contrast = (max_intensity - min_intensity) / (max_intensity + min_intensity)
            except:
                # Fallback contrast calculation
                if len(profile) > 0:
                    contrast = (np.max(profile) - np.min(profile)) / (np.max(profile) + np.min(profile))
        
        num_line_pairs = len(line_pair_widths)
        avg_line_pair_width = float(np.mean(line_pair_widths)) if line_pair_widths else 0.0
        
        # Calculate theoretical line pairs per mm for this USAF target element
        lp_per_mm = self.usaf_target.lp_per_mm(group, element)
        
        # Prepare results dictionary with comprehensive information
        results = {
            # USAF target information
            "group": group,
            "element": element,
            "lp_per_mm": float(lp_per_mm),
            "theoretical_lp_width_um": self.usaf_target.line_pair_width_microns(group, element),
            
            # Detection results
            "num_line_pairs": num_line_pairs,
            "num_boundaries": len(boundaries),
            "boundaries": boundaries,
            "line_pair_widths": line_pair_widths,
            "avg_line_pair_width": avg_line_pair_width,
            "contrast": float(contrast),
            
            # Algorithm parameters used
            "detection_threshold": self.threshold,
            "min_distance": self.min_distance,
            
            # Raw data for visualization
            "derivative": derivative.tolist() if hasattr(derivative, 'tolist') else None,
            "profile": profile.tolist() if hasattr(profile, 'tolist') else None,
        }
        
        return results 

#-----------------------------------------------------------
# Streamlit Helpers
#-----------------------------------------------------------
def process_uploaded_file(uploaded_file) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Process an uploaded file from Streamlit's file_uploader or a file path."""
    if uploaded_file is None:
        return None, None
    
    try:
        # Check if uploaded_file is a string (file path)
        if isinstance(uploaded_file, str):
            if not os.path.exists(uploaded_file):
                st.error(f"File not found: {uploaded_file}")
                return None, None
                
            # Load the image directly
            image = cv2.imread(uploaded_file)
            if image is not None:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image, uploaded_file
            else:
                st.error(f"Failed to load image: {uploaded_file}")
                return None, None
                
        # Handle uploaded file object from st.file_uploader
        else:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
                
            # Load the image
            image = cv2.imread(temp_path)
            if image is not None:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image, temp_path
            else:
                # Clean up the temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                st.error(f"Failed to load image: {uploaded_file.name}")
                return None, None
                
    except Exception as e:
        st.error(f"Error processing file: {e}")
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return None, None

def extract_roi_image(image, roi_coordinates: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Extract ROI from the image and return it."""
    try:
        if roi_coordinates is None:
            return None
            
        x, y, width, height = roi_coordinates
        
        # Check if image is an ImageProcessor object
        if hasattr(image, 'select_roi'):
            return image.select_roi(roi_coordinates)
        # Otherwise, assume it's a numpy array
        else:
            if image is not None and x >= 0 and y >= 0 and width > 0 and height > 0:
                return image[y:y+height, x:x+width]
            return None
    except Exception as e:
        st.error(f"Error extracting ROI: {e}")
        return None

def save_analysis_results(results: Dict[str, Any]) -> str:
    """Convert analysis results to CSV format."""
    # Prepare data for CSV
    data = {
        "Parameter": [],
        "Value": []
    }
    
    # Add results to data dictionary, excluding arrays
    for key, value in results.items():
        if isinstance(value, (int, float, str)) and not key.startswith("_"):
            data["Parameter"].append(key)
            data["Value"].append(value)
    
    # Convert to DataFrame and then to CSV
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

#-----------------------------------------------------------
# Streamlit UI Components
#-----------------------------------------------------------

# Default image path - modify this for your own default image
DEFAULT_IMAGE_PATH = "/Users/aaron/Library/CloudStorage/Box-Box/FOIL/Aaron/2025-05-12/airforcetarget_images/AF_2_2_00001.png"

def load_default_image():
    """Return the default image path if it exists, else None."""
    if os.path.exists(DEFAULT_IMAGE_PATH):
        return DEFAULT_IMAGE_PATH
    return None

def initialize_session_state():
    """Initialize or reset Streamlit session state variables."""
    if 'coordinates' not in st.session_state:
        st.session_state.coordinates = None
    if 'usaf_target' not in st.session_state:
        st.session_state.usaf_target = USAFTarget()

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
    lp_width_um = st.session_state.usaf_target.line_pair_width_microns(group, element) if group is not None and element is not None else None
    metric_cols = st.columns(4)
    metrics = [
        ("Line Pairs per mm", f"{results['lp_per_mm']:.2f}"),
        ("Line Pair Width (μm)", f"{lp_width_um:.2f}" if lp_width_um is not None else "-"),
        ("Contrast", f"{results['contrast']:.2f}" if 'contrast' in results else "-"),
        ("Line Pairs Detected", f"{results['num_line_pairs']}")
    ]
    for i, (label, value) in enumerate(metrics):
        with metric_cols[i]:
            st.markdown(f"<div style='font-size:1.1em'><b>{label}:</b> {value}</div>", unsafe_allow_html=True)

def display_roi_info():
    """Return ROI parameters as (x, y, width, height) or None."""
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
    """Handle interactive ROI selection on an image."""
    coords = streamlit_image_coordinates(image, key=key, click_and_drag=True)
    if coords is not None and coords.get("x1") is not None:
        point1 = (coords["x1"], coords["y1"])
        point2 = (coords["x2"], coords["y2"])
        if (point1[0] != point2[0] and point1[1] != point2[1] and st.session_state.coordinates != (point1, point2)):
            st.session_state.coordinates = (point1, point2)
            return True
    return False

def display_welcome_screen():
    """Display welcome screen when no images are loaded."""
    st.info("Please upload a USAF 1951 target image to begin analysis.")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/USAF-1951.svg/1024px-USAF-1951.svg.png", 
                caption="Example USAF 1951 Target", use_container_width=True)
    with col2:
        st.subheader("USAF 1951 Target Format")
        st.latex(r"\text{resolution (lp/mm)} = 2^{\text{group} + (\text{element} - 1)/6}")
        st.latex(r"\text{Line Pair Width (μm)} = \frac{1000}{2 \times \text{resolution (lp/mm)}}")

def display_analysis_details(results):
    """Display analysis details in tabs."""
    # Create tabs for different analysis views
    analysis_tab, plot_tab = st.tabs(["Line Pair Analysis", "Intensity Profile"])
    
    with analysis_tab:
        group = results.get('group')
        element = results.get('element')
        lp_width_um = st.session_state.usaf_target.line_pair_width_microns(group, element) if group is not None and element is not None else None
        lp_per_mm = results.get('lp_per_mm')
        widths = results.get('line_pair_widths', [])
        widths_str = ', '.join(str(int(w)) for w in widths)
        st.write(f"**Widths:** {widths_str}")
        st.markdown("""
        <b>A line pair</b> = one black bar + one white bar (one cycle)<br>
        <b>Line pair width</b> = distance from start of one black bar to the next
        """, unsafe_allow_html=True)
        st.latex(rf"Group = {group},\quad Element = {element}")
        st.latex(rf"\text{{Line Pairs per mm}} = 2^{{{group} + ({element} - 1)/6}} = {lp_per_mm:.2f}")
        st.latex(rf"\text{{Line Pair Width ($\mu m$)}} = \frac{{1000}}{{2 \times {lp_per_mm:.2f}}} = {lp_width_um:.2f}")
    
    with plot_tab:
        if 'profile' in results and results['profile'] is not None:
            profile = np.array(results['profile'])
            boundaries = results.get('boundaries', [])
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(profile, linewidth=2, color='#0072B2', label='Intensity Profile')
            
            # Mark detected boundaries
            for boundary in boundaries:
                ax.axvline(x=boundary, color='#CC79A7', linestyle='--', alpha=0.7)
            
            # Mark line pairs
            line_pair_widths = results.get('line_pair_widths', [])
            for i in range(len(line_pair_widths)):
                if i+2 < len(boundaries):
                    x0 = boundaries[i]
                    x2 = boundaries[i+2]
                    min_y = min(profile[x0], profile[x2]) if x0 < len(profile) and x2 < len(profile) else 0
                    ax.plot([x0, x2], [min_y-10, min_y-10], color='#CC79A7', linewidth=2, alpha=0.7)
            
            ax.set_xlabel("Position (pixels)")
            ax.set_ylabel("Intensity")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            st.pyplot(fig)
            st.markdown("<div style='text-align:center;'><b>Intensity Profile with Detected Boundaries</b></div>", unsafe_allow_html=True)

def analyze_image(image_path, roi, group, element):
    """Run analysis on an image with the given ROI and parameters."""
    img_proc = ImageProcessor()
    img_proc.load_image(image_path)
    img_proc.select_roi(roi)
    profile = img_proc.get_line_profile()
    
    analyzer = ProfileAnalyzer(st.session_state.usaf_target)
    results = analyzer.analyze_profile(
        profile=profile,
        group=group,
        element=element
    )
    
    return results, profile

def analyze_and_display_image(idx, uploaded_file):
    """Handle UI and analysis for a single image within an expander."""
    # Get image filename for the expander title
    if isinstance(uploaded_file, str):
        filename = os.path.basename(uploaded_file)
    else:
        filename = uploaded_file.name if hasattr(uploaded_file, 'name') else f"Image {idx+1}"
    
    # Create an expander for this image
    with st.expander(f"Image {idx+1}: {filename}", expanded=(idx == 0)):
        # Process the uploaded file
        image, temp_path = process_uploaded_file(uploaded_file)
        
        if image is None:
            st.error(f"Failed to load image: {filename}")
            return
        
        # Group and element selectors
        group, element = group_element_selectors(idx)
        
        # ROI selection
        roi_changed = handle_image_selection(image, key=f"usaf_image_{idx}")
        if roi_changed:
            st.rerun()
            
        roi = display_roi_info()
        keys = get_image_session_keys(idx)
        
        # Analysis section
        if roi and (roi != st.session_state.get(keys['analyzed_roi']) or 
                  group != st.session_state.get(keys['last_group']) or 
                  element != st.session_state.get(keys['last_element'])):
            with st.spinner("Analyzing image..."):
                results, _ = analyze_image(temp_path, roi, group, element)
                
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
                    st.image(roi_image, caption=f"Zoomed View - Group {group}, Element {element}", use_container_width=True)
            
            # Display metrics and analysis
            display_metrics_row(results)
            display_analysis_details(results)

def run_streamlit_app():
    """Main Streamlit application function."""
    try:
        st.set_page_config(page_title="USAF Target Analyzer", layout="wide")
        initialize_session_state()
        
        st.title("USAF Target Analyzer")
        st.markdown("""
        <style>
        .stExpander {margin-top: 0.5rem !important;}
        .plot-container {margin-bottom: 0.5rem;}
        </style>
        """, unsafe_allow_html=True)
        
        left_col, right_col = st.columns([1, 2])
        
        with left_col:
            # File uploader
            uploaded_files = st.file_uploader(
                "Upload USAF target image(s)",
                type=["jpg", "jpeg", "png", "tif", "tiff"],
                accept_multiple_files=True,
                help="Select one or more images containing a USAF 1951 resolution target"
            )
            
            # Default image support
            default_image_path = load_default_image()
            if not uploaded_files and default_image_path:
                st.info(f"Using default image: {os.path.basename(default_image_path)}")
                
        with right_col:
            # Process images
            if uploaded_files:
                for idx, uploaded_file in enumerate(uploaded_files):
                    analyze_and_display_image(idx, uploaded_file)
            elif default_image_path:
                analyze_and_display_image(0, default_image_path)
            else:
                display_welcome_screen()
    
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("For detailed error information, set DEBUG=1 in environment variables.")

if __name__ == "__main__":
    run_streamlit_app() 
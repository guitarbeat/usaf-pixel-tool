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
from PIL import Image, ImageDraw
import matplotlib.patheffects as PathEffects
import hashlib
import streamlit_nested_layout

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# --- Constants ---
SESSION_STATE_PREFIXES = [
    'group_', 'element_', 'analyzed_roi_', 'analysis_results_', 'last_group_', 'last_element_',
    'coordinates_', 'image_path_', 'image_name_', 'roi_valid_'
]
DEFAULT_IMAGE_PATH = "/Users/aaron/Library/CloudStorage/Box-Box/FOIL/Aaron/2025-05-12/airforcetarget_images/AF_2_2_00001.png"
# Add ROI color options
ROI_COLORS = ["#00FF00", "#FF00FF", "#00FFFF", "#FFFF00", "#FF8000", "#0080FF", "#8000FF", "#FF0080"]
INVALID_ROI_COLOR = "#FF0000"  # Red for invalid ROIs

# --- Utility Functions ---
def get_unique_id_for_image(image_file) -> str:
    try:
        if isinstance(image_file, str):
            filename = os.path.basename(image_file)
        else:
            filename = image_file.name if hasattr(image_file, 'name') else str(id(image_file))
        short_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        return f"img_{short_hash}"
    except Exception as e:
        logger.error(f"Error generating unique ID: {e}")
        return f"img_{int(time.time() * 1000)}"

def load_default_image():
    return DEFAULT_IMAGE_PATH if os.path.exists(DEFAULT_IMAGE_PATH) else None

def process_uploaded_file(uploaded_file) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if uploaded_file is None:
        return None, None
    try:
        if isinstance(uploaded_file, str):
            if not os.path.exists(uploaded_file):
                st.error(f"File not found: {uploaded_file}")
                return None, None
            image = cv2.imread(uploaded_file)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image, uploaded_file
            st.error(f"Failed to load image: {uploaded_file}")
            return None, None
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            image = cv2.imread(temp_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image, temp_path
            if os.path.exists(temp_path):
                os.remove(temp_path)
            st.error(f"Failed to load image: {uploaded_file.name}")
            return None, None
    except Exception as e:
        st.error(f"Error processing file: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return None, None

def extract_roi_image(image, roi_coordinates: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    try:
        if roi_coordinates is None:
            return None
        x, y, width, height = roi_coordinates
        if hasattr(image, 'select_roi'):
            return image.select_roi(roi_coordinates)
        if image is not None and x >= 0 and y >= 0 and width > 0 and height > 0:
            return image[y:y+height, x:x+width]
        return None
    except Exception as e:
        st.error(f"Error extracting ROI: {e}")
        return None

def save_analysis_results(results: Dict[str, Any]) -> str:
    data = {"Parameter": [], "Value": []}
    for key, value in results.items():
        if isinstance(value, (int, float, str)) and not key.startswith("_"):
            data["Parameter"].append(key)
            data["Value"].append(value)
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

def initialize_session_state():
    if 'usaf_target' not in st.session_state:
        st.session_state.usaf_target = USAFTarget()
    if 'uploaded_files_list' not in st.session_state:
        st.session_state.uploaded_files_list = []
    if 'default_image_added' not in st.session_state:
        st.session_state.default_image_added = False
    if 'image_index_to_id' not in st.session_state:
        st.session_state.image_index_to_id = {}
    
    # Handle cleanup of rotation and sensitivity session state when images are removed
    if 'rotation_state_cleanup' not in st.session_state:
        st.session_state.rotation_state_cleanup = set()
        
    # Check and clean up session state variables for images that no longer exist
    current_image_ids = set()
    for file in st.session_state.uploaded_files_list:
        current_image_ids.add(get_unique_id_for_image(file))
    
    # Store the current set of image IDs for next cleanup check
    old_image_ids = st.session_state.rotation_state_cleanup
    for image_id in old_image_ids:
        if image_id not in current_image_ids:
            # Remove session state variables for this image
            prefixes_to_clean = [
                f'rotation_{image_id}', 
                f'last_rotation_{image_id}',
                f'sensitivity_{image_id}',
                f'last_sensitivity_{image_id}',
                f'min_distance_{image_id}',
                f'last_min_distance_{image_id}'
            ]
            for prefix in prefixes_to_clean:
                if prefix in st.session_state:
                    del st.session_state[prefix]
    
    # Update the set of image IDs that need cleanup on next check
    st.session_state.rotation_state_cleanup = current_image_ids

def get_image_session_keys(idx, image_file=None):
    if image_file is not None:
        unique_id = get_unique_id_for_image(image_file)
        st.session_state.image_index_to_id[idx] = unique_id
    else:
        unique_id = st.session_state.image_index_to_id.get(idx, f"idx_{idx}")
    return {
        'group': f'group_{unique_id}',
        'element': f'element_{unique_id}',
        'analyzed_roi': f'analyzed_roi_{unique_id}',
        'analysis_results': f'analysis_results_{unique_id}',
        'last_group': f'last_group_{unique_id}',
        'last_element': f'last_element_{unique_id}',
        'coordinates': f'coordinates_{unique_id}',
        'image_path': f'image_path_{unique_id}',
        'image_name': f'image_name_{unique_id}',
        'roi_valid': f'roi_valid_{unique_id}',
    }

# --- Core Classes ---
class USAFTarget:
    def __init__(self):
        self.base_lp_per_mm = 1.0
    def lp_per_mm(self, group: int, element: int) -> float:
        return self.base_lp_per_mm * (2 ** (group + (element - 1) / 6))
    def line_pair_width_microns(self, group: int, element: int) -> float:
        return 1000.0 / self.lp_per_mm(group, element)
    def resolution_to_group_element(self, resolution_um: float) -> Dict[str, Union[int, float]]:
        if resolution_um <= 0:
            return {"group": 0, "element": 1, "exact_group": 0, "exact_element": 1}
        lp_per_mm = 1000.0 / resolution_um
        log2_result = math.log2(lp_per_mm / self.base_lp_per_mm)
        exact_group = int(log2_result)
        exact_element = round(((log2_result - exact_group) * 6) + 1)
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

def detect_line_pair_boundaries(profile, threshold=2, min_distance=3):
    """
    Detect line pair boundaries in an intensity profile with improved transition tracking.
    
    For two line pairs, we expect to detect a pattern like:
    [light-to-dark, dark-to-light, light-to-dark, dark-to-light]
    
    Args:
        profile: 1D numpy array of intensity values
        threshold: Sensitivity threshold for edge detection
        min_distance: Minimum distance between detected edges
        
    Returns:
        tuple of (boundaries, derivative, transition_types)
    """
    # Calculate derivative to find intensity changes
    derivative = np.diff(profile)
    
    # Apply stronger adaptive thresholding based on the profile's characteristics
    adaptive_threshold = max(threshold, np.std(derivative) * 0.5)
    
    # Find significant transitions with their directions
    pos_transitions = np.where(derivative > adaptive_threshold)[0]  # Dark-to-light transitions (positive derivative)
    neg_transitions = np.where(derivative < -adaptive_threshold)[0]  # Light-to-dark transitions (negative derivative)
    
    # Create arrays to track transition positions and types
    all_transitions = []
    transition_types = []  # 1 for dark-to-light, -1 for light-to-dark
    
    # Add positive transitions (dark-to-light)
    for pos in pos_transitions:
        all_transitions.append(pos)
        transition_types.append(1)
        
    # Add negative transitions (light-to-dark)
    for neg in neg_transitions:
        all_transitions.append(neg)
        transition_types.append(-1)
    
    # Sort transitions by position
    if all_transitions:
        # Get sort indices
        sort_indices = np.argsort(all_transitions)
        
        # Apply sorting to both arrays
        all_transitions = np.array(all_transitions)[sort_indices]
        transition_types = np.array(transition_types)[sort_indices]
    
    # Filter transitions that are too close and maintain transition type info
    filtered = []
    filtered_types = []
    last = -min_distance
    
    for i in range(len(all_transitions)):
        idx = all_transitions[i]
        if idx - last >= min_distance:
            filtered.append(idx)
            filtered_types.append(transition_types[i])
            last = idx
    
    # For two line pairs, analyze the pattern of transitions
    # Ideally, we want to have transitions like: light-to-dark, dark-to-light, light-to-dark, dark-to-light
    if len(filtered) > 2:
        # Try to identify proper line pair transitions by looking for alternating patterns
        proper_transitions = []
        proper_types = []
        
        # If first transition is dark-to-light, discard it as we want to start with light-to-dark
        start_idx = 0
        if len(filtered_types) > 0 and filtered_types[0] == 1:  # If first is dark-to-light
            start_idx = 1
            
        # Extract transitions by expected pattern
        i = start_idx
        while i < len(filtered) - 1:
            # Check for a light-to-dark followed by dark-to-light pattern
            if i+1 < len(filtered_types) and filtered_types[i] == -1 and filtered_types[i+1] == 1:
                proper_transitions.extend([filtered[i], filtered[i+1]])
                proper_types.extend([filtered_types[i], filtered_types[i+1]])
                i += 2
            else:
                # Skip this transition if it doesn't fit the pattern
                i += 1
        
        # If we found proper transitions, use them
        if len(proper_transitions) >= 2:
            filtered = proper_transitions
            filtered_types = proper_types
    
    # If we have too many transitions after filtering, keep only the strongest ones
    if len(filtered) > 5:
        # Sort transitions by derivative magnitude, but maintain alternating pattern
        transition_strengths = np.abs(derivative[filtered])
        strongest_indices = np.argsort(transition_strengths)[-5:]
        strongest_indices = np.sort(strongest_indices)  # Resort by position to maintain order
        
        filtered = [filtered[i] for i in strongest_indices]
        filtered_types = [filtered_types[i] for i in strongest_indices]
    
    # Return boundaries, derivative, and transition types
    return filtered, derivative, filtered_types

class ImageProcessor:
    def __init__(self):
        self.image = None
        self.grayscale = None
        self.roi = None
        self.profile = None
        self.individual_profiles = None
    def load_image(self, image_path: str) -> bool:
        try:
            if not os.path.isfile(image_path):
                logger.error(f"Image file not found: {image_path}")
                return False
            try:
                self.image = cv2.imread(image_path)
                if self.image is None:
                    raise ValueError("OpenCV couldn't load the image")
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.warning(f"OpenCV image loading failed: {e}. Trying with skimage...")
                from skimage import io
                self.image = io.imread(image_path)
            if len(self.image.shape) > 2:
                self.grayscale = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            else:
                self.grayscale = self.image
            return True
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return False
    def select_roi(self, roi: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
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
        if self.roi is None:
            return None
        try:
            use_roi = self.roi
            self.individual_profiles = use_roi.copy()
            self.profile = np.mean(use_roi, axis=0)
            return self.profile
        except Exception as e:
            logger.error(f"Error getting line profile: {e}")
            return None
    def get_individual_profiles(self) -> Optional[np.ndarray]:
        return self.individual_profiles

class ProfileAnalyzer:
    def __init__(self, usaf_target: USAFTarget = None, threshold: int = 20, min_distance: int = 3):
        self.usaf_target = usaf_target or USAFTarget()
        self.threshold = threshold
        self.min_distance = min_distance
    def analyze_profile(self, profile: np.ndarray, group: int, element: int) -> Dict:
        boundaries, derivative, transition_types = detect_line_pair_boundaries(
            profile, threshold=self.threshold, min_distance=self.min_distance
        )
        
        line_pair_widths = []
        contrast = 0.0
        
        # Special handling for two line pairs with transition type information
        if len(boundaries) >= 3:
            # Extract line pair widths using transition types
            line_pair_widths = []
            line_pair_start_end = []
            
            # CORRECTED: A line pair is from one light-to-dark transition to the NEXT light-to-dark transition
            # Find all light-to-dark transitions (red lines)
            red_indices = [i for i, trans_type in enumerate(transition_types) if trans_type == -1]
            
            # For each pair of consecutive red lines, check if there's a blue line in between
            for j in range(len(red_indices) - 1):
                start_idx = red_indices[j]
                end_idx = red_indices[j + 1]
                start_pos = boundaries[start_idx]
                end_pos = boundaries[end_idx]
                
                # Check if there's at least one blue line between these red lines
                has_blue_between = False
                for i in range(start_idx + 1, end_idx):
                    if i < len(transition_types) and transition_types[i] == 1:
                        has_blue_between = True
                        break
                
                if has_blue_between:
                    # This is a complete line pair: red-blue-red
                    width = end_pos - start_pos
                    if width > 0:
                        line_pair_widths.append(width)
                        line_pair_start_end.append((start_pos, end_pos))
            
            # If we couldn't find complete line pairs, try to estimate based on dark bar widths
            if not line_pair_widths and len(boundaries) >= 4:
                # Look for each dark bar (from light-to-dark to dark-to-light)
                dark_bar_widths = []
                dark_bar_positions = []
                
                for i in range(len(boundaries) - 1):
                    if i+1 < len(transition_types) and transition_types[i] == -1 and transition_types[i+1] == 1:
                        # This is a dark bar from light-to-dark (red) to dark-to-light (blue)
                        start_pos = boundaries[i]
                        end_pos = boundaries[i+1]
                        dark_width = end_pos - start_pos
                        
                        if dark_width > 0:
                            dark_bar_widths.append(dark_width)
                            dark_bar_positions.append((start_pos, end_pos))
                
                # Estimate line pair widths (each line pair = dark bar + light bar of equal width)
                if dark_bar_widths:
                    avg_dark_width = np.mean(dark_bar_widths)
                    for start_pos, _ in dark_bar_positions:
                        # Line pair width = 2 * dark bar width (dark + light bar)
                        estimated_lp_width = avg_dark_width * 2
                        line_pair_widths.append(estimated_lp_width)
            
            # Calculate contrast using more robust method
            try:
                # Identify dark and light regions based on transitions
                dark_regions = []  # Between light-to-dark and dark-to-light transitions (dark bars)
                light_regions = []  # Between dark-to-light and light-to-dark transitions (light bars)
                
                for i in range(len(boundaries) - 1):
                    if i+1 < len(transition_types):
                        start, end = boundaries[i], boundaries[i+1]
                        if start < end and start >= 0 and end < len(profile):
                            if transition_types[i] == -1 and transition_types[i+1] == 1:
                                # This is a dark bar (from L→D to D→L)
                                dark_regions.append(profile[start:end])
                            elif transition_types[i] == 1 and transition_types[i+1] == -1:
                                # This is a light bar (from D→L to L→D)
                                light_regions.append(profile[start:end])
                
                # Calculate contrast if we have both light and dark regions
                if dark_regions and light_regions:
                    min_intensity = np.mean([np.mean(region) for region in dark_regions])
                    max_intensity = np.mean([np.mean(region) for region in light_regions])
                    contrast = (max_intensity - min_intensity) / (max_intensity + min_intensity)
                else:
                    # Fallback if segmentation didn't work as expected
                    contrast = (np.max(profile) - np.min(profile)) / (np.max(profile) + np.min(profile))
            except:
                # Fallback calculation
                if len(profile) > 0:
                    contrast = (np.max(profile) - np.min(profile)) / (np.max(profile) + np.min(profile))
        
        num_line_pairs = len(line_pair_widths)
        avg_line_pair_width = float(np.mean(line_pair_widths)) if line_pair_widths else 0.0
        lp_per_mm = self.usaf_target.lp_per_mm(group, element)
        
        results = {
            "group": group,
            "element": element,
            "lp_per_mm": float(lp_per_mm),
            "theoretical_lp_width_um": self.usaf_target.line_pair_width_microns(group, element),
            "num_line_pairs": num_line_pairs,
            "num_boundaries": len(boundaries),
            "boundaries": boundaries,
            "transition_types": transition_types,
            "line_pair_widths": line_pair_widths,
            "avg_line_pair_width": avg_line_pair_width,
            "contrast": float(contrast),
            "detection_threshold": self.threshold,
            "min_distance": self.min_distance,
            "derivative": derivative.tolist() if hasattr(derivative, 'tolist') else None,
            "profile": profile.tolist() if hasattr(profile, 'tolist') else None,
        }
        return results

# --- Streamlit UI Functions ---
def group_element_selectors(idx, default_group=2, default_element=2):
    keys = get_image_session_keys(idx)
    if keys['group'] not in st.session_state:
        st.session_state[keys['group']] = default_group
    if keys['element'] not in st.session_state:
        st.session_state[keys['element']] = default_element
    if keys['coordinates'] not in st.session_state:
        st.session_state[keys['coordinates']] = None
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

def display_roi_info(idx: int, image=None) -> Optional[Tuple[int, int, int, int]]:
    keys = get_image_session_keys(idx)
    coordinates_key = keys['coordinates']
    roi_valid_key = keys['roi_valid']
    if coordinates_key not in st.session_state or st.session_state[coordinates_key] is None:
        st.session_state[roi_valid_key] = False
        return None
    try:
        point1, point2 = st.session_state[coordinates_key]
        roi_x = min(point1[0], point2[0])
        roi_y = min(point1[1], point2[1])
        roi_width = abs(point2[0] - point1[0])
        roi_height = abs(point2[1] - point1[1])
        if roi_width <= 0 or roi_height <= 0:
            logger.warning(f"Invalid ROI dimensions: width={roi_width}, height={roi_height}")
            st.session_state[roi_valid_key] = False
            return None
        if image is not None:
            img_height, img_width = None, None
            if hasattr(image, 'shape'):
                if len(image.shape) > 1:
                    img_height, img_width = image.shape[0], image.shape[1]
            elif hasattr(image, 'size'):
                img_width, img_height = image.size
            if img_width is not None and img_height is not None:
                if (roi_x < 0 or roi_y < 0 or
                    roi_x + roi_width > img_width or
                    roi_y + roi_height > img_height):
                    logger.warning(f"ROI extends beyond image dimensions: "
                                   f"roi=({roi_x},{roi_y},{roi_width},{roi_height}), "
                                   f"image=({img_width},{img_height})")
                    st.session_state[roi_valid_key] = False
                    return None
        st.session_state[roi_valid_key] = True
        return (int(roi_x), int(roi_y), int(roi_width), int(roi_height))
    except Exception as e:
        logger.error(f"Error processing ROI: {e}")
        st.session_state[roi_valid_key] = False
        return None

def handle_image_selection(idx: int, image_file, image_to_display: np.ndarray, key: str ="usaf_image") -> bool:
    keys = get_image_session_keys(idx, image_file)
    coordinates_key = keys['coordinates']
    roi_valid_key = keys['roi_valid']
    if coordinates_key not in st.session_state:
        st.session_state[coordinates_key] = None
        st.session_state[roi_valid_key] = False
    unique_id = keys['coordinates'].split('_')[1]
    component_key = f"{key}_{unique_id}"
    coords_component_output = streamlit_image_coordinates(
        image_to_display, key=component_key, click_and_drag=True
    )
    roi_changed = False
    if coords_component_output is not None and \
       coords_component_output.get("x1") is not None and \
       coords_component_output.get("x2") is not None and \
       coords_component_output.get("y1") is not None and \
       coords_component_output.get("y2") is not None:
        point1 = (coords_component_output["x1"], coords_component_output["y1"])
        point2 = (coords_component_output["x2"], coords_component_output["y2"])
        if point1[0] != point2[0] and point1[1] != point2[1]:
            current_coordinates = st.session_state.get(coordinates_key)
            if current_coordinates != (point1, point2):
                st.session_state[coordinates_key] = (point1, point2)
                is_valid = point1[0] >= 0 and point1[1] >= 0 and point2[0] >= 0 and point2[1] >= 0 and \
                          abs(point2[0] - point1[0]) > 0 and abs(point2[1] - point1[1]) > 0
                st.session_state[roi_valid_key] = is_valid
                roi_changed = True
                logger.debug(f"ROI updated for image {idx}: {point1} to {point2}, valid: {is_valid}")
    return roi_changed

def display_welcome_screen():
    st.info("Please upload a USAF 1951 target image to begin analysis.")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/d/d6/1951usaf_test_target.jpg",
                 caption="Example USAF 1951 Target", use_container_width=True)
    with col2:
        st.subheader("USAF 1951 Target Format")
        st.latex(r"\text{resolution (lp/mm)} = 2^{\text{group} + (\text{element} - 1)/6}")
        st.latex(r"\text{Line Pair Width (μm)} = \frac{1000}{2 \times \text{resolution (lp/mm)}}")

def display_analysis_details(results):
    group = results.get('group')
    element = results.get('element')
    lp_width_um = st.session_state.usaf_target.line_pair_width_microns(group, element)
    lp_per_mm = results.get('lp_per_mm')
    avg_measured_lp_width_px = results.get('avg_line_pair_width', 0.0)
    st.markdown("""
    <div style='text-align: center;'>
    <b>A line pair</b> = one black bar + one white bar (one cycle)<br>
    <b>Line pair width</b> = distance from start of one black bar to the next
    </div>
    """, unsafe_allow_html=True)
    st.latex(rf"Group = {group}, \quad Element = {element}")
    st.latex(rf"\text{{Line Pairs per mm (Theoretical)}} = 2^{{{group} + ({element} - 1)/6}} = {lp_per_mm:.2f}")
    st.latex(rf"\text{{Line Pair Width (µm, Theoretical)}} = \frac{{1000}}{{{lp_per_mm:.2f} \text{{ lp/mm}}}} = {lp_width_um:.2f} \text{{ µm}}")
    st.latex(rf"\text{{Avg. Measured Line Pair Width (pixels)}} = {avg_measured_lp_width_px:.2f} \text{{ px}}")
    if avg_measured_lp_width_px > 0 and lp_width_um is not None:
        implied_pixel_size = lp_width_um / avg_measured_lp_width_px
        st.latex(rf"\text{{Implied Pixel Size (µm/pixel)}} = \frac{{{lp_width_um:.2f} \text{{ µm}}}}{{{avg_measured_lp_width_px:.2f} \text{{ px}}}} = {implied_pixel_size:.3f} \text{{ µm/pixel}}")
    else:
        st.latex(rf"\text{{Implied Pixel Size (µm/pixel)}} = \text{{N/A (requires measurement)}}")

def plot_intensity_profile(results, annotated_roi_img=None, group=None, element=None):
    if 'profile' not in results or results['profile'] is None or annotated_roi_img is None:
        return
    avg_profile = np.array(results['profile'])
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino', 'serif'],
        'mathtext.fontset': 'stix',
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9
    })
    fig = plt.figure(figsize=(10, 7), dpi=150, facecolor='white')
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.05)
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(annotated_roi_img, cmap='gray', aspect='auto', interpolation='bicubic')
    if group is not None and element is not None:
        ax_img.set_title(f"USAF Target: Group {group}, Element {element}", fontweight='normal', pad=10)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_ylabel("")
    ax_profile = fig.add_subplot(gs[1], sharex=ax_img)
    if 'individual_profiles' in results and results['individual_profiles'] is not None:
        individual_profiles = results['individual_profiles']
        n_rows = individual_profiles.shape[0]
        step = max(1, n_rows // 20)
        for i in range(0, n_rows, step):
            ax_profile.plot(individual_profiles[i],
                   color='#6b88b6', alpha=0.12, linewidth=0.7, zorder=1)
    avg_measured_lp_width_px = results.get('avg_line_pair_width', 0.0)
    mean_intensity_label = 'Mean Intensity'
    if avg_measured_lp_width_px > 0:
        mean_intensity_label += f"\n(Avg. LP Width: {avg_measured_lp_width_px:.1f} px)"
    ax_profile.plot(avg_profile,
           color='#2c3e50', linewidth=2.0, alpha=1.0,
           label=mean_intensity_label, zorder=2)
    
    # Get transition information
    boundaries = results.get('boundaries', [])
    transition_types = results.get('transition_types', [])
    
    # Use different colors and annotations for different transition types
    light_to_dark_color = '#FF4500'  # Orange-red
    dark_to_light_color = '#00BFFF'  # Deep sky blue
    shadow_effect = [PathEffects.withSimplePatchShadow(
        offset=(1.0, -1.0), shadow_rgbFace='black', alpha=0.6
    )]
    
    # Draw boundary lines with transition type colors
    if len(boundaries) >= 3 and len(transition_types) == len(boundaries):
        # First pass: Draw the boundary lines
        for i, (boundary, trans_type) in enumerate(zip(boundaries, transition_types)):
            if trans_type == -1:  # Light-to-dark
                line_color = light_to_dark_color
                line_style = '-'
                label = "Light → Dark" if i == 0 else ""
            else:  # Dark-to-light
                line_color = dark_to_light_color
                line_style = '-'
                label = "Dark → Light" if i == 0 or (i == 1 and transition_types[0] != 1) else ""
            
            # Draw on profile plot
            ax_profile.axvline(x=boundary, color=line_color, linestyle=line_style, 
                              alpha=0.7, linewidth=1.5, zorder=4, label=label)
            
            # Draw on image
            ax_img.axvline(x=boundary, color=line_color, linestyle='--', 
                          alpha=0.6, linewidth=0.8, zorder=3)
        
        # Second pass: Identify and highlight complete line pairs
        # A complete line pair starts with light-to-dark (red), includes a dark-to-light (blue),
        # and ends at the next light-to-dark (red)
        annotation_image_color = '#FFFF99'  # Yellow
        bracket_line_width = 1.2
        
        # Find all light-to-dark transitions (red lines)
        red_lines = [(i, boundary) for i, (boundary, trans_type) in enumerate(zip(boundaries, transition_types)) 
                    if trans_type == -1]
        
        # For each pair of consecutive red lines, check if there's a blue line in between
        line_pairs = []
        for j in range(len(red_lines) - 1):
            start_idx, start_pos = red_lines[j]
            end_idx, end_pos = red_lines[j + 1]
            
            # Check if there's at least one blue line between these red lines
            has_blue_between = False
            for i in range(start_idx + 1, end_idx):
                if i < len(transition_types) and transition_types[i] == 1:
                    has_blue_between = True
                    break
            
            if has_blue_between:
                # This is a complete line pair: red-blue-red
                width_px = end_pos - start_pos
                if width_px >= 5:  # Only show if the line pair is wide enough
                    line_pairs.append((start_pos, end_pos))
                    
                    # Draw a bracket showing the complete line pair
                    y_pos = annotated_roi_img.shape[0] * 0.15
                    tick_height = annotated_roi_img.shape[0] * 0.035
                    
                    # Draw horizontal line
                    ax_img.annotate("", xy=(start_pos, y_pos), xytext=(end_pos, y_pos),
                                  arrowprops=dict(arrowstyle="-", color=annotation_image_color, linewidth=1.2))
                    
                    # Draw tick marks on ends
                    ax_img.annotate("", xy=(start_pos, y_pos), xytext=(start_pos, y_pos-tick_height),
                                  arrowprops=dict(arrowstyle="-", color=annotation_image_color, linewidth=1.2))
                    ax_img.annotate("", xy=(end_pos, y_pos), xytext=(end_pos, y_pos-tick_height),
                                  arrowprops=dict(arrowstyle="-", color=annotation_image_color, linewidth=1.2))
                    
                    # Add "Line Pair" label
                    mid_point = (start_pos + end_pos) / 2
                    line_pair_text = ax_img.text(mid_point, y_pos-tick_height, "Line Pair",
                                               color=annotation_image_color, ha="center", va="bottom", 
                                               fontsize=15, fontweight='bold')
                    line_pair_text.set_path_effects(shadow_effect)
                    
                    # Add measurement
                    measurement = f"{int(round(width_px))} px"
                    measurement_text = ax_img.text(mid_point, y_pos+tick_height*1.5, measurement,
                                                 color=annotation_image_color, ha="center", va="top", 
                                                 fontsize=12, style='normal', fontweight='bold')
                    measurement_text.set_path_effects(shadow_effect)
        
        # If we couldn't find complete line pairs, try to estimate them
        if not line_pairs:
            # Look for red-blue sequences (start of a line pair)
            for i in range(len(boundaries) - 2):
                if (i+1 < len(transition_types) and 
                    transition_types[i] == -1 and  # L→D (red)
                    transition_types[i+1] == 1):   # D→L (blue)
                    
                    start_pos = boundaries[i]
                    
                    # Estimate a complete line pair width
                    # For USAF targets, dark and light bars should have equal widths
                    dark_width = boundaries[i+1] - start_pos
                    estimated_full_width = dark_width * 2  # One dark + one light bar
                    
                    # Estimate the end position of the line pair
                    end_pos = start_pos + estimated_full_width
                    
                    # Check if it extends beyond the image
                    if end_pos < len(avg_profile):
                        # Draw estimated line pair
                        y_pos = annotated_roi_img.shape[0] * 0.15
                        tick_height = annotated_roi_img.shape[0] * 0.035
                        
                        # Draw horizontal line
                        ax_img.annotate("", xy=(start_pos, y_pos), xytext=(end_pos, y_pos),
                                      arrowprops=dict(arrowstyle="-", color=annotation_image_color, linewidth=1.2))
                        
                        # Draw tick marks on ends
                        ax_img.annotate("", xy=(start_pos, y_pos), xytext=(start_pos, y_pos-tick_height),
                                      arrowprops=dict(arrowstyle="-", color=annotation_image_color, linewidth=1.2))
                        ax_img.annotate("", xy=(end_pos, y_pos), xytext=(end_pos, y_pos-tick_height),
                                      arrowprops=dict(arrowstyle="-", color=annotation_image_color, linewidth=1.2))
                        
                        # Add "Line Pair" label
                        mid_point = (start_pos + end_pos) / 2
                        line_pair_text = ax_img.text(mid_point, y_pos-tick_height, "Line Pair (est.)",
                                                  color=annotation_image_color, ha="center", va="bottom", 
                                                  fontsize=15, fontweight='bold')
                        line_pair_text.set_path_effects(shadow_effect)
                        
                        # Add measurement
                        measurement = f"{int(round(estimated_full_width))} px"
                        measurement_text = ax_img.text(mid_point, y_pos+tick_height*1.5, measurement,
                                                     color=annotation_image_color, ha="center", va="top", 
                                                     fontsize=12, style='normal', fontweight='bold')
                        measurement_text.set_path_effects(shadow_effect)
    
    ax_profile.set_xlabel("Position (pixels)")
    ax_profile.set_ylabel("Intensity (a.u.)")
    ax_profile.spines['top'].set_visible(False)
    ax_profile.spines['right'].set_visible(False)
    ax_img.spines['top'].set_visible(False)
    ax_img.spines['right'].set_visible(False)
    ax_img.spines['left'].set_visible(False)
    ax_img.spines['bottom'].set_visible(False)
    ax_profile.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    ax_profile.legend(loc='upper right', frameon=True, framealpha=0.85, fontsize=8, edgecolor='none')
    x_max = len(avg_profile) if len(avg_profile) > 0 else 100
    ax_profile.set_xlim(0, x_max)
    ax_img.set_xlim(0, x_max)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Check if we have the necessary variables defined before using them
    lp_width_um = None
    if group is not None and element is not None and 'usaf_target' in st.session_state:
        lp_width_um = st.session_state.usaf_target.line_pair_width_microns(group, element)
        
    if group is not None and element is not None and lp_width_um is not None:
        caption = f"""
        <div style='text-align:center; font-family: "Times New Roman", Times, serif;'>
            <p style='margin-bottom:0.3rem; font-size:1.05rem;'><b>Figure: Intensity Profile Analysis of USAF Target</b></p>
            <p style='margin-top:0; font-size:0.9rem; color:#333;'>
                Group {group}, Element {element} with theoretical line pair width of {lp_width_um:.2f} µm.
                Each line pair consists of one complete dark bar and one complete light bar (from red line to red line).
                <span style='color:#FF4500;'>Orange</span> lines indicate Light→Dark transitions and 
                <span style='color:#00BFFF;'>Blue</span> lines indicate Dark→Light transitions.
            </p>
        </div>
        """
    else:
        caption = """
        <div style='text-align:center; font-family: "Times New Roman", Times, serif;'>
            <p style='margin-bottom:0.3rem; font-size:1.05rem;'><b>Figure: Aligned Visual Analysis</b></p>
            <p style='margin-top:0; font-size:0.9rem; color:#333;'>
                Each line pair consists of one complete dark bar and one complete light bar (from red line to red line).
                <span style='color:#FF4500;'>Orange</span> lines indicate Light→Dark transitions and 
                <span style='color:#00BFFF;'>Blue</span> lines indicate Dark→Light transitions.
            </p>
        </div>
        """
    st.markdown(caption, unsafe_allow_html=True)

def analyze_image_with_analyzer(image_path, roi, group, element, analyzer=None):
    """
    Analyze an image with a custom ProfileAnalyzer for custom sensitivity settings.
    
    Args:
        image_path: Path to the image file
        roi: Region of interest tuple (x, y, width, height)
        group: USAF group number
        element: USAF element number
        analyzer: Optional custom ProfileAnalyzer with threshold and min_distance settings
    
    Returns:
        Tuple of (analysis results dict, intensity profile)
    """
    img_proc = ImageProcessor()
    img_proc.load_image(image_path)
    img_proc.select_roi(roi)
    profile = img_proc.get_line_profile()
    individual_profiles = img_proc.get_individual_profiles()
    
    # Use provided analyzer or create a default one
    if analyzer is None:
        analyzer = ProfileAnalyzer(st.session_state.usaf_target)
    
    results = analyzer.analyze_profile(
        profile=profile,
        group=group,
        element=element
    )
    results['individual_profiles'] = individual_profiles
    return results, profile

def analyze_image(image_path, roi, group, element):
    """Original analyze_image function kept for backwards compatibility"""
    return analyze_image_with_analyzer(image_path, roi, group, element)

def analyze_and_display_image(idx, uploaded_file):
    filename = os.path.basename(uploaded_file) if isinstance(uploaded_file, str) else (
        uploaded_file.name if hasattr(uploaded_file, 'name') else f"Image {idx+1}"
    )
    keys = get_image_session_keys(idx, uploaded_file)
    st.session_state[keys['image_name']] = filename
    
    # Initialize rotation angle if not already in session state
    rotation_key = f'rotation_{get_unique_id_for_image(uploaded_file)}'
    if rotation_key not in st.session_state:
        st.session_state[rotation_key] = 0.0
    
    # Initialize edge detection sensitivity parameters
    sensitivity_key = f'sensitivity_{get_unique_id_for_image(uploaded_file)}'
    min_distance_key = f'min_distance_{get_unique_id_for_image(uploaded_file)}'
    if sensitivity_key not in st.session_state:
        st.session_state[sensitivity_key] = 20  # Default threshold value
    if min_distance_key not in st.session_state:
        st.session_state[min_distance_key] = 3  # Default minimum distance
        
    with st.expander(f"Image {idx+1}: {filename}", expanded=(idx == 0)):
        image, temp_path = process_uploaded_file(uploaded_file)
        if image is None:
            st.error(f"Failed to load image: {filename}")
            return
        st.session_state[keys['image_path']] = temp_path
        
        # Add all adjustments in a single expander
        with st.expander("Image Adjustments", expanded=True):
            # Add rotation control
            rotation_angle = st.slider(
                "Rotation angle (degrees)", 
                min_value=-45.0, 
                max_value=45.0, 
                value=st.session_state[rotation_key],
                step=0.5,
                key=rotation_key,
                help="Rotate the image to align line pairs horizontally"
            )
            
            # Add sensitivity controls
            st.markdown("#### Edge Detection Settings")
            
            sensitivity = st.slider(
                "Edge detection threshold", 
                min_value=5, 
                max_value=50, 
                value=st.session_state[sensitivity_key],
                step=1,
                key=sensitivity_key,
                help="Lower values detect more edges (more sensitive). Higher values detect only stronger edges."
            )
            
            min_distance = st.slider(
                "Minimum distance between edges", 
                min_value=1, 
                max_value=20, 
                value=st.session_state[min_distance_key],
                step=1,
                key=min_distance_key,
                help="Minimum distance in pixels between detected edges. Increase to avoid detecting noise."
            )
        
        # Apply rotation if needed
        if rotation_angle != 0:
            # Get image center
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            
            # Get rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            
            # Apply affine transform
            rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                          flags=cv2.INTER_LINEAR, 
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0, 0, 0))
            display_image = rotated_image
        else:
            display_image = image
        
        pil_img = Image.fromarray(display_image)
        draw = ImageDraw.Draw(pil_img)
        current_coords = st.session_state.get(keys['coordinates'])
        if current_coords:
            p1, p2 = current_coords
            coords = (min(p1[0], p2[0]), min(p1[1], p2[1]),
                      max(p1[0], p2[0]), max(p1[1], p2[1]))
            # Use a different color for each image's ROI based on index
            roi_valid = st.session_state.get(keys['roi_valid'], False)
            if roi_valid:
                # Use a color from the ROI_COLORS list for valid ROIs
                color_idx = idx % len(ROI_COLORS)
                outline_color = ROI_COLORS[color_idx]
            else:
                # Use red for invalid ROIs
                outline_color = INVALID_ROI_COLOR
            draw.rectangle(coords, outline=outline_color, width=3)
        roi_col, analysis_col = st.columns([1, 1])
        with roi_col:
            st.markdown("#### Select ROI on Image")
            roi_changed = handle_image_selection(
                idx, uploaded_file, pil_img, key=f"usaf_image_{idx}"
            )
            roi_valid = st.session_state.get(keys['roi_valid'], False)
            if current_coords is not None:
                if roi_valid:
                    st.success("ROI selection is valid")
                else:
                    st.error("ROI selection is not valid. Please select a valid region.")
        with analysis_col:
            settings_row = st.container()
            with settings_row:
                st.markdown("#### Analysis Settings")
                group, element = group_element_selectors(idx)
            analysis_results_for_plot = st.session_state.get(keys['analysis_results'])
            if analysis_results_for_plot:
                # Get the ROI from the rotated image if rotation is applied
                if rotation_angle != 0:
                    roi_for_display = extract_roi_image(display_image, st.session_state.get(keys['analyzed_roi']))
                else:
                    roi_for_display = extract_roi_image(image, st.session_state.get(keys['analyzed_roi']))
                    
                plot_intensity_profile(
                    analysis_results_for_plot,
                    annotated_roi_img=roi_for_display,
                    group=st.session_state.get(keys['last_group']),
                    element=st.session_state.get(keys['last_element'])
                )
            else:
                current_coords_for_preview = st.session_state.get(keys['coordinates'])
                if current_coords_for_preview:
                    try:
                        p1_preview, p2_preview = current_coords_for_preview
                        coords_preview = (min(p1_preview[0], p2_preview[0]), min(p1_preview[1], p2_preview[1]),
                                          max(p1_preview[0], p2_preview[0]), max(p1_preview[1], p2_preview[1]))
                        roi_img_preview = pil_img.crop(coords_preview)
                        st.image(roi_img_preview, caption="Selected ROI Preview", use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not display ROI preview: {str(e)}")
                else:
                    st.info("Select an ROI to view analysis.")
        keys = get_image_session_keys(idx, uploaded_file)
        current_selected_roi_tuple = display_roi_info(idx, display_image)  # Use rotated image for ROI validation
        group_for_trigger = st.session_state.get(keys['group'])
        element_for_trigger = st.session_state.get(keys['element'])
        roi_is_valid = st.session_state.get(keys['roi_valid'], False)
        
        # Check if rotation or sensitivity settings changed
        rotation_changed = rotation_angle != st.session_state.get(f'last_rotation_{get_unique_id_for_image(uploaded_file)}', 0.0)
        sensitivity_changed = (sensitivity != st.session_state.get(f'last_sensitivity_{get_unique_id_for_image(uploaded_file)}', 20) or
                             min_distance != st.session_state.get(f'last_min_distance_{get_unique_id_for_image(uploaded_file)}', 3))
        
        st.session_state[f'last_rotation_{get_unique_id_for_image(uploaded_file)}'] = rotation_angle
        st.session_state[f'last_sensitivity_{get_unique_id_for_image(uploaded_file)}'] = sensitivity
        st.session_state[f'last_min_distance_{get_unique_id_for_image(uploaded_file)}'] = min_distance
        
        should_analyze = (
            current_selected_roi_tuple is not None and
            roi_is_valid and
            group_for_trigger is not None and
            element_for_trigger is not None and
            (current_selected_roi_tuple != st.session_state.get(keys['analyzed_roi']) or
             group_for_trigger != st.session_state.get(keys['last_group']) or
             element_for_trigger != st.session_state.get(keys['last_element']) or
             rotation_changed or
             sensitivity_changed)
        )
        if should_analyze:
            with st.spinner("Analyzing image..."):
                try:
                    # Create a temporary file for the rotated image if needed
                    if rotation_angle != 0:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_rot_file:
                            cv2.imwrite(temp_rot_file.name, cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))
                            rotated_temp_path = temp_rot_file.name
                            
                        # Create analyzer with custom sensitivity
                        custom_analyzer = ProfileAnalyzer(
                            st.session_state.usaf_target, 
                            threshold=sensitivity,
                            min_distance=min_distance
                        )
                        
                        # Use custom analyzer with the analyze_image function
                        results_data, _ = analyze_image_with_analyzer(
                            rotated_temp_path, 
                            current_selected_roi_tuple, 
                            group_for_trigger, 
                            element_for_trigger,
                            custom_analyzer
                        )
                        
                        # Clean up the temporary file
                        os.remove(rotated_temp_path)
                    else:
                        # Create analyzer with custom sensitivity
                        custom_analyzer = ProfileAnalyzer(
                            st.session_state.usaf_target, 
                            threshold=sensitivity,
                            min_distance=min_distance
                        )
                        
                        # Use custom analyzer with the analyze_image function
                        results_data, _ = analyze_image_with_analyzer(
                            temp_path, 
                            current_selected_roi_tuple, 
                            group_for_trigger, 
                            element_for_trigger,
                            custom_analyzer
                        )
                    
                    st.session_state[keys['analyzed_roi']] = current_selected_roi_tuple
                    st.session_state[keys['analysis_results']] = results_data
                    st.session_state[keys['last_group']] = group_for_trigger
                    st.session_state[keys['last_element']] = element_for_trigger
                    if results_data:
                        st.rerun()
                except Exception as e:
                    logger.error(f"Analysis failed: {e}")
                    st.error(f"Analysis failed: {str(e)}")
        analysis_results_for_details = st.session_state.get(keys['analysis_results'])
        if analysis_results_for_details:
            st.markdown("<hr style='margin-top: 1.5rem; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: center; margin-top: 1.5rem;'>Analysis Details</h4>", unsafe_allow_html=True)
            display_analysis_details(analysis_results_for_details)

def run_streamlit_app():
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
        with st.sidebar:
            st.header("Controls")
            new_uploaded_files = st.file_uploader(
                "Upload USAF target image(s)",
                type=["jpg", "jpeg", "png", "tif", "tiff"],
                accept_multiple_files=True,
                help="Select one or more images containing a USAF 1951 resolution target"
            )
            if new_uploaded_files:
                for file in new_uploaded_files:
                    file_names = [f.name if hasattr(f, 'name') else os.path.basename(f)
                                  for f in st.session_state.uploaded_files_list]
                    new_file_name = file.name if hasattr(file, 'name') else os.path.basename(file)
                    if new_file_name not in file_names:
                        st.session_state.uploaded_files_list.append(file)
                        st.success(f"Added: {new_file_name}")
            
            # Add information about the features in the sidebar
            st.markdown("---")
            st.markdown("### Analysis Tips")
            st.info("""
            **Image Rotation**: Each image has a rotation slider to help align line pairs horizontally for better analysis. 
            Use this feature when your USAF target is tilted.
            """)
            
            st.info("""
            **Edge Detection Sensitivity**: Fine-tune the detection sensitivity for each image using the 'Edge Detection Sensitivity' 
            expander. Lower threshold values (more sensitive) detect more edges, while higher values detect only stronger edges. 
            Adjust the minimum distance to filter out noise.
            """)
            
            default_image_path = load_default_image()
            if not st.session_state.uploaded_files_list and default_image_path and not st.session_state.default_image_added:
                st.session_state.uploaded_files_list.append(default_image_path)
                st.session_state.default_image_added = True
                st.info(f"Using default image: {os.path.basename(default_image_path)}")
            if st.button("Clear All Images"):
                st.session_state.uploaded_files_list = []
                st.session_state.default_image_added = False
                st.session_state.image_index_to_id = {}
                st.success("All images cleared")
                for key in list(st.session_state.keys()):
                    if any(key.startswith(prefix) for prefix in SESSION_STATE_PREFIXES):
                        del st.session_state[key]
                    # Also clear rotation and sensitivity related keys
                    if any(key.startswith(prefix) for prefix in ['rotation_', 'last_rotation_', 'sensitivity_', 'last_sensitivity_', 'min_distance_', 'last_min_distance_']):
                        del st.session_state[key]
                st.rerun()
        main_container = st.container()
        with main_container:
            if st.session_state.uploaded_files_list:
                st.info(f"Currently analyzing {len(st.session_state.uploaded_files_list)} image(s)")
                for idx, uploaded_file in enumerate(st.session_state.uploaded_files_list):
                    analyze_and_display_image(idx, uploaded_file)
            else:
                display_welcome_screen()
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("For detailed error information, set DEBUG=1 in environment variables.")

if __name__ == "__main__":
    run_streamlit_app()
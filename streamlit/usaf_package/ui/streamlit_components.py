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
from ..main import run_analysis
from ..utils.image_tools import (
    process_uploaded_file, create_dummy_file_object, extract_roi_image
)
from ..utils.streamlit_helpers import (
    prepare_config_from_parameters, save_analysis_results
)

# Default image path
DEFAULT_IMAGE_PATH = "/Users/aaron/Library/CloudStorage/Box-Box/FOIL/Aaron/2025-05-12/airforcetarget_images/AF_2_2_00001.png"

@st.cache_data
def load_default_image():
    """Cache the default image loading."""
    if os.path.exists(DEFAULT_IMAGE_PATH):
        return create_dummy_file_object(DEFAULT_IMAGE_PATH)
    return None

def setup_sidebar():
    """Setup the sidebar with a clean, minimal interface."""
    with st.sidebar:
        st.title("Settings")
        
        # File uploader with improved styling
        uploaded_file = st.file_uploader(
        "Upload USAF target image",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="Select an image containing a USAF 1951 resolution target"
    )
    
        # Analysis parameters in a clean layout
        st.subheader("Target Parameters")
        
        # Use columns for related inputs
        col1, col2 = st.columns(2)
        with col1:
            group = st.number_input(
                "Group",
                value=2,
                min_value=-2,
                max_value=9,
                help="MIL-STD-150A target group number (-2 to 9)"
            )
        with col2:
            element = st.number_input(
                "Element",
                value=2,
                min_value=1,
                max_value=6,
                help="MIL-STD-150A target element number (1 to 6)"
            )
        
        # Profile extraction method
        profile_method = st.radio(
            "Profile Extraction Method",
            ["ROI Average", "Single Line"],
            index=0,
            help="ROI Average uses all rows/columns. Single Line uses one specific line."
        )
        
        # Show position slider only when Single Line is selected
        profile_position = None
        if profile_method == "Single Line":
            profile_position = st.slider(
                "Line Position (%)",
                min_value=0,
                max_value=100,
                value=50,
                step=1,
                help="Position of the scan line within the ROI (percentage)"
            )
        
        # Profile orientation
        orientation = st.radio(
            "Profile Orientation",
            ["Horizontal", "Vertical"],
            index=0,
            help="Horizontal profile analyzes line pairs along width. Vertical along height."
        )
        
        # Advanced settings in an expander
        with st.expander("Advanced Detection Settings"):
            col1, col2 = st.columns(2)
            with col1:
                sensitivity = st.slider(
                    "Sensitivity",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.2,
                    step=0.05,
                    help="Lower values = more sensitive detection"
                )
            with col2:
                min_distance = st.slider(
                    "Min Distance",
                    min_value=5,
                    max_value=30,
                    value=15,
                    step=1,
                    help="Minimum pixels between detected features"
                )
            smooth_profile = st.checkbox(
                "Smooth Profile",
                value=True,
                help="Apply smoothing to reduce noise"
            )
        
        # ROI controls
        st.subheader("ROI Selection")
        st.info("Click and drag on the image to select the region of interest")
        reset_roi = st.button("Reset Selection", use_container_width=True)
        
        # Help section in an expander
        with st.expander("About MIL-STD-150A"):
            st.markdown("""
            **Target Arrangement:**
            - Six groups in three layers
            - Largest groups on outer sides
            - Smaller layers toward center
            - Each group has six elements
            - Resolution doubles with each element
            """)
    
    return {
        "uploaded_file": uploaded_file,
        "reset_roi": reset_roi,
        "group": group,
        "element": element,
        "sensitivity": sensitivity,
        "min_distance": min_distance,
        "profile_method": profile_method,
        "profile_position": profile_position,
        "orientation": orientation.lower(),
        "smooth_profile": smooth_profile
    }

def initialize_session_state():
    """Initialize or reset Streamlit session state variables."""
    # ROI selection state
    if 'coordinates' not in st.session_state:
        st.session_state.coordinates = None
    # Analysis state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    # ROI change detection
    if 'analyzed_roi' not in st.session_state:
        st.session_state.analyzed_roi = None

def reset_session_state():
    """Reset session state variables."""
    st.session_state.coordinates = None
    st.session_state.analysis_results = None
    st.session_state.analyzed_roi = None

def handle_image_selection(image, key="usaf_image"):
    """
    Handle interactive ROI selection on an image.
    
    Args:
        image: The image to display
        key: Unique key for the streamlit component
        
    Returns:
        tuple: ROI coordinates as (x, y, width, height) or None
    """
    # Use streamlit_image_coordinates with click_and_drag feature
    coords = streamlit_image_coordinates(
        image,
        key=key,
        click_and_drag=True
    )
    
    # Process the coordinates
    if coords is not None and coords.get("x1") is not None:
        point1 = (coords["x1"], coords["y1"])
        point2 = (coords["x2"], coords["y2"])
        
        # Only update if this is a meaningful rectangle (not just a click)
        if (point1[0] != point2[0] and 
            point1[1] != point2[1] and 
            st.session_state.coordinates != (point1, point2)):
            
            st.session_state.coordinates = (point1, point2)
            return True  # Signal that coordinates changed
    
    return False  # No change

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
    
    # Calculate ROI parameters
    roi_x = min(point1[0], point2[0])
    roi_y = min(point1[1], point2[1])
    roi_width = abs(point2[0] - point1[0])
    roi_height = abs(point2[1] - point1[1])
    
    # Create a copy of the image with ROI rectangle
    img_with_roi = image.copy()
    
    # Draw rectangle
    cv2.rectangle(
        img_with_roi,
        (int(roi_x), int(roi_y)),
        (int(roi_x + roi_width), int(roi_y + roi_height)),
        (255, 0, 0),
        2
    )
    
    # Display image with ROI
    st.image(img_with_roi, caption="Selected Region", use_container_width=True)
    
    # Show zoomed ROI
    roi_image = image[int(roi_y):int(roi_y + roi_height), 
                     int(roi_x):int(roi_x + roi_width)]
    st.image(roi_image, caption="Zoomed View", use_container_width=True)
    
    return (int(roi_x), int(roi_y), int(roi_width), int(roi_height))

def display_welcome_screen():
    """Display the welcome screen with instructions when no image is loaded."""
    st.info("Please upload a USAF 1951 target image to begin analysis.")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/USAF-1951.svg/1024px-USAF-1951.svg.png", 
                caption="Example USAF 1951 Target", use_container_width=True)
    
    with col2:
        st.subheader("MIL-STD-150A Target Format")
        
        # Correct standard resolution formula
        st.latex(r"""
        \text{resolution (lp/mm)} = 2^{\text{group} + (\text{element} - 1)/6}
        """)
        
        # Line pair width formula
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

def display_analysis_results(results, roi_image):
    """Display analysis results in a clean, organized layout."""
    if 'error' in results:
        st.error(f"Analysis failed: {results['error']}")
        return
    
    # Key metrics in a clean layout
    st.header("Analysis Results")
    
    # Method and orientation info
    profile_method = results.get('profile_method', 'ROI Average')
    orientation = results.get('orientation', 'horizontal')
    position = results.get('profile_position')
    
    # Method description
    method_info = f"{orientation.capitalize()} profile using "
    if profile_method == "ROI Average":
        if orientation == "horizontal":
            method_info += "column averages (averaged across all rows)"
        else:
            method_info += "row averages (averaged across all columns)"
    else:
        if position is not None:
            if orientation == "horizontal":
                method_info += f"single line at {position}% height"
            else:
                method_info += f"single line at {position}% width"
    
    st.info(method_info)
    
    # Top row: Key metrics
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric(
            "Resolution",
            f"{results['resolution_microns']:.2f} µm",
            help="Width of one line pair"
        )
    with metric_cols[1]:
        st.metric(
            "Contrast",
            f"{results['contrast']:.2f}",
            help="Michelson contrast ratio"
        )
    with metric_cols[2]:
        st.metric(
            "MTF",
            f"{results['mtf']:.2f}",
            help="Modulation Transfer Function"
        )
    
    # Main content in tabs
    tab1, tab2, tab3 = st.tabs(["Profile Analysis", "Reference Table", "Technical Details"])
    
    with tab1:
        # Profile visualization
        fig = create_profile_visualization(roi_image, results['raw_profile'], results)
        st.pyplot(fig)
        
        # Line pair information with LaTeX
        st.subheader("Line Pair Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Number of valleys detected: {results['num_valleys']}")
            st.write(f"Number of peaks detected: {results['num_peaks']}")
            st.write(f"Number of line pairs (cycles): {results['num_line_pairs']}")
            st.latex(r"""
            \text{A line pair = one black bar + one white bar (one cycle)}
            """)
        with col2:
            st.write(f"Line pair width: {results['line_pair_width_microns']:.2f} µm")
            st.latex(r"""
            \text{Line Pair Width (μm)} = \frac{1000}{2^{\text{group} + (\text{element} - 1)/6}}
            """)
    
    with tab2:
        # Display the USAF 1951 resolution target reference table
        st.subheader("USAF 1951 Resolution Target Reference Table")
        st.markdown("""
        This table shows the standard values for line pairs per millimeter based on the group and element:
        """)
        
        # Create a pandas DataFrame with the standard values
        data = {
            "Group -2": [0.250, 0.281, 0.315, 0.354, 0.397, 0.445],
            "Group -1": [0.500, 0.561, 0.630, 0.707, 0.794, 0.891],
            "Group 0": [1.00, 1.12, 1.26, 1.41, 1.59, 1.78],
            "Group 1": [2.00, 2.24, 2.52, 2.83, 3.17, 3.56],
            "Group 2": [4.00, 4.49, 5.04, 5.66, 6.35, 7.13],
            "Group 3": [8.00, 8.98, 10.08, 11.31, 12.70, 14.25],
            "Group 4": [16.00, 17.96, 20.16, 22.63, 25.40, 28.51],
            "Group 5": [32.0, 35.9, 40.3, 45.3, 50.8, 57.0],
            "Group 6": [64.0, 71.8, 80.6, 90.5, 101.6, 114.0],
            "Group 7": [128.0, 143.7, 161.3, 181.0, 203.2, 228.1],
            "Group 8": [256.0, 287.4, 322.5, 362.0, 406.4, 456.1],
            "Group 9": [512.0, 574.7, 645.1, 724.1, 812.7, 912.3]
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add element numbers as the index
        df.index = [f"Element {i+1}" for i in range(6)]
        
        # Highlight the current group and element
        group = results.get('group', 0)
        element = results.get('element', 0)
        
        # Display the table
        st.dataframe(
            df,
            column_config={f"Group {group}": st.column_config.Column(
                f"Group {group}",
                help="Current selection",
                background="rgba(255, 255, 0, 0.3)",
            )},
            hide_index=False
        )
        
        # Calculate and display the theoretical values for the current selection
        st.subheader("Theoretical Values for Selected Group/Element")
        
        # Get the line pairs per mm value from the table
        if -2 <= group <= 9 and 1 <= element <= 6:
            group_key = f"Group {group}"
            if group_key in data:
                lp_per_mm = data[group_key][element-1]
                line_width_um = 1000 / (2 * lp_per_mm)
                st.write(f"Group {group}, Element {element}:")
                st.write(f"Line pairs per mm: {lp_per_mm:.2f} lp/mm")
                st.write(f"Line pair width: {1000/lp_per_mm:.2f} µm")
                st.write(f"Line width (single bar): {line_width_um:.2f} µm")
    
    with tab3:
        # Technical details in a clean format
        st.json(results)

def create_profile_visualization(roi_image, profile, results):
    """
    Create a matplotlib figure with the ROI image and intensity profile aligned to share the same x-axis.
    Shows both peaks and valleys, with line pairs as cycles (black+white bars).
    
    Args:
        roi_image: The ROI image
        profile: The intensity profile
        results: Analysis results dictionary with peak/valley positions
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Get image dimensions
    height, width = roi_image.shape[0], roi_image.shape[1]
    
    # Get info from results
    valley_positions = results.get('valley_positions', [])
    peak_positions = results.get('peak_positions', [])
    profile_method = results.get('profile_method', 'ROI Average')
    orientation = results.get('orientation', 'horizontal')
    position = results.get('profile_position')
    
    # Create figure with two subplots with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                  sharex=True,
                                  gridspec_kw={'height_ratios': [3, 1.5]})
    
    # Display the ROI image in the first subplot
    ax1.imshow(roi_image, cmap='gray', aspect='auto', extent=[0, width, height, 0])
    
    # Show scan line if using single line method
    if profile_method == "Single Line" and position is not None:
        if orientation == "horizontal":
            # For horizontal profile, line is at a specific y position
            line_pos = int((position / 100.0) * height)
            ax1.axhline(y=line_pos, color='yellow', linewidth=2, linestyle='--')
            ax1.set_title(f"ROI Image with Scan Line at {position}% Height")
        else:
            # For vertical profile, line is at a specific x position
            line_pos = int((position / 100.0) * width)
            ax1.axvline(x=line_pos, color='yellow', linewidth=2, linestyle='--')
            ax1.set_title(f"ROI Image with Scan Line at {position}% Width")
    else:
        ax1.set_title(f"ROI Image with {'Horizontal' if orientation == 'horizontal' else 'Vertical'} Line Pairs")
    
    # Remove x-axis labels from top plot since they're shared
    ax1.xaxis.set_tick_params(labelbottom=False)
    ax1.set_ylabel("Y position (pixels)")
    
    # Filter valley positions to ensure they're valid
    valid_valleys = [pos for pos in valley_positions if 0 <= pos < width]
    valid_peaks = [pos for pos in peak_positions if 0 <= pos < width]
    
    # Draw vertical line at each valley position on both plots
    for pos in valid_valleys:
        ax1.axvline(x=pos, color='red', linestyle='--', linewidth=1)
        # Note: We'll add lines to ax2 after plotting the profile
    
    # Draw vertical line at each peak position (using a different color)
    for pos in valid_peaks:
        ax1.axvline(x=pos, color='green', linestyle=':', linewidth=1)
    
    # Plot the profile in the second subplot with exact alignment to image x-coordinates
    x = np.arange(len(profile))
    ax2.plot(x, profile, linewidth=1.5, color='blue')
    ax2.set_xlim(0, width)
    
    # Set the plot limits to better match the data
    profile_min = min(profile) if len(profile) > 0 else 0
    profile_max = max(profile) if len(profile) > 0 else 100
    # Add 15% padding to the y-axis
    y_padding = (profile_max - profile_min) * 0.15
    ax2.set_ylim(profile_min - y_padding, profile_max + y_padding)
    
    # Now add valley lines to profile plot
    for pos in valid_valleys:
        if pos < len(profile):
            ax2.axvline(x=pos, color='red', linestyle='--', linewidth=1)
    
    # Add peak lines to profile plot
    for pos in valid_peaks:
        if pos < len(profile):
            ax2.axvline(x=pos, color='green', linestyle=':', linewidth=1)
    
    # Mark valleys on the profile
    if valid_valleys:
        valley_y = [profile[pos] for pos in valid_valleys if pos < len(profile)]
        valley_x = [pos for pos in valid_valleys if pos < len(profile)]
        
        # Plot valley points with higher visibility
        ax2.scatter(valley_x, valley_y, color='red', s=80, marker='v', label='Valleys (dark bars)', zorder=5)
        
        # Add position labels for valleys with improved visibility
        for x, y in zip(valley_x, valley_y):
            # White background for text
            ax2.annotate(f"{int(x)}", 
                       xy=(x, y), 
                       xytext=(0, -20), 
                       textcoords="offset points",
                       ha='center',
                       fontsize=9,
                       fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", 
                                fc='white', 
                                ec='black',
                                alpha=0.8))
        
    # Mark peaks on the profile
    if valid_peaks:
        peak_y = [profile[pos] for pos in valid_peaks if pos < len(profile)]
        peak_x = [pos for pos in valid_peaks if pos < len(profile)]
        
        # Plot peak points
        ax2.scatter(peak_x, peak_y, color='green', s=80, marker='^', label='Peaks (bright bars)', zorder=5)
        
        # Add position labels for peaks
        for x, y in zip(peak_x, peak_y):
            ax2.annotate(f"{int(x)}", 
                       xy=(x, y), 
                       xytext=(0, 20), 
                       textcoords="offset points",
                       ha='center',
                       fontsize=9,
                       fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", 
                                fc='white', 
                                ec='black',
                                alpha=0.8))
    
    # Identify line pairs (cycles of black+white)
    if len(valid_valleys) >= 1 and len(valid_peaks) >= 1:
        # Arrange all points (peaks and valleys) in order
        all_points = [(x, 'v') for x in valley_x] + [(x, 'p') for x in peak_x]
        all_points.sort(key=lambda point: point[0])
        
        # Find sequences that contain both a valley and peak
        line_pairs = []
        for i in range(len(all_points) - 1):
            point1 = all_points[i]
            point2 = all_points[i + 1]
            # If points are different types (one valley, one peak), they form a line pair
            if point1[1] != point2[1]:
                start_pos = point1[0]
                end_pos = point2[0]
                mid_pos = (start_pos + end_pos) / 2
                line_pairs.append((start_pos, end_pos, mid_pos))
            
        # Draw and label each line pair
        for i, (start_pos, end_pos, mid_pos) in enumerate(line_pairs):
            # Draw bracket marking the line pair
                lp_y = profile_min - y_padding * 0.5
                lp_height = y_padding * 0.2
                
                # Draw bracket
                ax2.plot([start_pos, end_pos], [lp_y, lp_y], 'g-', linewidth=2)
                ax2.plot([start_pos, start_pos], [lp_y-lp_height, lp_y+lp_height], 'g-', linewidth=2)
                ax2.plot([end_pos, end_pos], [lp_y-lp_height, lp_y+lp_height], 'g-', linewidth=2)
            
            # Label each line pair
                ax2.annotate(f"LP {i+1}", 
                           xy=(mid_pos, lp_y - lp_height/2),
                       ha='center',
                           color='g',
                       fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", 
                                    fc='white', 
                                    ec='black',
                                    alpha=0.8))
    
    # Explain the x and y axes
    profile_label = "Intensity Profile"
    if orientation == "horizontal":
        profile_label += " (Column Average)" if profile_method == "ROI Average" else f" (Row {position}%)"
    else:
        profile_label += " (Row Average)" if profile_method == "ROI Average" else f" (Column {position}%)"
    
    ax2.set_title(profile_label)
    ax2.set_xlabel("Position (pixels)")
    ax2.set_ylabel("Intensity Value")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Clear explanation of the intensity measurement
    if profile_method == "ROI Average":
        if orientation == "horizontal":
            info_text = "Each point represents the\naverage intensity of one column"
        else:
            info_text = "Each point represents the\naverage intensity of one row"
    else:
        if orientation == "horizontal":
            info_text = "Each point represents the\nintensity of one pixel in row at " + str(position) + "%"
        else:
            info_text = "Each point represents the\nintensity of one pixel in column at " + str(position) + "%"
    
    ax2.annotate(info_text, 
                xy=(width//2, profile_max * 0.95),
                xytext=(width//2, profile_max * 0.95),
                ha='center',
                bbox=dict(boxstyle="round,pad=0.3", 
                         fc='white', 
                         alpha=0.8))
    
    # Correct explanation of line pairs
    ax2.annotate("Line Pair = One Cycle:\nOne dark bar (valley) +\nOne bright space (peak)", 
                xy=(width * 0.1, profile_min + (profile_max-profile_min)*0.2),
                xytext=(width * 0.1, profile_min + (profile_max-profile_min)*0.2),
                ha='left',
                bbox=dict(boxstyle="round,pad=0.3", 
                         fc='white', 
                         alpha=0.8))
    
    # Tighten the layout and add space between subplots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    return fig 

def run_streamlit_app():
    """Main Streamlit application function with improved UI and correct line pair analysis."""
    try:
        # Setup page configuration
        st.set_page_config(
            page_title="USAF 1951 Resolution Target Analyzer",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        initialize_session_state()
        
        # Setup sidebar and get parameters
        params = setup_sidebar()
        
        # Handle reset button
        if params.get("reset_roi"):
            reset_session_state()
            st.rerun()
        
        # Main content area
        st.title("USAF 1951 Resolution Target Analyzer")
        
        # Try to load default image if no file uploaded
        uploaded_file = params.get("uploaded_file")
        if uploaded_file is None:
            uploaded_file = load_default_image()
            if uploaded_file:
                st.sidebar.success(f"Loaded default image: {os.path.basename(DEFAULT_IMAGE_PATH)}")
        
        # Process image if available
        if uploaded_file:
            # Process the uploaded image
            image, temp_path = process_uploaded_file(uploaded_file)
            
            if image is not None:
                # Main content in columns
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Image display with ROI selection
                    st.subheader("Image")
                    if handle_image_selection(image):
                        st.rerun()
                
                with col2:
                    # ROI information
                    st.subheader("Selected ROI")
                    roi = display_roi_info(image)
                
                # Run analysis if we have a valid ROI
                if roi and roi != st.session_state.analyzed_roi:
                    with st.spinner("Analyzing image..."):
                        # Prepare configuration and run analysis
                        config = prepare_config_from_parameters(params)
                        # Add profile extraction settings to config
                        config.update({
                            "profile_method": params.get("profile_method", "ROI Average"),
                            "smooth_profile": params.get("smooth_profile", True)
                        })
                        
                        from ..processing.image_processor import ImageProcessor
                        img_proc = ImageProcessor(config)
                        img_proc.load_image(temp_path)
                        img_proc.select_roi(roi)
                        
                        # Get profile with selected orientation and method
                        profile_position = params.get("profile_position")
                        if profile_position is not None:
                            # Convert percentage to pixel position
                            if params.get("orientation") == "horizontal":
                                # For horizontal profile, position is y-coordinate
                                roi_height = roi[3]  # roi = (x, y, width, height)
                                line_position = int((profile_position / 100.0) * roi_height)
                            else:
                                # For vertical profile, position is x-coordinate
                                roi_width = roi[2]
                                line_position = int((profile_position / 100.0) * roi_width)
                            
                            profile = img_proc.get_line_profile(
                                params.get("orientation"),
                                line_position=line_position
                            )
                        else:
                            profile = img_proc.get_line_profile(params.get("orientation"))
                        
                        # Use new peak/valley detection
                        peaks, valleys = img_proc.detect_peaks_and_valleys(
                            profile, 
                            min_distance=params.get("min_distance", 15),
                            prominence=params.get("sensitivity", 0.2)
                        )
                        
                        from ..processing.profile_analyzer import ProfileAnalyzer
                        analyzer = ProfileAnalyzer()
                        results = analyzer.analyze_profile(
                            profile=profile,
                            peak_positions=peaks,
                            valley_positions=valleys,
                            group=params.get("group", 2),
                            element=params.get("element", 3)
                        )
                        
                        # Add additional info to results
                        results["raw_profile"] = profile.tolist() if hasattr(profile, 'tolist') else None
                        results["peak_positions"] = peaks
                        results["valley_positions"] = valleys
                        results["orientation"] = params.get("orientation")
                        results["profile_method"] = params.get("profile_method")
                        results["profile_position"] = params.get("profile_position")
                        
                        st.session_state.analyzed_roi = roi
                        st.session_state.analysis_results = results
                
                # Display results if available
                if st.session_state.analysis_results:
                    results = st.session_state.analysis_results
                    roi_image = extract_roi_image(image, roi)
                    display_analysis_results(results, roi_image)
                    save_analysis_results(results)
            else:
                st.error("Failed to load image. Please check the file format.")
        else:
            # Welcome screen
            display_welcome_screen()
            
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("For detailed error information, set DEBUG=1 in environment variables.") 
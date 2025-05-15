"""
Streamlit-specific helper functions for the USAF application.

This module contains utility functions for handling file uploads, image processing,
and other Streamlit-specific operations.
"""

import os
import tempfile
import numpy as np
import cv2
import streamlit as st
from typing import Tuple, Optional, Any

def process_uploaded_file(uploaded_file):
    """
    Process an uploaded file from Streamlit's file uploader.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        tuple: (image_array, temp_file_path) or (None, None) if processing fails
    """
    if uploaded_file is None:
        return None, None
        
    try:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        
        # Read the image with OpenCV
        image = cv2.imread(temp_path)
        
        # Convert to grayscale if it's a color image
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
            
        return gray_image, temp_path
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return None, None
        
def create_dummy_file_object(default_path):
    """
    Create a dummy file object to simulate a file upload for default images.
    
    Args:
        default_path: Path to the default image file
        
    Returns:
        object: A file-like object compatible with Streamlit's file uploader
    """
    if not os.path.exists(default_path):
        return None
        
    class DummyUploadedFile:
        def __init__(self, path):
            self.name = os.path.basename(path)
            self.path = path
            
        def getvalue(self):
            with open(self.path, 'rb') as f:
                return f.read()
    
    return DummyUploadedFile(default_path)

def prepare_config_from_parameters(params):
    """
    Prepare a configuration dictionary from UI parameters.
    
    Args:
        params: Dictionary of UI parameters
        
    Returns:
        dict: Configuration dictionary for analysis
    """
    from ..core.config import DEFAULT_CONFIG
    
    # Start with the default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Update image enhancement settings
    config["image_enhancement"]["enabled"] = params.get("enhance_image", True)
    
    if params.get("enhancement_method"):
        if params["enhancement_method"] == "CLAHE":
            config["image_enhancement"]["apply_clahe"] = True
            config["image_enhancement"]["denoise_method"] = "none"
        elif params["enhancement_method"] == "Bilateral filter":
            config["image_enhancement"]["denoise_method"] = "bilateral"
        elif params["enhancement_method"] == "Non-local means":
            config["image_enhancement"]["denoise_method"] = "nl_means"
    
    if "enhancement_strength" in params:
        config["image_enhancement"]["denoise_strength"] = params["enhancement_strength"]
    
    # Add ML options
    if "use_ml" in params:
        config["profile_analysis"]["use_ml_quality_score"] = params["use_ml"]
        config["ml_options"]["use_anomaly_detection"] = params["use_ml"]
    
    return config

def extract_roi_image(image, roi):
    """
    Extract a region of interest from an image.
    
    Args:
        image: Source image array
        roi: Tuple of (x, y, width, height)
        
    Returns:
        numpy.ndarray: ROI image
    """
    if roi is None or image is None:
        return None
        
    x, y, width, height = roi
    return image[y:y+height, x:x+width]

def save_analysis_results(results):
    """
    Save analysis results to a CSV file via Streamlit download button.
    
    Args:
        results: Analysis results dictionary
    """
    if not results:
        return
        
    import pandas as pd
    import io
    
    # Extract relevant metrics
    metrics = {
        "Group": results.get("group"),
        "Element": results.get("element"),
        "Resolution (lp/mm)": 2**(results.get("group", 0) + (results.get("element", 1) - 1)/6),
        "Resolution (μm)": 1000 / (2**(results.get("group", 0) + (results.get("element", 1) - 1)/6)),
        "Contrast": results.get("contrast"),
        "MTF": results.get("mtf"),
        "SNR": results.get("snr"),
        "Quality Score": results.get("quality_score"),
    }
    
    # Add ML metrics if available
    if "pattern_score" in results:
        metrics["Pattern Score"] = results["pattern_score"]
        metrics["Amplitude Regularity"] = results.get("amplitude_regularity", 0)
        metrics["Spacing Regularity"] = results.get("spacing_regularity", 0)
    
    # Convert to DataFrame
    df = pd.DataFrame([metrics])
    
    # Create CSV buffer
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    # Offer download button
    st.download_button(
        label="Download Results CSV",
        data=csv_buffer.getvalue(),
        file_name="usaf_analysis_results.csv",
        mime="text/csv",
    ) 
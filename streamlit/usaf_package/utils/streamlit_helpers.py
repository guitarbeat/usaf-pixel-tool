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
from typing import Tuple, Optional, Any, Union

def process_uploaded_file(uploaded_file: Union[Any, str]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Process an uploaded file from Streamlit's file uploader or a file path.
    
    Args:
        uploaded_file: Streamlit UploadedFile object or a file path (str)
        
    Returns:
        tuple: (image_array, temp_file_path) or (None, None) if processing fails
    """
    if uploaded_file is None:
        return None, None
    
    try:
        if isinstance(uploaded_file, str):
            # uploaded_file is a file path
            if not os.path.exists(uploaded_file):
                return None, None
            image = cv2.imread(uploaded_file)
            temp_path = uploaded_file
        else:
            # Assume uploaded_file is a Streamlit UploadedFile
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            image = cv2.imread(temp_path)
        
        # Convert to grayscale if it's a color image
        if image is not None and len(image.shape) == 3:
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
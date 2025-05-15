#!/usr/bin/env python3
"""Image processing tools for the USAF analyzer streamlit app."""

import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from typing import Tuple, Optional, Union, Any
from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit.runtime.uploaded_file_manager import UploadedFile

class DummyUploadedFile:
    """A class to simulate a Streamlit UploadedFile for local files."""
    def __init__(self, file_path: str):
        self.name = os.path.basename(file_path)
        self.type = f"image/{os.path.splitext(file_path)[1][1:]}"
        self.size = os.path.getsize(file_path)
        self._file_path = file_path

    def getvalue(self) -> bytes:
        """Read the file content."""
        with open(self._file_path, 'rb') as f:
            return f.read()

@st.cache_data
def process_uploaded_file(_uploaded_file: Union[DummyUploadedFile, UploadedFile]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Process an uploaded image file.
    
    Args:
        _uploaded_file: Streamlit UploadedFile or DummyUploadedFile object (prefixed with _ to prevent hashing)
        
    Returns:
        Tuple of (image array, temporary file path)
    """
    try:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(_uploaded_file.name)[1])
        temp_path = temp_file.name
        
        # Write the uploaded file content to the temporary file
        temp_file.write(_uploaded_file.getvalue())
        temp_file.close()
        
        # Read the image
        image = cv2.imread(temp_path)
        if image is None:
            return None, None
            
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image, temp_path
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None, None

@st.cache_data
def create_dummy_file_object(_file_path: str) -> Optional[DummyUploadedFile]:
    """
    Create a dummy file object to simulate file uploads for default images.
    
    Args:
        _file_path: Path to the image file (prefixed with _ to prevent hashing)
        
    Returns:
        DummyUploadedFile object or None if file doesn't exist
    """
    if os.path.exists(_file_path):
        return DummyUploadedFile(_file_path)
    return None

def extract_roi_image(image: np.ndarray, roi: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Extract a region of interest from an image.
    
    Args:
        image: Source image
        roi: Region of interest as (x, y, width, height)
        
    Returns:
        Extracted ROI image or None if invalid
    """
    if image is None or roi is None:
        return None
        
    x, y, w, h = roi
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return None
        
    try:
        return image[y:y+h, x:x+w]
    except Exception:
        return None 
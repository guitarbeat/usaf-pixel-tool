#!/usr/bin/env python3
"""Image processing module for USAF target analysis."""

import os
import logging
import numpy as np
import cv2
from typing import Tuple, Optional

class ImageProcessor:
    """
    Process images to analyze USAF targets: load, select ROI, extract profile.
    """
    def __init__(self):
        self.image = None
        self.grayscale = None
        self.roi = None
        self.profile = None

    def load_image(self, image_path: str) -> bool:
        """
        Load image from path and convert to RGB and grayscale.
        Returns True if successful, False otherwise.
        """
        try:
            if not os.path.isfile(image_path):
                logger.error(f"Image file not found: {image_path}")
                return False
            # Try loading with OpenCV first
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
            return False

    def select_roi(self, roi: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Select region of interest from grayscale image.
        Returns the ROI array or None if error.
        """
        if self.grayscale is None:
            return None
        try:
            x, y, width, height = roi
            self.roi = self.grayscale[y:y+height, x:x+width]
            return self.roi
        except Exception as e:
            return None

    def get_line_profile(self) -> Optional[np.ndarray]:
        """
        Extract horizontal (column average) intensity profile from ROI.
        Returns the profile array or None if error.
        """
        if self.roi is None:
            return None
        try:
            use_roi = self.roi
            self.profile = np.mean(use_roi, axis=0)
            return self.profile
        except Exception as e:
            return None 
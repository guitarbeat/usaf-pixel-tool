#!/usr/bin/env python3
"""Image processing module for USAF target analysis."""

import os
import logging
import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
from scipy import signal
from scipy.signal import find_peaks, peak_prominences

# Import scikit-image functions
from skimage import filters, restoration, exposure, util

# Local imports
from ..core.config import logger

class ImageProcessor:
    """Process images to analyze USAF targets."""
    
    def __init__(self, config: dict = None):
        """Initialize image processor with configuration."""
        self.config = config or {}
        self.image = None
        self.grayscale = None
        self.roi = None
        self.profile = None
        
    def load_image(self, image_path: str) -> bool:
        """Load image from path."""
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
                # Use skimage for loading instead of matplotlib
                from skimage import io
                self.image = io.imread(image_path)
                
            # Convert to grayscale if image has multiple channels
            if len(self.image.shape) > 2:
                self.grayscale = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            else:
                self.grayscale = self.image
                
            logger.info(f"Successfully loaded image: {image_path}, shape: {self.image.shape}")
            return True
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return False
    
    def select_roi(self, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """Select region of interest from image."""
        if self.grayscale is None:
            logger.error("No image loaded")
            return None
            
        try:
            x, y, width, height = roi
            self.roi = self.grayscale[y:y+height, x:x+width]
            return self.roi
        except Exception as e:
            logger.error(f"Failed to select ROI: {e}")
            return None
    
    def enhance_image(self) -> np.ndarray:
        """Enhance image using scikit-image techniques."""
        if self.roi is None:
            logger.error("No ROI selected for enhancement")
            return None
            
        try:
            # Convert to float for processing
            img_float = util.img_as_float(self.roi)
            
            # Apply contrast enhancement
            p2, p98 = np.percentile(img_float, (2, 98))
            img_rescale = exposure.rescale_intensity(img_float, in_range=(p2, p98))
            
            # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
            img_eq = exposure.equalize_adapthist(img_rescale, clip_limit=0.03)
            
            # Apply denoising
            img_denoised = restoration.denoise_bilateral(
                img_eq, 
                sigma_color=0.05, 
                sigma_spatial=1
            )
            
            # Option to use Non-local means denoising (slower but can be better)
            if self.config.get('use_nl_means', False):
                img_denoised = restoration.denoise_nl_means(
                    img_eq,
                    patch_size=5,
                    patch_distance=6,
                    h=0.05
                )
            
            # Return enhanced image
            return img_denoised
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return self.roi
            
    def get_line_profile(self, orientation: str = 'horizontal', line_position: int = None) -> np.ndarray:
        """
        Extract intensity profile from ROI.
        
        Args:
            orientation: 'horizontal' or 'vertical'
            line_position: Specific line position for single line profile (optional)
                          If None, averages across all rows/columns (ROI Average method)
        
        For horizontal orientation:
            - Returns a column average plot (vertical average at each x position)
              or a single row profile at the specified y position
            - The x-axis represents horizontal distance through the selection
            - The y-axis represents pixel intensity
            
        For vertical orientation:
            - Returns a row average plot (horizontal average at each y position)
              or a single column profile at the specified x position
            - The x-axis represents vertical distance through the selection
            - The y-axis represents pixel intensity
        """
        if self.roi is None:
            logger.error("No ROI selected")
            return None
            
        try:
            # Optionally enhance the ROI for better profile
            if self.config.get('enhance_before_profile', False):
                enhanced_roi = self.enhance_image()
                if enhanced_roi is not None:
                    use_roi = enhanced_roi
                else:
                    use_roi = self.roi
            else:
                use_roi = self.roi
            
            # Determine whether to use single line or ROI average
            use_single_line = line_position is not None
            
            if orientation.lower() == 'horizontal':
                if use_single_line:
                    # Use a specific row (y position)
                    if 0 <= line_position < use_roi.shape[0]:
                        self.profile = use_roi[line_position, :].astype(float)
                        logger.info(f"Using single horizontal line at y={line_position}")
                    else:
                        logger.warning(f"Invalid line position {line_position}. Using middle row.")
                        # Use middle row if position is invalid
                        middle_row = use_roi.shape[0] // 2
                        self.profile = use_roi[middle_row, :].astype(float)
                else:
                    # Column average plot (average values vertically for each x position)
                    self.profile = np.mean(use_roi, axis=0)
                    logger.info("Using column average (averaged across all rows)")
            else:  # vertical
                if use_single_line:
                    # Use a specific column (x position)
                    if 0 <= line_position < use_roi.shape[1]:
                        self.profile = use_roi[:, line_position].astype(float)
                        logger.info(f"Using single vertical line at x={line_position}")
                    else:
                        logger.warning(f"Invalid line position {line_position}. Using middle column.")
                        # Use middle column if position is invalid
                        middle_col = use_roi.shape[1] // 2
                        self.profile = use_roi[:, middle_col].astype(float)
                else:
                    # Row average plot (average values horizontally for each y position)
                    self.profile = np.mean(use_roi, axis=1)
                    logger.info("Using row average (averaged across all columns)")
            
            # Apply optional smoothing
            if self.config.get('smooth_profile', False):
                try:
                    from scipy.signal import savgol_filter
                    # Calculate an appropriate window size based on profile length
                    window_length = min(51, max(5, (len(self.profile) // 10) * 2 + 1))
                    # Ensure window length is odd
                    if window_length % 2 == 0:
                        window_length += 1
                    self.profile = savgol_filter(self.profile, window_length, 3)
                    logger.info(f"Applied Savitzky-Golay smoothing with window size {window_length}")
                except Exception as e:
                    logger.warning(f"Failed to apply smoothing: {e}")
                
            return self.profile
        except Exception as e:
            logger.error(f"Failed to get line profile: {e}")
            return None
            
    def detect_valleys(self, profile: Optional[np.ndarray] = None, 
                      threshold_multiplier: float = None,
                      min_distance: int = None,
                      sensitivity: float = None) -> List[int]:
        """
        Enhanced valley detection with multiple strategies and fallbacks.
        
        Args:
            profile: Input intensity profile
            threshold_multiplier: Multiplier for threshold calculation
            min_distance: Minimum distance between valleys
            sensitivity: Detection sensitivity (lower = more sensitive)
            
        Returns:
            List of valley positions (indices in the profile).
        """
        if profile is None:
            profile = self.profile
            
        if profile is None or len(profile) == 0:
            logger.error("No profile data available for valley detection")
            return []
            
        # Get settings from config or use defaults
        if threshold_multiplier is None:
            # If sensitivity provided, convert it to threshold_multiplier
            # Higher sensitivity (lower value) = lower threshold
            if sensitivity is not None:
                threshold_multiplier = sensitivity
            else:
                threshold_multiplier = self.config.get('valley_detection', {}).get(
                    'threshold_multiplier', 0.75)
                
        if min_distance is None:
            min_distance = self.config.get('valley_detection', {}).get(
                'min_distance', 10)
        
        # Try scikit-image valley detection first (new method)
        try:
            # Invert profile to convert valleys to peaks
            inverted_profile = np.max(profile) - profile
            
            # Use skimage's peak_local_max for peak detection
            from skimage.feature import peak_local_max
            # Calculate the minimum height threshold
            height_threshold = np.mean(inverted_profile) + threshold_multiplier * np.std(inverted_profile)
            # Find peaks using skimage (indices argument removed in recent versions)
            peaks_idx = peak_local_max(
                inverted_profile,
                min_distance=min_distance,
                threshold_abs=height_threshold
            )
            # peaks_idx is an array of shape (N, 1) for 1D input; flatten as needed
            if peaks_idx.ndim == 2 and peaks_idx.shape[1] == 1:
                peaks = peaks_idx.flatten()
            else:
                peaks = peaks_idx
            
            if len(peaks) > 0:
                logger.info(f"Detected {len(peaks)} valleys with scikit-image method")
                return peaks.tolist()
        except Exception as e:
            logger.warning(f"scikit-image valley detection failed: {e}")
        
        # Try primary scipy-based method as fallback
        try:
            # Invert profile to convert valleys to peaks
            inverted_profile = np.max(profile) - profile
            
            # Calculate prominence threshold
            prominence_threshold = threshold_multiplier * np.std(inverted_profile)
            
            # Find peaks (valleys in original)
            peaks, _ = find_peaks(
                inverted_profile, 
                distance=min_distance,
                prominence=prominence_threshold
            )
            
            # If we found peaks, return them
            if len(peaks) > 0:
                # Get the prominences for quality assessment
                prominences = peak_prominences(inverted_profile, peaks)[0]
                
                # Filter out peaks with low prominence
                good_peaks = peaks[prominences > prominence_threshold]
                
                if len(good_peaks) > 0:
                    logger.info(f"Detected {len(good_peaks)} valleys with scipy method")
                    return good_peaks.tolist()
        except Exception as e:
            logger.warning(f"Primary valley detection failed: {e}")
            
        # Fallback to simpler method if primary fails or finds no peaks
        try:
            # Try with relaxed parameters
            relaxed_threshold = self.config.get('valley_detection', {}).get(
                'relaxed_threshold_multiplier', 0.5) * np.std(inverted_profile)
            relaxed_distance = self.config.get('valley_detection', {}).get(
                'relaxed_min_distance', 5)
                
            peaks, _ = find_peaks(
                inverted_profile, 
                distance=relaxed_distance,
                prominence=relaxed_threshold
            )
            
            if len(peaks) > 0:
                logger.info(f"Detected {len(peaks)} valleys with fallback method")
                return peaks.tolist()
        except Exception as e:
            logger.warning(f"Fallback valley detection failed: {e}")
        
        # Last resort: simple local minima detection
        try:
            # Smooth profile to reduce noise
            smoothed = signal.savgol_filter(profile, 11, 3)
            
            # Find local minima
            valleys = []
            for i in range(1, len(smoothed) - 1):
                if smoothed[i-1] > smoothed[i] < smoothed[i+1]:
                    valleys.append(i)
                    
            # Apply minimum distance constraint
            if min_distance > 1 and len(valleys) > 1:
                filtered_valleys = [valleys[0]]
                for v in valleys[1:]:
                    if v - filtered_valleys[-1] >= min_distance:
                        filtered_valleys.append(v)
                        
                valleys = filtered_valleys
                
            if valleys:
                logger.info(f"Detected {len(valleys)} valleys with simple minima method")
                return valleys
            else:
                logger.warning("No valleys detected with any method")
                return []
        except Exception as e:
            logger.error(f"All valley detection methods failed: {e}")
            return []

    def detect_peaks_and_valleys(self, profile: Optional[np.ndarray] = None, 
                                min_distance: int = 5, 
                                prominence: float = 0.1) -> Tuple[List[int], List[int]]:
        """
        Detect both peaks (bright bars) and valleys (dark bars) in the intensity profile.
        Returns two lists: peak positions and valley positions.
        """
        if profile is None:
            profile = self.profile
        if profile is None or len(profile) == 0:
            logger.error("No profile data available for peak/valley detection")
            return [], []
        try:
            # Smooth the profile to reduce noise
            smoothed = signal.savgol_filter(profile, 11, 3)
            # Detect peaks (bright bars)
            peaks, _ = find_peaks(smoothed, distance=min_distance, prominence=prominence)
            # Detect valleys (dark bars) by inverting the profile
            valleys, _ = find_peaks(-smoothed, distance=min_distance, prominence=prominence)
            return list(peaks), list(valleys)
        except Exception as e:
            logger.error(f"Failed to detect peaks/valleys: {e}")
            return [], [] 
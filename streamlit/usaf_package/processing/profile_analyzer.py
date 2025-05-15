#!/usr/bin/env python3
"""Profile analyzer for processing intensity profiles from USAF targets."""

import numpy as np
from typing import Dict
import scipy.ndimage
import scipy.signal

from ..core.usaf_target import USAFTarget

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

class ProfileAnalyzer:
    """
    Minimal analyzer for USAF intensity profiles: calculates contrast, counts line pairs, and computes resolution.
    """
    def __init__(self, usaf_target: USAFTarget = None):
        self.usaf_target = usaf_target or USAFTarget()

    def analyze_profile(self, profile: np.ndarray, group: int, element: int) -> Dict[str, float]:
        """
        Analyze profile to calculate basic metrics for lp resolution and pixel size using boundary detection.
        Args:
            profile: Intensity profile (1D array)
            group: USAF group number
            element: USAF element number
        Returns:
            Dictionary with contrast, lp/mm, line pair count, and line pair widths.
        """
        boundaries = detect_line_pair_boundaries(profile)
        # Calculate line pair widths: distance between every other boundary (start of black bar to start of next black bar)
        line_pair_widths = [boundaries[i+2] - boundaries[i] for i in range(len(boundaries)-2)]
        num_line_pairs = len(line_pair_widths)
        avg_line_pair_width = float(np.mean(line_pair_widths)) if line_pair_widths else 0.0
        if num_line_pairs < 1:
            return {
                "contrast": 0.0,
                "lp_per_mm": self.usaf_target.lp_per_mm(group, element),
                "num_line_pairs": 0,
                "num_boundaries": len(boundaries),
                "line_pair_widths": [],
                "avg_line_pair_width": 0.0,
                "boundaries": boundaries
            }
        # Simple contrast: difference between mean of high and low regions
        high = profile[boundaries[::2]] if len(boundaries) > 1 else [np.max(profile)]
        low = profile[boundaries[1::2]] if len(boundaries) > 1 else [np.min(profile)]
        avg_high = np.mean(high)
        avg_low = np.mean(low)
        contrast = (avg_high - avg_low) / (avg_high + avg_low) if (avg_high + avg_low) > 0 else 0.0
        lp_per_mm = self.usaf_target.lp_per_mm(group, element)
        return {
            "contrast": float(contrast),
            "lp_per_mm": float(lp_per_mm),
            "num_line_pairs": num_line_pairs,
            "num_boundaries": len(boundaries),
            "line_pair_widths": line_pair_widths,
            "avg_line_pair_width": avg_line_pair_width,
            "boundaries": boundaries
        }

def detect_valleys(profile, sensitivity=0.2, distance=None, width=None, threshold=None):
    """
    Detect valleys (dark bars) in the intensity profile with improved robustness.
    
    Args:
        profile: Array of intensity values
        sensitivity: Detection sensitivity (lower = more sensitive)
        distance: Minimum distance between valleys (in pixels)
        width: Expected width range of valleys
        threshold: Absolute intensity threshold below which to consider valleys
        
    Returns:
        List of valley positions
    """
    # Apply Gaussian smoothing to reduce noise sensitivity
    window_size = max(5, len(profile) // 50)  # Adaptive window size based on profile length
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd-sized window
    
    # Apply smoothing using a Gaussian filter
    sigma = window_size / 6.0
    smoothed_profile = scipy.ndimage.gaussian_filter1d(profile, sigma)
    
    # Invert the profile since we're looking for valleys (dark regions)
    inverted_profile = np.max(smoothed_profile) - smoothed_profile
    
    # Calculate default parameters if not provided
    if distance is None:
        distance = max(3, len(profile) // 30)  # Adapt to profile length
        
    if width is None:
        min_width = max(2, len(profile) // 100)
        max_width = max(10, len(profile) // 20)
        width = (min_width, max_width)
    
    if sensitivity is None:
        prominence = 0.2 * (np.max(inverted_profile) - np.min(inverted_profile))
    else:
        # Scale prominence based on the intensity range
        prominence = sensitivity * (np.max(inverted_profile) - np.min(inverted_profile))
    
    # Find peaks in the inverted profile (valleys in original)
    valley_indices = scipy.signal.find_peaks(
        inverted_profile,
        height=threshold,
        prominence=prominence,
        distance=distance,
        width=width
    )[0]
    
    # Additional outlier filtering
    if len(valley_indices) > 3:
        # Calculate spacing between valleys
        valley_spacings = np.diff(valley_indices)
        
        # Identify median spacing
        median_spacing = np.median(valley_spacings)
        
        # Filter out valleys that deviate too much from expected spacing
        filtered_indices = [valley_indices[0]]  # Always keep first valley
        
        for i in range(1, len(valley_indices)):
            spacing = valley_indices[i] - filtered_indices[-1]
            
            # Allow valleys if they're reasonably close to expected spacing
            # or if they're multiples of the expected spacing (might indicate missing valleys)
            if 0.6 * median_spacing <= spacing <= 1.4 * median_spacing or \
               (spacing >= 1.8 * median_spacing and spacing <= 2.2 * median_spacing):
                filtered_indices.append(valley_indices[i])
        
        valley_indices = np.array(filtered_indices)
    
    return valley_indices 
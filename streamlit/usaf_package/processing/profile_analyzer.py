#!/usr/bin/env python3
"""Profile analyzer for processing intensity profiles from USAF targets."""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import scipy.ndimage
import scipy.signal

# Import scikit-learn components for ML-enhanced analysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error

# Local imports
from ..core.config import logger
from ..core.usaf_target import USAFTarget

class ProfileAnalyzer:
    """Analyze intensity profiles from USAF targets with ML enhancement."""
    
    def __init__(self, usaf_target: USAFTarget = None):
        """Initialize profile analyzer with USAF target object."""
        self.usaf_target = usaf_target or USAFTarget()
        
    def analyze_profile(self, profile: np.ndarray, 
                        peak_positions: List[int],
                        valley_positions: List[int],
                        group: int, element: int) -> Dict[str, float]:
        """Analyze profile to calculate metrics using both peaks and valleys.
        
        Args:
            profile: Input intensity profile
            peak_positions: List of detected peak positions (bright bars)
            valley_positions: List of detected valley positions (dark bars)
            group: USAF group number
            element: USAF element number
            
        Returns:
            Dictionary with analysis metrics
        """
        num_periods = min(len(peak_positions), len(valley_positions))
        if num_periods < 1:
            logger.warning("Not enough peaks/valleys detected for analysis")
            return {
                "contrast": 0.0,
                "mtf": 0.0,
                "resolution_microns": float('inf'),
                "line_pair_width_microns": 0.0,
                "quality_score": 0.0,
                "num_line_pairs": 0,
                "num_peaks": len(peak_positions),
                "num_valleys": len(valley_positions)
            }
            
        try:
            # Calculate contrast using average peak and valley values
            peak_values = [profile[pos] for pos in peak_positions if 0 <= pos < len(profile)]
            valley_values = [profile[pos] for pos in valley_positions if 0 <= pos < len(profile)]
            avg_peak = np.mean(peak_values) if peak_values else 0.0
            avg_valley = np.mean(valley_values) if valley_values else 0.0
            contrast = (avg_peak - avg_valley) / (avg_peak + avg_valley) if (avg_peak + avg_valley) > 0 else 0.0
            
            # Calculate MTF
            mtf = self.usaf_target.calculate_mtf(contrast, group, element)
            
            # Get resolution metrics
            usaf_element = self.usaf_target.get_element(group, element)
            resolution_microns = 1000 / usaf_element.line_pairs_per_mm
            line_pair_width_microns = resolution_microns
            
            # Enhanced quality score calculation using scikit-learn
            quality_score = self.calculate_enhanced_quality_score(
                profile, valley_positions, peak_positions)
            
            # Compile results
            results = {
                "contrast": float(contrast),
                "mtf": float(mtf),
                "resolution_microns": float(resolution_microns),
                "line_pair_width_microns": float(line_pair_width_microns),
                "quality_score": float(quality_score),
                "num_line_pairs": num_periods,
                "num_peaks": len(peak_positions),
                "num_valleys": len(valley_positions)
            }
            
            return results
        except Exception as e:
            logger.error(f"Failed to analyze profile: {e}")
            return {
                "contrast": 0.0,
                "mtf": 0.0,
                "resolution_microns": float('inf'),
                "line_pair_width_microns": 0.0,
                "quality_score": 0.0,
                "num_line_pairs": 0,
                "num_peaks": len(peak_positions),
                "num_valleys": len(valley_positions),
                "error": str(e)
            }
    
    def calculate_enhanced_quality_score(self, profile: np.ndarray,
                                      valley_positions: List[int],
                                      peak_positions: List[int]) -> float:
        """
        Calculate an enhanced quality score using machine learning approaches.
        
        This uses multiple factors including:
        - Contrast
        - Regularity of spacing
        - Amplitude consistency
        - Signal-to-noise
        - Pattern fit quality
        """
        try:
            # Extract pattern features
            if not valley_positions or not peak_positions:
                return 0.0
                
            # Calculate basic metrics
            valley_values = [profile[pos] for pos in valley_positions if 0 <= pos < len(profile)]
            peak_values = [profile[pos] for pos in peak_positions if 0 <= pos < len(profile)]
            
            # Basic contrast
            avg_valley = np.mean(valley_values)
            avg_peak = np.mean(peak_values)
            contrast = (avg_peak - avg_valley) / (avg_peak + avg_valley) if avg_peak + avg_valley > 0 else 0.0
            
            # Spacing regularity - coefficient of variation of valley spacing
            valley_spacings = np.diff(valley_positions)
            spacing_cv = np.std(valley_spacings) / np.mean(valley_spacings) if np.mean(valley_spacings) > 0 else 1.0
            spacing_regularity = 1.0 - min(1.0, spacing_cv)
            
            # Amplitude regularity
            valley_amplitude_cv = np.std(valley_values) / np.mean(valley_values) if np.mean(valley_values) > 0 else 1.0
            peak_amplitude_cv = np.std(peak_values) / np.mean(peak_values) if np.mean(peak_values) > 0 else 1.0
            amplitude_regularity = 1.0 - min(1.0, (valley_amplitude_cv + peak_amplitude_cv) / 2)
            
            # Pattern fit quality
            if len(valley_positions) >= 3:
                # Use scikit-learn to fit a sine wave model to the data
                pattern_quality = self.calculate_pattern_fit(profile, valley_positions, peak_positions)
            else:
                pattern_quality = 0.5  # Default for too few valleys
                
            # Calculate SNR (signal to noise)
            signal = avg_peak - avg_valley
            noise = np.std(profile)
            snr = signal / noise if noise > 0 else 0
            snr_score = min(1.0, snr / 10.0)  # Normalize to 0-1 range
            
            # Combinatorial quality score (weighted average)
            # Weights can be adjusted based on importance of each factor
            weights = {
                'contrast': 0.35,
                'spacing_regularity': 0.25,
                'amplitude_regularity': 0.15,
                'pattern_quality': 0.15,
                'snr': 0.1
            }
            
            quality_score = (
                weights['contrast'] * contrast +
                weights['spacing_regularity'] * spacing_regularity +
                weights['amplitude_regularity'] * amplitude_regularity +
                weights['pattern_quality'] * pattern_quality +
                weights['snr'] * snr_score
            )
            
            return float(quality_score)
        except Exception as e:
            logger.error(f"Error calculating enhanced quality score: {e}")
            return 0.5  # Return a midpoint value on error
    
    def calculate_pattern_fit(self, profile: np.ndarray, 
                            valley_positions: List[int], 
                            peak_positions: List[int]) -> float:
        """
        Calculate how well the profile matches an ideal USAF pattern.
        
        Returns a score from 0.0 (poor fit) to 1.0 (perfect fit).
        """
        try:
            if len(valley_positions) < 3 or len(profile) < 10:
                return 0.5  # Default for too few points
                
            # Create a mask of expected pattern regions
            pattern_mask = np.ones(len(profile))
            
            # Mark valley regions
            valley_width = int(np.mean(np.diff(valley_positions)) * 0.3)
            for pos in valley_positions:
                start = max(0, pos - valley_width)
                end = min(len(profile), pos + valley_width + 1)
                pattern_mask[start:end] = 0  # Valleys should be dark (low value)
                
            # Mark peak regions
            for pos in peak_positions:
                start = max(0, pos - valley_width)
                end = min(len(profile), pos + valley_width + 1)
                pattern_mask[start:end] = 1  # Peaks should be bright (high value)
                
            # Normalize profile to 0-1 range for comparison
            normalized_profile = (profile - np.min(profile)) / (np.max(profile) - np.min(profile) + 1e-10)
            
            # Calculate mean squared error between normalized profile and pattern mask
            mse = mean_squared_error(pattern_mask, normalized_profile)
            
            # Convert MSE to a 0-1 score (higher is better)
            score = np.exp(-5 * mse)  # Exponential decay of error
            
            return float(score)
        except Exception as e:
            logger.error(f"Error calculating pattern fit: {e}")
            return 0.5
            
    def calculate_pattern_regularity(self, profile: np.ndarray, 
                                  valley_positions: List[int]) -> Dict[str, float]:
        """
        Calculate multiple metrics for pattern regularity.
        
        Returns dict with various regularity metrics.
        """
        try:
            if len(valley_positions) < 2:
                return {
                    "spacing_regularity": 0.0,
                    "amplitude_regularity": 0.0,
                    "pattern_score": 0.0
                }
                
            # Spacing regularity
            valley_spacings = np.diff(valley_positions)
            spacing_cv = np.std(valley_spacings) / np.mean(valley_spacings) if np.mean(valley_spacings) > 0 else 1.0
            spacing_regularity = 1.0 - min(1.0, spacing_cv)
            
            # Amplitude regularity
            valley_values = [profile[pos] for pos in valley_positions if 0 <= pos < len(profile)]
            amplitude_cv = np.std(valley_values) / np.mean(valley_values) if np.mean(valley_values) > 0 else 1.0
            amplitude_regularity = 1.0 - min(1.0, amplitude_cv)
            
            # Use Isolation Forest to detect anomalies in valley patterns
            if len(valley_positions) >= 5:  # Need enough data for ML
                try:
                    # Create feature matrix: [position, value]
                    X = np.array([[pos, profile[pos]] for pos in valley_positions if 0 <= pos < len(profile)])
                    
                    # Normalize features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Use Isolation Forest to detect anomalies
                    iso_forest = IsolationForest(contamination=0.2, random_state=42)
                    outlier_scores = iso_forest.fit_predict(X_scaled)
                    
                    # Convert to 0-1 score (percent of inliers)
                    anomaly_score = (outlier_scores == 1).sum() / len(outlier_scores)
                except Exception as e:
                    anomaly_score = 0.8  # Default if ML fails
            else:
                anomaly_score = 0.8  # Default for small datasets
                
            # Combined pattern score
            pattern_score = 0.4 * spacing_regularity + 0.4 * amplitude_regularity + 0.2 * anomaly_score
            
            return {
                "spacing_regularity": float(spacing_regularity),
                "amplitude_regularity": float(amplitude_regularity),
                "pattern_score": float(pattern_score)
            }
        except Exception as e:
            logger.error(f"Error calculating pattern regularity: {e}")
            return {
                "spacing_regularity": 0.5,
                "amplitude_regularity": 0.5,
                "pattern_score": 0.5
            }
            
    def estimate_usaf_parameters(self, valley_positions: List[int], 
                                profile_width: int) -> Dict:
        """
        Estimate USAF target parameters from valley positions 
        with ML enhancements.
        """
        if len(valley_positions) < 3:
            return {"estimated_elements": 0, "estimated_group": None}
            
        try:
            # Calculate average spacing and its std deviation
            spacings = np.diff(valley_positions)
            avg_spacing = np.mean(spacings)
            spacing_std = np.std(spacings)
            
            # Check for multiple patterns using PCA and clustering
            if len(valley_positions) >= 6 and spacing_std / avg_spacing > 0.15:
                # Create features: position, first derivative of position (spacing)
                features = []
                for i in range(1, len(valley_positions)):
                    pos = valley_positions[i]
                    spacing = pos - valley_positions[i-1]
                    features.append([pos, spacing])
                    
                features = np.array(features)
                
                # Normalize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                # Try to identify if there are multiple patterns using PCA
                pca = PCA(n_components=1)
                pca.fit(features_scaled)
                
                # Check variance explained - if low, could be multiple patterns
                variance_explained = pca.explained_variance_ratio_[0]
                
                if variance_explained < 0.8:
                    # Try clustering into 2 groups
                    kmeans = KMeans(n_clusters=2, random_state=42)
                    clusters = kmeans.fit_predict(features_scaled)
                    
                    # Check average spacing in each cluster
                    group1_indices = np.where(clusters == 0)[0]
                    group2_indices = np.where(clusters == 1)[0]
                    
                    if len(group1_indices) >= 2 and len(group2_indices) >= 2:
                        # Calculate spacing for each group
                        group1_positions = [valley_positions[i+1] for i in group1_indices]
                        group2_positions = [valley_positions[i+1] for i in group2_indices]
                        
                        group1_spacing = np.mean(np.diff(sorted(group1_positions)))
                        group2_spacing = np.mean(np.diff(sorted(group2_positions)))
                        
                        # Use the smaller spacing for estimation (higher frequency)
                        avg_spacing = min(group1_spacing, group2_spacing)
                        logger.info(f"Detected multiple patterns, using spacing: {avg_spacing:.2f}")
            
            # Estimate number of line pairs
            estimated_lp = profile_width / avg_spacing
            
            # Try to estimate group/element (would need calibration data)
            return {
                "estimated_elements": float(estimated_lp),
                "avg_spacing_pixels": float(avg_spacing),
                "spacing_uniformity": float(1.0 - min(1.0, spacing_std / avg_spacing))
            }
        except Exception as e:
            logger.error(f"Failed to estimate USAF parameters: {e}")
            return {
                "estimated_elements": 0, 
                "estimated_group": None,
                "error": str(e)
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
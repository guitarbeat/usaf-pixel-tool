#!/usr/bin/env python3
"""USAF 1951 Resolution Target Analyzer - Refactored for conciseness"""

import hashlib
import io
import logging
import os
import re
import tempfile
import time
from typing import Any

import config
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw
from skimage import exposure, img_as_ubyte
from streamlit_image_coordinates import streamlit_image_coordinates

import streamlit as st

# Setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Constants
SESSION_STATE_PREFIXES = config.SESSION_STATE_PREFIXES
DEFAULT_IMAGE_PATH = config.DEFAULT_IMAGE_PATH
ROI_COLORS = config.ROI_COLORS
INVALID_ROI_COLOR = config.INVALID_ROI_COLOR


# Utility Functions
def _get_effective_bit_depth(image: np.ndarray) -> int:
    """Estimate effective bit depth from max value"""
    if not hasattr(image, "dtype") or image.dtype == np.uint8:
        return 8
    max_val = np.max(image)
    for bits in (8, 10, 12, 14, 16, 32):
        if max_val <= (1 << bits) - 1:
            return bits
    return 16


def parse_filename_for_defaults(filename: str) -> dict[str, Any]:
    """Parse filename for magnification and USAF values (Zoom23_AFT74_00001.tif)"""
    result = {}
    try:
        base_name = os.path.basename(filename)
        if zoom_match := re.search(r"Zoom(\d+(?:\.\d+)?)", base_name, re.IGNORECASE):
            result["magnification"] = float(zoom_match.group(1))
        if aft_match := re.search(r"AFT(\d)(\d)", base_name, re.IGNORECASE):
            result.update(
                {"group": int(aft_match.group(1)), "element": int(aft_match.group(2))}
            )
    except Exception as e:
        logger.warning(f"Error parsing filename: {e}")
    return result


def rotate_image(image: np.ndarray, rotation_count: int) -> np.ndarray:
    """Rotate image by 90-degree increments"""
    return (
        np.rot90(image, k=rotation_count % 4)
        if image is not None and rotation_count % 4
        else image
    )


def normalize_to_uint8(
    image,
    autoscale=True,
    invert=False,
    normalize=False,
    saturated_pixels=0.5,
    equalize_histogram=False,
):
    """Normalize image to uint8 with various enhancement options"""
    if image is None or image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    image_copy = image.copy()

    # Handle float images
    if np.issubdtype(image_copy.dtype, np.floating) and (
        np.max(image_copy) > 1.0 or np.min(image_copy) < -1.0
    ):
        min_val, max_val = np.min(image_copy), np.max(image_copy)
        image_copy = (
            (image_copy - min_val) / (max_val - min_val)
            if max_val > min_val
            else np.zeros_like(image_copy)
        )

    is_multichannel = image_copy.ndim > 2

    def process_channel(channel, method):
        """Process single channel with given method"""
        try:
            if method == "equalize":
                return img_as_ubyte(exposure.equalize_hist(channel))
            elif method == "autoscale":
                p_low, p_high = saturated_pixels / 2, 100 - saturated_pixels / 2
                p_min, p_max = np.percentile(channel, (p_low, p_high))
                return (
                    img_as_ubyte(
                        exposure.rescale_intensity(
                            channel, in_range=(p_min, p_max), out_range=(0, 255)
                        )
                    )
                    if p_max > p_min
                    else np.zeros_like(channel, dtype=np.uint8)
                )
            elif method == "normalize":
                min_val, max_val = np.min(channel), np.max(channel)
                return (
                    img_as_ubyte(
                        exposure.rescale_intensity(
                            channel, in_range=(min_val, max_val), out_range=(0, 255)
                        )
                    )
                    if max_val > min_val
                    else np.zeros_like(channel, dtype=np.uint8)
                )
            else:  # bit_depth scaling
                max_val = (1 << _get_effective_bit_depth(image)) - 1
                return img_as_ubyte(
                    exposure.rescale_intensity(
                        channel, in_range=(0, max_val), out_range=(0, 255)
                    )
                )
        except Exception as e:
            logger.warning(f"Error processing channel: {e}")
            return np.zeros_like(channel, dtype=np.uint8)

    # Determine processing method
    if equalize_histogram:
        method = "equalize"
    elif autoscale:
        method = "autoscale"
    elif normalize:
        method = "normalize"
    else:
        method = "bit_depth"

    # Process image
    if image_copy.dtype != np.uint8 or normalize or equalize_histogram:
        if is_multichannel and image_copy.shape[-1] <= 4:
            result = np.zeros_like(image_copy, dtype=np.uint8)
            for c in range(image_copy.shape[-1]):
                result[..., c] = process_channel(image_copy[..., c], method)
            image_copy = result
        else:
            image_copy = process_channel(image_copy, method)

    return 255 - image_copy if invert else image_copy


def get_unique_id_for_image(image_file) -> str:
    """Generate unique ID for image file"""
    try:
        filename = (
            image_file.name
            if hasattr(image_file, "name")
            else (
                os.path.basename(image_file)
                if isinstance(image_file, str)
                else str(id(image_file))
            )
        )
        return f"img_{hashlib.md5(filename.encode()).hexdigest()[:8]}"
    except Exception:
        return f"img_{int(time.time() * 1000)}"


def load_default_image():
    """Load default image if it exists"""
    return (
        config.DEFAULT_IMAGE_PATH if os.path.exists(config.DEFAULT_IMAGE_PATH) else None
    )


def process_uploaded_file(uploaded_file) -> tuple[np.ndarray | None, str | None]:
    """Process uploaded file with session state settings"""
    if uploaded_file is None:
        return None, None

    try:
        unique_id = get_unique_id_for_image(uploaded_file)
        # Get processing settings from session state
        settings = {
            k: st.session_state.get(f"{k}_{unique_id}", v)
            for k, v in {
                "autoscale": True,
                "invert": False,
                "normalize": False,
                "saturated_pixels": 0.5,
                "equalize_histogram": False,
            }.items()
        }

        if isinstance(uploaded_file, str):
            if not os.path.exists(uploaded_file):
                st.error(f"File not found: {uploaded_file}")
                return None, None

            # Load image based on file type
            if uploaded_file.lower().endswith((".tif", ".tiff")):
                image = tifffile.imread(uploaded_file)
            else:
                image = np.array(Image.open(uploaded_file))
        else:
            # Handle uploaded file object
            file_bytes = uploaded_file.read()
            if uploaded_file.name.lower().endswith((".tif", ".tiff")):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".tif"
                ) as tmp_file:
                    tmp_file.write(file_bytes)
                    tmp_file.flush()
                    image = tifffile.imread(tmp_file.name)
                    os.unlink(tmp_file.name)
            else:
                image = np.array(Image.open(io.BytesIO(file_bytes)))

        # Apply processing
        processed_image = normalize_to_uint8(image, **settings)
        filename = (
            uploaded_file.name
            if hasattr(uploaded_file, "name")
            else os.path.basename(uploaded_file)
        )

        return processed_image, filename

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        st.error(f"Error loading image: {e}")
        return None, None


def extract_roi_image(
    image, roi_coordinates: tuple[int, int, int, int], rotation: int = 0
) -> np.ndarray | None:
    """Extract ROI from image with optional rotation"""
    if image is None or not roi_coordinates:
        return None

    try:
        x1, y1, x2, y2 = roi_coordinates
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        roi = image[y1:y2, x1:x2]
        return rotate_image(roi, rotation) if rotation else roi

    except Exception as e:
        logger.error(f"Error extracting ROI: {e}")
        return None


def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        "usaf_target": USAFTarget(),
        "uploaded_files_list": [],
        "default_image_added": False,
        "image_index_to_id": {},
        "rotation_state_cleanup": set(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Handle cleanup of rotation and sensitivity session state when images are removed
    current_image_ids = {
        get_unique_id_for_image(file) for file in st.session_state.uploaded_files_list
    }
    old_image_ids = st.session_state.rotation_state_cleanup
    for image_id in old_image_ids:
        if image_id not in current_image_ids:
            # Remove session state variables for this image
            prefixes_to_clean = [
                f"rotation_{image_id}",
                f"last_rotation_{image_id}",
                f"sensitivity_{image_id}",
                f"last_sensitivity_{image_id}",
                f"min_distance_{image_id}",
                f"last_min_distance_{image_id}",
                f"roi_rotation_{image_id}",
                f"last_roi_rotation_{image_id}",
            ]
            for prefix in prefixes_to_clean:
                if prefix in st.session_state:
                    del st.session_state[prefix]

    # Update the set of image IDs that need cleanup on next check
    st.session_state.rotation_state_cleanup = current_image_ids


def get_image_session_keys(idx, image_file=None):
    """Get session state keys for image"""
    if image_file is not None:
        unique_id = get_unique_id_for_image(image_file)
        st.session_state.image_index_to_id[idx] = unique_id
    else:
        unique_id = st.session_state.image_index_to_id.get(idx, f"idx_{idx}")
    return {
        "group": f"group_{unique_id}",
        "element": f"element_{unique_id}",
        "analyzed_roi": f"analyzed_roi_{unique_id}",
        "analysis_results": f"analysis_results_{unique_id}",
        "last_group": f"last_group_{unique_id}",
        "last_element": f"last_element_{unique_id}",
        "coordinates": f"coordinates_{unique_id}",
        "image_path": f"image_path_{unique_id}",
        "image_name": f"image_name_{unique_id}",
        "roi_valid": f"roi_valid_{unique_id}",
        "roi_rotation": f"roi_rotation_{unique_id}",
        "last_roi_rotation": f"last_roi_rotation_{unique_id}",
        "processed_image": f"processed_image_{unique_id}",
        "roi_coordinates": f"roi_coordinates_{unique_id}",
    }


def find_best_two_line_pairs(dark_bar_starts):
    """Find the best two line pairs from dark bar positions"""
    if len(dark_bar_starts) < 4:
        return []

    # Calculate all possible line pair widths
    line_pairs = []
    for i in range(len(dark_bar_starts) - 3):
        for j in range(i + 2, len(dark_bar_starts) - 1, 2):
            width1 = dark_bar_starts[i + 1] - dark_bar_starts[i]
            width2 = dark_bar_starts[j + 1] - dark_bar_starts[j]
            avg_width = (width1 + width2) / 2
            consistency = 1 - abs(width1 - width2) / max(width1, width2)
            line_pairs.append((i, j, avg_width, consistency))

    # Sort by consistency and return best two
    line_pairs.sort(key=lambda x: x[3], reverse=True)
    return line_pairs[:2] if len(line_pairs) >= 2 else line_pairs


class USAFTarget:
    """USAF 1951 target specifications"""

    def __init__(self):
        pass

    def lp_per_mm(self, group: int, element: int) -> float:
        return 2 ** (group + (element - 1) / 6.0)

    def line_pair_width_microns(self, group: int, element: int) -> float:
        return 1000.0 / self.lp_per_mm(group, element)


def detect_significant_transitions(profile):
    """Detect significant transitions in profile"""
    if len(profile) < 3:
        return [], []

    derivative = np.gradient(profile)
    abs_derivative = np.abs(derivative)
    threshold = np.std(abs_derivative) * 2

    transitions = []
    transition_types = []

    for i in range(1, len(derivative) - 1):
        if abs_derivative[i] > threshold:
            if derivative[i] > 0:
                transition_types.append("rising")
            else:
                transition_types.append("falling")
            transitions.append(i)

    return transitions, transition_types


def extract_alternating_patterns(transitions, transition_types):
    """Extract alternating dark-light patterns"""
    if len(transitions) < 2:
        return []

    patterns = []
    current_pattern = [transitions[0]]
    current_type = transition_types[0]

    for i in range(1, len(transitions)):
        if transition_types[i] != current_type:
            current_pattern.append(transitions[i])
            current_type = transition_types[i]

            if len(current_pattern) >= 4 and len(current_pattern) % 2 == 0:
                patterns.append(current_pattern.copy())
        else:
            if len(current_pattern) >= 4:
                patterns.append(current_pattern)
            current_pattern = [transitions[i]]
            current_type = transition_types[i]

    return patterns


def limit_transitions_to_strongest(
    transitions, transition_types, derivative, max_transitions=5, min_strength=10
):
    """Limit transitions to strongest ones"""
    if len(transitions) <= max_transitions:
        return transitions, transition_types

    strengths = [abs(derivative[t]) for t in transitions]
    indices = sorted(range(len(strengths)), key=lambda i: strengths[i], reverse=True)[
        :max_transitions
    ]
    indices.sort()

    return [transitions[i] for i in indices], [transition_types[i] for i in indices]


def find_line_pair_boundaries_derivative(profile):
    """Find line pair boundaries using derivative method"""
    transitions, transition_types = detect_significant_transitions(profile)
    if len(transitions) < 4:
        return []

    derivative = np.gradient(profile)
    transitions, transition_types = limit_transitions_to_strongest(
        transitions, transition_types, derivative
    )
    patterns = extract_alternating_patterns(transitions, transition_types)

    return patterns[0] if patterns else []


def find_line_pair_boundaries_windowed(profile, window=5):
    """Find boundaries using windowed approach"""
    if len(profile) < window * 2:
        return []

    smoothed = np.convolve(profile, np.ones(window) / window, mode="same")
    return find_line_pair_boundaries_derivative(smoothed)


def find_line_pair_boundaries_threshold(profile, threshold):
    """Find boundaries using threshold method"""
    if threshold is None:
        threshold = np.mean(profile)

    boundaries = []
    above_threshold = profile > threshold

    for i in range(1, len(above_threshold)):
        if above_threshold[i] != above_threshold[i - 1]:
            boundaries.append(i)

    return boundaries


class RoiManager:
    """Manages ROI selection and validation"""

    def __init__(self):
        self.point1 = None
        self.point2 = None
        self.coordinates = None
        self.is_valid = False

    def set_coordinates(self, point1, point2):
        self.point1, self.point2 = point1, point2
        self.validate_and_convert()

    def validate_and_convert(self):
        """Validate and convert coordinates"""
        if not (self.point1 and self.point2):
            self.is_valid = False
            return

        try:
            x1, y1 = map(int, self.point1)
            x2, y2 = map(int, self.point2)

            # Ensure proper ordering
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Check minimum size
            if (x2 - x1) >= 10 and (y2 - y1) >= 10:
                self.coordinates = (x1, y1, x2, y2)
                self.is_valid = True
            else:
                self.is_valid = False
        except (ValueError, TypeError):
            self.is_valid = False

    def validate_against_image(self, image):
        """Validate coordinates against image dimensions"""
        if not self.is_valid or image is None:
            return False

        h, w = image.shape[:2]
        x1, y1, x2, y2 = self.coordinates

        return 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h

    def extract_roi(self, image):
        """Extract ROI from image"""
        if not self.validate_against_image(image):
            return None

        x1, y1, x2, y2 = self.coordinates
        return image[y1:y2, x1:x2]


class ProfileVisualizer:
    """Handles profile visualization and analysis"""

    def __init__(self):
        self.configure_plot_style()

    def configure_plot_style(self):
        """Configure matplotlib style"""
        plt.style.use("default")
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "figure.titlesize": 14,
            }
        )

    def create_figure(self, figsize=(8, 8), dpi=150):
        """Create matplotlib figure"""
        return plt.subplots(
            2, 1, figsize=figsize, dpi=dpi, gridspec_kw={"height_ratios": [2, 1]}
        )

    def plot_image(
        self,
        ax,
        image,
        group=None,
        element=None,
        avg_line_pair_width=None,
        lp_width_um=None,
        magnification=None,
        lp_per_mm=None,
    ):
        """Plot ROI image with annotations"""
        ax.imshow(image, cmap="gray", aspect="equal")
        ax.set_title(
            f"ROI - Group {group}, Element {element}"
            if group and element
            else "ROI Image"
        )
        ax.axis("off")

        # Add scale bar if we have the information
        if avg_line_pair_width and lp_width_um:
            scale_length_um = 10  # 10 micron scale bar
            scale_length_px = scale_length_um * avg_line_pair_width / lp_width_um

            # Position scale bar in bottom right
            h, w = image.shape[:2]
            start_x = w - scale_length_px - 10
            start_y = h - 20

            ax.plot(
                [start_x, start_x + scale_length_px],
                [start_y, start_y],
                "white",
                linewidth=3,
            )
            ax.text(
                start_x + scale_length_px / 2,
                start_y - 10,
                f"{scale_length_um} μm",
                ha="center",
                va="top",
                color="white",
                fontsize=8,
                path_effects=[PathEffects.withStroke(linewidth=2, foreground="black")],
            )

    def plot_profiles(
        self,
        ax,
        profile,
        individual_profiles=None,
        avg_line_pair_width=0.0,
        profile_type="max",
        edge_method=None,
        threshold=None,
    ):
        """Plot intensity profiles"""
        x = np.arange(len(profile))
        ax.plot(x, profile, "b-", linewidth=2, label=f"{profile_type.title()} Profile")

        if individual_profiles:
            for i, prof in enumerate(individual_profiles):
                ax.plot(
                    x, prof, alpha=0.3, linewidth=1, label=f"Row {i+1}" if i < 3 else ""
                )

        # Plot threshold line if provided
        if threshold is not None:
            ax.axhline(
                y=threshold,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Threshold: {threshold:.1f}",
            )

        # Find and plot boundaries
        boundaries = self.find_boundaries(profile, edge_method, threshold)
        if boundaries:
            for boundary in boundaries:
                ax.axvline(x=boundary, color="red", linestyle=":", alpha=0.8)

        ax.set_xlabel("Position (pixels)")
        ax.set_ylabel("Intensity")
        ax.set_title(f'Intensity Profile ({edge_method or "original"} method)')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def find_boundaries(self, profile, edge_method, threshold):
        """Find boundaries using specified method"""
        if edge_method == "derivative":
            return find_line_pair_boundaries_derivative(profile)
        elif edge_method == "windowed":
            return find_line_pair_boundaries_windowed(profile)
        elif edge_method == "threshold":
            return find_line_pair_boundaries_threshold(profile, threshold)
        else:
            return find_line_pair_boundaries_derivative(profile)

    def find_line_pairs(self, boundaries, roi_img):
        """Find line pairs from boundaries"""
        if len(boundaries) < 4:
            return []

        line_pairs = []
        for i in range(0, len(boundaries) - 1, 2):
            if i + 1 < len(boundaries):
                start, end = boundaries[i], boundaries[i + 1]
                width = end - start
                center = (start + end) / 2
                line_pairs.append((start, end, width, center))

        return line_pairs

    def draw_bracket(self, ax, x_start, x_end, y_pos, tick_size, color):
        """Draw bracket annotation"""
        ax.plot(
            [x_start, x_start, x_end, x_end],
            [y_pos - tick_size, y_pos, y_pos, y_pos - tick_size],
            color=color,
            linewidth=1.5,
        )

    def annotate_line_pairs(self, ax, line_pairs, roi_img):
        """Annotate line pairs on the plot"""
        if not line_pairs:
            return

        h = roi_img.shape[0]
        y_pos = h + 5
        tick_size = 3

        for i, (start, end, width, center) in enumerate(line_pairs):
            color = "red" if i < 2 else "orange"
            self.draw_bracket(ax, start, end, y_pos, tick_size, color)
            ax.text(
                center,
                y_pos + 8,
                f"{width:.1f}px",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )

    def create_caption(
        self,
        group=None,
        element=None,
        lp_width_um=None,
        edge_method=None,
        lp_per_mm=None,
    ):
        """Create figure caption"""
        parts = []
        if group and element:
            parts.append(f"USAF Group {group}, Element {element}")
        if lp_per_mm:
            parts.append(f"{lp_per_mm:.2f} lp/mm")
        if lp_width_um:
            parts.append(f"{lp_width_um:.2f} μm line pair width")
        if edge_method:
            parts.append(f"Edge detection: {edge_method}")

        return " | ".join(parts) if parts else "USAF Target Analysis"

    def visualize_profile(
        self,
        results,
        roi_img,
        group=None,
        element=None,
        lp_width_um=None,
        magnification=None,
    ):
        """Create complete visualization"""
        fig, (ax1, ax2) = self.create_figure()

        # Plot image
        self.plot_image(
            ax1,
            roi_img,
            group,
            element,
            results.get("avg_line_pair_width"),
            lp_width_um,
            magnification,
            results.get("lp_per_mm"),
        )

        # Plot profiles
        self.plot_profiles(
            ax2,
            results["profile"],
            results.get("individual_profiles"),
            results.get("avg_line_pair_width", 0),
            results.get("profile_type", "max"),
            results.get("edge_method"),
            results.get("threshold"),
        )

        # Annotate line pairs
        boundaries = results.get("boundaries", [])
        if boundaries:
            line_pairs = self.find_line_pairs(boundaries, roi_img)
            self.annotate_line_pairs(ax1, line_pairs, roi_img)

        # Add caption
        caption = self.create_caption(
            group,
            element,
            lp_width_um,
            results.get("edge_method"),
            results.get("lp_per_mm"),
        )
        fig.suptitle(caption, fontsize=12, y=0.02)

        plt.tight_layout()
        return fig


class ImageProcessor:
    """Main image processing and analysis class"""

    def __init__(self, usaf_target: USAFTarget = None):
        self.usaf_target = usaf_target or USAFTarget()
        self.original_image = None
        self.processed_image = None
        self.roi_coordinates = None
        self.roi_rotation = 0
        self.processing_params = {
            "autoscale": True,
            "invert": False,
            "normalize": False,
            "saturated_pixels": 0.5,
            "equalize_histogram": False,
        }

    def load_image(self, image_path: str) -> bool:
        """Load image from path"""
        try:
            if image_path.lower().endswith((".tif", ".tiff")):
                self.original_image = tifffile.imread(image_path)
            else:
                self.original_image = np.array(Image.open(image_path))
            self.apply_processing()
            return True
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return False

    def apply_processing(self):
        """Apply processing to original image"""
        if self.original_image is not None:
            self.processed_image = normalize_to_uint8(
                self.original_image, **self.processing_params
            )

    def update_processing_params(self, **kwargs):
        """Update processing parameters"""
        self.processing_params.update(kwargs)
        self.apply_processing()

    def set_roi(self, roi_coordinates: tuple[int, int, int, int]) -> bool:
        """Set ROI coordinates"""
        if self.processed_image is None:
            return False

        roi_manager = RoiManager()
        roi_manager.set_coordinates(roi_coordinates[:2], roi_coordinates[2:])

        if roi_manager.validate_against_image(self.processed_image):
            self.roi_coordinates = roi_coordinates
            return True
        return False

    def set_roi_rotation(self, rotation_count: int) -> None:
        """Set ROI rotation"""
        self.roi_rotation = rotation_count % 4

    def select_roi(self) -> np.ndarray | None:
        """Extract ROI from processed image"""
        if not (self.processed_image is not None and self.roi_coordinates):
            return None
        return extract_roi_image(
            self.processed_image, self.roi_coordinates, self.roi_rotation
        )

    def get_line_profile(self, use_max=False) -> np.ndarray | None:
        """Get line profile from ROI"""
        roi_img = self.select_roi()
        if roi_img is None:
            return None

        return np.max(roi_img, axis=0) if use_max else np.mean(roi_img, axis=0)

    def detect_edges(self, edge_method="original"):
        """Detect edges in line profile"""
        profile = self.get_line_profile(use_max=True)
        if profile is None:
            return []

        methods = {
            "derivative": find_line_pair_boundaries_derivative,
            "windowed": find_line_pair_boundaries_windowed,
            "threshold": lambda p: find_line_pair_boundaries_threshold(p, np.mean(p)),
        }

        return methods.get(edge_method, find_line_pair_boundaries_derivative)(profile)

    def calculate_contrast(self):
        """Calculate Michelson contrast"""
        profile = self.get_line_profile(use_max=True)
        if profile is None or len(profile) == 0:
            return 0.0

        # Use percentiles to avoid outliers
        i_max = np.percentile(profile, 95)
        i_min = np.percentile(profile, 5)

        return (i_max - i_min) / (i_max + i_min) if (i_max + i_min) > 0 else 0.0

    def analyze_profile(self, group: int, element: int) -> dict:
        """Analyze line profile and return results"""
        roi_img = self.select_roi()
        if roi_img is None:
            return {"error": "No ROI selected"}

        # Get profiles
        max_profile = np.max(roi_img, axis=0)
        individual_profiles = [roi_img[i, :] for i in range(min(5, roi_img.shape[0]))]

        # Detect boundaries
        boundaries = find_line_pair_boundaries_derivative(max_profile)

        # Calculate line pair metrics
        line_pairs = []
        if len(boundaries) >= 4:
            for i in range(0, len(boundaries) - 1, 2):
                if i + 1 < len(boundaries):
                    width = boundaries[i + 1] - boundaries[i]
                    line_pairs.append(width)

        avg_line_pair_width = np.mean(line_pairs) if line_pairs else 0.0
        num_line_pairs = len(line_pairs)

        # Calculate theoretical values
        lp_per_mm = self.usaf_target.lp_per_mm(group, element)
        theoretical_lp_width_um = self.usaf_target.line_pair_width_microns(
            group, element
        )

        # Calculate contrast
        contrast = self.calculate_contrast()

        return {
            "profile": max_profile.tolist() if hasattr(max_profile, "tolist") else None,
            "individual_profiles": individual_profiles,
            "boundaries": boundaries,
            "line_pairs": line_pairs,
            "avg_line_pair_width": avg_line_pair_width,
            "num_line_pairs": num_line_pairs,
            "contrast": contrast,
            "lp_per_mm": lp_per_mm,
            "theoretical_lp_width_um": theoretical_lp_width_um,
            "group": group,
            "element": element,
            "roi_rotation": self.roi_rotation,
            "profile_type": "max",
            "edge_method": "derivative",
        }

    def process_and_analyze(
        self,
        image_path: str,
        roi: tuple[int, int, int, int],
        group: int,
        element: int,
        use_max: bool = True,
        edge_method: str = "original",
        threshold: float = None,
        roi_rotation: int = 0,
        **processing_params,
    ) -> dict:
        """Complete processing and analysis pipeline"""
        # Update parameters
        self.update_processing_params(**processing_params)
        self.set_roi_rotation(roi_rotation)

        # Load and process image
        if not self.load_image(image_path):
            return {"error": "Failed to load image"}

        # Set ROI
        if not self.set_roi(roi):
            return {"error": "Invalid ROI"}

        # Analyze
        results = self.analyze_profile(group, element)
        results.update({"edge_method": edge_method, "threshold": threshold})

        return results


# Streamlit UI Functions
def display_roi_info(idx: int, image=None) -> tuple[int, int, int, int] | None:
    """Display ROI selection interface"""
    keys = get_image_session_keys(idx)
    coordinates_key = keys["coordinates"]
    roi_valid_key = keys["roi_valid"]

    if (
        coordinates_key not in st.session_state
        or st.session_state[coordinates_key] is None
    ):
        st.session_state[roi_valid_key] = False
        return None

    try:
        point1, point2 = st.session_state[coordinates_key]
        roi_x = min(point1[0], point2[0])
        roi_y = min(point1[1], point2[1])
        roi_width = abs(point2[0] - point1[0])
        roi_height = abs(point2[1] - point1[1])

        if roi_width <= 0 or roi_height <= 0:
            logger.warning(
                f"Invalid ROI dimensions: width={roi_width}, height={roi_height}"
            )
            st.session_state[roi_valid_key] = False
            return None

        if image is not None:
            img_height, img_width = None, None
            if hasattr(image, "shape"):
                if len(image.shape) > 1:
                    img_height, img_width = image.shape[0], image.shape[1]
            elif hasattr(image, "size"):
                img_width, img_height = image.size

            if (
                img_width is not None
                and img_height is not None
                and (
                    roi_x < 0
                    or roi_y < 0
                    or roi_x + roi_width > img_width
                    or roi_y + roi_height > img_height
                )
            ):
                logger.warning(
                    f"ROI extends beyond image dimensions: "
                    f"roi=({roi_x},{roi_y},{roi_width},{roi_height}), "
                    f"image=({img_width},{img_height})"
                )
                st.session_state[roi_valid_key] = False
                return None

        st.session_state[roi_valid_key] = True
        return (int(roi_x), int(roi_y), int(roi_width), int(roi_height))

    except Exception as e:
        logger.error(f"Error processing ROI: {e}")
        st.session_state[roi_valid_key] = False
        return None


def handle_image_selection(
    idx: int,
    image_file,
    image_to_display,  # Can be PIL Image or np.ndarray
    key: str = "usaf_image",
    rotation: int = 0,
) -> bool:
    """Handle image selection and ROI definition"""
    if image_to_display is None:
        return False

    keys = get_image_session_keys(idx, image_file)
    coordinates_key = keys["coordinates"]
    roi_valid_key = keys["roi_valid"]

    if coordinates_key not in st.session_state:
        st.session_state[coordinates_key] = None
        st.session_state[roi_valid_key] = False

    unique_id = keys["coordinates"].split("_")[1]
    component_key = f"{key}_{unique_id}"

    # Convert PIL Image to numpy array if needed
    if isinstance(image_to_display, Image.Image):
        image_to_display = np.array(image_to_display)

    # Convert to uint8 if needed
    if hasattr(image_to_display, "dtype") and image_to_display.dtype != np.uint8:
        image_to_display = normalize_to_uint8(image_to_display)

    # No rotation is applied to the display image - we only want to rotate the extracted ROI
    coords_component_output = streamlit_image_coordinates(
        image_to_display, key=component_key, click_and_drag=True
    )

    roi_changed = False
    if (
        coords_component_output is not None
        and coords_component_output.get("x1") is not None
        and coords_component_output.get("x2") is not None
        and coords_component_output.get("y1") is not None
        and coords_component_output.get("y2") is not None
    ):
        # Get coordinates from the component output
        point1 = (coords_component_output["x1"], coords_component_output["y1"])
        point2 = (coords_component_output["x2"], coords_component_output["y2"])

        if point1[0] != point2[0] and point1[1] != point2[1]:
            current_coordinates = st.session_state.get(coordinates_key)
            if current_coordinates != (point1, point2):
                # Store the coordinates in the session state
                st.session_state[coordinates_key] = (point1, point2)
                is_valid = (
                    point1[0] >= 0
                    and point1[1] >= 0
                    and point2[0] >= 0
                    and point2[1] >= 0
                    and abs(point2[0] - point1[0]) > 0
                    and abs(point2[1] - point1[1]) > 0
                )
                st.session_state[roi_valid_key] = is_valid
                roi_changed = True
                logger.debug(
                    f"ROI updated for image {idx}: {point1} to {point2}, valid: {is_valid}"
                )

    return roi_changed


def display_welcome_screen():
    """Display welcome screen"""
    st.markdown("""
    ## Welcome to USAF Target Analyzer
    
    This tool analyzes USAF 1951 resolution targets to measure:
    - Line pair resolution
    - Contrast measurements  
    - Pixel size calibration
    
    **To get started:**
    1. Upload one or more images using the sidebar
    2. Select a region of interest (ROI) around the target
    3. Specify the group and element numbers
    4. View the analysis results
    
    **Supported formats:** TIFF, PNG, JPEG
    """)


def display_analysis_details(results):
    """Display detailed analysis results"""
    if "error" in results:
        st.error(f"Analysis error: {results['error']}")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Line Pairs/mm", f"{results.get('lp_per_mm', 0):.2f}")
        st.metric("Contrast", f"{results.get('contrast', 0):.3f}")

    with col2:
        st.metric("Line Pairs Detected", results.get("num_line_pairs", 0))
        st.metric(
            "Avg Line Pair Width", f"{results.get('avg_line_pair_width', 0):.1f} px"
        )

    with col3:
        st.metric(
            "Theoretical Width", f"{results.get('theoretical_lp_width_um', 0):.2f} μm"
        )
        st.metric("ROI Rotation", f"{results.get('roi_rotation', 0) * 90}°")


def analyze_and_display_image(idx, uploaded_file):
    """Main function to analyze and display a single image"""
    try:
        # Process uploaded file
        image, filename = process_uploaded_file(uploaded_file)
        if image is None:
            st.error(f"Failed to load image {idx + 1}")
            return

        # Get session keys
        unique_id = get_unique_id_for_image(uploaded_file)
        keys = get_image_session_keys(idx, uploaded_file)

        # Store processed image and filename
        st.session_state[keys["processed_image"]] = image
        st.session_state[keys["image_name"]] = filename

        # Create expander for this image
        with st.expander(f"📸 Image {idx + 1}: {filename}", expanded=True):
            # Image processing controls
            st.subheader("Processing Controls")
            col_proc1, col_proc2, col_proc3 = st.columns(3)

            with col_proc1:
                # Rotation control
                rotation = st.slider(
                    "Rotation (90° steps)", 0, 3, 0, key=f"rotation_{unique_id}"
                )
                autoscale = st.checkbox(
                    "Auto-scale contrast", True, key=f"autoscale_{unique_id}"
                )

            with col_proc2:
                invert = st.checkbox("Invert image", False, key=f"invert_{unique_id}")
                normalize = st.checkbox(
                    "Normalize", False, key=f"normalize_{unique_id}"
                )

            with col_proc3:
                if autoscale:
                    saturated_pixels = st.slider(
                        "Saturated pixels (%)",
                        0.0,
                        5.0,
                        0.5,
                        0.1,
                        key=f"saturated_pixels_{unique_id}",
                    )
                else:
                    saturated_pixels = 0.5

                equalize_histogram = st.checkbox(
                    "Equalize histogram", False, key=f"equalize_histogram_{unique_id}"
                )

            # Update processing if parameters changed
            current_params = {
                "autoscale": autoscale,
                "invert": invert,
                "normalize": normalize,
                "saturated_pixels": saturated_pixels,
                "equalize_histogram": equalize_histogram,
            }

            # Reprocess if needed
            if any(
                st.session_state.get(f"{k}_{unique_id}") != v
                for k, v in current_params.items()
            ):
                image = normalize_to_uint8(image, **current_params)
                st.session_state[keys["processed_image"]] = image

            # Main layout: ROI selection and plot display
            roi_col, plot_col = st.columns([1, 1])

            with roi_col:
                st.markdown("#### Select ROI on Image")

                # Create PIL image for display with ROI rectangle
                # Ensure image is numpy array first
                if isinstance(image, Image.Image):
                    image_array = np.array(image)
                else:
                    image_array = image

                pil_img = Image.fromarray(image_array)
                draw = ImageDraw.Draw(pil_img)

                # Draw ROI rectangle if coordinates exist
                current_coords = st.session_state.get(keys["coordinates"])
                if current_coords:
                    p1, p2 = current_coords
                    coords = (
                        min(p1[0], p2[0]),
                        min(p1[1], p2[1]),
                        max(p1[0], p2[0]),
                        max(p1[1], p2[1]),
                    )
                    # Choose color based on ROI validity
                    if roi_valid := st.session_state.get(keys["roi_valid"], False):
                        color_idx = idx % len(ROI_COLORS)
                        outline_color = ROI_COLORS[color_idx]
                    else:
                        outline_color = INVALID_ROI_COLOR
                    draw.rectangle(coords, outline=outline_color, width=3)

                # Handle image selection and ROI - pass PIL image for display
                roi_changed = handle_image_selection(
                    idx, uploaded_file, pil_img, f"usaf_image_{unique_id}", rotation
                )

                # Status display for ROI validity
                roi_valid = st.session_state.get(keys["roi_valid"], False)
                if current_coords is not None:
                    if roi_valid:
                        st.success("ROI selection is valid")
                    else:
                        st.error(
                            "ROI selection is not valid. Please select a valid region."
                        )

            with plot_col:
                # Show processed image/plot and analysis results
                if analysis_results := st.session_state.get(keys["analysis_results"]):
                    # Get ROI coordinates and extract ROI for display - use numpy array
                    roi_coords = display_roi_info(idx, image_array)
                    if roi_coords:
                        roi_img = extract_roi_image(image_array, roi_coords, rotation)
                        if roi_img is not None:
                            # Generate the figure using ProfileVisualizer
                            visualizer = ProfileVisualizer()

                            # Get analysis parameters
                            group_val = analysis_results.get("group")
                            element_val = analysis_results.get("element")
                            lp_width_um = analysis_results.get(
                                "theoretical_lp_width_um"
                            )
                            magnification = st.session_state.get(
                                f"magnification_{unique_id}", 10.0
                            )

                            fig = visualizer.visualize_profile(
                                analysis_results,
                                roi_img,
                                group=group_val,
                                element=element_val,
                                lp_width_um=lp_width_um,
                                magnification=magnification,
                            )
                            if fig is not None:
                                st.pyplot(fig)
                                plt.close(fig)

                                # Show caption
                                caption = visualizer.create_caption(
                                    group_val,
                                    element_val,
                                    lp_width_um,
                                    edge_method=analysis_results.get(
                                        "edge_method", "original"
                                    ),
                                )
                                st.markdown(caption, unsafe_allow_html=True)
                elif current_coords_for_preview := st.session_state.get(
                    keys["coordinates"]
                ):
                    try:
                        p1_preview, p2_preview = current_coords_for_preview
                        coords_preview = (
                            min(p1_preview[0], p2_preview[0]),
                            min(p1_preview[1], p2_preview[1]),
                            max(p1_preview[0], p2_preview[0]),
                            max(p1_preview[1], p2_preview[1]),
                        )

                        # Extract the ROI from the PIL image for preview
                        roi_img_preview = pil_img.crop(coords_preview)

                        # Apply rotation to the preview if needed
                        if rotation > 0:
                            roi_img_preview = Image.fromarray(
                                rotate_image(np.array(roi_img_preview), rotation)
                            )

                        st.image(
                            roi_img_preview,
                            caption="Selected ROI Preview",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.warning(f"Could not display ROI preview: {e!s}")
                else:
                    st.info("Select an ROI to view analysis.")

            # Analysis controls (below the main layout)
            if roi_valid and current_coords:
                st.subheader("Analysis Parameters")
                col_a, col_b = st.columns(2)

                with col_a:
                    # Parse defaults from filename
                    defaults = parse_filename_for_defaults(filename)

                    group = st.number_input(
                        "Group",
                        1,
                        7,
                        defaults.get("group", 4),
                        key=f"group_{unique_id}",
                    )
                    element = st.number_input(
                        "Element",
                        1,
                        6,
                        defaults.get("element", 1),
                        key=f"element_{unique_id}",
                    )
                    magnification = st.number_input(
                        "Magnification",
                        0.1,
                        100.0,
                        defaults.get("magnification", 10.0),
                        0.1,
                        key=f"magnification_{unique_id}",
                    )

                with col_b:
                    edge_method = st.selectbox(
                        "Edge Detection",
                        ["derivative", "windowed", "threshold"],
                        key=f"edge_method_{unique_id}",
                    )
                    use_max = st.checkbox(
                        "Use max projection", True, key=f"use_max_{unique_id}"
                    )

                    threshold = None
                    if edge_method == "threshold":
                        threshold = st.number_input(
                            "Threshold", 0, 255, 128, key=f"threshold_{unique_id}"
                        )

                # Analyze button
                if st.button("🔍 Analyze", key=f"analyze_{unique_id}"):
                    with st.spinner("Analyzing..."):
                        # Get ROI coordinates using display_roi_info - use numpy array
                        roi_coords = display_roi_info(idx, image_array)

                        if roi_coords:
                            # Create processor and analyze
                            processor = ImageProcessor()

                            # Create temporary file for analysis
                            with tempfile.NamedTemporaryFile(
                                suffix=".tif", delete=False
                            ) as tmp_file:
                                if isinstance(uploaded_file, str):
                                    # Copy file
                                    import shutil

                                    shutil.copy2(uploaded_file, tmp_file.name)
                                else:
                                    # Save uploaded file
                                    uploaded_file.seek(0)
                                    tmp_file.write(uploaded_file.read())

                                tmp_file.flush()

                                # Analyze
                                results = processor.process_and_analyze(
                                    tmp_file.name,
                                    roi_coords,
                                    group,
                                    element,
                                    use_max=use_max,
                                    edge_method=edge_method,
                                    threshold=threshold,
                                    roi_rotation=rotation,
                                    **current_params,
                                )

                                # Clean up
                                os.unlink(tmp_file.name)

                            # Store results
                            st.session_state[keys["analysis_results"]] = results

                            # Display results
                            if "error" not in results:
                                st.success("✅ Analysis complete!")
                                display_analysis_details(results)
                                st.rerun()  # Refresh to show the plot
                            else:
                                st.error(f"Analysis failed: {results['error']}")
                        else:
                            st.error("Please select an ROI first")

                # Show existing results if available
                if existing_results := st.session_state.get(keys["analysis_results"]):
                    if "error" not in existing_results:
                        st.info("Previous analysis results:")
                        display_analysis_details(existing_results)
            else:
                st.info("👆 Click and drag on the image to select ROI")

    except Exception as e:
        logger.error(f"Error in analyze_and_display_image: {e}")
        st.error(f"Error processing image {idx + 1}: {e}")


def collect_analysis_data():
    """Collect analysis data for CSV export"""
    data = {
        "Filename": [],
        "Magnification": [],
        "Group": [],
        "Element": [],
        "Line Pairs/mm": [],
        "Line Pair Width (µm)": [],
        "Pixel Size (µm/pixel)": [],
        "Contrast": [],
        "Line Pairs Detected": [],
        "Avg Line Pair Width (px)": [],
        "ROI Rotation": [],
    }

    for idx, uploaded_file in enumerate(st.session_state.uploaded_files_list):
        unique_id = get_unique_id_for_image(uploaded_file)
        keys = get_image_session_keys(idx, uploaded_file)

        if analysis_results := st.session_state.get(keys["analysis_results"]):
            filename = st.session_state.get(keys["image_name"], f"Image {idx+1}")
            magnification = st.session_state.get(f"magnification_{unique_id}", 0)

            # Extract data
            data["Filename"].append(filename)
            data["Magnification"].append(magnification)
            data["Group"].append(analysis_results.get("group", 0))
            data["Element"].append(analysis_results.get("element", 0))
            data["Line Pairs/mm"].append(analysis_results.get("lp_per_mm", 0))
            data["Line Pair Width (µm)"].append(
                analysis_results.get("theoretical_lp_width_um", 0)
            )

            # Calculate pixel size
            avg_lp_width_px = analysis_results.get("avg_line_pair_width", 0)
            lp_width_um = analysis_results.get("theoretical_lp_width_um", 0)
            pixel_size = (
                lp_width_um / avg_lp_width_px
                if avg_lp_width_px > 0 and lp_width_um > 0
                else 0
            )
            data["Pixel Size (µm/pixel)"].append(pixel_size)

            data["Contrast"].append(analysis_results.get("contrast", 0))
            data["Line Pairs Detected"].append(
                analysis_results.get("num_line_pairs", 0)
            )
            data["Avg Line Pair Width (px)"].append(avg_lp_width_px)
            data["ROI Rotation"].append(
                f"{analysis_results.get('roi_rotation', 0) * 90}°"
            )

    return pd.DataFrame(data)


def run_streamlit_app():
    """Main Streamlit application"""
    try:
        st.set_page_config(page_title="USAF Target Analyzer", layout="wide")
        initialize_session_state()

        st.title("USAF Target Analyzer")
        st.markdown(
            "<style>.stExpander {margin-top: 0.5rem !important;} .plot-container {margin-bottom: 0.5rem;}</style>",
            unsafe_allow_html=True,
        )

        # Sidebar controls
        with st.sidebar:
            st.header("Controls")

            # File uploader
            if new_uploaded_files := st.file_uploader(
                "Upload USAF target image(s)",
                type=["jpg", "jpeg", "png", "tif", "tiff"],
                accept_multiple_files=True,
                help="Select one or more images containing a USAF 1951 resolution target",
            ):
                for file in new_uploaded_files:
                    file_names = [
                        f.name if hasattr(f, "name") else os.path.basename(f)
                        for f in st.session_state.uploaded_files_list
                    ]
                    new_file_name = (
                        file.name if hasattr(file, "name") else os.path.basename(file)
                    )
                    if new_file_name not in file_names:
                        st.session_state.uploaded_files_list.append(file)
                        st.success(f"Added: {new_file_name}")

            st.markdown("---")
            st.markdown("### Analysis Tips")
            st.info(
                "**Image Rotation**: Use rotation slider to align line pairs horizontally for better analysis."
            )

            # CSV export
            st.markdown("---")
            st.markdown("### Export Results")

            if st.button("Generate Analysis CSV"):
                if not st.session_state.uploaded_files_list:
                    st.warning("No images uploaded.")
                else:
                    df = collect_analysis_data()
                    if df.empty:
                        st.warning("No analysis data available.")
                    else:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "Download Analysis CSV",
                            csv,
                            "usaf_analysis_results.csv",
                            "text/csv",
                        )
                        st.markdown("#### CSV Preview")
                        st.dataframe(df, use_container_width=True)

            # Load default image
            default_image_path = load_default_image()
            if (
                not st.session_state.uploaded_files_list
                and default_image_path
                and not st.session_state.default_image_added
            ):
                st.session_state.uploaded_files_list.append(default_image_path)
                st.session_state.default_image_added = True
                st.info(f"Using default image: {os.path.basename(default_image_path)}")

            # Clear all button
            if st.button("Clear All Images"):
                st.session_state.uploaded_files_list = []
                st.session_state.default_image_added = False
                st.session_state.image_index_to_id = {}
                for key in list(st.session_state.keys()):
                    if any(
                        key.startswith(prefix)
                        for prefix in config.SESSION_STATE_PREFIXES
                    ):
                        del st.session_state[key]
                st.success("All images cleared")
                st.rerun()

        # Main content
        main_container = st.container()
        with main_container:
            if st.session_state.uploaded_files_list:
                st.info(
                    f"Currently analyzing {len(st.session_state.uploaded_files_list)} image(s)"
                )
                for idx, uploaded_file in enumerate(
                    st.session_state.uploaded_files_list
                ):
                    analyze_and_display_image(idx, uploaded_file)
            else:
                display_welcome_screen()

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("For detailed error information, set DEBUG=1 in environment variables.")


if __name__ == "__main__":
    run_streamlit_app()

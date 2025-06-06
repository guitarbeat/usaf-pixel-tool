#!/usr/bin/env python3
"""
USAF 1951 Resolution Target Analyzer

A comprehensive tool for analyzing USAF 1951 resolution targets in microscopy and imaging systems.
"""

import hashlib
import io
import logging
import os
import re  # Add import for regex
import tempfile
import time
from typing import Any

import cv2
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit_nested_layout  # noqa: F401
import tifffile
from PIL import Image, ImageDraw
from skimage import exposure, img_as_ubyte
from streamlit_image_coordinates import streamlit_image_coordinates

import streamlit as st

# --- Logging Setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# --- Constants ---
# File Paths
DEFAULT_IMAGE_PATH = os.path.expanduser(
    "~/Library/CloudStorage/Box-Box/FOIL/Aaron/2025-05-12/airforcetarget_images/AF_2_2_00001.png"
)

# ROI Colors
ROI_COLORS = [
    "#00FF00",
    "#FF00FF",
    "#00FFFF",
    "#FFFF00",
    "#FF8000",
    "#0080FF",
    "#8000FF",
    "#FF0080",
]
INVALID_ROI_COLOR = "#FF0000"  # Red for invalid ROIs

# Session State Prefixes
SESSION_STATE_PREFIXES = [
    "group_",
    "element_",
    "analyzed_roi_",
    "analysis_results_",
    "last_group_",
    "last_element_",
    "coordinates_",
    "image_path_",
    "image_name_",
    "roi_valid_",
]

# UI Defaults
DEFAULT_GROUP = 2
DEFAULT_ELEMENT = 2

# Welcome screen constants
WELCOME_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/d/d6/1951usaf_test_target.jpg"
)
WELCOME_IMAGE_CAPTION = "Example USAF 1951 Target"

# --- Utility Functions ---


def _get_effective_bit_depth(image: np.ndarray) -> int:
    """
    Estimate the effective bit depth of an image by examining its maximum value.
    For example, a 16-bit image with max value 4095 is likely 12-bit digitized.
    """
    if not hasattr(image, "dtype"):
        return 8
    if image.dtype == np.uint8:
        return 8
    max_val = np.max(image)
    # Find the smallest bit depth that can represent max_val
    for bits in (8, 10, 12, 14, 16, 32):
        if max_val <= (1 << bits) - 1:
            return bits
    return 16  # fallback


def parse_filename_for_defaults(filename: str) -> dict[str, Any]:
    """
    Parse filename to extract magnification and USAF target values.

    Expected pattern examples:
    - Zoom23_AFT74_00001.tif - Zoom=23.0, AFT=7.4 (Group 7, Element 4)
    - Zoom7.6_AFT56_00001.tif - Zoom=7.6, AFT=5.6 (Group 5, Element 6)

    Args:
        filename: The filename to parse

    Returns:
        Dictionary with 'magnification', 'group', and 'element' if found
    """
    result = {}

    try:
        # Extract just the filename if a full path is given
        base_name = os.path.basename(filename)

        # Look for magnification pattern: Zoom<number>
        zoom_match = re.search(r"Zoom(\d+(?:\.\d+)?)", base_name, re.IGNORECASE)
        if zoom_match:
            try:
                magnification = float(zoom_match.group(1))
                result["magnification"] = magnification
                logger.debug(f"Found magnification in filename: {magnification}")
            except (ValueError, TypeError):
                pass

        # Look for AFT pattern: AFT<group><element>
        aft_match = re.search(r"AFT(\d)(\d)", base_name, re.IGNORECASE)
        if aft_match:
            try:
                group = int(aft_match.group(1))
                element = int(aft_match.group(2))
                result["group"] = group
                result["element"] = element
                logger.debug(f"Found group {group}, element {element} in filename")
            except (ValueError, TypeError, IndexError):
                pass
    except Exception as e:
        logger.warning(f"Error parsing filename for defaults: {e}")

    return result


def rotate_image(image: np.ndarray, rotation_count: int) -> np.ndarray:
    """
    Rotate an image by 90-degree increments.

    Args:
        image: The image to rotate
        rotation_count: Number of 90-degree rotations (0-3)

    Returns:
        Rotated image
    """
    if image is None:
        return None

    try:
        # Normalize rotation count to 0-3
        rotation_count = rotation_count % 4

        if rotation_count == 0:
            return image

        # Apply rotation
        return np.rot90(image, k=rotation_count)
    except Exception as e:
        logger.error(f"Error rotating image: {e}")
        return image  # Return original image on error


def normalize_to_uint8(
    image,
    autoscale=True,
    invert=False,
    normalize=False,
    saturated_pixels=0.5,
    equalize_histogram=False,
):
    """
    Normalize image to uint8 (0-255) range with ImageJ-like contrast enhancement options.

    Args:
        image: Input image array
        autoscale: If True, uses percentile-based contrast stretching based on saturated_pixels value.
                  If False, scales to the image's digitization bit depth.
        invert: If True, invert the image (useful for microscopy where dark = high signal)
        normalize: If True, recalculates pixel values to use the full range (0-255 for 8-bit)
        saturated_pixels: Percentage of pixels allowed to become saturated (0.0-100.0)
        equalize_histogram: If True, enhances image using histogram equalization

    Returns:
        Normalized uint8 image
    """
    # Handle empty or invalid images
    if image is None or image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    # Make a copy to avoid modifying the original
    image_copy = image.copy()

    # Handle float images - scikit-image expects float images to be between -1 and 1
    if np.issubdtype(image_copy.dtype, np.floating):
        # Check if values are outside the expected range for float images
        if np.max(image_copy) > 1.0 or np.min(image_copy) < -1.0:
            # Normalize float values to 0-1 range
            min_val = np.min(image_copy)
            max_val = np.max(image_copy)
            if max_val > min_val:
                image_copy = (image_copy - min_val) / (max_val - min_val)
            else:
                image_copy = np.zeros_like(image_copy)

    # Check if we're dealing with a multi-channel image
    is_multichannel = image_copy.ndim > 2

    # Apply histogram equalization if requested (overrides other contrast adjustments)
    if equalize_histogram:
        # Handle RGB/RGBA images
        if is_multichannel and image_copy.shape[-1] <= 4:
            # Process each channel separately for equalization
            result = np.zeros_like(image_copy, dtype=np.uint8)
            for c in range(image_copy.shape[-1]):
                channel = image_copy[..., c]
                try:
                    # Use skimage's equalize_hist function
                    equalized = exposure.equalize_hist(channel)
                    result[..., c] = img_as_ubyte(equalized)
                except Exception as e:
                    logger.warning(f"Error equalizing histogram for channel {c}: {e}")
                    result[..., c] = img_as_ubyte(channel)
            image_copy = result
        else:
            try:
                # Apply equalization to grayscale image
                equalized = exposure.equalize_hist(image_copy)
                image_copy = img_as_ubyte(equalized)
            except Exception as e:
                logger.warning(f"Error equalizing histogram: {e}")

    # Only normalize if not already uint8 and not equalized
    elif image_copy.dtype != np.uint8 or normalize:
        # Determine effective bit depth (not just dtype, but actual data range)
        bit_depth = _get_effective_bit_depth(image_copy)

        if autoscale:
            # For multi-channel images, process each channel separately
            # Handle RGB/RGBA images
            if is_multichannel and image_copy.shape[-1] <= 4:
                # Process each channel separately
                result = np.zeros_like(image_copy, dtype=np.uint8)
                for c in range(image_copy.shape[-1]):
                    channel = image_copy[..., c]
                    try:
                        # Use percentiles based on saturated_pixels parameter (ImageJ-like)
                        p_low, p_high = saturated_pixels / 2, 100 - saturated_pixels / 2
                        p_min, p_max = np.percentile(channel, (p_low, p_high))

                        # Ensure we don't divide by zero
                        if p_max > p_min:
                            # Rescale using skimage's exposure module
                            channel_rescaled = exposure.rescale_intensity(
                                channel, in_range=(p_min, p_max), out_range=(0, 255)
                            )
                            result[..., c] = img_as_ubyte(channel_rescaled)
                        else:
                            result[..., c] = np.zeros_like(channel, dtype=np.uint8)
                    except Exception as e:
                        logger.warning(f"Error normalizing channel {c}: {e}")
                        # Fallback to simple normalization
                        min_val = np.min(channel)
                        max_val = np.max(channel)
                        if max_val > min_val:
                            result[..., c] = np.clip(
                                ((channel - min_val) / (max_val - min_val) * 255),
                                0,
                                255,
                            ).astype(np.uint8)
                        else:
                            result[..., c] = np.zeros_like(channel, dtype=np.uint8)
                image_copy = result
            else:
                # For grayscale or other images, process the whole array
                try:
                    # Use percentiles based on saturated_pixels parameter (ImageJ-like)
                    p_low, p_high = saturated_pixels / 2, 100 - saturated_pixels / 2
                    p_min, p_max = np.percentile(image_copy, (p_low, p_high))

                    # Ensure we don't divide by zero
                    if p_max > p_min:
                        # Rescale using skimage's exposure module
                        image_rescaled = exposure.rescale_intensity(
                            image_copy, in_range=(p_min, p_max), out_range=(0, 255)
                        )
                        image_copy = img_as_ubyte(image_rescaled)
                    else:
                        # Fallback if the image has no contrast
                        image_copy = np.zeros_like(image_copy, dtype=np.uint8)
                except Exception as e:
                    logger.warning(f"Error in rescale_intensity: {e}")
                    # Fallback to simple normalization
                    min_val = np.min(image_copy)
                    max_val = np.max(image_copy)
                    if max_val > min_val:
                        image_copy = np.clip(
                            ((image_copy - min_val) / (max_val - min_val) * 255), 0, 255
                        ).astype(np.uint8)
                    else:
                        image_copy = np.zeros_like(image_copy, dtype=np.uint8)
        elif normalize:
            # Normalize to full range without using percentiles
            # For multi-channel images, process each channel separately
            if is_multichannel and image_copy.shape[-1] <= 4:
                result = np.zeros_like(image_copy, dtype=np.uint8)
                for c in range(image_copy.shape[-1]):
                    channel = image_copy[..., c]
                    try:
                        # Get actual min and max for normalization
                        min_val = np.min(channel)
                        max_val = np.max(channel)
                        if max_val > min_val:
                            # Normalize to full range
                            channel_normalized = exposure.rescale_intensity(
                                channel, in_range=(min_val, max_val), out_range=(0, 255)
                            )
                            result[..., c] = img_as_ubyte(channel_normalized)
                        else:
                            result[..., c] = np.zeros_like(channel, dtype=np.uint8)
                    except Exception as e:
                        logger.warning(f"Error normalizing channel {c}: {e}")
                        result[..., c] = np.zeros_like(channel, dtype=np.uint8)
                image_copy = result
            else:
                try:
                    # Get actual min and max for normalization
                    min_val = np.min(image_copy)
                    max_val = np.max(image_copy)
                    if max_val > min_val:
                        # Normalize to full range
                        image_normalized = exposure.rescale_intensity(
                            image_copy, in_range=(min_val, max_val), out_range=(0, 255)
                        )
                        image_copy = img_as_ubyte(image_normalized)
                    else:
                        image_copy = np.zeros_like(image_copy, dtype=np.uint8)
                except Exception as e:
                    logger.warning(f"Error in normalization: {e}")
                    image_copy = np.zeros_like(image_copy, dtype=np.uint8)
        else:
            # Scale to the image's digitization bit depth (effective, not just dtype)
            max_val = (1 << bit_depth) - 1  # 2^bit_depth - 1

            # For multi-channel images, process each channel separately
            # Handle RGB/RGBA images
            if is_multichannel and image_copy.shape[-1] <= 4:
                # Process each channel separately
                result = np.zeros_like(image_copy, dtype=np.uint8)
                for c in range(image_copy.shape[-1]):
                    channel = image_copy[..., c]
                    try:
                        # Use skimage's rescale_intensity for more robust scaling
                        channel_rescaled = exposure.rescale_intensity(
                            channel, in_range=(0, max_val), out_range=(0, 255)
                        )
                        result[..., c] = img_as_ubyte(channel_rescaled)
                    except Exception as e:
                        logger.warning(f"Error normalizing channel {c}: {e}")
                        # Fallback to simple normalization
                        result[..., c] = np.clip(
                            (channel / max_val * 255), 0, 255
                        ).astype(np.uint8)
                image_copy = result
            else:
                # For grayscale or other images, process the whole array
                try:
                    image_rescaled = exposure.rescale_intensity(
                        image_copy, in_range=(0, max_val), out_range=(0, 255)
                    )
                    image_copy = img_as_ubyte(image_rescaled)
                except Exception as e:
                    logger.warning(f"Error in rescale_intensity: {e}")
                    # Fallback to simple normalization
                    image_copy = np.clip((image_copy / max_val * 255), 0, 255).astype(
                        np.uint8
                    )

    # Invert if requested (useful for some microscopy images)
    if invert:
        image_copy = 255 - image_copy

    return image_copy


def get_unique_id_for_image(image_file) -> str:
    try:
        if isinstance(image_file, str):
            filename = os.path.basename(image_file)
        else:
            filename = (
                image_file.name if hasattr(image_file, "name") else id(image_file)
            )
        short_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        return f"img_{short_hash}"
    except Exception as e:
        logger.error(f"Error generating unique ID: {e}")
        return f"img_{int(time.time() * 1000)}"


def load_default_image():
    return DEFAULT_IMAGE_PATH if os.path.exists(DEFAULT_IMAGE_PATH) else None


def process_uploaded_file(uploaded_file) -> tuple[np.ndarray | None, str | None]:
    if uploaded_file is None:
        return None, None
    try:
        # Get image processing settings from session state if available
        unique_id = get_unique_id_for_image(uploaded_file)
        autoscale = st.session_state.get(f"autoscale_{unique_id}", True)
        invert = st.session_state.get(f"invert_{unique_id}", False)
        normalize = st.session_state.get(f"normalize_{unique_id}", False)
        saturated_pixels = st.session_state.get(f"saturated_pixels_{unique_id}", 0.5)
        equalize_histogram = st.session_state.get(
            f"equalize_histogram_{unique_id}", False
        )

        if isinstance(uploaded_file, str):
            if not os.path.exists(uploaded_file):
                st.error(f"File not found: {uploaded_file}")
                return None, None
            ext = os.path.splitext(uploaded_file)[1].lower()
            if ext in [".tif", ".tiff"]:
                try:
                    with tifffile.TiffFile(uploaded_file) as tif:
                        if len(tif.pages) == 0:
                            st.error(f"TIFF file has no pages: {uploaded_file}")
                            return None, None
                        image = tif.pages[0].asarray()
                    logger.info(
                        f"Loaded TIFF image: shape={image.shape}, dtype={image.dtype}, range={np.min(image)}-{np.max(image)}"
                    )
                    # Store the effective bit depth in session state
                    bit_depth = _get_effective_bit_depth(image)
                    st.session_state[f"bit_depth_{unique_id}"] = bit_depth
                    # Normalize using user settings
                    try:
                        image = normalize_to_uint8(
                            image,
                            autoscale=autoscale,
                            invert=invert,
                            normalize=normalize,
                            saturated_pixels=saturated_pixels,
                            equalize_histogram=equalize_histogram,
                        )
                        if image.ndim == 2:
                            image = np.stack([image] * 3, axis=-1)
                        elif image.shape[-1] == 1:
                            image = np.repeat(image, 3, axis=-1)
                        return image, uploaded_file
                    except Exception as e:
                        logger.error(f"Error normalizing TIFF image: {e}")
                        st.error(f"Error normalizing TIFF image: {e}")
                        return None, None
                except Exception as e:
                    logger.error(f"Failed to load TIFF: {uploaded_file} ({e})")
                    st.error(f"Failed to load TIFF: {uploaded_file} ({e})")
                    return None, None
            else:
                try:
                    # Try OpenCV first
                    image = cv2.imread(uploaded_file, cv2.IMREAD_UNCHANGED)
                    if image is None:
                        # If OpenCV fails, try PIL
                        logger.info(
                            f"OpenCV failed to load image, trying PIL: {uploaded_file}"
                        )
                        pil_image = Image.open(uploaded_file)
                        image = np.array(pil_image)

                    if image is not None:
                        logger.info(
                            f"Loaded image: shape={image.shape}, dtype={image.dtype}, range={np.min(image)}-{np.max(image)}"
                        )
                        # Convert BGR to RGB if needed (OpenCV loads as BGR)
                        if len(image.shape) == 3 and image.shape[2] == 3:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        # Determine bit depth from loaded image
                        bit_depth = _get_effective_bit_depth(image)
                        st.session_state[f"bit_depth_{unique_id}"] = bit_depth
                        try:
                            image = normalize_to_uint8(
                                image,
                                autoscale=autoscale,
                                invert=invert,
                                normalize=normalize,
                                saturated_pixels=saturated_pixels,
                                equalize_histogram=equalize_histogram,
                            )
                            if image.ndim == 2:
                                image = np.stack([image] * 3, axis=-1)
                            elif image.shape[-1] == 1:
                                image = np.repeat(image, 3, axis=-1)
                            return image, uploaded_file
                        except Exception as e:
                            logger.error(f"Error normalizing image: {e}")
                            st.error(f"Error normalizing image: {e}")
                            return None, None
                    else:
                        logger.error(
                            f"Failed to load image with both OpenCV and PIL: {uploaded_file}"
                        )
                        st.error(f"Failed to load image: {uploaded_file}")
                        return None, None
                except Exception as e:
                    logger.error(f"Error loading image: {uploaded_file} ({e})")
                    st.error(f"Error loading image: {uploaded_file} ({e})")
                    return None, None
        else:
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name
                ext = os.path.splitext(temp_path)[1].lower()
                if ext in [".tif", ".tiff"]:
                    try:
                        with tifffile.TiffFile(temp_path) as tif:
                            if len(tif.pages) == 0:
                                st.error(
                                    f"TIFF file has no pages: {uploaded_file.name}"
                                )
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                                return None, None
                            image = tif.pages[0].asarray()
                        logger.info(
                            f"Loaded TIFF image: shape={image.shape}, dtype={image.dtype}, range={np.min(image)}-{np.max(image)}"
                        )
                        bit_depth = _get_effective_bit_depth(image)
                        st.session_state[f"bit_depth_{unique_id}"] = bit_depth
                        try:
                            image = normalize_to_uint8(
                                image,
                                autoscale=autoscale,
                                invert=invert,
                                normalize=normalize,
                                saturated_pixels=saturated_pixels,
                                equalize_histogram=equalize_histogram,
                            )
                            if image.ndim == 2:
                                image = np.stack([image] * 3, axis=-1)
                            elif image.shape[-1] == 1:
                                image = np.repeat(image, 3, axis=-1)
                            return image, temp_path
                        except Exception as e:
                            logger.error(f"Error normalizing TIFF image: {e}")
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            st.error(f"Error normalizing TIFF image: {e}")
                            return None, None
                    except Exception as e:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        logger.error(f"Failed to load TIFF: {uploaded_file.name} ({e})")
                        st.error(f"Failed to load TIFF: {uploaded_file.name} ({e})")
                        return None, None
                else:
                    try:
                        # Try OpenCV first
                        image = cv2.imread(temp_path, cv2.IMREAD_UNCHANGED)
                        if image is None:
                            # If OpenCV fails, try PIL
                            logger.info(
                                f"OpenCV failed to load image, trying PIL: {uploaded_file.name}"
                            )
                            pil_image = Image.open(temp_path)
                            image = np.array(pil_image)

                        if image is not None:
                            logger.info(
                                f"Loaded image: shape={image.shape}, dtype={image.dtype}, range={np.min(image)}-{np.max(image)}"
                            )
                            # Convert BGR to RGB if needed (OpenCV loads as BGR)
                            if len(image.shape) == 3 and image.shape[2] == 3:
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            bit_depth = _get_effective_bit_depth(image)
                            st.session_state[f"bit_depth_{unique_id}"] = bit_depth
                            try:
                                image = normalize_to_uint8(
                                    image,
                                    autoscale=autoscale,
                                    invert=invert,
                                    normalize=normalize,
                                    saturated_pixels=saturated_pixels,
                                    equalize_histogram=equalize_histogram,
                                )
                                if image.ndim == 2:
                                    image = np.stack([image] * 3, axis=-1)
                                elif image.shape[-1] == 1:
                                    image = np.repeat(image, 3, axis=-1)
                                return image, temp_path
                            except Exception as e:
                                logger.error(f"Error normalizing image: {e}")
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                                st.error(f"Error normalizing image: {e}")
                                return None, None
                        else:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            logger.error(
                                f"Failed to load image with both OpenCV and PIL: {uploaded_file.name}"
                            )
                            st.error(f"Failed to load image: {uploaded_file.name}")
                            return None, None
                    except Exception as e:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        logger.error(f"Error loading image: {uploaded_file.name} ({e})")
                        st.error(f"Error loading image: {uploaded_file.name} ({e})")
                        return None, None
            except Exception as e:
                logger.error(f"Error processing uploaded file: {e}")
                st.error(f"Error processing uploaded file: {e}")
                return None, None
        return None, None
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        st.error(f"Error processing file: {e}")
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return None, None


def extract_roi_image(
    image, roi_coordinates: tuple[int, int, int, int], rotation: int = 0
) -> np.ndarray | None:
    try:
        if roi_coordinates is None:
            return None
        x, y, width, height = roi_coordinates
        if hasattr(image, "select_roi"):
            roi = image.select_roi(roi_coordinates)
        elif image is not None and x >= 0 and y >= 0 and width > 0 and height > 0:
            roi = image[y : y + height, x : x + width]
        else:
            return None

        # Apply rotation if specified
        if rotation > 0 and roi is not None:
            roi = rotate_image(roi, rotation)

        return roi
    except Exception as e:
        st.error(f"Error extracting ROI: {e}")
        return None


def initialize_session_state():
    if "usaf_target" not in st.session_state:
        st.session_state.usaf_target = USAFTarget()
    if "uploaded_files_list" not in st.session_state:
        st.session_state.uploaded_files_list = []
    if "default_image_added" not in st.session_state:
        st.session_state.default_image_added = False
    if "image_index_to_id" not in st.session_state:
        st.session_state.image_index_to_id = {}

    # Handle cleanup of rotation and sensitivity session state when images are removed
    if "rotation_state_cleanup" not in st.session_state:
        st.session_state.rotation_state_cleanup = set()

    current_image_ids = {
        get_unique_id_for_image(file) for file in st.session_state.uploaded_files_list
    }
    # Store the current set of image IDs for next cleanup check
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
                f"roi_rotation_{image_id}",  # Add ROI rotation state cleanup
                # Add last ROI rotation state cleanup
                f"last_roi_rotation_{image_id}",
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
        # Add key for ROI rotation
        "roi_rotation": f"roi_rotation_{unique_id}",
        # Add key for last ROI rotation
        "last_roi_rotation": f"last_roi_rotation_{unique_id}",
    }


def find_best_two_line_pairs(dark_bar_starts):
    """
    Given a list of dark bar starts, find the two consecutive pairs whose widths are most similar.
    Returns the two pairs and their average width.
    """
    pairs = [
        (dark_bar_starts[i], dark_bar_starts[i + 1])
        for i in range(len(dark_bar_starts) - 1)
    ]
    widths = [end - start for start, end in pairs]
    if len(widths) < 2:
        return [], 0.0  # Not enough pairs
    # Find the two widths that are closest to each other
    min_diff = float("inf")
    best_indices = (0, 1)
    for i in range(len(widths)):
        for j in range(i + 1, len(widths)):
            diff = abs(widths[i] - widths[j])
            if diff < min_diff:
                min_diff = diff
                best_indices = (i, j)
    # Get the best two pairs and their average width
    best_pairs = [pairs[best_indices[0]], pairs[best_indices[1]]]
    avg_width = (widths[best_indices[0]] + widths[best_indices[1]]) / 2
    return best_pairs, avg_width


# --- Core Classes ---


class USAFTarget:
    def __init__(self):
        self.base_lp_per_mm = 1.0

    def lp_per_mm(self, group: int, element: int) -> float:
        return self.base_lp_per_mm * (2 ** (group + (element - 1) / 6))

    def line_pair_width_microns(self, group: int, element: int) -> float:
        return 1000.0 / self.lp_per_mm(group, element)


def detect_significant_transitions(profile):
    """
    Detect significant intensity transitions in a profile using only sign changes in the derivative.
    Returns:
        tuple of (all_transitions, transition_types, derivative)
    """
    derivative = np.diff(profile)
    # Find zero crossings in the derivative (sign changes)
    sign_changes = np.where(np.diff(np.sign(derivative)) != 0)[0] + 1
    all_transitions = sign_changes.tolist()
    # Determine transition type: 1 for positive slope, -1 for negative slope
    transition_types = [
        1 if derivative[i - 1] < derivative[i] else -1 for i in sign_changes
    ]
    return all_transitions, transition_types, derivative


def extract_alternating_patterns(transitions, transition_types):
    """
    Extract alternating light-to-dark and dark-to-light transition patterns.

    Args:
        transitions: Array of transition positions
        transition_types: Array of transition types (1: dark-to-light, -1: light-to-dark)

    Returns:
        tuple of (pattern_transitions, pattern_types)
    """
    if len(transitions) <= 2:
        return transitions, transition_types

    # Try to identify proper line pair transitions by looking for alternating patterns
    proper_transitions = []
    proper_types = []

    start_idx = 1 if transition_types and transition_types[0] == 1 else 0
    # Extract transitions by expected pattern
    i = start_idx
    while i < len(transitions) - 1:
        # Check for a light-to-dark followed by dark-to-light pattern
        if (
            i + 1 < len(transition_types)
            and transition_types[i] == -1
            and transition_types[i + 1] == 1
        ):
            proper_transitions.extend([transitions[i], transitions[i + 1]])
            proper_types.extend([transition_types[i], transition_types[i + 1]])
            i += 2
        else:
            # Skip this transition if it doesn't fit the pattern
            i += 1

    # If we found proper transitions, use them
    if len(proper_transitions) >= 2:
        return proper_transitions, proper_types

    return transitions, transition_types


def limit_transitions_to_strongest(
    transitions, transition_types, derivative, max_transitions=5, min_strength=10
):
    """
    If there are too many transitions, keep only the strongest ones above a minimum strength threshold.

    Args:
        transitions: Array of transition positions
        transition_types: Array of transition types
        derivative: The derivative array
        max_transitions: Maximum number of transitions to keep
        min_strength: Minimum absolute derivative value to consider a transition

    Returns:
        tuple of (strongest_transitions, strongest_types)
    """
    # Filter out transitions below the minimum strength
    filtered = [
        (t, typ)
        for t, typ in zip(transitions, transition_types, strict=False)
        if abs(derivative[t]) >= min_strength
    ]
    if not filtered:
        return [], []
    filtered_transitions, filtered_types = zip(*filtered, strict=False)
    filtered_transitions = list(filtered_transitions)
    filtered_types = list(filtered_types)
    if len(filtered_transitions) <= max_transitions:
        return filtered_transitions, filtered_types
    # Sort transitions by derivative magnitude
    transition_strengths = np.abs([derivative[t] for t in filtered_transitions])
    strongest_indices = np.argsort(transition_strengths)[-max_transitions:]
    # Resort by position to maintain order
    strongest_indices = np.sort(strongest_indices)
    strongest_transitions = [filtered_transitions[i] for i in strongest_indices]
    strongest_types = [filtered_types[i] for i in strongest_indices]
    return strongest_transitions, strongest_types


def find_line_pair_boundaries_derivative(profile):
    """
    Find line pair boundaries using sign changes in the derivative.
    Returns:
        (dark_bar_starts, derivative, transition_types)
    Only -1 (light-to-dark) transitions are returned as boundaries.
    """
    all_transitions, all_types, derivative = detect_significant_transitions(profile)
    pattern_transitions, pattern_types = extract_alternating_patterns(
        all_transitions, all_types
    )
    # Adaptive threshold: 20% of max derivative
    max_deriv = np.max(np.abs(derivative)) if len(derivative) > 0 else 0
    min_strength = 0.2 * max_deriv if max_deriv > 0 else 0
    final_transitions, final_types = limit_transitions_to_strongest(
        pattern_transitions, pattern_types, derivative, min_strength=min_strength
    )
    # Only keep -1 transitions (light-to-dark, i.e., dark bar starts)
    dark_bar_starts = [
        t for t, typ in zip(final_transitions, final_types, strict=False) if typ == -1
    ]
    dark_bar_types = [-1] * len(dark_bar_starts)
    return dark_bar_starts, derivative, dark_bar_types


def find_line_pair_boundaries_windowed(profile, window=5):
    """
    Find line pair boundaries using sign changes in a windowed mean difference.
    Returns:
        (dark_bar_starts, pseudo_derivative, transition_types)
    Only -1 (light-to-dark) transitions are returned as boundaries.
    """
    profile = np.asarray(profile)
    pseudo_derivative = np.zeros_like(profile, dtype=float)
    edges = []
    transition_types = []
    for i in range(window, len(profile) - window):
        left = np.mean(profile[i - window : i])
        right = np.mean(profile[i : i + window])
        diff = right - left
        pseudo_derivative[i] = diff
        if i > window and np.sign(pseudo_derivative[i - 1]) != np.sign(diff):
            edges.append(i)
            transition_types.append(1 if diff > 0 else -1)
    pattern_transitions, pattern_types = extract_alternating_patterns(
        edges, transition_types
    )
    # Adaptive threshold: 20% of max pseudo_derivative
    max_deriv = np.max(np.abs(pseudo_derivative)) if len(pseudo_derivative) > 0 else 0
    min_strength = 0.2 * max_deriv if max_deriv > 0 else 0
    final_transitions, final_types = limit_transitions_to_strongest(
        pattern_transitions, pattern_types, pseudo_derivative, min_strength=min_strength
    )
    # Only keep -1 transitions (light-to-dark, i.e., dark bar starts)
    dark_bar_starts = [
        t for t, typ in zip(final_transitions, final_types, strict=False) if typ == -1
    ]
    dark_bar_types = [-1] * len(dark_bar_starts)
    return dark_bar_starts, pseudo_derivative, dark_bar_types


def find_line_pair_boundaries_threshold(profile, threshold):
    """
    Find line pair boundaries by locating where the profile crosses a threshold value.

    Args:
        profile: The intensity profile array
        threshold: The threshold value to use

    Returns:
        (dark_bar_starts, thresholded_profile, transition_types)
    Only -1 (light-to-dark) transitions are returned as boundaries.
    """
    # Convert profile to numpy array
    profile_array = np.array(profile)

    # Ensure threshold is within valid range for uint8 data (0-255)
    threshold = max(0, min(255, threshold))

    # Create a binary mask where True is above threshold
    above_threshold = profile_array > threshold

    # Find light-to-dark transitions directly
    dark_bar_starts = []
    for i in range(1, len(above_threshold)):
        # Look for transitions from above threshold to below threshold (light-to-dark)
        if above_threshold[i - 1] == True and above_threshold[i] == False:
            dark_bar_starts.append(i)

    # Create corresponding transition types (all -1 for light-to-dark)
    transition_types = [-1] * len(dark_bar_starts)

    # Debug output about what was found
    logger.debug(
        f"Profile range: {np.min(profile_array)} to {np.max(profile_array)}, threshold: {threshold}"
    )
    logger.debug(
        f"Found {len(dark_bar_starts)} dark bar starts (light-to-dark transitions)"
    )
    if len(dark_bar_starts) > 0:
        logger.debug(f"Dark bar starts at positions: {dark_bar_starts[:10]}...")
    else:
        logger.warning(f"No dark bar starts found with threshold {threshold}!")

    # Create a pseudo derivative for compatibility with the rest of the code
    thresholded_profile = np.ones_like(profile_array) * threshold

    return dark_bar_starts, thresholded_profile, transition_types


class RoiManager:
    """
    Class for managing Regions of Interest (ROIs) in images.
    Handles selection, validation, and extraction of ROIs.
    """

    def __init__(self):
        self.coordinates = None  # (point1, point2) tuple
        self.roi_tuple = None  # (x, y, width, height) tuple
        self.is_valid = False

    def set_coordinates(self, point1, point2):
        """Set ROI coordinates from two points and validate the selection"""
        self.coordinates = (point1, point2)
        self.validate_and_convert()
        return self.is_valid

    def validate_and_convert(self):
        """Convert corner points to (x, y, width, height) format and validate"""
        if self.coordinates is None:
            self.is_valid = False
            self.roi_tuple = None
            return

        point1, point2 = self.coordinates
        roi_x = min(point1[0], point2[0])
        roi_y = min(point1[1], point2[1])
        roi_width = abs(point2[0] - point1[0])
        roi_height = abs(point2[1] - point1[1])

        # Basic validation: ensure non-zero dimensions
        if roi_width <= 0 or roi_height <= 0:
            logger.warning(
                f"Invalid ROI dimensions: width={roi_width}, height={roi_height}"
            )
            self.is_valid = False
            self.roi_tuple = None
        else:
            self.roi_tuple = (int(roi_x), int(roi_y), int(roi_width), int(roi_height))
            self.is_valid = True

    def validate_against_image(self, image):
        """
        Validate ROI against image dimensions

        Args:
            image: Image to validate against (numpy array or PIL Image)

        Returns:
            bool: True if valid, False otherwise
        """
        if not self.is_valid or self.roi_tuple is None:
            return False

        roi_x, roi_y, roi_width, roi_height = self.roi_tuple

        # Get image dimensions
        img_height, img_width = None, None
        if hasattr(image, "shape"):
            if len(image.shape) > 1:
                img_height, img_width = image.shape[0], image.shape[1]
        elif hasattr(image, "size"):
            img_width, img_height = image.size

        # Validate ROI is within image bounds
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
            self.is_valid = False

        return self.is_valid

    def extract_roi(self, image):
        """
        Extract ROI from image

        Args:
            image: Image to extract ROI from (numpy array)

        Returns:
            numpy.ndarray: Extracted ROI or None if invalid
        """
        if not self.is_valid or self.roi_tuple is None:
            return None

        roi_x, roi_y, roi_width, roi_height = self.roi_tuple

        try:
            if hasattr(image, "select_roi"):
                return image.select_roi(self.roi_tuple)

            if (
                image is not None
                and roi_x >= 0
                and roi_y >= 0
                and roi_width > 0
                and roi_height > 0
            ):
                return image[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]

            return None
        except Exception as e:
            logger.error(f"Error extracting ROI: {e}")
            return None


class ProfileVisualizer:
    """
    Class for visualizing profile analysis results.
    Handles generating plots and visualizations of intensity profiles and transition analysis.
    """

    def __init__(self):
        # Configure default plot style
        self.configure_plot_style()

        # Define colors for different elements
        self.light_to_dark_color = "#FF4500"  # Orange-red

        self.annotation_color = "#FFFF99"  # Yellow
        self.profile_color = "#2c3e50"  # Dark blue-gray
        self.individual_profile_color = "#6b88b6"  # Light blue

        # Shadow effect for text
        self.shadow_effect = [
            PathEffects.withSimplePatchShadow(
                offset=(1.0, -1.0), shadow_rgbFace="black", alpha=0.6
            )
        ]

    def configure_plot_style(self):
        """Configure matplotlib plot style"""
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "DejaVu Serif", "Palatino", "serif"],
                "mathtext.fontset": "stix",
                "axes.titlesize": 11,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
            }
        )

    def create_figure(self, figsize=(8, 8), dpi=150):
        """Create a square matplotlib figure with properly configured layout"""
        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.05)

        # Set explicit figure margins to avoid tight_layout issues
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)

        # Create subplot axes
        ax_img = fig.add_subplot(gs[0])
        ax_profile = fig.add_subplot(gs[1], sharex=ax_img)

        # Configure axes appearance
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_ylabel("")

        # Increased font size
        ax_profile.set_xlabel("Position (pixels)", fontsize=12)
        # Increased font size
        ax_profile.set_ylabel("Intensity (a.u.)", fontsize=12)
        ax_profile.grid(True, alpha=0.15, linestyle="-", linewidth=0.5)
        # Increased tick font size
        ax_profile.tick_params(axis="both", which="major", labelsize=11)

        # Remove unnecessary spines
        for ax in [ax_img, ax_profile]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax_img.spines["left"].set_visible(False)
        ax_img.spines["bottom"].set_visible(False)

        return fig, ax_img, ax_profile

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
        """Plot the ROI image"""
        ax.imshow(image, cmap="gray", aspect="auto", interpolation="bicubic")

        if group is not None and element is not None:
            title_lines = []
            group_str = f"$\\mathbf{{{group}}}$"
            element_str = f"$\\mathbf{{{element}}}$"
            title_lines.append(f"USAF Target: Group {group_str}, Element {element_str}")

            # Combine Line Pairs per mm and Avg Line Pair Width on one line
            lp_per_mm_str = ""
            avg_width_str = ""

            if lp_per_mm is not None:
                lp_per_mm_str = f"Line Pairs/mm: $\\mathbf{{{lp_per_mm:.2f}}}$"

            if avg_line_pair_width is not None and avg_line_pair_width > 0:
                avg_width_str = (
                    f"Avg. LP Width: $\\mathbf{{{avg_line_pair_width:.2f}}}$ px"
                )

            # Add combined line if either value is available
            if lp_per_mm_str or avg_width_str:
                combined_str = ""
                if lp_per_mm_str and avg_width_str:
                    combined_str = f"{lp_per_mm_str}  |  {avg_width_str}"
                else:
                    combined_str = lp_per_mm_str or avg_width_str
                title_lines.append(combined_str)

            # Combine Pixel Size and Magnification on one line
            if (
                avg_line_pair_width is not None
                and avg_line_pair_width > 0
                and lp_width_um is not None
            ):
                pixel_size = lp_width_um / avg_line_pair_width
                pixel_size_str = f"Pixel Size: $\\mathbf{{{pixel_size:.3f}}}$ µm/pixel"

                mag_str = ""
                if magnification is not None:
                    mag_str = f"Magnification: $\\mathbf{{{magnification:.1f}\\times}}$"

                # Add combined line if both values are available
                if pixel_size_str and mag_str:
                    title_lines.append(f"{pixel_size_str}  |  {mag_str}")
                else:
                    if pixel_size_str:
                        title_lines.append(pixel_size_str)
                    if mag_str:
                        title_lines.append(mag_str)

            title_text = "\n".join(title_lines)
            ax.set_title(
                title_text,
                fontweight="normal",
                pad=15,
                fontsize=20,
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
        """Plot the intensity profile and individual profiles
        Args:
            ax: The matplotlib axis to plot on
            profile: The 1D intensity profile
            individual_profiles: Optional, the 2D array of individual row profiles
            avg_line_pair_width: Optional, average line pair width for annotation
            profile_type: 'mean' or 'max', used for labeling
        """
        # Plot individual profiles if available
        if individual_profiles is not None:
            n_rows = individual_profiles.shape[0]
            step = max(1, n_rows // 20)
            for i in range(0, n_rows, step):
                ax.plot(
                    individual_profiles[i],
                    color=self.individual_profile_color,
                    alpha=0.12,
                    linewidth=0.7,
                    zorder=1,
                )
            # Create label for intensity
            intensity_label = "Max Intensity"
        if avg_line_pair_width > 0:
            intensity_label += f"\n(Avg. LP Width: {avg_line_pair_width:.1f} px)"

        # Plot profile
        ax.plot(
            profile,
            color=self.profile_color,
            linewidth=2.5,
            alpha=1.0,  # Increased linewidth
            label=intensity_label,
            zorder=2,
        )

        # Plot threshold line if applicable
        if threshold is not None and edge_method == "threshold":
            ax.axhline(
                y=threshold,
                color="r",
                linestyle="--",
                alpha=0.8,
                linewidth=1.5,
                zorder=3,
                label="Threshold",
            )  # Increased linewidth

        # Increase font size for axis labels and ticks
        # Increased tick font size
        ax.tick_params(axis="both", which="major", labelsize=11)
        # Increased label font size
        ax.set_xlabel("Position (pixels)", fontsize=12)
        # Increased label font size
        ax.set_ylabel("Intensity (a.u.)", fontsize=12)

    def find_line_pairs(self, boundaries, roi_img):
        """
        Find and return only the two best-matching line pairs for annotation.
        """
        best_pairs, _ = find_best_two_line_pairs(boundaries)
        line_pairs = []
        for j, (start_pos, end_pos) in enumerate(best_pairs):
            width_px = end_pos - start_pos
            if width_px >= 5:
                line_pairs.append((start_pos, end_pos, width_px, j))
        return line_pairs

    def draw_bracket(self, ax, x_start, x_end, y_pos, tick_size, color):
        """Draw a bracket with ticks between two x-positions"""
        line_width = 1.5

        # Draw horizontal line
        ax.annotate(
            "",
            xy=(x_start, y_pos),
            xytext=(x_end, y_pos),
            arrowprops=dict(arrowstyle="-", color=color, linewidth=line_width),
        )

        # Draw tick marks on ends
        ax.annotate(
            "",
            xy=(x_start, y_pos),
            xytext=(x_start, y_pos + tick_size),
            arrowprops=dict(arrowstyle="-", color=color, linewidth=line_width),
        )
        ax.annotate(
            "",
            xy=(x_end, y_pos),
            xytext=(x_end, y_pos + tick_size),
            arrowprops=dict(arrowstyle="-", color=color, linewidth=line_width),
        )

    def annotate_line_pairs(self, ax, line_pairs, roi_img):
        """
        Annotate line pairs with brackets and labels using dark bar starts.
        """
        vertical_position = (
            0.35  # Initial vertical position (0-1 relative to image height)
        )
        tick_height = 0.04  # Height of the tick marks
        for start_pos, end_pos, width_px, j in line_pairs:
            y_offset = 0.0 if j == 0 else 0.15 * j
            y_pos = roi_img.shape[0] * (vertical_position + y_offset)
            tick_size = roi_img.shape[0] * tick_height
            mid_point = (start_pos + end_pos) / 2
            self.draw_bracket(
                ax, start_pos, end_pos, y_pos, tick_size, self.annotation_color
            )
            label_y_pos = y_pos + tick_size * 0.5
            line_pair_text = ax.text(
                mid_point,
                label_y_pos,
                "Line Pair",
                color=self.annotation_color,
                ha="center",
                va="top",
                fontsize=16,
                fontweight="bold",
            )  # Increased font size
            line_pair_text.set_path_effects(self.shadow_effect)
            measurement = f"{int(round(width_px))} px"
            measurement_text = ax.text(
                mid_point,
                label_y_pos + tick_size * 1.5,
                measurement,
                color=self.annotation_color,
                ha="center",
                va="top",
                fontsize=14,
                fontweight="bold",
            )  # Increased font size
            measurement_text.set_path_effects(self.shadow_effect)

    def create_caption(
        self,
        group=None,
        element=None,
        lp_width_um=None,
        edge_method=None,
        lp_per_mm=None,
    ):
        """Create an HTML caption for the plot, including edge detection method."""
        if edge_method == "parallel":
            method_str = "Windowed Step (Robust)"
        elif edge_method == "threshold":
            method_str = "Threshold-based"
        else:
            method_str = "Original"

        if group is not None and element is not None and lp_width_um is not None:
            lp_per_mm_str = f"{lp_per_mm:.2f} lp/mm" if lp_per_mm is not None else ""

            caption = f"""
            <div style='text-align:center; font-family: "Times New Roman", Times, serif;'>
                <p style='margin-bottom:0.3rem; font-size:1.25rem;'><b>Figure: Intensity Profile Analysis of USAF Target</b></p>
                <p style='margin-top:0; font-size:1.0rem; color:#333;'>
                    Group {group}, Element {element} with theoretical line pair width of {lp_width_um:.2f} µm 
                    {f"({lp_per_mm_str})" if lp_per_mm_str else ""}.<br>
                    <b>Max Intensity Profile</b> with <b>Edge Detection Method:</b> <span style='color:#0074D9'>{method_str}</span><br>
                    Each line pair consists of one complete dark bar and one complete light bar.<br>
                    <span style='color:{self.light_to_dark_color};'>Orange</span> lines indicate the start of dark bars (threshold crossings).
                </p>
            </div>
            """
        else:
            caption = f"""
            <div style='text-align:center; font-family: "Times New Roman", Times, serif;'>
                <p style='margin-bottom:0.3rem; font-size:1.25rem;'><b>Figure: Aligned Visual Analysis</b></p>
                <p style='margin-top:0; font-size:1.0rem; color:#333;'>
                    <b>Max Intensity Profile</b> with <b>Edge Detection Method:</b> <span style='color:#0074D9'>{method_str}</span><br>
                    Each line pair consists of one complete dark bar and one complete light bar.<br>
                    <span style='color:{self.light_to_dark_color};'>Orange</span> lines indicate the start of dark bars (threshold crossings).
                </p>
            </div>
            """
        return caption

    def visualize_profile(
        self,
        results,
        roi_img,
        group=None,
        element=None,
        lp_width_um=None,
        magnification=None,
    ):
        """
        Create a complete visualization of profile analysis
        Args:
            results: Analysis results dictionary
            roi_img: The ROI image to display
            group: USAF target group
            element: USAF target element
            lp_width_um: Theoretical line pair width in microns
            magnification: User-provided magnification value
        Returns:
            fig: Matplotlib figure
        """
        # Validate inputs
        if "profile" not in results or results["profile"] is None or roi_img is None:
            return None
        # Convert profile to numpy array
        profile = np.array(results["profile"])
        # Create figure
        fig, ax_img, ax_profile = self.create_figure()
        # Plot ROI image
        avg_line_pair_width = results.get("avg_line_pair_width", 0.0)
        # Get Line Pairs per mm from results
        lp_per_mm = results.get("lp_per_mm", None)
        self.plot_image(
            ax_img,
            roi_img,
            group,
            element,
            avg_line_pair_width,
            lp_width_um,
            magnification,
            lp_per_mm,
        )
        # Plot profiles
        individual_profiles = results.get("individual_profiles")
        profile_type = results.get("profile_type", "max")
        edge_method = results.get("edge_method", "original")
        threshold = results.get("threshold", None)
        self.plot_profiles(
            ax_profile,
            profile,
            individual_profiles,
            avg_line_pair_width,
            profile_type=profile_type,
            edge_method=edge_method,
            threshold=threshold,
        )
        # Get transition information
        boundaries = results.get("boundaries", [])

        # Debug info about boundaries
        logger.debug(f"Visualizing profile with {len(boundaries)} boundaries")
        if len(boundaries) > 0:
            logger.debug(f"Boundary positions: {boundaries[:10]}...")

        # Initialize line_pairs before conditional code
        line_pairs = []

        # Draw transition lines and annotations if we have any boundaries
        if len(boundaries) > 0:
            # Draw transition boundary lines (all are dark bar starts)
            for boundary in boundaries:
                ax_profile.axvline(
                    x=boundary,
                    color=self.light_to_dark_color,
                    linestyle="-",
                    alpha=0.7,
                    linewidth=1.8,
                    zorder=4,
                )  # Increased linewidth
                ax_img.axvline(
                    x=boundary,
                    color=self.light_to_dark_color,
                    linestyle="--",
                    alpha=0.6,
                    linewidth=1.0,
                    zorder=3,
                )  # Increased linewidth

            # Find and annotate line pairs if we have enough boundaries
            if len(boundaries) >= 2:
                line_pairs = self.find_line_pairs(boundaries, roi_img)
            self.annotate_line_pairs(ax_img, line_pairs, roi_img)
        else:
            ax_profile.text(
                0.5,
                0.5,
                "No boundaries detected!\nTry adjusting the threshold.",
                transform=ax_profile.transAxes,
                ha="center",
                va="center",
                color="red",
                fontsize=14,
                fontweight="bold",
            )  # Increased font size
            logger.warning("No boundaries detected for visualization")
        # Set x-axis limits
        x_max = len(profile) if len(profile) > 0 else 100
        ax_profile.set_xlim(0, x_max)
        ax_img.set_xlim(0, x_max)
        # Return the figure - the caller is responsible for displaying it using st.pyplot(fig)
        return fig


class ImageProcessor:
    def __init__(self, usaf_target: USAFTarget = None):
        self.image = None
        self.original_image = None  # Store the original unprocessed image
        self.grayscale = None
        self.original_grayscale = None  # Store the original grayscale image
        self.roi_manager = RoiManager()
        self.roi = None
        self.original_roi = None  # Store the original ROI before processing
        self.profile = None
        self.individual_profiles = None
        self.usaf_target = usaf_target or USAFTarget()
        self.boundaries = None
        self.transition_types = None
        self.derivative = None
        self.line_pair_widths = []

        self.dark_regions = []
        self.light_regions = []
        self.contrast = 0.0
        self.processing_params = {
            "autoscale": True,
            "invert": False,
            "normalize": False,
            "saturated_pixels": 0.5,
            "equalize_histogram": True,  # Changed default to True
        }
        self.roi_rotation = (
            0  # Store ROI rotation (0, 1, 2, or 3 for 0°, 90°, 180°, 270°)
        )

    def load_image(self, image_path: str) -> bool:
        try:
            if not os.path.isfile(image_path):
                logger.error(f"Image file not found: {image_path}")
                return False
            try:
                # Load the image without any processing first
                if image_path.lower().endswith((".tif", ".tiff")):
                    try:
                        with tifffile.TiffFile(image_path) as tif:
                            if len(tif.pages) == 0:
                                logger.error(f"TIFF file has no pages: {image_path}")
                                return False
                            self.original_image = tif.pages[0].asarray()
                    except Exception as e:
                        logger.error(f"Failed to load TIFF: {image_path} ({e})")
                        return False
                else:
                    self.original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    if self.original_image is None:
                        # If OpenCV fails, try PIL
                        logger.info(
                            f"OpenCV failed to load image, trying PIL: {image_path}"
                        )
                        pil_image = Image.open(image_path)
                        self.original_image = np.array(pil_image)

                if self.original_image is None:
                    logger.error(f"Failed to load image: {image_path}")
                    return False

                # Convert BGR to RGB if needed (OpenCV loads as BGR)
                if (
                    len(self.original_image.shape) == 3
                    and self.original_image.shape[2] == 3
                ):
                    self.original_image = cv2.cvtColor(
                        self.original_image, cv2.COLOR_BGR2RGB
                    )

                # Create grayscale version of the original image
                if len(self.original_image.shape) > 2:
                    self.original_grayscale = cv2.cvtColor(
                        self.original_image, cv2.COLOR_RGB2GRAY
                    )
                else:
                    self.original_grayscale = self.original_image

                # Create display version with default processing
                self.apply_processing()

                return True
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                return False
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return False

    def apply_processing(self):
        """Apply current processing parameters to the original image"""
        try:
            # Apply processing to the original image
            self.image = normalize_to_uint8(
                self.original_image,
                autoscale=self.processing_params["autoscale"],
                invert=self.processing_params["invert"],
                normalize=self.processing_params["normalize"],
                saturated_pixels=self.processing_params["saturated_pixels"],
                equalize_histogram=self.processing_params["equalize_histogram"],
            )

            # Create grayscale version of the processed image
            if len(self.image.shape) > 2:
                self.grayscale = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            else:
                self.grayscale = self.image

            # If we have an ROI, reapply processing to it
            if self.original_roi is not None:
                self.roi = normalize_to_uint8(
                    self.original_roi,
                    autoscale=self.processing_params["autoscale"],
                    invert=self.processing_params["invert"],
                    normalize=self.processing_params["normalize"],
                    saturated_pixels=self.processing_params["saturated_pixels"],
                    equalize_histogram=self.processing_params["equalize_histogram"],
                )

            return True
        except Exception as e:
            logger.error(f"Error applying processing: {e}")
            return False

    def update_processing_params(self, **kwargs):
        """Update processing parameters and reapply processing"""
        for key, value in kwargs.items():
            if key in self.processing_params:
                self.processing_params[key] = value

        return self.apply_processing()

    def set_roi(self, roi_coordinates: tuple[int, int, int, int]) -> bool:
        """
        Set and validate ROI coordinates

        Args:
            roi_coordinates: ROI tuple (x, y, width, height)

        Returns:
            bool: True if ROI is valid, False otherwise
        """
        if not isinstance(roi_coordinates, tuple) or len(roi_coordinates) != 4:
            logger.error(f"Invalid ROI coordinates format: {roi_coordinates}")
            return False

        x, y, width, height = roi_coordinates
        point1 = (x, y)
        point2 = (x + width, y + height)

        # Use ROI manager to set and validate coordinates
        valid = self.roi_manager.set_coordinates(point1, point2)
        valid = valid and self.roi_manager.validate_against_image(self.grayscale)

        if valid:
            self.select_roi()

        return valid

    def set_roi_rotation(self, rotation_count: int) -> None:
        """
        Set ROI rotation count (number of 90° rotations).

        Args:
            rotation_count: Number of 90-degree rotations (0-3)
        """
        self.roi_rotation = rotation_count % 4

    def select_roi(self) -> np.ndarray | None:
        """
        Extract the ROI based on the current roi_manager settings

        Returns:
            Optional[np.ndarray]: The extracted ROI or None
        """
        if self.grayscale is None or not self.roi_manager.is_valid:
            return None

        try:
            # Extract ROI from both original and processed grayscale images
            self.original_roi = self.roi_manager.extract_roi(self.original_grayscale)
            self.roi = self.roi_manager.extract_roi(self.grayscale)

            # Apply rotation if needed
            if self.roi_rotation > 0:
                self.original_roi = rotate_image(self.original_roi, self.roi_rotation)
                self.roi = rotate_image(self.roi, self.roi_rotation)

            return self.roi
        except Exception as e:
            logger.error(f"Error selecting ROI: {e}")
            return None

    def get_line_profile(self, use_max=False) -> np.ndarray | None:
        if self.roi is None:
            return None
        try:
            use_roi = self.roi
            self.individual_profiles = use_roi.copy()
            if use_max:
                self.profile = np.max(use_roi, axis=0)
            else:
                self.profile = np.mean(use_roi, axis=0)
            return self.profile
        except Exception as e:
            logger.error(f"Error getting line profile: {e}")
            return None

    def detect_edges(self, edge_method="original"):
        if self.profile is None:
            logger.error("No profile available for edge detection")
            return False
        if edge_method == "parallel":
            self.boundaries, self.derivative, self.transition_types = (
                find_line_pair_boundaries_windowed(self.profile)
            )
        else:
            self.boundaries, self.derivative, self.transition_types = (
                find_line_pair_boundaries_derivative(self.profile)
            )
        return len(self.boundaries) > 0

    def calculate_contrast(self):
        """
        Calculate contrast from the detected light and dark regions.

        Returns:
            float: The calculated contrast value
        """
        if (
            self.profile is None
            or self.boundaries is None
            or self.transition_types is None
        ):
            return 0.0

        try:
            # Identify dark and light regions based on transitions
            self.dark_regions = []
            self.light_regions = []

            for i in range(len(self.boundaries) - 1):
                if i + 1 < len(self.transition_types):
                    start, end = self.boundaries[i], self.boundaries[i + 1]
                    if start < end and start >= 0 and end < len(self.profile):
                        if (
                            self.transition_types[i] == -1
                            and self.transition_types[i + 1] == 1
                        ):
                            # This is a dark bar (from L→D to D→L)
                            self.dark_regions.append(self.profile[start:end])
                        elif (
                            self.transition_types[i] == 1
                            and self.transition_types[i + 1] == -1
                        ):
                            # This is a light bar (from D→L to L→D)
                            self.light_regions.append(self.profile[start:end])

            # Calculate contrast if we have both light and dark regions
            if self.dark_regions and self.light_regions:
                min_intensity = np.mean(
                    [np.mean(region) for region in self.dark_regions]
                )
                max_intensity = np.mean(
                    [np.mean(region) for region in self.light_regions]
                )
                self.contrast = (max_intensity - min_intensity) / (
                    max_intensity + min_intensity
                )
            else:
                # Fallback if segmentation didn't work as expected
                self.contrast = (np.max(self.profile) - np.min(self.profile)) / (
                    np.max(self.profile) + np.min(self.profile)
                )
        except Exception:
            # Fallback calculation
            if len(self.profile) > 0:
                self.contrast = (np.max(self.profile) - np.min(self.profile)) / (
                    np.max(self.profile) + np.min(self.profile)
                )

        return self.contrast

    def analyze_profile(self, group: int, element: int) -> dict:
        """
        Analyze the current profile for the specified USAF target group and element.

        Args:
            group: USAF group number
            element: USAF element number

        Returns:
            Dictionary with analysis results
        """
        # Ensure we have a profile to analyze
        if self.profile is None:
            logger.error("No profile available for analysis")
            return {"error": "No profile available for analysis"}

        # Step 1: Detect edges in the profile if not already detected
        # (skip if boundaries are already set, e.g., by threshold detection)
        if self.boundaries is None or len(self.boundaries) == 0:
            if self.boundaries is None or len(self.boundaries) == 0:
                logger.debug("No boundaries detected yet, using detect_edges")
                self.detect_edges()
            else:
                logger.debug(f"Using pre-existing {len(self.boundaries)} boundaries")

            # Step 2: Use only the best two line pairs
            self.line_pair_widths = []

        if self.boundaries is not None and len(self.boundaries) >= 3:
            best_pairs, avg_width = find_best_two_line_pairs(self.boundaries)

            self.line_pair_widths = [end - start for start, end in best_pairs]
            self.avg_line_pair_width = avg_width
            logger.debug(
                f"Found {len(best_pairs)} best line pairs with avg width: {avg_width}"
            )
        else:
            self.avg_line_pair_width = 0.0
            logger.debug("Not enough boundaries for best line pairs")

        # Step 3: Calculate contrast
        self.calculate_contrast()

        # Calculate theoretical values based on USAF target
        num_line_pairs = len(self.line_pair_widths)
        lp_per_mm = self.usaf_target.lp_per_mm(group, element)

        # Create results dictionary
        results = {
            "group": group,
            "element": element,
            "lp_per_mm": float(lp_per_mm),
            "theoretical_lp_width_um": self.usaf_target.line_pair_width_microns(
                group, element
            ),
            "num_line_pairs": num_line_pairs,
            "num_boundaries": len(self.boundaries)
            if self.boundaries is not None
            else 0,
            "boundaries": self.boundaries,
            "transition_types": self.transition_types,
            "line_pair_widths": self.line_pair_widths,
            "avg_line_pair_width": self.avg_line_pair_width,
            "contrast": self.contrast,
            "derivative": (
                self.derivative.tolist() if hasattr(self.derivative, "tolist") else None
            ),
            "profile": self.profile.tolist()
            if hasattr(self.profile, "tolist")
            else None,
            "processing_params": self.processing_params.copy(),  # Include processing parameters
        }

        # Add individual profiles to the results
        if self.individual_profiles is not None:
            results["individual_profiles"] = self.individual_profiles

        return results

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
        """
        Complete pipeline: load image, select ROI, and analyze profile
        Args:
            image_path: Path to the image file
            roi: Region of interest tuple (x, y, width, height)
            group: USAF group number
            element: USAF group element
            use_max: If True, use max for profile; else mean (defaults to True)
            edge_method: 'original' or 'parallel', for legend
            threshold: Threshold value for edge detection (if None, use edge_method)
            roi_rotation: Number of 90-degree rotations to apply to the ROI (0-3)
            **processing_params: Additional processing parameters (autoscale, invert, etc.)
        Returns:
            Dictionary with analysis results
        """
        # Update processing parameters if provided
        if processing_params:
            self.update_processing_params(**processing_params)

        # Set ROI rotation
        self.set_roi_rotation(roi_rotation)

        if not self.load_image(image_path):
            return {"error": f"Failed to load image: {image_path}"}
        if not self.set_roi(roi):
            return {"error": f"Failed to set ROI: {roi}"}

        # Get the profile (using max intensity)
        self.get_line_profile(use_max=True)

        # Debug the profile range
        min_val = (
            np.min(self.profile)
            if self.profile is not None and len(self.profile) > 0
            else 0
        )
        max_val = (
            np.max(self.profile)
            if self.profile is not None and len(self.profile) > 0
            else 0
        )
        logger.debug(f"Profile range: {min_val} to {max_val}")

        # Use threshold-based edge detection if threshold is provided
        if threshold is not None:
            logger.debug(f"Using threshold detection with value {threshold}")
            self.boundaries, self.derivative, self.transition_types = (
                find_line_pair_boundaries_threshold(self.profile, threshold)
            )
            edge_method = "threshold"
            logger.debug(
                f"Found {len(self.boundaries)} boundaries with threshold {threshold}"
            )
            # Analyze the profile after setting boundaries
            results = self.analyze_profile(group, element)
            results["profile_type"] = "max"
            results["edge_method"] = edge_method
        else:
            # Use the specified edge method
            results = self.analyze_profile_with_edge_method(edge_method, group, element)

        # Add threshold information to results
        results["threshold"] = threshold if threshold is not None else 0

        # Add rotation information to results
        results["roi_rotation"] = self.roi_rotation

        return results

    def analyze_profile_with_edge_method(self, edge_method, group, element):
        """
        Analyze the profile using the specified edge detection method.
        Args:
            edge_method: The edge detection method to use ('original', 'parallel', etc.)
            group: USAF group number
            element: USAF group element
        Returns:
            Dictionary with analysis results, including profile type and edge method.
        """
        logger.debug(f"Using {edge_method} edge detection method")
        self.detect_edges(edge_method=edge_method)
        logger.debug(
            f"Found {len(self.boundaries) if self.boundaries else 0} boundaries with {edge_method} method"
        )

        result = self.analyze_profile(group, element)
        result["profile_type"] = "max"
        result["edge_method"] = edge_method
        return result


# --- Streamlit UI Functions ---


def display_roi_info(idx: int, image=None) -> tuple[int, int, int, int] | None:
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
    image_to_display: np.ndarray,
    key: str = "usaf_image",
    rotation: int = 0,
) -> bool:
    keys = get_image_session_keys(idx, image_file)
    coordinates_key = keys["coordinates"]
    roi_valid_key = keys["roi_valid"]
    if coordinates_key not in st.session_state:
        st.session_state[coordinates_key] = None
        st.session_state[roi_valid_key] = False
    unique_id = keys["coordinates"].split("_")[1]
    component_key = f"{key}_{unique_id}"

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
    st.info("Please upload a USAF 1951 target image to begin analysis.")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(
            WELCOME_IMAGE_URL,
            caption=WELCOME_IMAGE_CAPTION,
            use_container_width=True,
        )
    with col2:
        st.subheader("USAF 1951 Target Format")
        st.latex(
            r"\text{resolution (lp/mm)} = 2^{\text{group} + (\text{element} - 1)/6}"
        )
        st.latex(
            r"\text{Line Pair Width (μm)} = \frac{1000}{2 \times \text{resolution (lp/mm)}}"
        )


def display_analysis_details(results):
    group = results.get("group")
    element = results.get("element")
    lp_width_um = st.session_state.usaf_target.line_pair_width_microns(group, element)
    lp_per_mm = results.get("lp_per_mm")
    avg_measured_lp_width_px = results.get("avg_line_pair_width", 0.0)
    st.markdown(
        """
    <div style='text-align: center;'>
    <b>A line pair</b> = one black bar + one white bar (one cycle)<br>
    <b>Line pair width</b> = distance from start of one black bar to the next
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.latex(rf"Group = {group}, \quad Element = {element}")
    st.latex(
        rf"\text{{Line Pairs per mm (Theoretical)}} = 2^{{{group} + ({element} - 1)/6}} = {lp_per_mm:.2f}"
    )
    st.latex(
        rf"\text{{Line Pair Width (µm, Theoretical)}} = \frac{{1000}}{{{lp_per_mm:.2f} \text{{ lp/mm}}}} = {lp_width_um:.2f} \text{{ µm}}"
    )
    st.latex(
        rf"\text{{Avg. Measured Line Pair Width (pixels)}} = {avg_measured_lp_width_px:.2f} \text{{ px}}"
    )
    if avg_measured_lp_width_px > 0 and lp_width_um is not None:
        implied_pixel_size = lp_width_um / avg_measured_lp_width_px
        st.latex(
            rf"\text{{Implied Pixel Size (µm/pixel)}} = \frac{{{lp_width_um:.2f} \text{{ µm}}}}{{{avg_measured_lp_width_px:.2f} \text{{ px}}}} = {implied_pixel_size:.3f} \text{{ µm/pixel}}"
        )
    else:
        st.latex(
            r"\text{Implied Pixel Size (µm/pixel)} = \text{N/A (requires measurement)}"
        )


def analyze_and_display_image(idx, uploaded_file):
    filename = (
        os.path.basename(uploaded_file)
        if isinstance(uploaded_file, str)
        else (
            uploaded_file.name if hasattr(uploaded_file, "name") else f"Image {idx+1}"
        )
    )
    keys = get_image_session_keys(idx, uploaded_file)
    st.session_state[keys["image_name"]] = filename

    # Parse filename to get default values for magnification, group, and element
    default_values = parse_filename_for_defaults(filename)
    default_group = default_values.get("group", DEFAULT_GROUP)
    default_element = default_values.get("element", DEFAULT_ELEMENT)
    default_magnification = default_values.get("magnification", 10.0)

    # Get unique ID for this image
    unique_id = get_unique_id_for_image(uploaded_file)

    # Keep ROI rotation state for extracted ROI only, not for display
    roi_rotation_key = keys["roi_rotation"]
    last_roi_rotation_key = keys["last_roi_rotation"]

    if roi_rotation_key not in st.session_state:
        st.session_state[roi_rotation_key] = 0
    if last_roi_rotation_key not in st.session_state:
        st.session_state[last_roi_rotation_key] = 0

    # Define keys for image processing settings
    autoscale_key = f"autoscale_{unique_id}"
    invert_key = f"invert_{unique_id}"
    normalize_key = f"normalize_{unique_id}"
    saturated_pixels_key = f"saturated_pixels_{unique_id}"
    equalize_histogram_key = f"equalize_histogram_{unique_id}"
    bit_depth_key = f"bit_depth_{unique_id}"
    magnification_key = f"magnification_{unique_id}"
    threshold_key = f"threshold_{unique_id}"
    settings_changed_key = f"settings_changed_{unique_id}"

    # Initialize settings in session state if they don't exist yet
    if autoscale_key not in st.session_state:
        st.session_state[autoscale_key] = True
    if invert_key not in st.session_state:
        st.session_state[invert_key] = False
    if normalize_key not in st.session_state:
        st.session_state[normalize_key] = False
    if saturated_pixels_key not in st.session_state:
        st.session_state[saturated_pixels_key] = 0.5
    if equalize_histogram_key not in st.session_state:
        st.session_state[equalize_histogram_key] = True
    if magnification_key not in st.session_state:
        st.session_state[magnification_key] = default_magnification
    if settings_changed_key not in st.session_state:
        st.session_state[settings_changed_key] = False

    # Reset settings_changed flag if we're in a rerun triggered by settings changing
    if st.session_state.get(settings_changed_key, False):
        st.session_state[settings_changed_key] = False

    with st.expander(f"📸 Image {idx+1}: {filename}", expanded=(idx == 0)):
        # Process the image first
        image, temp_path = process_uploaded_file(uploaded_file)
        if image is None:
            st.error(f"❌ Failed to load image: {filename}")
            return
        st.session_state[keys["image_path"]] = temp_path

        # Enhanced compact header with key information in 3 columns
        header_col1, header_col2, header_col3 = st.columns([2, 1, 1])

        with header_col1:
            # Display parsed defaults if found with enhanced styling
            if default_values:
                values_list = []
                if "magnification" in default_values:
                    values_list.append(f"🔍 {default_values['magnification']}×")
                if "group" in default_values:
                    values_list.append(f"📊 G{default_values['group']}")
                if "element" in default_values:
                    values_list.append(f"🎯 E{default_values['element']}")

                if values_list:
                    st.success(f"**Auto-detected:** {' • '.join(values_list)}")
            else:
                st.info("**Ready for analysis** - Configure settings below")

        with header_col2:
            # Display bit depth information compactly
            bit_depth = st.session_state.get(bit_depth_key, 8)
            st.info(f"**{bit_depth}-bit** (0-{(1 << bit_depth)-1})")

        with header_col3:
            # Extract ROI info for threshold calculation and display profile range
            roi_tuple = display_roi_info(idx, image)
            default_threshold = 50
            max_threshold = 255
            if roi_tuple:
                temp_roi = extract_roi_image(image, roi_tuple)
                if temp_roi is not None:
                    profile_max = np.max(temp_roi, axis=0)
                    if len(profile_max) > 0:
                        min_val = np.min(profile_max)
                        max_val = np.max(profile_max)
                        default_threshold = int(min_val + (max_val - min_val) * 0.4)
                        default_threshold = max(0, min(255, default_threshold))
                        st.metric("Profile Range", f"{int(min_val)}-{int(max_val)}")
                    else:
                        st.info("**Profile:** Not available")
                else:
                    st.info("**Profile:** Not available")
            else:
                st.info("**Profile:** Select ROI")

        st.markdown("---")  # Visual separator

        # Enhanced tabs for better organization
        settings_tab, roi_tab = st.tabs(["⚙️ Settings", "🎯 ROI & Analysis"])

        with settings_tab:
            # Organize settings in a more compact and logical layout
            target_col, processing_col = st.columns([1, 1])

            with target_col:
                st.markdown("##### 🎯 Target Parameters")

                # Group selector with horizontal radio buttons
                selected_group = st.radio(
                    "**Group**",
                    options=[
                        "-2",
                        "-1",
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                    ],
                    index=[
                        "-2",
                        "-1",
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                    ].index(str(default_group))
                    if str(default_group)
                    in ["-2", "-1", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                    else 2,
                    key=f"group_radio_{unique_id}",
                    horizontal=True,
                    help="USAF target group number",
                )

                # Element selector with radio buttons
                selected_element = st.radio(
                    "**Element**",
                    options=["1", "2", "3", "4", "5", "6"],
                    index=int(default_element) - 1
                    if 0 < int(default_element) <= 6
                    else 0,
                    key=f"element_radio_{unique_id}",
                    horizontal=True,
                    help="USAF target element number",
                )

                # Magnification with number input
                magnification = st.number_input(
                    "**Magnification (×)**",
                    min_value=0.1,
                    max_value=1000.0,
                    value=st.session_state[magnification_key],
                    step=0.1,
                    format="%.1f",
                    key=f"magnification_widget_{unique_id}",
                    help="Optical magnification for display",
                )

            with processing_col:
                st.markdown("##### 🖼️ Image Processing")

                # Organize toggles in a compact grid
                toggle_col1, toggle_col2 = st.columns(2)

                with toggle_col1:
                    autoscale = st.toggle(
                        "**Autoscale**",
                        value=st.session_state[autoscale_key],
                        key=f"autoscale_widget_{unique_id}",
                        help="Percentile-based contrast",
                    )

                    normalize = st.toggle(
                        "**Normalize**",
                        value=st.session_state[normalize_key],
                        key=f"normalize_widget_{unique_id}",
                        help="Full range (0-255)",
                    )

                with toggle_col2:
                    invert = st.toggle(
                        "**Invert**",
                        value=st.session_state[invert_key],
                        key=f"invert_widget_{unique_id}",
                        help="Invert colors",
                    )

                    equalize_histogram = st.toggle(
                        "**Equalize**",
                        value=st.session_state[equalize_histogram_key],
                        key=f"equalize_histogram_widget_{unique_id}",
                        help="Histogram equalization",
                    )

                # Saturated pixels slider (compact)
                saturated_pixels = st.slider(
                    "**Saturated Pixels (%)**",
                    min_value=0.0,
                    max_value=20.0,
                    value=st.session_state[saturated_pixels_key],
                    step=0.1,
                    format="%.1f",
                    key=f"saturated_pixels_widget_{unique_id}",
                    disabled=not autoscale,
                    help="Percentage of pixels to saturate",
                )

            # Analysis options in a more compact layout
            st.markdown("##### 🔍 Analysis Options")
            analysis_col1, analysis_col2 = st.columns([2, 1])

            with analysis_col1:
                threshold = st.slider(
                    "**Threshold Line**",
                    min_value=0,
                    max_value=max_threshold,
                    value=int(st.session_state.get(threshold_key, default_threshold)),
                    key=f"threshold_widget_{unique_id}",
                    help="Edge detection threshold",
                )

            with analysis_col2:
                prev_rotation = st.session_state.get(roi_rotation_key, 0)
                rotation_options = ["0°", "90°", "180°", "270°"]

                selected_rotation = st.radio(
                    "**ROI Rotation**",
                    options=rotation_options,
                    index=prev_rotation,
                    horizontal=True,
                    key=f"roi_rotation_radio_{unique_id}",
                    help="Rotate extracted ROI",
                )
                new_rotation = rotation_options.index(selected_rotation)

        with roi_tab:
            # Enhanced main analysis area with improved layout
            roi_col, result_col = st.columns([1, 1])

            with roi_col:
                st.markdown("##### 🎯 Select Region of Interest")

                # Prepare image for ROI selection
                pil_img = Image.fromarray(image)
                draw = ImageDraw.Draw(pil_img)
                current_coords = st.session_state.get(keys["coordinates"])

                if current_coords:
                    p1, p2 = current_coords
                    coords = (
                        min(p1[0], p2[0]),
                        min(p1[1], p2[1]),
                        max(p1[0], p2[0]),
                        max(p1[1], p2[1]),
                    )
                    roi_valid = st.session_state.get(keys["roi_valid"], False)
                    if roi_valid:
                        color_idx = idx % len(ROI_COLORS)
                        outline_color = ROI_COLORS[color_idx]
                    else:
                        outline_color = INVALID_ROI_COLOR
                    draw.rectangle(coords, outline=outline_color, width=3)

                # Display image for ROI selection
                roi_changed = handle_image_selection(
                    idx, uploaded_file, pil_img, key=f"usaf_image_{idx}", rotation=0
                )

                # Trigger analysis if ROI changed
                if roi_changed:
                    st.session_state[settings_changed_key] = True

                # Enhanced compact status display
                roi_valid = st.session_state.get(keys["roi_valid"], False)
                if current_coords is not None:
                    if roi_valid:
                        st.success("✅ **Valid ROI selected** - Ready for analysis")
                    else:
                        st.error("❌ **Invalid ROI** - Please reselect area")
                else:
                    st.info("👆 **Click and drag** to select analysis region")

            with result_col:
                st.markdown("##### 📊 Analysis Results")

                # Show analysis results or preview with enhanced layout
                if analysis_results_for_plot := st.session_state.get(
                    keys["analysis_results"]
                ):
                    # Get ROI rotation from analysis results
                    roi_rotation = analysis_results_for_plot.get("roi_rotation", 0)

                    # Extract ROI with rotation
                    roi_for_display = extract_roi_image(
                        image,
                        st.session_state.get(keys["analyzed_roi"]),
                        rotation=roi_rotation,
                    )

                    magnification = st.session_state.get(magnification_key, 10.0)
                    # Generate the figure using ProfileVisualizer
                    visualizer = ProfileVisualizer()
                    lp_width_um = None
                    group_val = st.session_state.get(keys["last_group"])
                    element_val = st.session_state.get(keys["last_element"])
                    if (
                        group_val is not None
                        and element_val is not None
                        and "usaf_target" in st.session_state
                    ):
                        lp_width_um = (
                            st.session_state.usaf_target.line_pair_width_microns(
                                group_val, element_val
                            )
                        )
                    fig = visualizer.visualize_profile(
                        analysis_results_for_plot,
                        roi_for_display,
                        group=group_val,
                        element=element_val,
                        lp_width_um=lp_width_um,
                        magnification=magnification,
                    )
                    if fig is not None:
                        st.pyplot(fig)

                        # Enhanced compact download section
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                        buf.seek(0)

                        # Generate filename
                        pixel_size_str = "NA"
                        if (
                            lp_width_um is not None
                            and analysis_results_for_plot.get("avg_line_pair_width", 0)
                            > 0
                        ):
                            pixel_size = (
                                lp_width_um
                                / analysis_results_for_plot["avg_line_pair_width"]
                            )
                            pixel_size_str = f"{pixel_size:.3f}um"
                        group_str = (
                            f"group{group_val}" if group_val is not None else "groupNA"
                        )
                        mag_str = (
                            f"mag{int(round(magnification))}x"
                            if magnification is not None
                            else "magNA"
                        )
                        file_name = f"usaf_processed_{group_str}_pix{pixel_size_str}_{mag_str}.png"

                        st.download_button(
                            label="📥 Download Plot",
                            data=buf,
                            file_name=file_name,
                            mime="image/png",
                            use_container_width=True,
                        )

                        # Show caption below download button with better formatting
                        caption = visualizer.create_caption(
                            group_val,
                            element_val,
                            lp_width_um,
                            edge_method=analysis_results_for_plot.get(
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

                        # Get current ROI rotation for preview
                        roi_rotation = st.session_state.get(roi_rotation_key, 0)

                        # Extract the ROI from the unrotated image
                        roi_img_preview = pil_img.crop(coords_preview)

                        # Apply rotation to the preview if needed
                        if roi_rotation > 0:
                            roi_img_preview = Image.fromarray(
                                rotate_image(np.array(roi_img_preview), roi_rotation)
                            )

                        st.image(
                            roi_img_preview,
                            caption="🔍 ROI Preview",
                            use_container_width=True,
                        )
                        st.info(
                            "💡 **Adjust settings** and the analysis will update automatically"
                        )
                    except Exception as e:
                        st.warning(f"⚠️ Could not display ROI preview: {e!s}")
                else:
                    st.info("👆 **Select an ROI above** to view analysis results")

        # Check for changes and update session state
        settings_changed = False

        # Check target parameters
        if st.session_state.get(keys["group"]) != int(selected_group):
            st.session_state[keys["group"]] = int(selected_group)
            settings_changed = True

        if st.session_state.get(keys["element"]) != int(selected_element):
            st.session_state[keys["element"]] = int(selected_element)
            settings_changed = True

        if st.session_state.get(magnification_key) != magnification:
            st.session_state[magnification_key] = magnification
            settings_changed = True

        # Check image processing settings
        if st.session_state.get(autoscale_key) != autoscale:
            st.session_state[autoscale_key] = autoscale
            settings_changed = True

        if st.session_state.get(invert_key) != invert:
            st.session_state[invert_key] = invert
            settings_changed = True

        if st.session_state.get(normalize_key) != normalize:
            st.session_state[normalize_key] = normalize
            settings_changed = True

        if st.session_state.get(saturated_pixels_key) != saturated_pixels:
            st.session_state[saturated_pixels_key] = saturated_pixels
            settings_changed = True

        if st.session_state.get(equalize_histogram_key) != equalize_histogram:
            st.session_state[equalize_histogram_key] = equalize_histogram
            settings_changed = True

        # Check analysis settings
        if st.session_state.get(threshold_key) != threshold:
            st.session_state[threshold_key] = threshold
            settings_changed = True

        if st.session_state.get(roi_rotation_key, 0) != new_rotation:
            st.session_state[roi_rotation_key] = new_rotation
            settings_changed = True

        # Set flag to trigger analysis if any settings changed
        if settings_changed:
            st.session_state[settings_changed_key] = True

        # Get parameters for analysis
        current_selected_roi_tuple = display_roi_info(idx, image)
        group_for_trigger = st.session_state.get(keys["group"])
        element_for_trigger = st.session_state.get(keys["element"])
        roi_is_valid = st.session_state.get(keys["roi_valid"], False)
        threshold = st.session_state.get(threshold_key, 50)
        threshold = max(0, min(255, threshold))

        # Get current ROI rotation
        roi_rotation = st.session_state.get(roi_rotation_key, 0)

        # Determine if analysis should run - triggered by settings changes
        should_analyze = (
            st.session_state.get(settings_changed_key, False)
            and current_selected_roi_tuple is not None
            and roi_is_valid
            and group_for_trigger is not None
            and element_for_trigger is not None
        )

        if should_analyze:
            with st.spinner("🔄 Analyzing image..."):
                try:
                    logging.getLogger().setLevel(logging.DEBUG)
                    img_proc = ImageProcessor(
                        usaf_target=st.session_state.usaf_target,
                    )

                    # Set the processing parameters
                    processing_params = {
                        "autoscale": st.session_state[autoscale_key],
                        "invert": st.session_state[invert_key],
                        "normalize": st.session_state[normalize_key],
                        "saturated_pixels": st.session_state[saturated_pixels_key],
                        "equalize_histogram": st.session_state[equalize_histogram_key],
                    }

                    # Ensure threshold is within valid range before passing to the processor
                    threshold = max(0, min(255, threshold))

                    # Process and analyze the image with the current settings and rotation
                    results_data = img_proc.process_and_analyze(
                        temp_path,
                        current_selected_roi_tuple,
                        group_for_trigger,
                        element_for_trigger,
                        use_max=True,
                        threshold=threshold,
                        roi_rotation=roi_rotation,
                        **processing_params,
                    )

                    st.session_state[keys["analyzed_roi"]] = current_selected_roi_tuple
                    st.session_state[keys["analysis_results"]] = results_data
                    st.session_state[keys["last_group"]] = group_for_trigger
                    st.session_state[keys["last_element"]] = element_for_trigger
                    st.session_state[last_roi_rotation_key] = roi_rotation

                    # Clear settings changed flag after processing
                    st.session_state[settings_changed_key] = False

                    # Show success message
                    st.success("✅ **Analysis completed successfully!**")

                    # Rerun once to display results
                    st.rerun()
                except Exception as e:
                    logger.error(f"Analysis failed: {e}")
                    st.error(f"❌ **Analysis failed:** {e!s}")
                    st.error(f"**Error details:** {type(e).__name__} - {e!s}")

        # Display analysis details in a collapsible section with better organization
        if analysis_results_for_details := st.session_state.get(
            keys["analysis_results"]
        ):
            with st.expander("📈 **Detailed Analysis Results**", expanded=False):
                display_analysis_details(analysis_results_for_details)


def collect_analysis_data():
    """
    Collect analysis data for all processed images

    Returns:
        pandas.DataFrame: DataFrame with analysis data
    """
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
        # Get unique ID and keys for this image
        unique_id = get_unique_id_for_image(uploaded_file)
        keys = get_image_session_keys(idx, uploaded_file)

        # Check if we have analysis results for this image
        if analysis_results := st.session_state.get(keys["analysis_results"]):
            # Get filename
            filename = st.session_state.get(keys["image_name"], f"Image {idx+1}")
            data["Filename"].append(filename)

            # Get magnification
            magnification = st.session_state.get(f"magnification_{unique_id}", 0)
            data["Magnification"].append(magnification)

            # Get group and element
            group = analysis_results.get("group", 0)
            element = analysis_results.get("element", 0)
            data["Group"].append(group)
            data["Element"].append(element)

            # Get line pairs per mm
            lp_per_mm = analysis_results.get("lp_per_mm", 0)
            data["Line Pairs/mm"].append(lp_per_mm)

            # Get theoretical line pair width in microns
            lp_width_um = analysis_results.get("theoretical_lp_width_um", 0)
            data["Line Pair Width (µm)"].append(lp_width_um)

            # Calculate pixel size
            avg_lp_width_px = analysis_results.get("avg_line_pair_width", 0)
            if avg_lp_width_px > 0 and lp_width_um > 0:
                pixel_size = lp_width_um / avg_lp_width_px
            else:
                pixel_size = 0
            data["Pixel Size (µm/pixel)"].append(pixel_size)

            # Get contrast
            contrast = analysis_results.get("contrast", 0)
            data["Contrast"].append(contrast)

            # Get number of line pairs detected
            num_line_pairs = analysis_results.get("num_line_pairs", 0)
            data["Line Pairs Detected"].append(num_line_pairs)

            # Get average line pair width in pixels
            data["Avg Line Pair Width (px)"].append(avg_lp_width_px)

            # Get ROI rotation
            roi_rotation = analysis_results.get("roi_rotation", 0)
            data["ROI Rotation"].append(f"{roi_rotation * 90}°")

    # Create DataFrame
    df = pd.DataFrame(data)
    return df


def run_streamlit_app():
    try:
        st.set_page_config(
            page_title="USAF Target Analyzer",
            layout="wide",
            page_icon="🎯",
            initial_sidebar_state="collapsed",  # Collapse sidebar since we're not using it
        )
        initialize_session_state()

        # Enhanced page header with better styling
        st.title("🎯 USAF Target Analyzer")
        st.subheader(
            """Comprehensive analysis tool for USAF 1951 resolution targets in microscopy and imaging systems
            """,
        )

        # Control Panel at the top of the main area
        with st.container():
            st.markdown('<div class="control-panel">', unsafe_allow_html=True)

            # Create tabs for different control sections
            upload_tab, manage_tab, export_tab, help_tab = st.tabs(
                [
                    "📁 Upload & Status",
                    "🗂️ Manage Images",
                    "📤 Export Results",
                    "💡 Help & Tips",
                ]
            )

            with upload_tab:
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("#### 📁 **Upload Images**")
                    if new_uploaded_files := st.file_uploader(
                        "Select USAF target image(s)",
                        type=["jpg", "jpeg", "png", "tif", "tiff"],
                        accept_multiple_files=True,
                        help="Upload one or more images containing a USAF 1951 resolution target",
                    ):
                        for file in new_uploaded_files:
                            file_names = [
                                f.name if hasattr(f, "name") else os.path.basename(f)
                                for f in st.session_state.uploaded_files_list
                            ]
                            new_file_name = (
                                file.name
                                if hasattr(file, "name")
                                else os.path.basename(file)
                            )
                            if new_file_name not in file_names:
                                st.session_state.uploaded_files_list.append(file)
                                st.success(f"✅ **Added:** {new_file_name}")

                with col2:
                    st.markdown("#### 📊 **Current Status**")
                    if st.session_state.uploaded_files_list:
                        st.info(
                            f"**{len(st.session_state.uploaded_files_list)}** image(s) loaded"
                        )

                        # Show analysis progress
                        analyzed_count = 0
                        for idx, uploaded_file in enumerate(
                            st.session_state.uploaded_files_list
                        ):
                            keys = get_image_session_keys(idx, uploaded_file)
                            if st.session_state.get(keys["analysis_results"]):
                                analyzed_count += 1

                        if analyzed_count > 0:
                            st.success(f"**{analyzed_count}** image(s) analyzed")
                        else:
                            st.warning("**No images analyzed yet**")
                    else:
                        st.info("**No images loaded**")

            with manage_tab:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 🗂️ **Image Management**")

                    # Load default image
                    default_image_path = load_default_image()
                    if (
                        not st.session_state.uploaded_files_list
                        and default_image_path
                        and not st.session_state.default_image_added
                    ):
                        st.session_state.uploaded_files_list.append(default_image_path)
                        st.session_state.default_image_added = True
                        st.info(
                            f"📷 **Default image loaded:** {os.path.basename(default_image_path)}"
                        )

                    if st.button("🗑️ **Clear All Images**", use_container_width=True):
                        st.session_state.uploaded_files_list = []
                        st.session_state.default_image_added = False
                        st.session_state.image_index_to_id = {}
                        st.success("✅ **All images cleared**")
                        for key in list(st.session_state.keys()):
                            if any(
                                key.startswith(prefix)
                                for prefix in SESSION_STATE_PREFIXES
                            ):
                                del st.session_state[key]
                        st.rerun()

                with col2:
                    if st.session_state.uploaded_files_list:
                        st.markdown("#### 📋 **Loaded Images**")
                        for idx, uploaded_file in enumerate(
                            st.session_state.uploaded_files_list
                        ):
                            filename = (
                                uploaded_file.name
                                if hasattr(uploaded_file, "name")
                                else os.path.basename(uploaded_file)
                                if isinstance(uploaded_file, str)
                                else f"Image {idx+1}"
                            )
                            keys = get_image_session_keys(idx, uploaded_file)
                            status = (
                                "✅ Analyzed"
                                if st.session_state.get(keys["analysis_results"])
                                else "⏳ Pending"
                            )
                            st.text(f"{idx+1}. {filename} - {status}")

            with export_tab:
                st.markdown("#### 📤 **Export Analysis Results**")

                col1, col2 = st.columns([1, 1])

                with col1:
                    if st.button(
                        "📊 **Generate Analysis CSV**", use_container_width=True
                    ):
                        if not st.session_state.uploaded_files_list:
                            st.warning("⚠️ No images uploaded for analysis.")
                        else:
                            # Collect data and create DataFrame
                            df = collect_analysis_data()

                            if df.empty:
                                st.warning(
                                    "⚠️ No analysis data available. Please analyze images first."
                                )
                            else:
                                # Create CSV string
                                csv = df.to_csv(index=False)

                                # Store in session state for download
                                st.session_state["csv_data"] = csv
                                st.session_state["csv_df"] = df
                                st.success(
                                    f"✅ **CSV generated with {len(df)} results**"
                                )

                with col2:
                    if "csv_data" in st.session_state:
                        st.download_button(
                            label="📥 **Download CSV**",
                            data=st.session_state["csv_data"],
                            file_name="usaf_analysis_results.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                        # Show preview
                        if st.checkbox("👀 **Show CSV Preview**"):
                            st.dataframe(
                                st.session_state["csv_df"], use_container_width=True
                            )

            with help_tab:
                help_col1, help_col2 = st.columns(2)

                with help_col1:
                    st.markdown("#### 🎯 **ROI Selection Guide**")
                    st.markdown("""
                    - **Click and drag** on the image to select your region of interest
                    - Select an area containing **clear line pairs**
                    - Ensure the ROI is **large enough** to capture multiple line pairs
                    - The ROI outline will be **green** when valid, **red** when invalid
                    
                    **Best practices:**
                    - Include at least 3-5 line pairs in your ROI
                    - Avoid edges and artifacts
                    - Center the ROI on the clearest part of the target
                    """)

                    st.markdown("#### 🔄 **Image Rotation**")
                    st.markdown("""
                    Use the **ROI Rotation** controls to align line pairs horizontally 
                    for optimal analysis when your USAF target appears tilted in the image.
                    
                    **Tip:** Most accurate results occur when line pairs are horizontal.
                    """)

                with help_col2:
                    st.markdown("#### ⚙️ **Settings Guide**")
                    st.markdown("""
                    **Image Processing:**
                    - **Autoscale**: Automatic contrast adjustment (recommended)
                    - **Normalize**: Use full intensity range
                    - **Invert**: Flip dark/light (useful for some microscopy images)
                    - **Equalize**: Enhance contrast using histogram equalization
                    
                    **Analysis:**
                    - **Threshold**: Adjust edge detection sensitivity
                    - **Group/Element**: Select the USAF target pattern to analyze
                    """)

                    st.markdown("#### 📏 **Understanding Results**")
                    st.markdown("""
                    **Key Metrics:**
                    - **Line Pairs/mm**: Spatial frequency of the target
                    - **Pixel Size**: Physical size per pixel in micrometers
                    - **Contrast**: Measure of image sharpness
                    - **Line Pair Width**: Theoretical width in micrometers
                    
                    **Quality Indicators:**
                    - Higher contrast = better image quality
                    - More detected line pairs = better resolution
                    """)

            st.markdown("</div>", unsafe_allow_html=True)

        # Enhanced main content area
        main_container = st.container()
        with main_container:
            if st.session_state.uploaded_files_list:
                # Enhanced status banner
                # Process each image with enhanced organization
                for idx, uploaded_file in enumerate(
                    st.session_state.uploaded_files_list
                ):
                    analyze_and_display_image(idx, uploaded_file)
            else:
                display_welcome_screen()

    except Exception as e:
        st.error(f"❌ **Application Error:** {e}")
        st.info(
            "💡 For detailed error information, set DEBUG=1 in environment variables."
        )


if __name__ == "__main__":
    run_streamlit_app()

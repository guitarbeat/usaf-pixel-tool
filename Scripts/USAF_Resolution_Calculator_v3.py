#!/usr/bin/env python

"""
USAF 1951 Resolution Target Calculator (v3)
A Python-native version using PyImageJ for image processing and analysis.
"""

import datetime
import logging
import math
import platform
import shutil
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import imagej
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import signal

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed."""
    missing_deps = []

    # Check Python version
    if sys.version_info < (3, 8):
        missing_deps.append("Python 3.8 or higher")

    # Check for Java
    try:
        java_version = subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT)
        if b'openjdk version "11' not in java_version.lower() and b'openjdk version "8' not in java_version.lower():
            missing_deps.append("OpenJDK 8 or 11")
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing_deps.append("Java (OpenJDK 8 or 11)")

    # Check for Maven
    if not shutil.which('mvn'):
        missing_deps.append("Maven")

    # Check for conda/mamba
    if not shutil.which('conda') and not shutil.which('mamba'):
        missing_deps.append("Conda or Mamba")

    return missing_deps

def print_installation_instructions():
    """Print detailed installation instructions."""
    logger.info("\n=== PyImageJ Installation Instructions ===")
    logger.info("\n1. Install Miniforge3 or ensure conda-forge is configured:")
    logger.info("   conda config --add channels conda-forge")
    logger.info("   conda config --set channel_priority strict")

    logger.info("\n2. Create and activate a new environment:")
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        # M1 Mac specific instructions
        logger.info("   mamba create -n pyimagej pyimagej openjdk=11")
    else:
        logger.info("   mamba create -n pyimagej pyimagej openjdk=8")
    logger.info("   mamba activate pyimagej")

    logger.info("\n3. Verify installation:")
    logger.info("   python -c 'import imagej; ij = imagej.init(); print(ij.getVersion())'")

    logger.info("\n4. If using pip instead of conda/mamba:")
    logger.info("   pip install pyimagej")
    logger.info("   Note: You'll need to install Java and Maven separately")

    logger.info("\nFor more details, visit: https://py.imagej.net/en/latest/Install.html")

def verify_installation():
    """Verify PyImageJ installation and dependencies."""
    missing_deps = check_dependencies()

    if missing_deps:
        logger.error("Missing required dependencies:")
        for dep in missing_deps:
            logger.error(f"- {dep}")
        print_installation_instructions()
        sys.exit(1)

    try:
        # Test PyImageJ installation
        ij = imagej.init()
        version = ij.getVersion()
        logger.info(f"PyImageJ initialized successfully with ImageJ version: {version}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize PyImageJ: {e!s}")
        print_installation_instructions()
        return False

# Constants
DEBUG = True
SAVE_RESULTS = True
ASK_FOR_SAVE_DIRECTORY = True

class USAFTarget:
    """USAF 1951 target specifications and calculations."""
    MIN_GROUP = -2
    MAX_GROUP = 7
    MIN_ELEMENT = 1
    MAX_ELEMENT = 6

    def __init__(self, group, element):
        self.group = group
        self.element = element
        self.line_pairs_per_mm = self.calculate_line_pairs_per_mm()

    def calculate_line_pairs_per_mm(self):
        return math.pow(2, self.group + (self.element-1)/6)

    def get_line_pair_spacing_microns(self):
        return (1.0 / self.line_pairs_per_mm) * 1000

    @staticmethod
    def validate_parameters(group, element):
        if not (USAFTarget.MIN_GROUP <= group <= USAFTarget.MAX_GROUP):
            raise ValueError(f"Group must be between {USAFTarget.MIN_GROUP} and {USAFTarget.MAX_GROUP}")
        if not (USAFTarget.MIN_ELEMENT <= element <= USAFTarget.MAX_ELEMENT):
            raise ValueError(f"Element must be between {USAFTarget.MIN_ELEMENT} and {USAFTarget.MAX_ELEMENT}")
        return True

class ProfileAnalyzer:
    """Analyzes intensity profiles from USAF target images."""

    SMOOTHING_LEVELS = {
        'Light': {'window_size': 3, 'iterations': 1},
        'Medium': {'window_size': 5, 'iterations': 2},
        'Heavy': {'window_size': 7, 'iterations': 3}
    }

    def __init__(self, profile, smoothing_level='Medium', bar_type='Dark Bars'):
        self.raw_profile = np.array(profile)
        self.smoothing_params = self.SMOOTHING_LEVELS[smoothing_level]
        self.bar_type = bar_type
        self.profile = self.preprocess_profile()

    def preprocess_profile(self):
        profile = self.smooth_profile(self.raw_profile)
        if self.bar_type == 'Bright Bars':
            profile = np.max(profile) - profile
        return profile

    def smooth_profile(self, profile, window_size=None, iterations=None):
        if window_size is None:
            window_size = self.smoothing_params['window_size']
        if iterations is None:
            iterations = self.smoothing_params['iterations']

        smoothed = profile.copy()
        for _ in range(iterations):
            smoothed = signal.convolve(smoothed,
                                     np.ones(window_size)/window_size,
                                     mode='same')
        return smoothed

    def find_valleys(self):
        profile = self.profile
        mean = np.mean(profile)
        std_dev = np.std(profile)
        valley_threshold = mean - 0.75 * std_dev

        # Find valleys using peak finding
        valleys = signal.find_peaks(-profile,
                                  height=-valley_threshold,
                                  distance=10)[0]

        # Sort valleys by intensity and take the three deepest
        valley_intensities = profile[valleys]
        sorted_indices = np.argsort(valley_intensities)
        valleys = valleys[sorted_indices[:3]]
        valleys.sort()

        return valleys

class ROIHandler:
    """Handles ROI selection and profile extraction."""

    def __init__(self, ij, jimage, image, selection_type='Line'):
        self.ij = ij
        self.jimage = jimage  # Java image object
        self.image = image    # xarray/numpy image
        self.selection_type = selection_type
        self.roi = None
        self.profile = None
        self.selection_length = None
        self.angle = 0.0
        self.valleys = None
        self.roi_image = None
        self.last_profile_plot = None

    def get_profile(self):
        """Get intensity profile from the current ROI."""
        if self.roi is None:
            raise ValueError("No ROI selected. Please draw a line selection first.")

        if self.selection_type == 'Line':
            return self._get_line_profile()
        else:
            return self._get_rectangle_profile()

    def _get_line_profile(self):
        """Extract profile from a line ROI."""
        # Get the line profile using ImageJ's built-in functionality
        profile = self.ij.py.from_java(
            self.ij.op().run('profile.line', self.jimage, self.roi)
        )
        self.selection_length = len(profile)
        return profile, 0.0

    def _get_rectangle_profile(self):
        """Extract and average profiles from a rectangle ROI."""
        # Get rectangle bounds
        rect = self.roi.getBounds()
        self.selection_length = rect.width
        self.angle = self.roi.getAngle()

        # Crop the Java image using the ROI
        cropped = self.ij.op().run('transform.crop', self.jimage, self.roi)
        roi_data = self.ij.py.from_java(cropped)

        # Handle rotation if needed
        if abs(self.angle) > 0.01:
            roi_data = self._rotate_image(roi_data, -self.angle)

        # Average profiles
        height = roi_data.shape[0]
        num_profiles = min(height, 10)
        step = height // num_profiles

        profiles = []
        for y in range(0, height, step):
            profiles.append(roi_data[y, :])

        averaged_profile = np.mean(profiles, axis=0)

        # Create profile plot
        self.last_profile_plot = self._create_profile_plot(profiles, averaged_profile)

        return averaged_profile, self.angle

    def _rotate_image(self, image, angle):
        """Rotate image using scipy."""
        from scipy.ndimage import rotate
        return rotate(image, angle, reshape=False)

    def _create_profile_plot(self, individual_profiles, averaged_profile):
        """Create a matplotlib plot of the profiles."""
        plt.figure(figsize=(10, 6))

        # Plot individual profiles
        for profile in individual_profiles:
            plt.plot(profile, color='gray', alpha=0.3)

        # Plot averaged profile
        plt.plot(averaged_profile, color='blue', linewidth=2)

        # Find and mark valleys
        analyzer = ProfileAnalyzer(averaged_profile, 'Light')
        valleys = analyzer.find_valleys()

        # Mark valleys
        for valley in valleys:
            plt.axvline(x=valley, color='red', linestyle='--')
            plt.text(valley, min(averaged_profile), 'V', color='red')

        # Add measurements
        for i in range(len(valleys)-1):
            x1, x2 = valleys[i], valleys[i+1]
            plt.plot([x1, x2], [min(averaged_profile)]*2, color='green')
            plt.text((x1+x2)/2, min(averaged_profile),
                    f'{x2-x1:.1f}px', color='green')

        plt.title('Line Profiles')
        plt.xlabel('Distance (pixels)')
        plt.ylabel('Intensity')
        plt.legend(['Individual Profiles', 'Averaged Profile',
                   'Valley Centers', 'Line Pair Width'])

        return plt.gcf()

class ResultsCalculator:
    """Calculates resolution and other metrics from the analysis."""

    def __init__(self, usaf_target, profile_analyzer, roi_handler):
        self.target = usaf_target
        self.analyzer = profile_analyzer
        self.roi_handler = roi_handler
        self.results = {}

    def calculate_results(self):
        """Calculate all metrics and return results dictionary."""
        valleys = self.analyzer.find_valleys()
        if len(valleys) < 3:
            raise ValueError("Could not detect 3 dark bars")

        # Calculate basic metrics
        self.results['line_pairs_per_mm'] = self.target.line_pairs_per_mm
        self.results['microns_per_pair'] = self.target.get_line_pair_spacing_microns()

        # Calculate spacings
        spacings = np.diff(valleys)
        self.results['valley_positions'] = valleys
        self.results['valley_spacings'] = spacings
        self.results['avg_spacing'] = np.mean(spacings)
        self.results['spacing_std_dev'] = np.std(spacings)

        # Calculate resolution
        self.results['pixels_per_pair'] = self.results['avg_spacing']
        self.results['microns_per_pixel'] = (
            self.results['microns_per_pair'] / self.results['pixels_per_pair']
        )

        # Calculate contrast
        profile = self.analyzer.profile
        valley_intensities = profile[valleys]
        peak_intensities = self._find_peak_intensities(profile, valleys)
        self.results['contrast_ratio'] = (
            (np.max(peak_intensities) - np.min(valley_intensities)) /
            (np.max(peak_intensities) + np.min(valley_intensities))
        )

        self.results['selection_angle'] = self.roi_handler.angle

        return self.results

    def _find_peak_intensities(self, profile, valleys):
        """Find peak intensities between valleys."""
        peaks = []
        for i in range(len(valleys)-1):
            start, end = valleys[i], valleys[i+1]
            peak_pos = np.argmax(profile[start:end]) + start
            peaks.append(profile[peak_pos])
        return peaks

def get_save_directory(image_path=None):
    """Get directory for saving results."""
    if ASK_FOR_SAVE_DIRECTORY:
        root = tk.Tk()
        root.withdraw()
        directory = filedialog.askdirectory(
            title="Select Directory to Save Results",
            initialdir=image_path if image_path else None
        )
        if directory:
            return directory

    if image_path:
        return str(Path(image_path).parent)

    return str(Path.cwd())

def get_output_subdir(base_dir):
    """Create and return output subdirectory."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(base_dir) / f'USAF_Results_{timestamp}'
    output_dir.mkdir(exist_ok=True)
    return output_dir

def save_results_to_csv(results, params, output_dir):
    """Save results to CSV file."""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = output_dir / f'USAF_Resolution_Results_{timestamp}.csv'

    # Create DataFrame
    data = {
        'Parameter': [
            'Date',
            'Group',
            'Element',
            'Selection Type',
            'Bar Type',
            'Target Line Pairs/mm',
            'Microns per Line Pair',
            'Average Bar Spacing (pixels)',
            'Complete Line Pair Width (pixels)',
            'Microns per Pixel',
            'Contrast Ratio',
            'Selection Angle',
            'Tilt Correction'
        ],
        'Value': [
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            params['group'],
            params['element'],
            params['selection_type'],
            params['bar_type'],
            f"{results['line_pairs_per_mm']:.4f}",
            f"{results['microns_per_pair']:.4f}",
            f"{results['avg_spacing']:.1f}",
            f"{results['pixels_per_pair']:.1f}",
            f"{results['microns_per_pixel']:.4f}",
            f"{results['contrast_ratio']:.3f}",
            f"{results['selection_angle']:.2f}",
            'Applied' if abs(results['selection_angle']) > 0.1 else 'Not needed'
        ]
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return filename

def create_summary_image(image, results, params, roi_handler, output_dir):
    """Create a summary image with analysis results."""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = output_dir / f'USAF_Summary_{timestamp}.png'

    # Convert ImageJ image to numpy array
    img_array = roi_handler.ij.py.from_java(
        roi_handler.ij.py.to_java(image.getProcessor())
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img_array, cmap='gray')

    # Add info box
    info_text = (
        f"USAF Target Analysis Results\n"
        f"Group: {params['group']}, Element: {params['element']}\n"
        f"Line Pairs/mm: {results['line_pairs_per_mm']:.4f}\n"
        f"Microns per Pixel: {results['microns_per_pixel']:.4f}\n"
        f"Contrast Ratio: {results['contrast_ratio']:.3f}\n"
        f"Selection Method: {params['selection_type']}"
    )

    # Add text box
    props = dict(boxstyle='round', facecolor='black', alpha=0.7)
    ax.text(0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            color='white',
            bbox=props)

    # Add scale bar
    pixels_100um = 100 / results['microns_per_pixel']
    ax.add_patch(Rectangle((img_array.shape[1]-pixels_100um-20,
                           img_array.shape[0]-40),
                          pixels_100um, 5,
                          facecolor='white'))
    ax.text(img_array.shape[1]-pixels_100um/2-20,
            img_array.shape[0]-50,
            '100 µm',
            color='white',
            ha='center')

    # Add valley markers
    valleys = results['valley_positions']
    for valley in valleys:
        ax.axvline(x=valley, color='cyan', linestyle='--')
        ax.text(valley, 10, 'V', color='cyan')

    # Add measurements
    for i in range(len(valleys)-1):
        x1, x2 = valleys[i], valleys[i+1]
        ax.plot([x1, x2], [img_array.shape[0]/2]*2, color='yellow')
        ax.text((x1+x2)/2, img_array.shape[0]/2-20,
                f'{x2-x1:.1f}px',
                color='yellow',
                ha='center')

    # Add profile plot inset if available
    if roi_handler.last_profile_plot:
        ax_inset = fig.add_axes([0.6, 0.6, 0.35, 0.25])
        roi_handler.last_profile_plot.savefig(filename)
        plt.close(roi_handler.last_profile_plot)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

def initialize_imagej():
    """Initialize ImageJ with proper error handling."""
    try:
        logger.info("Initializing ImageJ...")

        # Check if running on macOS
        if sys.platform == 'darwin':
            logger.info("Running on macOS - using GUI mode for better compatibility")
            ij = imagej.init(mode='gui')
        else:
            # Use interactive mode for other platforms
            ij = imagej.init(mode='interactive')

        logger.info("ImageJ initialized successfully")
        return ij

    except Exception as e:
        logger.error(f"Failed to initialize ImageJ: {e!s}")
        logger.error("Please ensure you have installed all required dependencies:")
        logger.error("1. Install conda and mamba:")
        logger.error("   conda install mamba -n base -c conda-forge")
        logger.error("2. Create environment:")
        logger.error("   mamba create -n pyimagej -c conda-forge pyimagej openjdk=8")
        logger.error("3. Activate environment:")
        logger.error("   conda activate pyimagej")

        # Check for specific error types
        if "Not enough memory" in str(e):
            logger.error("\nMemory error detected. Try increasing Java heap size:")
            logger.error("export JAVA_OPTS='-Xmx4g'  # or higher value")
        elif "Class not found" in str(e):
            logger.error("\nImageJ classes not found. Make sure you have the correct version of ImageJ installed.")
        elif "mvn not found" in str(e):
            logger.error("\nMaven not found. Please install Maven and ensure it's in your PATH.")

        sys.exit(1)

def run_script():
    """Main script entry point."""
    try:
        # Verify installation and dependencies first
        if not verify_installation():
            return

        # Initialize ImageJ with proper error handling
        ij = initialize_imagej()

        # Get parameters
        params = {
            'group': 2,
            'element': 2,
            'selection_type': 'Line',
            'bar_type': 'Dark Bars',
            'smoothing': 'Medium',
            'show_details': True
        }

        # Validate parameters
        USAFTarget.validate_parameters(params['group'], params['element'])

        # Get image using tkinter file dialog
        root = tk.Tk()
        root.attributes('-topmost', True)  # Make sure the dialog appears on top
        root.lift()  # Lift the window
        root.focus_force()  # Force focus

        # Center the dialog
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - 400) // 2
        y = (screen_height - 300) // 2
        root.geometry(f'400x300+{x}+{y}')

        image_path = filedialog.askopenfilename(
            parent=root,
            title="Select USAF Target Image",
            filetypes=[("Image files", "*.tif *.tiff *.jpg *.jpeg *.png")]
        )
        root.destroy()  # Clean up the tkinter window

        if not image_path:
            logger.info("No image selected. Exiting...")
            return

        logger.info(f"Opening image: {image_path}")
        # Open image using ImageJ's io module
        jimage = ij.io().open(image_path)

        # Convert the image from ImageJ2 to xarray
        image = ij.py.from_java(jimage)

        # Display the image using matplotlib
        plt.figure(figsize=(10, 8))
        plt.imshow(image, cmap='gray')
        plt.title("Click two points to define the line selection")
        plt.axis('on')

        # Get line coordinates from user
        points = plt.ginput(2, timeout=0)
        plt.close()

        if len(points) != 2:
            logger.error("Please select exactly two points for the line.")
            return

        x1, y1 = points[0]
        x2, y2 = points[1]

        # Create a line ROI
        from ij.gui import Line
        roi = Line(int(x1), int(y1), int(x2), int(y2))

        # Create ROI handler with the selected ROI
        roi_handler = ROIHandler(ij, jimage, image, params['selection_type'])
        roi_handler.roi = roi

        # Set up save directory
        save_dir = get_save_directory(image_path)
        output_dir = get_output_subdir(save_dir)

        # Get profile
        logger.info("Extracting profile...")
        profile, angle = roi_handler.get_profile()

        # Create analyzer
        analyzer = ProfileAnalyzer(profile, params['smoothing'], params['bar_type'])

        # Create target
        target = USAFTarget(params['group'], params['element'])

        # Calculate results
        logger.info("Calculating results...")
        calculator = ResultsCalculator(target, analyzer, roi_handler)
        results = calculator.calculate_results()

        # Save results
        if SAVE_RESULTS:
            logger.info("Saving results...")
            save_results_to_csv(results, params, output_dir)
            create_summary_image(image, results, params, roi_handler, output_dir)

        # Show results
        print("\nUSAF Resolution Analysis Results:")
        print(f"Line Pairs/mm: {results['line_pairs_per_mm']:.4f}")
        print(f"Microns per Pixel: {results['microns_per_pixel']:.4f}")
        print(f"Contrast Ratio: {results['contrast_ratio']:.3f}")

        if SAVE_RESULTS:
            print(f"\nResults saved to: {output_dir}")

    except Exception as e:
        if DEBUG:
            logger.error(f"Error: {e!s}", exc_info=True)
        else:
            logger.error(f"Error: {e!s}")
        sys.exit(1)

if __name__ == '__main__':
    run_script()

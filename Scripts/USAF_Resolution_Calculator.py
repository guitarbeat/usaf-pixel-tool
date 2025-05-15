#@ String (label='USAF Resolution Target Calculator', visibility='MESSAGE', required=False) message
#@ Integer (label='Group Number', value=2, min=-2, max=7, style='spinner', persist=False, required=True) group
#@ Integer (label='Element Number', value=2, min=1, max=6, style='spinner', persist=False, required=True) element
#@ String (label='Selection Type', choices={'Line', 'Rectangle'}, style='radioButtonHorizontal', persist=False, required=True) selection_type
#@ String (label='Bar Type', choices={'Dark Bars', 'Bright Bars'}, style='radioButtonHorizontal', persist=False, required=True) bar_type
#@ String (label='Smoothing Level', choices={'Light', 'Medium', 'Heavy'}, style='radioButtonHorizontal', persist=False, required=True) smoothing
#@ Boolean (label='Show detailed analysis', value=True, persist=False, required=False) show_details

'''
USAF 1951 Resolution Target Calculator
This script calculates pixel size using a USAF resolution target.

Key Concepts:
- A line pair consists of one dark bar plus one bright space
- Three dark bars form two complete line pairs
- Measurements are made between centers of dark bars
- Calibration uses the known USAF target spacing
'''

# Standard imports
from __future__ import division
import math
import sys
import time
import shutil

# ImageJ imports
from ij import IJ, ImagePlus, WindowManager, Prefs
from ij.gui import GenericDialog, Plot, Line, Overlay, PointRoi, WaitForUserDialog, ProfilePlot, Roi, NonBlockingGenericDialog, TextRoi
from ij.process import ImageProcessor, FloatProcessor
from ij.measure import ResultsTable
from java.awt import Color
from java.awt.event import AdjustmentListener
from java.awt import Scrollbar
from ij.io import OpenDialog, FileSaver
from java.io import File
import os
import datetime
from ij.io import DirectoryChooser
from java.lang import System

# Constants
DEBUG = True
SAVE_RESULTS = True  # Set to True to automatically save results
ASK_FOR_SAVE_DIRECTORY = True  # Set to True to ask user for save directory

# Global variables
save_directory = None  # Will store the user-selected save directory or the source image directory
output_subdir = None   # Will store the subdirectory for this run

class USAFTarget:
    """Handles USAF target calculations and parameters."""
    MIN_GROUP = -2
    MAX_GROUP = 7
    MIN_ELEMENT = 1
    MAX_ELEMENT = 6
    
    def __init__(self, group, element):
        """Initialize USAF target with group and element numbers."""
        self.group = group
        self.element = element
        self.line_pairs_per_mm = self.calculate_line_pairs_per_mm()
        
    def calculate_line_pairs_per_mm(self):
        """Calculate line pairs per mm for given group and element."""
        return math.pow(2, self.group + (self.element-1)/6)
        
    def get_line_pair_spacing_microns(self):
        """Get spacing between line pairs in microns."""
        return (1.0 / self.line_pairs_per_mm) * 1000
        
    @staticmethod
    def validate_parameters(group, element):
        """Validate group and element numbers."""
        if not (USAFTarget.MIN_GROUP <= group <= USAFTarget.MAX_GROUP):
            raise ValueError("Group must be between {} and {}".format(USAFTarget.MIN_GROUP, USAFTarget.MAX_GROUP))
        if not (USAFTarget.MIN_ELEMENT <= element <= USAFTarget.MAX_ELEMENT):
            raise ValueError("Element must be between {} and {}".format(USAFTarget.MIN_ELEMENT, USAFTarget.MAX_ELEMENT))
        return True

class ProfileAnalyzer:
    """Handles intensity profile analysis."""
    
    SMOOTHING_LEVELS = {
        'Light': {'window_size': 3, 'iterations': 1},
        'Medium': {'window_size': 5, 'iterations': 2},
        'Heavy': {'window_size': 7, 'iterations': 3}
    }
    
    def __init__(self, profile, smoothing_level='Medium', bar_type='Dark Bars'):
        self.raw_profile = profile
        self.smoothing_params = self.SMOOTHING_LEVELS[smoothing_level]
        self.bar_type = bar_type
        self.profile = self.preprocess_profile()
        
    def preprocess_profile(self):
        """Preprocess the profile including smoothing and potential inversion."""
        profile = self.smooth_profile(self.raw_profile)
        if self.bar_type == 'Bright Bars':
            profile = [max(profile) - x for x in profile]
        return profile
        
    @staticmethod
    def smooth_profile(profile, window_size=5, iterations=2):
        """Apply moving average smoothing to profile."""
        smoothed = list(profile)
        for _ in range(iterations):
            temp = list(smoothed)
            for i in range(window_size, len(profile) - window_size):
                smoothed[i] = sum(temp[i-window_size:i+window_size+1]) / (2*window_size + 1)
        return smoothed
        
    def find_valleys(self):
        """Find valley positions (dark bars) in profile."""
        profile_length = len(self.profile)
        
        # Calculate profile statistics
        mean = sum(self.profile) / profile_length
        std_dev = math.sqrt(sum((x - mean) ** 2 for x in self.profile) / profile_length)
        
        # More aggressive valley detection
        valley_threshold = mean - 0.75 * std_dev  # Changed from 0.5 to 0.75
        
        # Find all potential valleys
        potential_valleys = []
        min_valley_width = 10  # Minimum width of a valley in pixels
        
        i = 0
        while i < profile_length - min_valley_width:
            if self.profile[i] < valley_threshold:
                # Found start of a valley, find its center
                valley_start = i
                while i < profile_length and self.profile[i] < valley_threshold:
                    i += 1
                valley_end = i
                
                # Use center of valley
                valley_center = valley_start + (valley_end - valley_start) // 2
                potential_valleys.append((valley_center, self.profile[valley_center]))
            i += 1
        
        # Sort valleys by depth and take the 3 deepest
        potential_valleys.sort(key=lambda x: x[1])  # Sort by intensity (depth)
        valleys = [v[0] for v in potential_valleys[:3]]
        
        # Sort by position
        valleys.sort()
        
        return valleys
    
    def is_local_minimum(self, pos, window=3):
        """Check if position is a local minimum."""
        center_value = self.profile[pos]
        for i in range(max(0, pos - window), min(len(self.profile), pos + window + 1)):
            if i != pos and self.profile[i] < center_value:
                return False
        return True

class ROIHandler:
    """Handles ROI selection and profile extraction."""
    
    def __init__(self, image, selection_type='Line'):
        self.image = image
        self.selection_type = selection_type
        self.roi = None
        self.profile = None
        self.selection_length = None
        self.angle = 0.0
        self.valleys = None
        self.roi_image = None
    
    def get_profile(self):
        """Get intensity profile from ROI."""
        self.roi = self.image.getRoi()
        if self.roi is None:
            raise ValueError("No selection found")
            
        if isinstance(self.roi, Line):
            profile = self._get_line_profile()
            return profile, 0.0  # Line tool doesn't handle tilt
        elif isinstance(self.roi, Roi):
            profile, angle = self._get_rectangle_profile()
            return profile, angle
        else:
            raise ValueError("Invalid selection type")
    
    def save_roi_image(self, valleys=None):
        """Save the ROI image with overlay showing the valleys."""
        if self.roi_image is None:
            return
            
        # Create a timestamp for the filename
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        base_filename = "USAF_ROI_Image_{}.png".format(timestamp)
        
        # Get the full save path
        roi_filename = get_save_path(base_filename)
        
        # If valleys are provided, add markers to the overlay
        if valleys is not None and len(valleys) > 0:
            overlay = Overlay()
            
            # Get the ROI image dimensions
            width = self.roi_image.getWidth()
            height = self.roi_image.getHeight()
            
            # Add lines for the valleys
            for valley in valleys:
                line = Line(valley, 0, valley, height-1)
                line.setStrokeColor(Color.RED)
                line.setStrokeWidth(2)
                overlay.add(line)
            
            # Set the overlay
            self.roi_image.setOverlay(overlay)
        
        # Save the image
        FileSaver(self.roi_image).saveAsPng(roi_filename)
        IJ.log("ROI image saved to: " + roi_filename)
        return roi_filename
        
    def _get_line_profile(self):
        """Get profile from line selection."""
        self.selection_length = self.roi.getLength()
        profile_plot = ProfilePlot(self.image)
        
        # Create ROI image for saving
        ip = self.image.getProcessor().crop()
        self.roi_image = ImagePlus("ROI", ip)
        
        return profile_plot.getProfile()
    
    def _get_rectangle_profile(self):
        """Get profile from rectangle selection with averaging."""
        rect = self.roi.getBounds()
        self.selection_length = rect.width
        self.angle = self.roi.getAngle()
        
        # Create ROI image
        ip = self.image.getProcessor().crop()
        roi_imp = ImagePlus("ROI", ip)
        roi_imp.show()
        roi_imp.setTitle("ROI Analysis")
        
        # Create non-blocking dialog with slider
        gd = NonBlockingGenericDialog("Tilt Adjustment")
        gd.addSlider("Rotation angle (degrees)", -45, 45, self.angle)
        gd.addCheckbox("Auto-detect tilt", True)
        
        # Get the slider
        slider = gd.getSliders().get(0)
        
        # Create preview image
        preview_imp = roi_imp.duplicate()
        preview_imp.setTitle("Interactive Tilt Preview")
        preview_imp.show()
        
        # Add adjustment listener to slider
        class SliderListener(AdjustmentListener):
            def adjustmentValueChanged(self, event):
                angle = slider.getValue()
                # Update preview
                preview_imp.setProcessor(roi_imp.getProcessor().duplicate())
                IJ.run(preview_imp, "Rotate...", "angle={} grid=1 interpolation=Bilinear".format(-angle))
                preview_imp.updateAndDraw()
        
        slider.addAdjustmentListener(SliderListener())
        
        # Show dialog
        gd.showDialog()
        
        if gd.wasCanceled():
            preview_imp.close()
            roi_imp.close()
            raise ValueError("Tilt correction cancelled")
        
        # Get final angle
        final_angle = slider.getValue()
        use_auto = gd.getNextBoolean()
        
        if use_auto:
            final_angle = self.angle
        
        # Apply final rotation to original ROI
        if abs(final_angle) > 0.01:
            IJ.run(roi_imp, "Rotate...", "angle={} grid=1 interpolation=Bilinear".format(-final_angle))
        
        preview_imp.close()
        
        self.angle = final_angle
        self.roi_image = roi_imp  # Save for later use
        
        # Get dimensions after rotation
        width = roi_imp.getWidth()
        height = roi_imp.getHeight()
        
        # Take multiple profiles and average
        num_profiles = min(height, 10)
        step = height // num_profiles
        
        # Create overlay for visualization
        overlay = Overlay()
        profiles = []
        averaged_profile = [0.0] * width
        
        for y in range(0, height, step):
            # Add sampling line to overlay
            line_roi = Line(0, y, width-1, y)
            line_roi.setStrokeColor(Color.YELLOW)
            overlay.add(line_roi)
            
            # Get profile
            profile = [roi_imp.getProcessor().getPixel(x, y) for x in range(width)]
            profiles.append(profile)
            
            # Add to average
            for x in range(width):
                averaged_profile[x] += profile[x] / num_profiles
        
        # Show sampling lines
        roi_imp.setOverlay(overlay)
        
        # Create profile visualization and get the valleys
        self.valleys = self._show_profiles_plot(profiles, averaged_profile)
        
        return averaged_profile, self.angle
    
    def _show_profiles_plot(self, individual_profiles, averaged_profile):
        """Show plot of individual and averaged profiles with measurements."""
        # Create a timestamp for the filename
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        base_filename = "USAF_Profile_Plot_{}.png".format(timestamp)
        
        # Get the full save path if we're saving results
        if SAVE_RESULTS:
            plot_filename = get_save_path(base_filename)
        
        # Create the plot
        plot = Plot("Line Profiles", "Distance (pixels)", "Intensity")
        plot.setLimits(0, len(averaged_profile), 
                      min(averaged_profile)*0.9, 
                      max(averaged_profile)*1.1)
        
        # Add individual profiles in gray
        for profile in individual_profiles:
            plot.setColor(Color(200, 200, 200))
            plot.addPoints(range(len(profile)), profile, Plot.LINE)
        
        # Add averaged profile in blue
        plot.setColor(Color(0, 0, 255))
        plot.setLineWidth(2)
        plot.addPoints(range(len(averaged_profile)), averaged_profile, Plot.LINE)
        
        # Add valley markers and measurements
        analyzer = ProfileAnalyzer(averaged_profile, 'Light')  # Create temporary analyzer
        valleys = analyzer.find_valleys()
        
        # Mark valleys (dark bars)
        plot.setColor(Color(255, 0, 0))  # Red
        for valley in valleys:
            # Draw vertical line at valley
            plot.drawLine(valley, min(averaged_profile)*0.9, 
                         valley, averaged_profile[valley])
            # Add label
            plot.addLabel(valley, averaged_profile[valley], "V")
        
        # Draw line pair measurements
        plot.setColor(Color(0, 255, 0))  # Green
        for i in range(len(valleys)-1):
            # Draw line between valleys showing line pair
            y_pos = min(averaged_profile)*0.95
            x1 = valleys[i]
            x2 = valleys[i+1]
            plot.drawLine(x1, y_pos, x2, y_pos)
            # Add measurement label
            plot.addLabel((x1+x2)/2, y_pos, 
                         "{:.1f}px".format(x2-x1))
        
        plot.addLegend(
            "Individual Profiles\n" +
            "Averaged Profile\n" +
            "Valley Centers\n" +
            "Line Pair Width")
        
        # Save the plot if SAVE_RESULTS is true
        if SAVE_RESULTS:
            plotImage = plot.makeHighResolution("USAF Profile", 4.0, True)
            FileSaver(plotImage).saveAsPng(plot_filename)
            IJ.log("Profile plot saved to: " + plot_filename)
        
        # Show the plot
        plot.show()
        
        return valleys

    def get_roi_overlay(self, valleys=None):
        """Return an overlay with valley markers for the ROI image."""
        if self.roi_image is None:
            return None
        overlay = Overlay()
        width = self.roi_image.getWidth()
        height = self.roi_image.getHeight()
        if valleys is not None and len(valleys) > 0:
            for valley in valleys:
                line = Line(valley, 0, valley, height-1)
                line.setStrokeColor(Color.RED)
                line.setStrokeWidth(2)
                overlay.add(line)
        return overlay

class ResultsCalculator:
    """Handles measurement calculations and analysis."""
    
    def __init__(self, usaf_target, profile_analyzer, roi_handler):
        self.target = usaf_target
        self.analyzer = profile_analyzer
        self.roi_handler = roi_handler
        self.valleys = None
        self.results = {}
        
    def calculate_results(self):
        """Perform all calculations and store results."""
        self.valleys = self.analyzer.find_valleys()
        if len(self.valleys) < 3:
            raise ValueError("Could not detect 3 dark bars")
            
        # Basic USAF target calculations
        self.results['line_pairs_per_mm'] = self.target.line_pairs_per_mm
        self.results['microns_per_pair'] = self.target.get_line_pair_spacing_microns()
        
        # Calculate spacings between valleys (dark bars)
        spacings = []
        for i in range(len(self.valleys)-1):
            spacing = self.valleys[i+1] - self.valleys[i]
            spacings.append(spacing)
            
        # Calculate statistics
        self.results['valley_positions'] = self.valleys
        self.results['valley_spacings'] = spacings
        self.results['avg_spacing'] = sum(spacings) / len(spacings)
        self.results['spacing_std_dev'] = math.sqrt(
            sum((x - self.results['avg_spacing'])**2 for x in spacings) / len(spacings))
            
        # Calculate line pair width (one valley spacing = one line pair)
        self.results['pixels_per_pair'] = self.results['avg_spacing']  # One spacing is one pair
        self.results['microns_per_pixel'] = (self.results['microns_per_pair'] / 
                                           self.results['pixels_per_pair'])
                                           
        # Calculate contrast
        profile = self.analyzer.profile
        valley_intensities = [profile[v] for v in self.valleys]
        peak_intensities = self._find_peak_intensities(profile, self.valleys)
        
        self.results['contrast_ratio'] = (max(peak_intensities) - min(valley_intensities)) / (
            max(peak_intensities) + min(valley_intensities))
            
        # Store selection angle
        self.results['selection_angle'] = self.roi_handler.angle
        
        return self.results
    
    def _find_peak_intensities(self, profile, valleys):
        """Find intensity peaks between valleys."""
        peaks = []
        for i in range(len(valleys)-1):
            start = valleys[i]
            end = valleys[i+1]
            peak_pos = max(range(start, end), key=lambda x: profile[x])
            peaks.append(profile[peak_pos])
        return peaks

class ResultsDisplay:
    """Handles display and formatting of results."""
    
    def __init__(self, results, params):
        self.results = results
        self.params = params
        
    def show_results(self):
        """Display formatted results."""
        title = 'USAF Target Analysis Results'
        
        # Close existing window if present
        window = WindowManager.getWindow(title)
        if window is not None:
            window.close()
        
        output = []
        output.extend(self._format_header())
        output.extend(self._format_input_params())
        output.extend(self._format_calculations())
        output.extend(self._format_analysis())
        
        if abs(self._get_measurement_error()) > 20:
            output.extend(self._format_warnings())
            
        self._show_results_window(title, output)
        
    def _format_header(self):
        return [
            '================================================================',
            '                   USAF Target Analysis Results                    ',
            '================================================================\n'
        ]
        
    def _format_input_params(self):
        return [
            'Input Parameters:',
            '-----------------',
            'Group Number: {}'.format(self.params["group"]),
            'Element Number: {}'.format(self.params["element"]),
            'Selection Type: {}'.format(self.params["selection_type"]),
            'Bar Type: {}'.format(self.params["bar_type"]),
            'Smoothing Level: {}'.format(self.params["smoothing"]),
            'Target Line Pairs/mm: {:.4f}\n'.format(self.results["line_pairs_per_mm"])
        ]
        
    def _format_calculations(self):
        return [
            'Resolution Calculations:',
            '-----------------------',
            'Line Pairs/mm: {:.4f}'.format(self.results["line_pairs_per_mm"]),
            'Dark Bars Detected: 3 (forming 2 complete line pairs)',
            'Microns per Line Pair: {:.4f}'.format(self.results["microns_per_pair"]),
            'Average Bar Spacing: {:.1f} pixels'.format(self.results["avg_spacing"]),
            'Complete Line Pair Width: {:.1f} pixels'.format(self.results["pixels_per_pair"]),
            'Microns per Pixel: {:.4f}\n'.format(self.results["microns_per_pixel"])
        ]
        
    def _format_analysis(self):
        output = [
            'Profile Analysis:',
            '-----------------',
            'Bar-to-Bar Measurements:']
            
        # Use 'um' instead of trying to encode µm
        for i, spacing in enumerate(self.results['valley_spacings']):
            output.append('  Bars {}-{}: {:.1f} pixels ({:.1f} um)'.format(i+1, i+2, spacing, spacing * self.results["microns_per_pixel"]))
                
        output.extend([
            '\nMeasurement Statistics:',
            '  Bar-to-bar average: {:.1f} pixels ({:.1f} um)'.format(self.results["avg_spacing"], self.results["avg_spacing"] * self.results["microns_per_pixel"]),
            '  Standard deviation: {:.1f} pixels ({:.1f} um)'.format(self.results["spacing_std_dev"], self.results["spacing_std_dev"] * self.results["microns_per_pixel"]),
            '  Coefficient of variation: {:.1f}%'.format((self.results["spacing_std_dev"] / self.results["avg_spacing"]) * 100),
            '  Contrast ratio: {:.3f}'.format(abs(self.results["contrast_ratio"])),
            '\nSelection Information:',
            '  Method: {}'.format(self.params["selection_type"]),
            '  Profile averaging: {}'.format("Yes (multiple profiles)" if self.params["selection_type"] == "Rectangle" else "No"),
            '  Selection angle: {:.2f} deg'.format(self.results.get("selection_angle", 0.0)),
            '  Tilt correction: {}'.format("Applied" if abs(self.results.get("selection_angle", 0.0)) > 0.1 else "Not needed")
        ])
        
        return output
        
    def _format_warnings(self):
        return [
            '\nWARNING: Large measurement error detected!',
            'Suggestions:',
            '1. Verify Group and Element numbers are correct',
            '2. Check selection alignment and positioning',
            '3. Try different smoothing level',
            '4. Consider using Rectangle tool for better averaging'
        ]
        
    def _get_measurement_error(self):
        measured = self.results['avg_spacing'] * self.results['microns_per_pixel']
        expected = self.results['microns_per_pair']
        return ((measured - expected) / expected) * 100
        
    def _show_results_window(self, title, output):
        rt = ResultsTable()
        rt.reset()
        rt.showRowNumbers(False)
        
        for line in output:
            rt.incrementCounter()
            rt.addLabel("Results", line)
        
        rt.show(title)

def run_script():
    """Main script entry point."""
    try:
        imp = WindowManager.getCurrentImage()
        if imp is None:
            image_titles = WindowManager.getImageTitles()
            if len(image_titles) > 0:
                imp = WindowManager.getImage(image_titles[0])
            else:
                choice = IJ.showMessageWithCancel(
                    "No Image Open", 
                    "No image is currently open. Would you like to open an image now?")
                if choice:
                    IJ.run("Open...", "")
                    imp = WindowManager.getCurrentImage()
                    if imp is None:
                        IJ.log("No image was opened. Script canceled.")
                        return
                else:
                    IJ.log("Script canceled: No image open")
                    return
        params = get_script_parameters()
        if not params:
            return
        try:
            target = USAFTarget(params['group'], params['element'])
        except ValueError as e:
            IJ.error(str(e))
            return
        if SAVE_RESULTS:
            get_save_directory(imp)
            get_output_subdir()
        roi_handler = ROIHandler(imp, params['selection_type'])
        show_selection_dialog(params['selection_type'])
        try:
            profile, angle = roi_handler.get_profile()
        except ValueError as e:
            IJ.error(str(e))
            return
        analyzer = ProfileAnalyzer(profile, params['smoothing'], params['bar_type'])
        calculator = ResultsCalculator(target, analyzer, roi_handler)
        try:
            results = calculator.calculate_results()
        except ValueError as e:
            IJ.error(str(e))
            return
        display = ResultsDisplay(results, params)
        display.show_results()
        update_image_calibration(imp, results['microns_per_pixel'])
        if SAVE_RESULTS:
            csv_file = save_results_to_csv(results, params)
            if roi_handler.roi_image is not None:
                roi_handler.save_roi_image(results['valley_positions'])
            create_summary_image(imp, results, params, roi_handler)
            IJ.showMessage("USAF Resolution Calculator", 
                          "Analysis complete!\n\n" +
                          "Results and images have been saved to:\n" +
                          get_output_subdir())
    except Exception as e:
        if DEBUG:
            import traceback
            IJ.log('Error: ' + str(e))
            IJ.log('Traceback:')
            traceback.print_exc(file=sys.stdout)
        else:
            IJ.error('Error', str(e))

def show_selection_dialog(selection_type):
    """Show dialog with selection instructions."""
    message = ('Draw {} across ALL THREE line pairs\n'.format(selection_type) +
              '1. Include all three dark bars\n' +
              '2. Draw perpendicular to the bars\n' +
              '3. Selection should be centered on the bars')
    dialog = WaitForUserDialog('Make Selection', message)
    dialog.show()

def update_image_calibration(imp, microns_per_pixel):
    """Update image spatial calibration."""
    cal = imp.getCalibration()
    cal.setUnit("um")  # Changed from 'µm' to 'um'
    cal.pixelWidth = microns_per_pixel
    cal.pixelHeight = microns_per_pixel
    imp.updateAndDraw()

def get_script_parameters():
    '''Get parameters from script parameters or user input.'''
    try:
        # These variables are defined by the script parameters
        params = {
            'group': group,
            'element': element,
            'selection_type': selection_type,
            'bar_type': bar_type,
            'smoothing': smoothing,
            'show_details': show_details
        }
        return params
    except NameError:
        # If script parameters aren't defined, get from dialog
        return get_user_parameters()

def get_user_parameters():
    '''Get parameters from user dialog if script parameters aren't available.'''
    gd = GenericDialog('USAF 1951 Target Calculator')

    gd.addNumericField('Group Number:', 2, 0)
    gd.addNumericField('Element Number:', 2, 0)
    gd.addChoice('Selection Type:', ['Line', 'Rectangle'], 'Line')
    gd.addChoice('Bar Type:', ['Dark Bars', 'Bright Bars'], 'Dark Bars')
    gd.addChoice('Smoothing Level:', ['Light', 'Medium', 'Heavy'], 'Medium')
    gd.addCheckbox('Show detailed analysis', True)
    gd.showDialog()
    
    if gd.wasCanceled():
        return None
        
    return {
        'group': int(gd.getNextNumber()),
        'element': int(gd.getNextNumber()),
        'selection_type': gd.getNextChoice(),
        'bar_type': gd.getNextChoice(),
        'smoothing': gd.getNextChoice(),
        'show_details': gd.getNextBoolean()
    }

def save_results_to_csv(results, params):
    """Save results to a CSV file."""
    # Define the CSV file name
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_filename = "USAF_Resolution_Results_{}.csv".format(timestamp)
    
    # Get the full save path
    filename = get_save_path(base_filename)
    
    # Define the CSV content
    content = [
        "USAF Resolution Target Calculator Results",
        "Date: {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        "Group: {}".format(params["group"]),
        "Element: {}".format(params["element"]),
        "Selection Type: {}".format(params["selection_type"]),
        "Bar Type: {}".format(params["bar_type"]),
        "Smoothing Level: {}".format(params["smoothing"]),
        "Target Line Pairs/mm: {:.4f}".format(results["line_pairs_per_mm"]),
        "Microns per Line Pair: {:.4f}".format(results["microns_per_pair"]),
        "Average Bar Spacing: {:.1f} pixels".format(results["avg_spacing"]),
        "Complete Line Pair Width: {:.1f} pixels".format(results["pixels_per_pair"]),
        "Microns per Pixel: {:.4f}".format(results["microns_per_pixel"]),
        "Contrast Ratio: {:.3f}".format(results["contrast_ratio"]),
        "Selection Angle: {:.2f} degrees".format(results["selection_angle"]),
        "Tilt Correction: {}".format('Applied' if abs(results["selection_angle"]) > 0.1 else 'Not needed')
    ]
    
    # Save the CSV file
    with open(filename, 'w') as file:
        for line in content:
            file.write(line + "\n")
    
    IJ.log("Results saved to CSV file: " + filename)
    return filename

def create_summary_image(imp, results, params, roi_handler):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_filename = "USAF_Summary_{}.png".format(timestamp)
    summary_filename = get_save_path(base_filename)
    summary_imp = imp.duplicate()
    summary_imp.setTitle("USAF Summary")
    overlay = Overlay()
    # Add semi-transparent background for text
    roi = Roi(10, 10, 400, 240)
    roi.setFillColor(Color(0, 0, 0, 150))
    overlay.add(roi)
    # Add text annotations
    text_roi = TextRoi(15, 15, "USAF Target Analysis Results")
    text_roi.setColor(Color.WHITE)
    text_roi.setStrokeWidth(2)
    text_roi.setFontSize(16)
    overlay.add(text_roi)
    result_lines = [
        "Group: {}, Element: {}".format(params["group"], params["element"]),
        "Line Pairs/mm: {:.4f}".format(results["line_pairs_per_mm"]),
        "Microns per Pixel: {:.4f}".format(results["microns_per_pixel"]),
        "Contrast Ratio: {:.3f}".format(results["contrast_ratio"]),
        "Selection Method: {}".format(params["selection_type"])
    ]
    for i, line in enumerate(result_lines):
        y_pos = 40 + i * 22
        text_roi = TextRoi(20, y_pos, line)
        text_roi.setColor(Color.WHITE)
        text_roi.setFontSize(14)
        overlay.add(text_roi)
    # Add calibration bar
    cal_height = summary_imp.getHeight() - 40
    cal_x = summary_imp.getWidth() - 160
    pixels_100um = 100 / results["microns_per_pixel"]
    scale_roi = Line(cal_x, cal_height, cal_x + pixels_100um, cal_height)
    scale_roi.setStrokeColor(Color.WHITE)
    scale_roi.setStrokeWidth(3)
    overlay.add(scale_roi)
    scale_text = TextRoi(cal_x, cal_height - 20, "100 µm")
    scale_text.setColor(Color.WHITE)
    scale_text.setFontSize(12)
    overlay.add(scale_text)
    # --- Add ROI overlay (valleys) to summary image ---
    roi_overlay = roi_handler.get_roi_overlay(results.get('valley_positions', []))
    if roi_overlay is not None:
        for i in range(roi_overlay.size()):
            overlay.add(roi_overlay.get(i))
    summary_imp.setOverlay(overlay)
    flatImp = summary_imp.flatten()
    FileSaver(flatImp).saveAsPng(summary_filename)
    IJ.log("Summary image saved to: " + summary_filename)
    summary_imp.close()
    return summary_filename

def get_save_directory(imp):
    """Determine the directory where results should be saved.
    
    Args:
        imp: The ImagePlus object of the active image
        
    Returns:
        A string path to the save directory
    """
    global save_directory, output_subdir
    
    # If we already have a save directory, use it
    if save_directory:
        return save_directory
        
    # If we should ask the user for a directory
    if ASK_FOR_SAVE_DIRECTORY:
        # Use ImageJ's directory chooser dialog
        directory_chooser = DirectoryChooser("Select Directory to Save Results")
        
        # Try to start in the source image directory if possible
        if imp is not None and imp.getOriginalFileInfo() is not None:
            directory_chooser.setDefaultDirectory(imp.getOriginalFileInfo().directory)
        
        # Show the dialog and get the selected directory
        selected_dir = directory_chooser.getDirectory()
        
        if selected_dir is not None:
            save_directory = selected_dir
            IJ.log("Results will be saved to: " + save_directory)
            return save_directory
        else:
            # User canceled, use the image directory or current directory
            IJ.log("Directory selection canceled. Using default directory.")
    
    # Try to use the source image directory
    if imp is not None and imp.getOriginalFileInfo() is not None:
        save_directory = imp.getOriginalFileInfo().directory
        IJ.log("Using source image directory: " + save_directory)
        return save_directory
    
    # If all else fails, use current directory
    save_directory = System.getProperty("user.dir") + File.separator
    IJ.log("Using current working directory: " + save_directory)
    return save_directory

def get_output_subdir():
    global output_subdir, save_directory
    if output_subdir is not None:
        return output_subdir
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_subdir = os.path.join(save_directory, 'USAF_Results_' + timestamp) + File.separator
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    IJ.log("All output will be saved in: " + output_subdir)
    return output_subdir

def get_save_path(base_filename):
    """Get the full path for saving a file.
    
    Args:
        base_filename: The base filename without directory
        
    Returns:
        The full path including directory and filename
    """
    global save_directory
    
    # Make sure we have a valid save directory
    if save_directory is None:
        save_directory = System.getProperty("user.dir") + File.separator
        
    # Create the full path
    subdir = get_output_subdir()
    full_path = os.path.join(subdir, base_filename)
    
    return full_path

if __name__ in ['__builtin__','__main__']:
    run_script() 
#@ String (label='USAF Resolution Target Calculator', visibility='MESSAGE', required=False) message
#@ Integer (label='Group Number', value=2, min=-2, max=7, style='spinner', persist=False, required=True) group
#@ Integer (label='Element Number', value=2, min=1, max=6, style='spinner', persist=False, required=True) element
#@ String (label='Selection Type', choices={'Line', 'Rectangle'}, style='radioButtonHorizontal', persist=False, required=True) selection_type
#@ String (label='Bar Type', choices={'Dark Bars', 'Bright Bars'}, style='radioButtonHorizontal', persist=False, required=True) bar_type
#@ String (label='Smoothing Level', choices={'Light', 'Medium', 'Heavy'}, style='radioButtonHorizontal', persist=False, required=True) smoothing
#@ Boolean (label='Show detailed analysis', value=True, persist=False, required=False) show_details

'''
USAF 1951 Resolution Target Calculator (v2)
A visually improved version for summary images and output organization.
'''

import datetime
import math
import os
import sys

from ij import IJ, ImagePlus, WindowManager
from ij.gui import (
    GenericDialog,
    Line,
    NonBlockingGenericDialog,
    Overlay,
    Plot,
    ProfilePlot,
    Roi,
    TextRoi,
    WaitForUserDialog,
)
from ij.io import DirectoryChooser, FileSaver
from ij.process import Blitter
from java.awt import Color, Font
from java.awt.event import AdjustmentListener
from java.io import File
from java.lang import System

DEBUG = True
SAVE_RESULTS = True
ASK_FOR_SAVE_DIRECTORY = True
save_directory = None
output_subdir = None

# --- Directory and Path Management ---
def get_save_directory(imp):
    global save_directory, output_subdir
    if save_directory:
        return save_directory
    if ASK_FOR_SAVE_DIRECTORY:
        directory_chooser = DirectoryChooser("Select Directory to Save Results")
        if imp is not None and imp.getOriginalFileInfo() is not None:
            directory_chooser.setDefaultDirectory(imp.getOriginalFileInfo().directory)
        selected_dir = directory_chooser.getDirectory()
        if selected_dir is not None:
            save_directory = selected_dir
            IJ.log("Results will be saved to: " + save_directory)
            return save_directory
        else:
            IJ.log("Directory selection canceled. Using default directory.")
    if imp is not None and imp.getOriginalFileInfo() is not None:
        save_directory = imp.getOriginalFileInfo().directory
        IJ.log("Using source image directory: " + save_directory)
        return save_directory
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
    global save_directory
    if save_directory is None:
        save_directory = System.getProperty("user.dir") + File.separator
    subdir = get_output_subdir()
    return os.path.join(subdir, base_filename)

# --- ProfileAnalyzer, ROIHandler, ResultsCalculator, ResultsDisplay, and utility functions ---
# (Copied from v1, with minor changes for v2)

class ProfileAnalyzer:
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
        profile = self.smooth_profile(self.raw_profile)
        if self.bar_type == 'Bright Bars':
            profile = [max(profile) - x for x in profile]
        return profile
    @staticmethod
    def smooth_profile(profile, window_size=5, iterations=2):
        smoothed = list(profile)
        for _ in range(iterations):
            temp = list(smoothed)
            for i in range(window_size, len(profile) - window_size):
                smoothed[i] = sum(temp[i-window_size:i+window_size+1]) / (2*window_size + 1)
        return smoothed
    def find_valleys(self):
        profile_length = len(self.profile)
        mean = sum(self.profile) / profile_length
        std_dev = math.sqrt(sum((x - mean) ** 2 for x in self.profile) / profile_length)
        valley_threshold = mean - 0.75 * std_dev
        potential_valleys = []
        min_valley_width = 10
        i = 0
        while i < profile_length - min_valley_width:
            if self.profile[i] < valley_threshold:
                valley_start = i
                while i < profile_length and self.profile[i] < valley_threshold:
                    i += 1
                valley_end = i
                valley_center = valley_start + (valley_end - valley_start) // 2
                potential_valleys.append((valley_center, self.profile[valley_center]))
            i += 1
        potential_valleys.sort(key=lambda x: x[1])
        valleys = [v[0] for v in potential_valleys[:3]]
        valleys.sort()
        return valleys

class ROIHandler:
    def __init__(self, image, selection_type='Line'):
        self.image = image
        self.selection_type = selection_type
        self.roi = None
        self.profile = None
        self.selection_length = None
        self.angle = 0.0
        self.valleys = None
        self.roi_image = None
        self.last_profile_plot_img = None
    def get_profile(self):
        self.roi = self.image.getRoi()
        if self.roi is None:
            raise ValueError("No selection found")
        if isinstance(self.roi, Line):
            profile = self._get_line_profile()
            return profile, 0.0
        elif isinstance(self.roi, Roi):
            profile, angle = self._get_rectangle_profile()
            return profile, angle
        else:
            raise ValueError("Invalid selection type")
    def _get_line_profile(self):
        self.selection_length = self.roi.getLength()
        profile_plot = ProfilePlot(self.image)
        ip = self.image.getProcessor().crop()
        self.roi_image = ImagePlus("ROI", ip)
        return profile_plot.getProfile()
    def _get_rectangle_profile(self):
        rect = self.roi.getBounds()
        self.selection_length = rect.width
        self.angle = self.roi.getAngle()
        ip = self.image.getProcessor().crop()
        roi_imp = ImagePlus("ROI", ip)
        roi_imp.show()
        roi_imp.setTitle("ROI Analysis")
        gd = NonBlockingGenericDialog("Tilt Adjustment")
        gd.addSlider("Rotation angle (degrees)", -45, 45, self.angle)
        gd.addCheckbox("Auto-detect tilt", True)
        slider = gd.getSliders().get(0)
        preview_imp = roi_imp.duplicate()
        preview_imp.setTitle("Interactive Tilt Preview")
        preview_imp.show()
        class SliderListener(AdjustmentListener):
            def adjustmentValueChanged(self, event):
                angle = slider.getValue()
                preview_imp.setProcessor(roi_imp.getProcessor().duplicate())
                IJ.run(preview_imp, "Rotate...", f"angle={-angle} grid=1 interpolation=Bilinear")
                preview_imp.updateAndDraw()
        slider.addAdjustmentListener(SliderListener())
        gd.showDialog()
        if gd.wasCanceled():
            preview_imp.close()
            roi_imp.close()
            raise ValueError("Tilt correction cancelled")
        final_angle = slider.getValue()
        use_auto = gd.getNextBoolean()
        if use_auto:
            final_angle = self.angle
        if abs(final_angle) > 0.01:
            IJ.run(roi_imp, "Rotate...", f"angle={-final_angle} grid=1 interpolation=Bilinear")
        preview_imp.close()
        self.angle = final_angle
        self.roi_image = roi_imp
        width = roi_imp.getWidth()
        height = roi_imp.getHeight()
        num_profiles = min(height, 10)
        step = height // num_profiles
        overlay = Overlay()
        profiles = []
        averaged_profile = [0.0] * width
        for y in range(0, height, step):
            line_roi = Line(0, y, width-1, y)
            line_roi.setStrokeColor(Color.YELLOW)
            overlay.add(line_roi)
            profile = [roi_imp.getProcessor().getPixel(x, y) for x in range(width)]
            profiles.append(profile)
            for x in range(width):
                averaged_profile[x] += profile[x] / num_profiles
        roi_imp.setOverlay(overlay)
        self.valleys, self.last_profile_plot_img = self._show_profiles_plot(profiles, averaged_profile)
        return averaged_profile, self.angle
    def _show_profiles_plot(self, individual_profiles, averaged_profile):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        base_filename = f"USAF_Profile_Plot_{timestamp}.png"
        plot_filename = get_save_path(base_filename) if SAVE_RESULTS else None
        plot = Plot("Line Profiles", "Distance (pixels)", "Intensity")
        plot.setLimits(0, len(averaged_profile), min(averaged_profile)*0.9, max(averaged_profile)*1.1)
        for profile in individual_profiles:
            plot.setColor(Color(200, 200, 200))
            plot.addPoints(range(len(profile)), profile, Plot.LINE)
        plot.setColor(Color(0, 0, 255))
        plot.setLineWidth(2)
        plot.addPoints(range(len(averaged_profile)), averaged_profile, Plot.LINE)
        analyzer = ProfileAnalyzer(averaged_profile, 'Light')
        valleys = analyzer.find_valleys()
        plot.setColor(Color(255, 0, 0))
        for valley in valleys:
            plot.drawLine(valley, min(averaged_profile)*0.9, valley, averaged_profile[valley])
            plot.addLabel(valley, averaged_profile[valley], "V")
        plot.setColor(Color(0, 255, 0))
        for i in range(len(valleys)-1):
            y_pos = min(averaged_profile)*0.95
            x1 = valleys[i]
            x2 = valleys[i+1]
            plot.drawLine(x1, y_pos, x2, y_pos)
            plot.addLabel((x1+x2)/2, y_pos, f"{x2-x1:.1f}px")
        plot.addLegend("Individual Profiles\nAveraged Profile\nValley Centers\nLine Pair Width")
        if SAVE_RESULTS:
            plotImage = plot.makeHighResolution("USAF Profile", 4.0, True, ImagePlus.COLOR_RGB)
            FileSaver(plotImage).saveAsPng(plot_filename)
            IJ.log("Profile plot saved to: " + plot_filename)
            return valleys, plot.getImagePlus()
        plot.show()
        return valleys, None
    def get_roi_overlay(self, valleys=None):
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

class USAFTarget:
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

class ResultsCalculator:
    def __init__(self, usaf_target, profile_analyzer, roi_handler):
        self.target = usaf_target
        self.analyzer = profile_analyzer
        self.roi_handler = roi_handler
        self.valleys = None
        self.results = {}
    def calculate_results(self):
        self.valleys = self.analyzer.find_valleys()
        if len(self.valleys) < 3:
            raise ValueError("Could not detect 3 dark bars")
        self.results['line_pairs_per_mm'] = self.target.line_pairs_per_mm
        self.results['microns_per_pair'] = self.target.get_line_pair_spacing_microns()
        spacings = []
        for i in range(len(self.valleys)-1):
            spacing = self.valleys[i+1] - self.valleys[i]
            spacings.append(spacing)
        self.results['valley_positions'] = self.valleys
        self.results['valley_spacings'] = spacings
        self.results['avg_spacing'] = sum(spacings) / len(spacings)
        self.results['spacing_std_dev'] = math.sqrt(sum((x - self.results['avg_spacing'])**2 for x in spacings) / len(spacings))
        self.results['pixels_per_pair'] = self.results['avg_spacing']
        self.results['microns_per_pixel'] = (self.results['microns_per_pair'] / self.results['pixels_per_pair'])
        profile = self.analyzer.profile
        valley_intensities = [profile[v] for v in self.valleys]
        peak_intensities = self._find_peak_intensities(profile, self.valleys)
        self.results['contrast_ratio'] = (max(peak_intensities) - min(valley_intensities)) / (max(peak_intensities) + min(valley_intensities))
        self.results['selection_angle'] = self.roi_handler.angle
        return self.results
    def _find_peak_intensities(self, profile, valleys):
        peaks = []
        for i in range(len(valleys)-1):
            start = valleys[i]
            end = valleys[i+1]
            peak_pos = max(range(start, end), key=lambda x: profile[x])
            peaks.append(profile[peak_pos])
        return peaks

# --- Utility functions (get_script_parameters, etc.) ---
def get_script_parameters():
    try:
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
        return get_user_parameters()

def get_user_parameters():
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

def show_selection_dialog(selection_type):
    message = (f'Draw {selection_type} across ALL THREE line pairs\n' +
              '1. Include all three dark bars\n' +
              '2. Draw perpendicular to the bars\n' +
              '3. Selection should be centered on the bars')
    dialog = WaitForUserDialog('Make Selection', message)
    dialog.show()

def update_image_calibration(imp, microns_per_pixel):
    cal = imp.getCalibration()
    cal.setUnit("um")
    cal.pixelWidth = microns_per_pixel
    cal.pixelHeight = microns_per_pixel
    imp.updateAndDraw()

def save_results_to_csv(results, params):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_filename = f"USAF_Resolution_Results_{timestamp}.csv"
    filename = get_save_path(base_filename)
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
    with open(filename, 'w') as file:
        for line in content:
            file.write(line + "\n")
    IJ.log("Results saved to CSV file: " + filename)
    return filename

# --- Main script entry point ---
def run_script():
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
        update_image_calibration(imp, results['microns_per_pixel'])
        if SAVE_RESULTS:
            save_results_to_csv(results, params)
            # Use the new beautiful summary image
            profile_plot_img = getattr(roi_handler, 'last_profile_plot_img', None)
            create_beautiful_summary_image(imp, results, params, roi_handler, profile_plot_img)
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


# --- Main summary image creation ---
def create_beautiful_summary_image(imp, results, params, roi_handler, profile_plot_img=None):
    """
    Create a visually improved summary image with:
    - Modern, semi-transparent info box
    - Large, clear fonts
    - Labeled colored lines/arrows for valleys and bar-to-bar measurements
    - Modern scale bar
    - Optional: inset profile plot
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_filename = f"USAF_Summary_Beautiful_{timestamp}.png"
    summary_filename = get_save_path(base_filename)
    summary_imp = imp.duplicate()
    summary_imp.setTitle("USAF Summary (Beautiful)")
    overlay = Overlay()
    width = summary_imp.getWidth()
    height = summary_imp.getHeight()
    # --- Info box ---
    box_w, box_h = 420, 260
    box_x, box_y = 30, 30
    info_box = Roi(box_x, box_y, box_w, box_h)
    info_box.setFillColor(Color(0, 0, 0, 180))
    overlay.add(info_box)
    # --- Title and subtitle ---
    title = TextRoi(box_x+15, box_y+20, "USAF Target Analysis Results")
    title.setColor(Color.WHITE)
    title.setFont(Font("SansSerif", Font.BOLD, 22))
    overlay.add(title)
    subtitle = TextRoi(box_x+15, box_y+50, "(Automated Summary)")
    subtitle.setColor(Color(200,200,200))
    subtitle.setFont(Font("SansSerif", Font.ITALIC, 14))
    overlay.add(subtitle)
    # --- Key results ---
    result_lines = [
        "Group: {}, Element: {}".format(params["group"], params["element"]),
        "Line Pairs/mm: {:.4f}".format(results["line_pairs_per_mm"]),
        "Microns per Pixel: {:.4f}".format(results["microns_per_pixel"]),
        "Contrast Ratio: {:.3f}".format(results["contrast_ratio"]),
        "Selection Method: {}".format(params["selection_type"])
    ]
    for i, line in enumerate(result_lines):
        y_pos = box_y + 80 + i * 28
        text_roi = TextRoi(box_x+25, y_pos, line)
        text_roi.setColor(Color.WHITE)
        text_roi.setFont(Font("Monospaced", Font.PLAIN, 18))
        overlay.add(text_roi)
    # --- Modern scale bar ---
    cal_height = height - 60
    cal_x = width - 180
    pixels_100um = 100 / results["microns_per_pixel"]
    scale_roi = Line(cal_x, cal_height, cal_x + pixels_100um, cal_height)
    scale_roi.setStrokeColor(Color.WHITE)
    scale_roi.setStrokeWidth(5)
    overlay.add(scale_roi)
    scale_text = TextRoi(cal_x, cal_height - 28, "100 µm")
    scale_text.setColor(Color.WHITE)
    scale_text.setFont(Font("SansSerif", Font.BOLD, 16))
    overlay.add(scale_text)
    # --- Valley lines and labels ---
    valleys = results.get('valley_positions', [])
    for idx, valley in enumerate(valleys):
        vline = Line(valley, 0, valley, height)
        vline.setStrokeColor(Color.CYAN)
        vline.setStrokeWidth(3)
        overlay.add(vline)
        vlabel = TextRoi(valley+4, 10, f"V{idx+1}")
        vlabel.setColor(Color.CYAN)
        vlabel.setFont(Font("SansSerif", Font.BOLD, 16))
        overlay.add(vlabel)
    # --- Bar-to-bar measurement arrows and labels ---
    for i in range(len(valleys)-1):
        x1 = valleys[i]
        x2 = valleys[i+1]
        y_arrow = height//2
        arrow = Line(x1, y_arrow, x2, y_arrow)
        arrow.setStrokeColor(Color.YELLOW)
        arrow.setStrokeWidth(4)
        overlay.add(arrow)
        dist = x2-x1
        dist_label = TextRoi((x1+x2)//2-10, y_arrow-30, f"{dist:.1f}px")
        dist_label.setColor(Color.YELLOW)
        dist_label.setFont(Font("SansSerif", Font.BOLD, 16))
        overlay.add(dist_label)
    # --- Optional: Inset profile plot ---
    if profile_plot_img is not None:
        inset_w, inset_h = 320, 180
        inset_x, inset_y = width-inset_w-40, height-inset_h-40
        ip = profile_plot_img.getProcessor().resize(inset_w, inset_h, True)
        profile_inset = ImagePlus("Profile Inset", ip)
        summary_ip = summary_imp.getProcessor()
        summary_ip.copyBits(profile_inset.getProcessor(), inset_x, inset_y, Blitter.COPY)
        border = Roi(inset_x, inset_y, inset_w, inset_h)
        border.setStrokeColor(Color.WHITE)
        border.setStrokeWidth(3)
        overlay.add(border)
        label = TextRoi(inset_x+10, inset_y+10, "Profile Plot")
        label.setColor(Color.WHITE)
        label.setFont(Font("SansSerif", Font.BOLD, 14))
        overlay.add(label)
    summary_imp.setOverlay(overlay)
    flatImp = summary_imp.flatten()
    FileSaver(flatImp).saveAsPng(summary_filename)
    IJ.log("Beautiful summary image saved to: " + summary_filename)
    summary_imp.close()
    return summary_filename

if __name__ in ['__builtin__','__main__']:
    run_script()

// ImageProcessingMaster.ijm - Author: Aaron Woods - Updated: June 12, 2024
// Comprehensive image processing macro that combines and enhances functionality from:
// - Z-stack processing with MIP creation
// - Temporal color coding
// - Batch file conversion with contrast enhancement
// 
// Enhancements:
// - Split the dialog into multiple steps to improve usability
// - Removed Gaussian blur feature
// - Removed try...catch blocks (unsupported in ImageJ macro language)
// - Each dialog collects a subset of parameters
// - Compatible with ImageJ macro language
// - Added option to customize temporal color coding LUT
// - Added help buttons to dialog
// - Fixed issue with potential double processing of files
// - Added auto-detection of microscope calibration data

// Initialization
requires("1.53c");
run("Options...", "iterations=3 count=1 black edm=Overwrite");
run("Colors...", "foreground=white background=black selection=yellow");
run("Clear Results");
run("Close All");
close("Log");

// Global variables
var totalFiles = 0;
var processedFiles = 0;
var skippedFiles = 0;
var errorFiles = 0;
var startTime = getTime();
var errorLog = "";
var helpURL = "https://imagej.net/ij/macros/";

// --- Step 1: Basic Settings Dialog ---
Dialog.create("ImageProcessingMaster - Step 1 of 3");
Dialog.addMessage("Basic Settings");
Dialog.addDirectory("Input Directory:", "");
Dialog.addDirectory("Output Directory (leave empty to use input directory):", "");
Dialog.addString("File Extension:", ".tif");
Dialog.addCheckbox("Process Subdirectories Recursively", false);
Dialog.addCheckbox("Process Z-Stacks", true);
Dialog.addCheckbox("Bulk Convert and Enhance Contrast", true);
Dialog.addCheckbox("Overwrite Existing Files", false);
Dialog.addHelp(helpURL);
Dialog.show();

// Get values from Step 1
inputDir = Dialog.getString();
outputDir = Dialog.getString();
ext = Dialog.getString();
processSubdirs = Dialog.getCheckbox();
processZStacks = Dialog.getCheckbox();
bulkConvertContrast = Dialog.getCheckbox();
overwriteExisting = Dialog.getCheckbox();

// Validate that at least one processing option is selected
if (!processZStacks && !bulkConvertContrast) {
    exit("Error: You must select at least one processing option (Z-Stacks or Bulk Convert).");
}

// --- Step 2: Image Properties Dialog ---
Dialog.create("ImageProcessingMaster - Step 2 of 3");
Dialog.addCheckbox("Apply Image Properties", true);
Dialog.addCheckbox("Auto-detect from first image (overrides values below)", true);
Dialog.addMessage("Image Properties");
Dialog.addString("X Unit:", "um");
Dialog.addString("Y Unit:", "um");
Dialog.addString("Z Unit:", "um");
Dialog.addNumber("Pixel Width:", 0.7843412);
Dialog.addNumber("Pixel Height:", 0.7843412);
Dialog.addNumber("Voxel Depth:", 1.0);
Dialog.addHelp(helpURL);
Dialog.show();

// Get values from Step 2
applyImageProperties = Dialog.getCheckbox();
autoDetectProperties = Dialog.getCheckbox();
xUnit = Dialog.getString();
yUnit = Dialog.getString();
zUnit = Dialog.getString();
pixelWidth = Dialog.getNumber();
pixelHeight = Dialog.getNumber();
voxelDepth = Dialog.getNumber();

// --- Step 3: Processing Parameters Dialog ---
Dialog.create("ImageProcessingMaster - Step 3 of 4");
Dialog.addMessage("Z-Stack Processing Parameters");
Dialog.addNumber("Background Subtraction Radius (px, 0 for none):", 5);
Dialog.addCheckbox("Enhance Contrast on Projections", true);
Dialog.addCheckbox("Save Color-coded Image", true);
Dialog.addCheckbox("Apply 3D Median Filter", true);
Dialog.addNumber("Median Filter X (pixels):", 2);
Dialog.addNumber("Median Filter Y (pixels):", 2);
Dialog.addNumber("Median Filter Z (slices):", 2);
Dialog.addNumber("Enhance Contrast Saturation Level (%):", 0.35);
Dialog.addChoice("Color Coding LUT:", newArray("smart", "Fire", "Ice", "Spectrum", "Rainbow RGB", "physics", "thal", "brgbcmyw"));

Dialog.addMessage("Bulk Processing Parameters");
Dialog.addCheckbox("Apply Contrast to Bulk Images", true);
Dialog.addNumber("Bulk Contrast Saturation Level (%):", 0.35);
Dialog.addChoice("Bulk Output Format:", newArray("PNG", "TIFF", "JPEG"));
Dialog.addHelp(helpURL);
Dialog.show();

// Get values from Step 3
BGSub = round(Dialog.getNumber());
enhanceContrastProjections = Dialog.getCheckbox();
saveColorCoded = Dialog.getCheckbox();
applyMedianFilter = Dialog.getCheckbox();
medianX = round(Dialog.getNumber());
medianY = round(Dialog.getNumber());
medianZ = round(Dialog.getNumber());
contrastSaturation = Dialog.getNumber();
colorLUT = Dialog.getChoice();

// Get bulk processing parameters
applyContrastToBulk = Dialog.getCheckbox();
contrastSaturationForBulk = Dialog.getNumber();
bulkOutputFormat = Dialog.getChoice();

// --- Step 4: Scale Bar Options Dialog ---
Dialog.create("ImageProcessingMaster - Step 4 of 4");
Dialog.addMessage("Scale Bar Options");
Dialog.addNumber("Scale Bar Width:", 50);
Dialog.addNumber("Scale Bar Height:", 4);
Dialog.addNumber("Scale Bar Font Size:", 12);
Dialog.addChoice("Scale Bar Color:", newArray("White", "Black", "Gray", "Red", "Green", "Blue", "Yellow"));
Dialog.addChoice("Scale Bar Location:", newArray("Bottom Right", "Bottom Left", "Top Right", "Top Left"));
Dialog.addChoice("Scale Bar Label Option:", newArray("%unit", "Custom", "None"));
Dialog.addString("Scale Bar Custom Label:", "Scale");
Dialog.addNumber("Scale Bar Decimal Places:", 0);
Dialog.addHelp(helpURL);
Dialog.show();

// Get scale bar options
scaleBarWidth = Dialog.getNumber();
scaleBarHeight = Dialog.getNumber();
scaleBarFontSize = Dialog.getNumber();
scaleBarColor = Dialog.getChoice();
scaleBarLocation = Dialog.getChoice();
scaleBarLabelOption = Dialog.getChoice();
scaleBarCustomLabel = Dialog.getString();
scaleBarDecimalPlaces = Dialog.getNumber();

// Ensure output directory is set
if (outputDir == "") {
    outputDir = inputDir;
}

// Validate input parameters
if (!validateParameters()) {
    exit("Parameter validation failed.");
}

// Main Execution
setBatchMode(true);
processFolder(inputDir, outputDir);
setBatchMode(false);

// Generate summary report
generateReport();

print("Batch processing completed.");
showMessage("Batch Processing Complete", "Processed: " + processedFiles + " files\nSkipped: " + skippedFiles + " files\nErrors: " + errorFiles + " files");

// --- Function Definitions ---

// Function to validate input parameters
function validateParameters() {
    // Check if input directory exists
    if (!File.exists(inputDir)) {
        showError("Input directory does not exist:\n" + inputDir);
        return false;
    }
    // Check if output directory exists or can be created
    if (!File.exists(outputDir)) {
        if (!File.makeDirectory(outputDir)) {
            showError("Cannot create output directory:\n" + outputDir);
            return false;
        }
    }
    // Validate numeric parameters
    if (BGSub < 0) {
        showError("Background Subtraction Radius cannot be negative.");
        return false;
    }
    if (medianX <= 0 || medianY <= 0 || medianZ <= 0) {
        showError("Median Filter dimensions must be positive integers.");
        return false;
    }
    if (contrastSaturation < 0 || contrastSaturation > 100) {
        showError("Contrast Saturation Level must be between 0 and 100.");
        return false;
    }
    if (contrastSaturationForBulk < 0 || contrastSaturationForBulk > 100) {
        showError("Bulk Contrast Saturation Level must be between 0 and 100.");
        return false;
    }
    // If neither processing option is selected, show error
    if (!processZStacks && !bulkConvertContrast) {
        showError("You must select at least one processing option.");
        return false;
    }
    return true;
}

// Function to display error messages
function showError(message) {
    print("Error: " + message);
    errorLog += message + "\n";
    showMessage("Error", message);
    exit("Error: " + message);
}

// Function to process folders recursively
function processFolder(input, output) {
    var fileList = getFileList(input);
    fileList = Array.sort(fileList);
    for (var i = 0; i < fileList.length; i++) {
        var file = fileList[i];
        var filePath = input + File.separator + file;
        if (File.isDirectory(filePath)) {
            if (processSubdirs) {
                var subOutput = output + File.separator + file;
                if (!File.exists(subOutput)) {
                    File.makeDirectory(subOutput);
                }
                processFolder(filePath, subOutput);
            }
            continue;
        }
        if (endsWith(file.toLowerCase(), ext.toLowerCase())) {
            totalFiles++;
            // Process file based on user selection, but don't process same file twice
            if (processZStacks) {
                processZStackFile(input, output, file);
            } else if (bulkConvertContrast) {
                processBulkConvert(input, output, file);
            }
        }
    }
}

// Process individual z-stack file
function processZStackFile(input, output, fileName) {
    var filePath = input + File.separator + fileName;
    var baseName = File.getNameWithoutExtension(fileName);
    var processedDir = output + File.separator + baseName;
    var mipOutputPath = processedDir + File.separator + baseName + "_MAX.png";
    if (!overwriteExisting && File.exists(mipOutputPath)) {
        print("Skipping existing file: " + mipOutputPath);
        skippedFiles++;
        return;
    }
    
    // Open file and check if it's valid
    if (!File.exists(filePath)) {
        print("File not found: " + filePath);
        errorLog += "File not found: " + filePath + "\n";
        errorFiles++;
        return;
    }
    
    open(filePath);
    if (nImages == 0) {
        print("Failed to open: " + filePath);
        errorLog += "Failed to open: " + filePath + "\n";
        errorFiles++;
        return;
    }
    
    if (!isStack()) {
        print("File " + fileName + " is not a stack. Skipping.");
        close();
        skippedFiles++;
        return;
    }
    
    // Auto-detect image properties if selected
    if (applyImageProperties) {
        if (autoDetectProperties && totalFiles == 1) {
            // Only detect from first image to avoid slowdown
            detectImageProperties();
        }
        applyImagePropertiesToImage();
    }
    
    // Apply filters and projections
    var colorCodedImageTitle = applyFilters(fileName);
    var mipTitle = applyZProjection(fileName);
    if (enhanceContrastProjections) {
        enhanceContrastOnImage(mipTitle, contrastSaturation);
    }
    saveProcessedImagesAsPNG(output, fileName, mipTitle, colorCodedImageTitle);
    // Close the images
    closeAllImages();
    processedFiles++;
    updateProgress(processedFiles, totalFiles);
}

// Detect image properties from current image
function detectImageProperties() {
    print("Auto-detecting image properties...");
    // Store original units and calibration from the first image
    xUnit = Stack.getXUnit();
    yUnit = Stack.getYUnit();
    zUnit = Stack.getZUnit();
    pixelWidth = Stack.getVoxelWidth();
    pixelHeight = Stack.getVoxelHeight();
    voxelDepth = Stack.getVoxelDepth();
    
    print("Detected: Width=" + pixelWidth + xUnit + 
          ", Height=" + pixelHeight + yUnit + 
          ", Depth=" + voxelDepth + zUnit);
}

// Process individual file for bulk conversion and contrast enhancement
function processBulkConvert(input, output, fileName) {
    var inputFilePath = input + File.separator + fileName;
    var outputFileName = replace(fileName, ext, "." + toLowerCase(bulkOutputFormat));
    var outputFilePath = output + File.separator + outputFileName;

    if (!overwriteExisting && File.exists(outputFilePath)) {
        print("Skipping existing file: " + outputFilePath);
        skippedFiles++;
        return;
    }
    
    // Check if file exists and can be opened
    if (!File.exists(inputFilePath)) {
        print("File not found: " + inputFilePath);
        errorLog += "File not found: " + inputFilePath + "\n";
        errorFiles++;
        return;
    }
    
    open(inputFilePath);
    if (nImages == 0) {
        print("Failed to open: " + inputFilePath);
        errorLog += "Failed to open: " + inputFilePath + "\n";
        errorFiles++;
        return;
    }
    
    // Auto-detect image properties if selected
    if (applyImageProperties) {
        if (autoDetectProperties && totalFiles == 1) {
            // Only detect from first image to avoid slowdown
            detectImageProperties();
        }
        applyImagePropertiesToImage();
    }
    
    // Enhance contrast if selected
    if (applyContrastToBulk) {
        enhanceContrastOnImage(getTitle(), contrastSaturationForBulk);
    }
    
    // Add scale bar if needed and if we're applying image properties
    if (applyImageProperties && scaleBarWidth > 0) {
        addScaleBar();
    }
    
    // Save the image in the selected format in the output directory
    saveAs(bulkOutputFormat, outputFilePath);
    // Close the image
    close();
    print("Processed and saved: " + outputFilePath);
    processedFiles++;
    updateProgress(processedFiles, totalFiles);
}

// Add scale bar to the current image
function addScaleBar() {
    // Set scale bar color
    if (scaleBarColor == "White") {
        run("Colors...", "foreground=white");
    } else if (scaleBarColor == "Black") {
        run("Colors...", "foreground=black");
    } else if (scaleBarColor == "Gray") {
        run("Colors...", "foreground=gray");
    } else if (scaleBarColor == "Red") {
        run("Colors...", "foreground=red");
    } else if (scaleBarColor == "Green") {
        run("Colors...", "foreground=green");
    } else if (scaleBarColor == "Blue") {
        run("Colors...", "foreground=blue");
    } else if (scaleBarColor == "Yellow") {
        run("Colors...", "foreground=yellow");
    }
    
    // Set scale bar location
    var location = "";
    if (scaleBarLocation == "Bottom Right") {
        location = "lower right";
    } else if (scaleBarLocation == "Bottom Left") {
        location = "lower left";
    } else if (scaleBarLocation == "Top Right") {
        location = "upper right";
    } else if (scaleBarLocation == "Top Left") {
        location = "upper left";
    }
    
    // Set scale bar label
    var label = "";
    if (scaleBarLabelOption == "%unit") {
        label = "";  // This will use the default unit
    } else if (scaleBarLabelOption == "Custom") {
        label = scaleBarCustomLabel;
    } else {
        label = " ";  // Space for no label
    }
    
    // Add scale bar
    run("Scale Bar...", "width=" + scaleBarWidth + " height=" + scaleBarHeight + 
        " font=" + scaleBarFontSize + " color=" + toLowerCase(scaleBarColor) + 
        " background=None location=[" + location + "] label=" + label + 
        " bold overlay show");
}

// Apply image properties to the current image
function applyImagePropertiesToImage() {
    Stack.setXUnit(xUnit);
    Stack.setYUnit(yUnit);
    Stack.setZUnit(zUnit);
    var channels = 1; // Assuming single-channel images
    var slices = nSlices(); // Automatically set slices to the number of slices in the stack
    var frames = 1; // Assuming no time frames
    run("Properties...", "channels=" + channels + " slices=" + slices + " frames=" + frames +
        " pixel_width=" + pixelWidth + " pixel_height=" + pixelHeight + " voxel_depth=" + voxelDepth);
}

// Check if the opened image is a stack
function isStack() {
    return (nSlices() > 1);
}

// Apply 3D median filter, background subtraction, and temporal color coding
function applyFilters(originalImageTitle) {
    selectWindow(originalImageTitle);
    if (applyMedianFilter) {
        run("Median 3D...", "x=" + medianX + " y=" + medianY + " z=" + medianZ);
    }
    if (BGSub > 0) {
        run("Subtract Background...", "rolling=" + BGSub + " stack");
    }
    run("Temporal-Color Code", "lut=" + colorLUT + " start=1 end=" + nSlices() + " create");
    setOption("ScaleConversions", true);
    var colorCodedImageTitle = getTitle(); // Get the title of the new color-coded image
    return colorCodedImageTitle;
}

// Apply Z projection and return the title of the resulting image
function applyZProjection(originalImageTitle) {
    selectWindow(originalImageTitle);
    run("Z Project...", "projection=[Max Intensity]");
    var projectedImageTitle = getTitle(); // Get the title of the new projected image
    return projectedImageTitle;
}

// Enhance contrast on an image
function enhanceContrastOnImage(imageTitle, saturationLevel) {
    selectWindow(imageTitle);
    run("Enhance Contrast...", "saturated=" + saturationLevel + " normalize equalize");
}

// Save all processed images as PNG files
function saveProcessedImagesAsPNG(output, fileName, mipTitle, colorCodedImageTitle) {
    var baseName = File.getNameWithoutExtension(fileName);
    var processedDir = output + File.separator + baseName;
    if (!File.exists(processedDir)) {
        File.makeDirectory(processedDir);
    }
    saveImage(mipTitle, processedDir, baseName + "_MAX.png");
    if (saveColorCoded) {
        saveImage(colorCodedImageTitle, processedDir, baseName + "_Color.png");
        if (isWindowOpen("MAX_colored")) {
            saveImage("MAX_colored", processedDir, baseName + "_MAX_colored.png");
        }
        if (isWindowOpen("color time scale")) {
            saveImage("color time scale", processedDir, baseName + "_ColorBar.png");
        }
    }
}

// Save a specific image with the given name
function saveImage(imageTitle, dir, fileName) {
    if (!isWindowOpen(imageTitle)) {
        print("Warning: Cannot save " + imageTitle + ", window not found.");
        return;
    }
    selectWindow(imageTitle);
    saveAs("PNG", dir + File.separator + fileName);
    close();
}

// Check if a window with the specified title is open
function isWindowOpen(title) {
    var windowList = getList("image.titles");
    for (var i = 0; i < windowList.length; i++) {
        if (windowList[i] == title) {
            return true;
        }
    }
    return false;
}

// Close all open images
function closeAllImages() {
    while (nImages > 0) {
        selectImage(nImages);
        close();
    }
}

// Show progress in the status bar with time estimation
function updateProgress(current, total) {
    if (current <= 0) return; // Avoid division by zero
    
    var progress = current / total;
    var elapsed = (getTime() - startTime) / 1000; // in seconds
    var estimatedTotalTime = (elapsed / current) * total;
    var remainingTime = estimatedTotalTime - elapsed;
    var remainingMinutes = floor(remainingTime / 60);
    var remainingSeconds = floor(remainingTime % 60);
    showStatus("Processing... " + current + " of " + total + " files completed. Estimated time left: " + remainingMinutes + "m " + remainingSeconds + "s");
    showProgress(progress);
}

// Function to generate a summary report
function generateReport() {
    var endTime = getTime();
    var totalTime = (endTime - startTime) / 1000; // in seconds
    var totalMinutes = floor(totalTime / 60);
    var totalSeconds = floor(totalTime % 60);
    print("\n--- Summary Report ---");
    print("Total files found: " + totalFiles);
    print("Processed files: " + processedFiles);
    print("Skipped files: " + skippedFiles);
    print("Files with errors: " + errorFiles);
    print("Total processing time: " + totalMinutes + " minutes " + totalSeconds + " seconds");
    
    // Save log to output directory
    saveLog(outputDir + File.separator + "ImageProcessingMaster_log.txt");
    
    if (errorFiles > 0) {
        print("\nErrors encountered:");
        print(errorLog);
    }
}

// Function to save the log to a file
function saveLog(logPath) {
    selectWindow("Log");
    saveAs("Text", logPath);
    print("Log saved to: " + logPath);
}

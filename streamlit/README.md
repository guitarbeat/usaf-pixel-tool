# USAF Target Analyzer

A comprehensive tool for analyzing USAF 1951 resolution targets in microscopy and imaging systems.

## Features

- **Modular Design**: Well-organized package structure with clear separation of concerns
- **Flexible Operation Modes**:
  - Interactive web interface with Streamlit
  - Headless mode for automated processing
  - Command-line interface for scripting
- **Enhanced Valley Detection**:
  - Robust algorithm with multiple fallback methods
  - Automatic parameter adjustment
  - Prominence-based filtering
- **Comprehensive Analysis**:
  - Contrast calculation
  - MTF (Modulation Transfer Function)
  - Resolution in line pairs/mm and microns
  - Signal-to-noise ratio
  - Quality metrics
- **Rich Visualization**:
  - Column average intensity plots
  - Interactive ROI selection
  - Zoomed preview of selected region
  - CSV export of results
- **Intuitive Interface**:
  - Click-and-drag ROI selection
  - Real-time visual feedback
  - Live preview of the profile line

## Installation

### Prerequisites

- Python 3.6 or higher
- Required packages (installed automatically):
  - streamlit
  - streamlit-image-coordinates
  - numpy
  - opencv-python
  - matplotlib
  - pandas
  - scipy

### Direct Installation

```bash
# Install from the repository
pip install -r requirements.txt
```

## Usage

### Streamlit Web Interface

The application provides a modern web interface powered by Streamlit with interactive image selection capabilities.

1. Install the required dependencies: `pip install -r requirements.txt`
2. Run the Streamlit app: `streamlit run usaf.py`
3. Your default web browser will open with the application interface
4. Upload your USAF target image using the file uploader
5. Use the interactive selection by clicking and dragging on the image to define your ROI
6. See a zoomed preview of your selection in real-time
7. Configure USAF parameters (group and element) and profile settings
8. Click "Analyze Image" to process the image
9. View the results, including resolution metrics and profile visualization
10. Download the results as a CSV file for further analysis

This interface is particularly useful for:

- Users who prefer a graphical interface over command-line tools
- Quick analysis without writing code
- Sharing the tool with non-technical colleagues
- Remote access when deployed on a server

### Profile Analysis

The tool analyzes the intensity profile across the selected ROI. For rectangular selections:

- **Horizontal Profile**: Displays a "column average plot", where the x-axis represents the horizontal distance through the selection and the y-axis represents the vertically averaged pixel intensity.
- **Vertical Profile**: Displays a "row average plot", where the x-axis represents the vertical distance through the selection and the y-axis represents the horizontally averaged pixel intensity.

This averaging approach reduces noise and provides more reliable measurements of contrast and resolution.

### As a Python Package

```python
from usaf_package.main import run_analysis

# Run analysis with custom parameters
results = run_analysis(
    image_path='path/to/image.tif',
    roi=(100, 100, 200, 200),
    profile_position=50,
    orientation='horizontal',
    group=2,
    element=3
)

# Process results
print(f"Contrast: {results['contrast']}")
print(f"Resolution: {results['resolution_microns']} µm")
```

## Package Structure

```mermaid
usaf.py                     # Main streamlit application
usaf_package/
├── __init__.py
├── main.py                  # Main analysis entry point
├── core/                    # Core functionality
│   ├── __init__.py
│   ├── config.py            # Configuration and constants
│   └── usaf_target.py       # USAF target definitions and calculations
├── processing/              # Image processing components
│   ├── __init__.py
│   ├── image_processor.py   # Image loading and processing
│   └── profile_analyzer.py  # Profile analysis algorithms
├── ui/                      # User interface components
│   ├── __init__.py
│   ├── interactive.py       # Interactive UI for GUI mode
│   └── output_manager.py    # Visualization and reporting
└── utils/                   # Utility functions
    ├── __init__.py
    ├── cli.py               # Command-line interface utilities
    ├── helpers.py           # Streamlit UI helper functions
    └── streamlit_helpers.py # Streamlit-specific utility functions (file upload, ROI, CSV export)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Based on the USAF 1951 resolution target standard
- Inspired by the need for better resolution and contrast analysis in microscopy
- Uses streamlit-image-coordinates for interactive image selection

## Recent Refactoring Improvements

The codebase has been refactored to increase modularity, reduce complexity, and improve maintainability:

### Architectural Improvements

1. **Separation of UI from Core Logic**:
   - Created a dedicated `streamlit_components.py` module for all Streamlit-specific UI components
   - Moved visualization code from the main app into appropriate UI modules
   - Isolated business logic from presentation logic

2. **Improved Code Organization**:
   - Extracted complex functionality into smaller, focused functions
   - Grouped related functionality into appropriate modules
   - Reduced duplication by centralizing common operations

3. **Enhanced Maintainability**:
   - Simplified the main app file from over 600 lines to under 100 lines
   - Added comprehensive docstrings throughout the codebase
   - Created a more hierarchical, logical package structure

### Technical Improvements

1. **Parameter Management**:
   - Centralized UI parameter handling
   - Created helper functions for parameter validation and processing
   - Improved parameter passing between components

2. **Image Processing Workflow**:
   - Streamlined the image loading and processing pipeline
   - Created dedicated helpers for ROI selection and manipulation
   - Improved error handling throughout the pipeline

3. **Session State Management**:
   - Created dedicated functions for managing Streamlit session state
   - Made state operations more consistent and predictable
   - Improved handling of application state transitions

### Usage Notes

No changes in functionality or user experience were introduced. All existing features continue to work as before, but the codebase is now more modular, easier to maintain, and better organized.

*Note: The previously unused `image_tools.py` has been removed as part of this cleanup. All Streamlit-specific helpers are now in `streamlit_helpers.py`.*

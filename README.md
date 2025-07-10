# Sentinel-2 LULC Classification System

## Overview

This is a Streamlit-based web application for processing Sentinel-2 satellite imagery and performing Land Use Land Cover (LULC) classification using machine learning algorithms. The system allows users to upload GeoTIFF files, preprocess satellite imagery, apply various classification algorithms, and visualize results with area calculations and interactive maps.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

- **Frontend**: Streamlit web interface for user interaction
- **Backend**: Python-based processing modules for image processing, ML classification, and visualization
- **Data Processing**: Rasterio for geospatial data handling and NumPy for numerical operations
- **Machine Learning**: Scikit-learn for classification algorithms
- **Visualization**: Plotly for interactive charts, Folium for maps, and Matplotlib for image display

## Key Components

### 1. Main Application (`app.py`)
- **Purpose**: Entry point and user interface
- **Technology**: Streamlit with custom CSS styling
- **Features**: File upload, parameter configuration, results display
- **Architecture Decision**: Streamlit was chosen for rapid prototyping and built-in web interface capabilities

### 2. Image Processor (`utils/image_processor.py`)
- **Purpose**: Handles Sentinel-2 imagery loading and preprocessing
- **Key Features**:
  - Multiple band combinations (RGB, False Color, SWIR)
  - Spatial resolution adjustment
  - Smoothing filters and cloud masking
- **Architecture Decision**: Rasterio used for robust geospatial data handling

### 3. ML Classifier (`utils/ml_classifier.py`)
- **Purpose**: Machine learning classification for LULC mapping
- **Algorithms**: Random Forest, K-Means clustering, SVM
- **Features**: 10 predefined land cover classes with standardized preprocessing
- **Architecture Decision**: Scikit-learn chosen for its comprehensive ML toolkit and ease of use

### 4. Area Calculator (`utils/area_calculator.py`)
- **Purpose**: Calculates area statistics for classified land cover maps
- **Features**: Per-class area calculations, percentage coverage, statistical summaries
- **Architecture Decision**: NumPy-based calculations for efficient processing of large raster datasets

### 5. Visualization (`utils/visualization.py`)
- **Purpose**: Creates visual outputs for imagery and classification results
- **Features**: RGB image creation, classification maps, interactive folium maps with proper geographic positioning
- **Architecture Decision**: Multiple visualization libraries used for different purposes (Matplotlib for static images, Folium for interactive maps)

### 6. Accuracy Assessment (`utils/accuracy_assessment.py`)
- **Purpose**: Evaluates classification accuracy using statistical metrics
- **Features**: Overall accuracy, Kappa coefficient, confusion matrix, producer's/user's accuracy, F1 scores
- **Architecture Decision**: Scikit-learn metrics with custom visualization using Plotly for interactive confusion matrices

## Data Flow

1. **Input**: User uploads GeoTIFF Sentinel-2 imagery file
2. **Preprocessing**: Image processor handles band selection, resampling, and filtering
3. **Feature Extraction**: Spectral indices and features extracted from satellite bands
4. **Classification**: ML classifier processes features to generate LULC map
5. **Post-processing**: Area calculator computes statistics and coverage metrics
6. **Visualization**: Results displayed through multiple visualization components
7. **Output**: Interactive maps, charts, and downloadable results

## External Dependencies

### Core Dependencies
- **Streamlit**: Web application framework
- **Rasterio**: Geospatial data I/O and processing
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **PIL (Pillow)**: Image processing
- **Plotly**: Interactive plotting
- **Folium**: Interactive maps
- **Matplotlib**: Static plotting
- **SciPy**: Scientific computing

### Rationale for Key Choices
- **Rasterio over GDAL**: Higher-level Python interface for geospatial operations
- **Streamlit over Flask/Django**: Rapid development for data science applications
- **Scikit-learn**: Comprehensive ML library with consistent API
- **Folium**: JavaScript-based interactive maps within Python environment

## Deployment Strategy

The application is designed for deployment on Replit with the following considerations:

1. **Environment**: Python 3.x with pip package management
2. **File Handling**: Temporary file processing for uploaded GeoTIFF files
3. **Memory Management**: Efficient processing for large satellite imagery datasets
4. **Scalability**: Modular design allows for easy extension of classification algorithms and visualization options

## Recent Changes

- July 03, 2025: Removed all export/download options from Classification Results and Area Analysis pages per user request
- July 03, 2025: Added accuracy assessment functionality with confusion matrix, overall accuracy, and Kappa coefficient
- July 03, 2025: Fixed map coordinates to show actual satellite image location instead of default location
- July 03, 2025: Fixed deprecation warnings for image display parameters
- July 03, 2025: Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.

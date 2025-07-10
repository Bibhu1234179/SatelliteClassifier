import streamlit as st
import rasterio
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
import io
import tempfile
import os
import matplotlib.pyplot as plt
from utils.image_processor import SentinelImageProcessor
from utils.ml_classifier import LULCClassifier
from utils.area_calculator import AreaCalculator
from utils.visualization import Visualizer
from utils.accuracy_assessment import AccuracyAssessment

# Page configuration
st.set_page_config(
    page_title="Sentinel-2 LULC Classification",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = None
if 'area_stats' not in st.session_state:
    st.session_state.area_stats = None
if 'accuracy_metrics' not in st.session_state:
    st.session_state.accuracy_metrics = None

def main():
    # Header
    st.markdown('<div class="main-header">🛰️ Sentinel-2 LULC Classification Tool</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Upload & Process", "Classification Results", "Area Analysis", "Accuracy Assessment", "Visualization"]
    )
    
    # Information box
    st.markdown("""
    <div class="info-box">
        <strong>About this tool:</strong><br>
        This application processes Sentinel-2 satellite imagery to generate Land Use Land Cover (LULC) classifications.
        Upload a GeoTIFF file, and the system will classify it into different land cover types and provide area analysis.
    </div>
    """, unsafe_allow_html=True)
    
    if page == "Upload & Process":
        upload_and_process_page()
    elif page == "Classification Results":
        classification_results_page()
    elif page == "Area Analysis":
        area_analysis_page()
    elif page == "Accuracy Assessment":
        accuracy_assessment_page()
    elif page == "Visualization":
        visualization_page()

def upload_and_process_page():
    st.markdown('<div class="section-header">📁 Upload Sentinel-2 Imagery</div>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a Sentinel-2 GeoTIFF file",
        type=['tif', 'tiff'],
        help="Upload a Sentinel-2 satellite image in GeoTIFF format"
    )
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Processing options
        st.markdown('<div class="section-header">⚙️ Processing Options</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            band_combination = st.selectbox(
                "Select Band Combination",
                ["RGB (4,3,2)", "False Color (8,4,3)", "SWIR (12,8,4)", "NDVI Enhanced (8,4,3)"],
                help="Choose the band combination for processing"
            )
        
        with col2:
            classification_method = st.selectbox(
                "Classification Method",
                ["Random Forest", "K-Means", "Support Vector Machine"],
                help="Select the machine learning algorithm for classification"
            )
        
        # Advanced options
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                n_classes = st.slider("Number of Classes", 3, 10, 6)
                spatial_resolution = st.number_input("Spatial Resolution (m)", 10, 60, 10)
            with col2:
                apply_smoothing = st.checkbox("Apply Smoothing Filter", value=True)
                cloud_mask = st.checkbox("Apply Cloud Masking", value=True)
        
        # Process button
        if st.button("🚀 Process Image", type="primary"):
            process_image(uploaded_file, band_combination, classification_method, n_classes, 
                         spatial_resolution, apply_smoothing, cloud_mask)

def process_image(uploaded_file, band_combination, classification_method, n_classes, 
                 spatial_resolution, apply_smoothing, cloud_mask):
    """Process the uploaded Sentinel-2 image"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Initialize processors
        image_processor = SentinelImageProcessor()
        classifier = LULCClassifier()
        area_calculator = AreaCalculator()
        
        # Step 1: Load and preprocess image
        status_text.text("Loading and preprocessing image...")
        progress_bar.progress(10)
        
        processed_data = image_processor.load_and_preprocess(
            tmp_file_path, band_combination, spatial_resolution, apply_smoothing, cloud_mask
        )
        
        # Step 2: Prepare data for classification
        status_text.text("Preparing data for classification...")
        progress_bar.progress(30)
        
        features = image_processor.extract_features(processed_data)
        
        # Step 3: Classify image
        status_text.text(f"Classifying image using {classification_method}...")
        progress_bar.progress(50)
        
        classification_result = classifier.classify(
            features, method=classification_method, n_classes=n_classes
        )
        
        # Step 4: Calculate areas
        status_text.text("Calculating area statistics...")
        progress_bar.progress(70)
        
        area_stats = area_calculator.calculate_areas(
            classification_result, processed_data['transform'], spatial_resolution
        )
        
        # Step 5: Perform accuracy assessment
        status_text.text("Performing accuracy assessment...")
        progress_bar.progress(80)
        
        accuracy_assessor = AccuracyAssessment()
        validation_data = accuracy_assessor.generate_reference_data(classification_result)
        accuracy_metrics = accuracy_assessor.calculate_accuracy_metrics(validation_data)
        
        # Step 6: Store results
        status_text.text("Finalizing results...")
        progress_bar.progress(90)
        
        st.session_state.processed_image = processed_data
        st.session_state.processed_data = processed_data
        st.session_state.classification_result = classification_result
        st.session_state.area_stats = area_stats
        st.session_state.accuracy_metrics = accuracy_metrics
        
        progress_bar.progress(100)
        status_text.text("Processing completed successfully!")
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        st.success("✅ Image processed successfully! Navigate to other tabs to view results.")
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        progress_bar.empty()
        status_text.empty()

def classification_results_page():
    st.markdown('<div class="section-header">🗺️ Classification Results</div>', unsafe_allow_html=True)
    
    if st.session_state.classification_result is None:
        st.warning("⚠️ No classification results available. Please process an image first.")
        return
    
    # Display original and classified images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        if st.session_state.processed_image is not None:
            visualizer = Visualizer()
            original_img = visualizer.create_rgb_image(st.session_state.processed_image)
            st.image(original_img, caption="Original Sentinel-2 Image", use_container_width=True)
    
    with col2:
        st.subheader("LULC Classification")
        if st.session_state.classification_result is not None:
            visualizer = Visualizer()
            classified_img = visualizer.create_classification_image(st.session_state.classification_result)
            st.image(classified_img, caption="LULC Classification Result", use_container_width=True)
    
    # Classification legend
    st.markdown('<div class="section-header">🏷️ Land Cover Classes</div>', unsafe_allow_html=True)
    
    if st.session_state.area_stats is not None:
        legend_df = pd.DataFrame({
            'Class': st.session_state.area_stats['class_names'],
            'Color': st.session_state.area_stats['colors'],
            'Description': st.session_state.area_stats['descriptions']
        })
        
        for idx, row in legend_df.iterrows():
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                <div style="width: 30px; height: 20px; background-color: {row['Color']}; 
                           border: 1px solid #ccc; margin-right: 10px;"></div>
                <strong>{row['Class']}</strong>: {row['Description']}
            </div>
            """, unsafe_allow_html=True)
    


def area_analysis_page():
    st.markdown('<div class="section-header">📊 Area Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.area_stats is None:
        st.warning("⚠️ No area statistics available. Please process an image first.")
        return
    
    # Display area statistics
    stats_df = pd.DataFrame(st.session_state.area_stats['statistics'])
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_area = stats_df['Area_km2'].sum()
        st.metric("Total Area", f"{total_area:.2f} km²")
    
    with col2:
        largest_class = stats_df.loc[stats_df['Area_km2'].idxmax(), 'Class']
        st.metric("Dominant Class", largest_class)
    
    with col3:
        n_classes = len(stats_df)
        st.metric("Number of Classes", n_classes)
    
    with col4:
        avg_area = stats_df['Area_km2'].mean()
        st.metric("Average Class Area", f"{avg_area:.2f} km²")
    
    # Detailed statistics table
    st.markdown('<div class="section-header">📋 Detailed Statistics</div>', unsafe_allow_html=True)
    
    # Format the dataframe for better display
    display_df = stats_df.copy()
    display_df['Area_km2'] = display_df['Area_km2'].round(2)
    display_df['Area_hectares'] = display_df['Area_hectares'].round(2)
    display_df['Percentage'] = display_df['Percentage'].round(2)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Area comparison charts
    st.markdown('<div class="section-header">📈 Area Comparison</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig_pie = px.pie(
            stats_df, 
            values='Area_km2', 
            names='Class',
            title='Land Cover Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart
        fig_bar = px.bar(
            stats_df, 
            x='Class', 
            y='Area_km2',
            title='Area by Land Cover Class',
            color='Class',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Horizontal bar chart for better readability
    fig_horizontal = px.bar(
        stats_df.sort_values('Area_km2', ascending=True), 
        x='Area_km2', 
        y='Class',
        title='Land Cover Classes by Area (km²)',
        orientation='h',
        color='Area_km2',
        color_continuous_scale='viridis'
    )
    st.plotly_chart(fig_horizontal, use_container_width=True)

def visualization_page():
    st.markdown('<div class="section-header">🗺️ Interactive Visualization</div>', unsafe_allow_html=True)
    
    if st.session_state.classification_result is None:
        st.warning("⚠️ No visualization data available. Please process an image first.")
        return
    
    # Interactive map
    st.markdown('<div class="section-header">🌍 Interactive Map</div>', unsafe_allow_html=True)
    
    try:
        visualizer = Visualizer()
        
        # Create Folium map
        if st.session_state.processed_image is not None:
            folium_map = visualizer.create_interactive_map(
                st.session_state.processed_image,
                st.session_state.classification_result,
                st.session_state.area_stats
            )
            
            # Display the map
            folium_static(folium_map, width=1200, height=600)
        
        # Classification overview
        st.markdown('<div class="section-header">📊 Classification Overview</div>', unsafe_allow_html=True)
        
        if st.session_state.area_stats is not None:
            stats_df = pd.DataFrame(st.session_state.area_stats['statistics'])
            
            # Create a comprehensive overview chart
            fig = go.Figure()
            
            # Add bar trace
            fig.add_trace(go.Bar(
                x=stats_df['Class'],
                y=stats_df['Area_km2'],
                name='Area (km²)',
                marker_color='skyblue',
                text=stats_df['Percentage'].round(1).astype(str) + '%',
                textposition='outside'
            ))
            
            fig.update_layout(
                title='Land Cover Classification Summary',
                xaxis_title='Land Cover Class',
                yaxis_title='Area (km²)',
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        

    
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")

def accuracy_assessment_page():
    st.markdown('<div class="section-header">📊 Accuracy Assessment</div>', unsafe_allow_html=True)
    
    if st.session_state.accuracy_metrics is None:
        st.warning("⚠️ No accuracy assessment available. Please process an image first.")
        return
    
    accuracy_metrics = st.session_state.accuracy_metrics
    
    # Overall metrics summary
    st.markdown('<div class="section-header">📈 Overall Accuracy Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        overall_acc = accuracy_metrics['overall_accuracy']
        st.metric("Overall Accuracy", f"{overall_acc:.1%}", f"{overall_acc:.3f}")
    
    with col2:
        kappa = accuracy_metrics['kappa_coefficient']
        st.metric("Kappa Coefficient", f"{kappa:.3f}")
    
    with col3:
        n_samples = accuracy_metrics['n_samples']
        st.metric("Validation Samples", f"{n_samples:,}")
    
    with col4:
        # Kappa interpretation
        if kappa < 0.20:
            kappa_level = "Slight"
        elif kappa < 0.40:
            kappa_level = "Fair"
        elif kappa < 0.60:
            kappa_level = "Moderate"
        elif kappa < 0.80:
            kappa_level = "Substantial"
        else:
            kappa_level = "Almost Perfect"
        st.metric("Agreement Level", kappa_level)
    
    # Accuracy interpretation
    if overall_acc >= 0.85:
        st.success("✅ Excellent classification accuracy achieved!")
    elif overall_acc >= 0.70:
        st.info("ℹ️ Good classification accuracy. Consider improvements for better results.")
    else:
        st.warning("⚠️ Classification accuracy needs improvement. Review parameters and training data.")
    
    # Confusion Matrix
    st.markdown('<div class="section-header">🔢 Confusion Matrix</div>', unsafe_allow_html=True)
    
    accuracy_assessor = AccuracyAssessment()
    
    # Create and display confusion matrix
    cm_fig = accuracy_assessor.create_confusion_matrix_plot(accuracy_metrics)
    st.plotly_chart(cm_fig, use_container_width=True)
    
    # Class-wise accuracy metrics
    st.markdown('<div class="section-header">📋 Class-wise Accuracy</div>', unsafe_allow_html=True)
    
    # Create accuracy summary plot
    accuracy_fig = accuracy_assessor.create_accuracy_summary_plot(accuracy_metrics)
    st.plotly_chart(accuracy_fig, use_container_width=True)
    
    # Detailed accuracy table
    st.markdown('<div class="section-header">📊 Detailed Accuracy Table</div>', unsafe_allow_html=True)
    
    # Create accuracy dataframe
    accuracy_df = pd.DataFrame({
        'Class': accuracy_metrics['class_names'],
        'Producer\'s Accuracy': [f"{accuracy_metrics['producers_accuracy'][cls]:.3f}" for cls in accuracy_metrics['class_labels']],
        'User\'s Accuracy': [f"{accuracy_metrics['users_accuracy'][cls]:.3f}" for cls in accuracy_metrics['class_labels']],
        'F1 Score': [f"{accuracy_metrics['f1_scores'][cls]:.3f}" for cls in accuracy_metrics['class_labels']]
    })
    
    st.dataframe(accuracy_df, use_container_width=True)
    
    # Accuracy report
    st.markdown('<div class="section-header">📄 Accuracy Assessment Report</div>', unsafe_allow_html=True)
    
    with st.expander("View Detailed Report"):
        report = accuracy_assessor.create_accuracy_report(accuracy_metrics)
        st.text(report)
    
    # Export options
    st.markdown('<div class="section-header">💾 Export Accuracy Results</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Accuracy Table"):
            csv_data = accuracy_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="accuracy_assessment.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📋 Export Full Report"):
            report = accuracy_assessor.create_accuracy_report(accuracy_metrics)
            st.download_button(
                label="Download Report",
                data=report,
                file_name="accuracy_report.txt",
                mime="text/plain"
            )
    
    with col3:
        if st.button("🔢 Export Confusion Matrix"):
            cm_df = pd.DataFrame(
                accuracy_metrics['confusion_matrix'],
                index=accuracy_metrics['class_names'],
                columns=accuracy_metrics['class_names']
            )
            csv_data = cm_df.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="confusion_matrix.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()

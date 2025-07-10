import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from PIL import Image
import folium
from folium import plugins
import rasterio
from rasterio.plot import show
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
import io
import base64

class Visualizer:
    """
    Handles visualization of satellite imagery and classification results
    """
    
    def __init__(self):
        self.class_colors = {
            0: "#0077BE",  # Water - Blue
            1: "#228B22",  # Forest - Dark Green
            2: "#FFFF00",  # Agriculture - Yellow
            3: "#FF0000",  # Urban - Red
            4: "#D2691E",  # Bare Land - Brown
            5: "#90EE90",  # Grassland - Light Green
            6: "#4169E1",  # Wetland - Royal Blue
            7: "#FFFFFF",  # Snow/Ice - White
            8: "#C0C0C0",  # Cloud - Light Gray
            9: "#696969"   # Shadow - Dark Gray
        }
        
        self.class_names = {
            0: "Water",
            1: "Forest",
            2: "Agriculture", 
            3: "Urban",
            4: "Bare Land",
            5: "Grassland",
            6: "Wetland",
            7: "Snow/Ice",
            8: "Cloud",
            9: "Shadow"
        }
    
    def create_rgb_image(self, processed_data, enhance=True):
        """
        Create RGB image from processed satellite data
        
        Args:
            processed_data: Dictionary containing processed image data
            enhance: Whether to enhance contrast
            
        Returns:
            PIL.Image: RGB image
        """
        image = processed_data['image']
        
        # Use first 3 bands for RGB
        rgb_bands = image[:3]
        
        # Handle NaN values
        rgb_bands = np.nan_to_num(rgb_bands, nan=0.0)
        
        # Enhance contrast if requested
        if enhance:
            rgb_bands = self._enhance_contrast(rgb_bands)
        
        # Ensure values are in 0-1 range
        rgb_bands = np.clip(rgb_bands, 0, 1)
        
        # Convert to 0-255 range and proper shape for PIL
        rgb_image = (rgb_bands * 255).astype(np.uint8)
        rgb_image = np.transpose(rgb_image, (1, 2, 0))  # Change from (bands, height, width) to (height, width, bands)
        
        # Create PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        return pil_image
    
    def create_classification_image(self, classification_result):
        """
        Create colored classification image
        
        Args:
            classification_result: Dictionary containing classification results
            
        Returns:
            PIL.Image: Colored classification image
        """
        classification_map = classification_result['classification_map']
        shape = classification_result['shape']
        
        # Create RGB image
        rgb_image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        
        # Get unique classes
        unique_classes = np.unique(classification_map)
        valid_classes = unique_classes[unique_classes >= 0]
        
        # Color each class
        for class_id in valid_classes:
            mask = classification_map == class_id
            color = self.class_colors.get(class_id, "#808080")
            
            # Convert hex color to RGB
            rgb_color = self._hex_to_rgb(color)
            
            rgb_image[mask] = rgb_color
        
        # Create PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        return pil_image
    
    def create_interactive_map(self, processed_data, classification_result, area_stats):
        """
        Create interactive Folium map with classification overlay
        
        Args:
            processed_data: Dictionary containing processed image data
            classification_result: Dictionary containing classification results
            area_stats: Area statistics dictionary
            
        Returns:
            folium.Map: Interactive map
        """
        # Extract geographic center from the processed data
        transform = processed_data['transform']
        metadata = processed_data['metadata']
        
        # Calculate center coordinates from bounds
        bounds = metadata['bounds']
        
        # Extract coordinates safely
        try:
            left = bounds.left
            bottom = bounds.bottom
            right = bounds.right
            top = bounds.top
        except AttributeError:
            # Fallback to array-like access
            left, bottom, right, top = bounds[0], bounds[1], bounds[2], bounds[3]
        
        center_lat = (bottom + top) / 2
        center_lon = (left + right) / 2
        
        # Convert to lat/lon if needed
        try:
            if processed_data['crs'] and not processed_data['crs'].is_geographic:
                # Transform bounds to WGS84 (geographic coordinates)
                geographic_bounds = transform_bounds(
                    processed_data['crs'], 
                    CRS.from_epsg(4326),  # WGS84
                    left, bottom, right, top
                )
                # Update center coordinates with transformed values
                center_lat = (geographic_bounds[1] + geographic_bounds[3]) / 2
                center_lon = (geographic_bounds[0] + geographic_bounds[2]) / 2
                # Update bounds values
                left, bottom, right, top = geographic_bounds
        except Exception as e:
            print(f"Could not transform coordinates: {e}")
            # Use original bounds for fallback
        
        # Calculate appropriate zoom level based on image extent
        lat_diff = abs(top - bottom)
        lon_diff = abs(right - left)
        
        # Rough zoom level calculation
        if max(lat_diff, lon_diff) > 1:
            zoom_level = 8
        elif max(lat_diff, lon_diff) > 0.1:
            zoom_level = 10
        elif max(lat_diff, lon_diff) > 0.01:
            zoom_level = 12
        else:
            zoom_level = 14
        
        # Create base map centered on the actual data
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_level,
            tiles='OpenStreetMap'
        )
        
        # Add satellite tile layer
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Create classification overlay (simplified)
        classification_image = self.create_classification_image(classification_result)
        
        # Convert PIL image to base64 for embedding
        buffer = io.BytesIO()
        classification_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Add image overlay (this is simplified - real implementation would handle geospatial coordinates)
        # For demonstration, we'll add a marker with popup information
        
        # Skip legend for now to avoid compatibility issues
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add statistics popup at the center of the actual data
        stats_html = self._create_stats_popup_html(area_stats)
        folium.Marker(
            location=[center_lat, center_lon],
            popup=folium.Popup(stats_html, max_width=400),
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
        
        # Add image bounds as a rectangle overlay
        try:
            bounds_coords = [
                [bottom, left],  # Southwest corner
                [top, right]     # Northeast corner
            ]
            
            folium.Rectangle(
                bounds=bounds_coords,
                popup=f"Sentinel-2 Image Bounds<br>Area: {area_stats['summary']['total_area_km2']:.2f} km²",
                color='red',
                weight=2,
                fill=False
            ).add_to(m)
        except Exception as e:
            print(f"Could not add bounds rectangle: {e}")
        
        return m
    
    def _enhance_contrast(self, image):
        """Enhance image contrast using histogram stretching"""
        enhanced = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            band = image[i]
            
            # Calculate percentiles for stretching
            p2, p98 = np.percentile(band[band > 0], (2, 98))
            
            # Stretch contrast
            enhanced[i] = np.clip((band - p2) / (p98 - p2), 0, 1)
        
        return enhanced
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _create_legend_html(self, area_stats):
        """Create HTML legend for the map"""
        legend_html = '''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 200px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px;">
        <h4>Land Cover Classes</h4>
        '''
        
        for i, (name, color) in enumerate(zip(area_stats['class_names'], area_stats['colors'])):
            legend_html += f'''
            <div style="margin: 5px 0;">
                <span style="display: inline-block; width: 20px; height: 15px; 
                           background-color: {color}; border: 1px solid black; margin-right: 5px;"></span>
                <span>{name}</span>
            </div>
            '''
        
        legend_html += '</div>'
        return legend_html
    
    def _create_stats_popup_html(self, area_stats):
        """Create HTML popup with statistics"""
        stats_html = '<h4>Area Statistics</h4><table style="width:100%">'
        stats_html += '<tr><th>Class</th><th>Area (km²)</th><th>%</th></tr>'
        
        for stat in area_stats['statistics'][:5]:  # Show top 5 classes
            stats_html += f'''
            <tr>
                <td>{stat['Class']}</td>
                <td>{stat['Area_km2']:.2f}</td>
                <td>{stat['Percentage']:.1f}%</td>
            </tr>
            '''
        
        stats_html += '</table>'
        return stats_html
    
    def create_comparison_plot(self, original_image, classified_image, area_stats):
        """
        Create side-by-side comparison plot
        
        Args:
            original_image: Original satellite image
            classified_image: Classified image
            area_stats: Area statistics
            
        Returns:
            matplotlib.figure.Figure: Comparison plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Sentinel-2 Image')
        axes[0].axis('off')
        
        # Classified image
        axes[1].imshow(classified_image)
        axes[1].set_title('LULC Classification')
        axes[1].axis('off')
        
        # Add legend
        legend_elements = []
        for name, color in zip(area_stats['class_names'], area_stats['colors']):
            legend_elements.append(Rectangle((0,0),1,1, facecolor=color, label=name))
        
        axes[1].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        return fig
    
    def create_downloadable_map(self, classification_result, area_stats, processed_data):
        """
        Create a high-quality downloadable map with legend
        
        Args:
            classification_result: Dictionary containing classification results
            area_stats: Area statistics dictionary
            processed_data: Dictionary containing processed image data
            
        Returns:
            matplotlib.figure.Figure: High-quality map with legend
        """
        # Create figure with specific size for good quality
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Create classification image
        classified_array = classification_result['classified_image']
        class_labels = classification_result['class_labels']
        
        # Create color map for visualization
        colors = area_stats['colors']
        from matplotlib.colors import ListedColormap
        # Convert hex colors to RGB tuples for matplotlib
        rgb_colors = []
        for color in colors:
            hex_color = color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))
            rgb_colors.append(rgb)
        cmap = ListedColormap(rgb_colors)
        
        # Display the classification
        im = ax.imshow(classified_array, cmap=cmap, vmin=0, vmax=len(colors)-1)
        
        # Set title
        ax.set_title('Land Use Land Cover Classification', fontsize=16, fontweight='bold', pad=20)
        
        # Remove axis ticks but keep the frame
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Create legend with area statistics
        legend_elements = []
        for i, (name, color, stat) in enumerate(zip(area_stats['class_names'], colors, area_stats['statistics'])):
            area_km2 = stat['Area_km2']
            percentage = stat['Percentage']
            label = f'{name}\n{area_km2:.1f} km² ({percentage:.1f}%)'
            legend_elements.append(Rectangle((0,0),1,1, facecolor=color, label=label))
        
        # Position legend outside the plot area
        legend = ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                          fontsize=10, title='Land Cover Classes', title_fontsize=12)
        legend.get_title().set_fontweight('bold')
        
        # Add metadata text
        total_area = sum(stat['Area_km2'] for stat in area_stats['statistics'])
        method = classification_result.get('method', 'Unknown')
        
        metadata_text = f'Classification Method: {method}\nTotal Area: {total_area:.2f} km²\nSpatial Resolution: {processed_data.get("spatial_resolution", "Unknown")} m'
        ax.text(0.02, 0.02, metadata_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout to accommodate legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        
        return fig
    
    def save_map_as_image(self, classification_result, area_stats, processed_data, filename='lulc_classification_map.png', dpi=300):
        """
        Save classification map as high-quality image file
        
        Args:
            classification_result: Dictionary containing classification results
            area_stats: Area statistics dictionary  
            processed_data: Dictionary containing processed image data
            filename: Output filename
            dpi: Resolution for output image
            
        Returns:
            str: Path to saved file
        """
        fig = self.create_downloadable_map(classification_result, area_stats, processed_data)
        fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)  # Free memory
        return filename
    
    def create_area_chart(self, area_stats):
        """
        Create area statistics chart
        
        Args:
            area_stats: Area statistics dictionary
            
        Returns:
            matplotlib.figure.Figure: Area chart
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        areas = [stat['Area_km2'] for stat in area_stats['statistics']]
        labels = [stat['Class'] for stat in area_stats['statistics']]
        colors = [stat['Color'] for stat in area_stats['statistics']]
        
        ax1.pie(areas, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Land Cover Distribution')
        
        # Bar chart
        ax2.bar(labels, areas, color=colors)
        ax2.set_title('Area by Land Cover Class')
        ax2.set_ylabel('Area (km²)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig

import numpy as np
from rasterio.transform import from_bounds
import pandas as pd

class AreaCalculator:
    """
    Calculates area statistics for classified land cover maps
    """
    
    def __init__(self):
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
        
        self.class_descriptions = {
            0: "Water bodies including rivers, lakes, and ponds",
            1: "Dense forest and tree cover areas",
            2: "Agricultural lands and croplands",
            3: "Urban areas and built-up land",
            4: "Bare soil and exposed land",
            5: "Grasslands and pastures",
            6: "Wetlands and marshy areas",
            7: "Snow and ice covered areas",
            8: "Cloud cover",
            9: "Shadow areas"
        }
    
    def calculate_areas(self, classification_result, transform, spatial_resolution):
        """
        Calculate area statistics for each land cover class
        
        Args:
            classification_result: Dictionary containing classification results
            transform: Rasterio transform object
            spatial_resolution: Spatial resolution in meters
            
        Returns:
            dict: Area statistics and metadata
        """
        classification_map = classification_result['classification_map']
        shape = classification_result['shape']
        
        # Calculate pixel area in square meters
        pixel_area_m2 = spatial_resolution * spatial_resolution
        
        # Get unique classes (excluding invalid pixels marked as -1)
        unique_classes = np.unique(classification_map)
        valid_classes = unique_classes[unique_classes >= 0]
        
        # Calculate statistics for each class
        statistics = []
        
        for class_id in valid_classes:
            # Count pixels for this class
            pixel_count = np.sum(classification_map == class_id)
            
            # Calculate areas
            area_m2 = pixel_count * pixel_area_m2
            area_km2 = area_m2 / 1_000_000  # Convert to km²
            area_hectares = area_m2 / 10_000  # Convert to hectares
            
            # Calculate percentage
            total_valid_pixels = np.sum(classification_map >= 0)
            percentage = (pixel_count / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0
            
            # Get class information
            class_name = self.class_names.get(class_id, f"Class {class_id}")
            class_color = self.class_colors.get(class_id, "#808080")
            class_description = self.class_descriptions.get(class_id, f"Land cover class {class_id}")
            
            statistics.append({
                'Class_ID': class_id,
                'Class': class_name,
                'Pixel_Count': pixel_count,
                'Area_m2': area_m2,
                'Area_km2': area_km2,
                'Area_hectares': area_hectares,
                'Percentage': percentage,
                'Color': class_color,
                'Description': class_description
            })
        
        # Sort by area (descending)
        statistics.sort(key=lambda x: x['Area_km2'], reverse=True)
        
        # Calculate total statistics
        total_area_m2 = sum(stat['Area_m2'] for stat in statistics)
        total_area_km2 = total_area_m2 / 1_000_000
        total_area_hectares = total_area_m2 / 10_000
        
        # Create summary
        summary = {
            'total_area_m2': total_area_m2,
            'total_area_km2': total_area_km2,
            'total_area_hectares': total_area_hectares,
            'n_classes': len(valid_classes),
            'spatial_resolution': spatial_resolution,
            'image_dimensions': shape
        }
        
        # Extract arrays for easier access
        class_names = [stat['Class'] for stat in statistics]
        class_colors = [stat['Color'] for stat in statistics]
        class_descriptions = [stat['Description'] for stat in statistics]
        
        return {
            'statistics': statistics,
            'summary': summary,
            'class_names': class_names,
            'colors': class_colors,
            'descriptions': class_descriptions
        }
    
    def calculate_change_analysis(self, classification_result_1, classification_result_2, 
                                transform, spatial_resolution):
        """
        Calculate change analysis between two classification results
        
        Args:
            classification_result_1: First classification result
            classification_result_2: Second classification result
            transform: Rasterio transform object
            spatial_resolution: Spatial resolution in meters
            
        Returns:
            dict: Change analysis results
        """
        map1 = classification_result_1['classification_map']
        map2 = classification_result_2['classification_map']
        
        # Ensure both maps have the same shape
        if map1.shape != map2.shape:
            raise ValueError("Classification maps must have the same dimensions")
        
        # Calculate pixel area
        pixel_area_m2 = spatial_resolution * spatial_resolution
        
        # Find valid pixels in both maps
        valid_mask = (map1 >= 0) & (map2 >= 0)
        
        # Calculate change matrix
        unique_classes_1 = np.unique(map1[valid_mask])
        unique_classes_2 = np.unique(map2[valid_mask])
        all_classes = np.unique(np.concatenate([unique_classes_1, unique_classes_2]))
        
        change_matrix = np.zeros((len(all_classes), len(all_classes)), dtype=int)
        
        for i, class1 in enumerate(all_classes):
            for j, class2 in enumerate(all_classes):
                change_count = np.sum((map1 == class1) & (map2 == class2) & valid_mask)
                change_matrix[i, j] = change_count
        
        # Calculate change statistics
        change_stats = []
        for i, class1 in enumerate(all_classes):
            for j, class2 in enumerate(all_classes):
                if i != j and change_matrix[i, j] > 0:  # Only changes, not stable areas
                    pixel_count = change_matrix[i, j]
                    area_m2 = pixel_count * pixel_area_m2
                    area_km2 = area_m2 / 1_000_000
                    
                    class1_name = self.class_names.get(class1, f"Class {class1}")
                    class2_name = self.class_names.get(class2, f"Class {class2}")
                    
                    change_stats.append({
                        'From_Class': class1_name,
                        'To_Class': class2_name,
                        'From_Class_ID': class1,
                        'To_Class_ID': class2,
                        'Pixel_Count': pixel_count,
                        'Area_m2': area_m2,
                        'Area_km2': area_km2
                    })
        
        # Sort by area (descending)
        change_stats.sort(key=lambda x: x['Area_km2'], reverse=True)
        
        return {
            'change_matrix': change_matrix,
            'change_statistics': change_stats,
            'class_labels': [self.class_names.get(c, f"Class {c}") for c in all_classes]
        }
    
    def export_statistics_to_csv(self, area_stats, filename):
        """
        Export area statistics to CSV file
        
        Args:
            area_stats: Area statistics dictionary
            filename: Output filename
        """
        df = pd.DataFrame(area_stats['statistics'])
        df.to_csv(filename, index=False)
        return filename
    
    def create_area_report(self, area_stats):
        """
        Create a comprehensive area analysis report
        
        Args:
            area_stats: Area statistics dictionary
            
        Returns:
            str: Formatted report text
        """
        statistics = area_stats['statistics']
        summary = area_stats['summary']
        
        report = []
        report.append("LAND USE LAND COVER CLASSIFICATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Summary section
        report.append("SUMMARY")
        report.append("-" * 20)
        report.append(f"Total Area: {summary['total_area_km2']:.2f} km²")
        report.append(f"Total Area: {summary['total_area_hectares']:.2f} hectares")
        report.append(f"Number of Classes: {summary['n_classes']}")
        report.append(f"Spatial Resolution: {summary['spatial_resolution']} m")
        report.append(f"Image Dimensions: {summary['image_dimensions'][0]} x {summary['image_dimensions'][1]} pixels")
        report.append("")
        
        # Detailed statistics
        report.append("DETAILED STATISTICS")
        report.append("-" * 20)
        
        for stat in statistics:
            report.append(f"\n{stat['Class']}:")
            report.append(f"  Area: {stat['Area_km2']:.2f} km² ({stat['Area_hectares']:.2f} hectares)")
            report.append(f"  Percentage: {stat['Percentage']:.1f}%")
            report.append(f"  Pixel Count: {stat['Pixel_Count']:,}")
            report.append(f"  Description: {stat['Description']}")
        
        return "\n".join(report)

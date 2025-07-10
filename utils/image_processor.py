import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings('ignore')

class SentinelImageProcessor:
    """
    Handles Sentinel-2 satellite imagery processing including loading,
    preprocessing, and feature extraction.
    """
    
    def __init__(self):
        self.band_combinations = {
            "RGB (4,3,2)": [4, 3, 2],  # True color
            "False Color (8,4,3)": [8, 4, 3],  # False color infrared
            "SWIR (12,8,4)": [12, 8, 4],  # Short wave infrared
            "NDVI Enhanced (8,4,3)": [8, 4, 3]  # Enhanced with NDVI
        }
        
        self.sentinel_bands = {
            1: "Coastal aerosol",
            2: "Blue",
            3: "Green", 
            4: "Red",
            5: "Red edge 1",
            6: "Red edge 2",
            7: "Red edge 3",
            8: "NIR",
            8: "NIR narrow",
            9: "Water vapour",
            10: "SWIR - Cirrus",
            11: "SWIR 1",
            12: "SWIR 2"
        }
    
    def load_and_preprocess(self, file_path, band_combination, spatial_resolution, 
                          apply_smoothing, cloud_mask):
        """
        Load and preprocess Sentinel-2 imagery
        
        Args:
            file_path: Path to the GeoTIFF file
            band_combination: Selected band combination
            spatial_resolution: Target spatial resolution in meters
            apply_smoothing: Whether to apply smoothing filter
            cloud_mask: Whether to apply cloud masking
            
        Returns:
            dict: Processed image data with bands, transform, and metadata
        """
        try:
            with rasterio.open(file_path) as src:
                # Read metadata
                metadata = {
                    'crs': src.crs,
                    'transform': src.transform,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'bounds': src.bounds
                }
                
                # Determine available bands
                available_bands = src.count
                
                # Get band indices for the selected combination
                if band_combination in self.band_combinations:
                    band_indices = self.band_combinations[band_combination]
                else:
                    # Default to first 3 bands if combination not found
                    band_indices = [1, 2, 3]
                
                # Ensure band indices are within available range
                band_indices = [min(b, available_bands) for b in band_indices]
                
                # Read the selected bands
                bands_data = []
                for band_idx in band_indices:
                    band_data = src.read(band_idx)
                    
                    # Handle nodata values
                    if src.nodata is not None:
                        band_data = np.where(band_data == src.nodata, np.nan, band_data)
                    
                    # Apply smoothing if requested
                    if apply_smoothing:
                        band_data = median_filter(band_data, size=3)
                    
                    bands_data.append(band_data)
                
                # Stack bands
                image_array = np.stack(bands_data, axis=0)
                
                # Apply cloud masking (simple approach using high reflectance values)
                if cloud_mask:
                    image_array = self._apply_cloud_mask(image_array)
                
                # Calculate additional indices if needed
                if available_bands >= 8:  # Ensure we have NIR and Red bands
                    try:
                        # Read NIR (band 8) and Red (band 4) for NDVI
                        nir = src.read(min(8, available_bands))
                        red = src.read(min(4, available_bands))
                        
                        # Calculate NDVI
                        ndvi = self._calculate_ndvi(nir, red)
                        
                        # Add NDVI as additional band
                        image_array = np.vstack([image_array, ndvi[np.newaxis, :, :]])
                        
                    except Exception as e:
                        print(f"Could not calculate NDVI: {e}")
                
                # Normalize pixel values (0-1 range)
                image_array = self._normalize_image(image_array)
                
                processed_data = {
                    'image': image_array,
                    'transform': src.transform,
                    'crs': src.crs,
                    'metadata': metadata,
                    'band_names': [f"Band_{i}" for i in band_indices] + 
                                 (["NDVI"] if available_bands >= 8 else [])
                }
                
                return processed_data
                
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")
    
    def _apply_cloud_mask(self, image_array):
        """Apply simple cloud masking based on high reflectance values"""
        # Simple cloud detection using high reflectance in visible bands
        if image_array.shape[0] >= 3:
            # Calculate brightness
            brightness = np.mean(image_array[:3], axis=0)
            
            # Create cloud mask (pixels with high brightness)
            cloud_threshold = np.percentile(brightness[~np.isnan(brightness)], 95)
            cloud_mask = brightness > cloud_threshold
            
            # Apply mask to all bands
            image_array[:, cloud_mask] = np.nan
        
        return image_array
    
    def _calculate_ndvi(self, nir, red):
        """Calculate Normalized Difference Vegetation Index"""
        # Handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red)
            ndvi[np.isnan(ndvi)] = 0
            ndvi[np.isinf(ndvi)] = 0
            
        return ndvi
    
    def _normalize_image(self, image_array):
        """Normalize image values to 0-1 range"""
        normalized = np.zeros_like(image_array, dtype=np.float32)
        
        for i in range(image_array.shape[0]):
            band = image_array[i]
            
            # Remove NaN and infinite values for percentile calculation
            valid_pixels = band[~np.isnan(band) & ~np.isinf(band)]
            
            if len(valid_pixels) > 0:
                # Use percentile-based normalization to handle outliers
                p2, p98 = np.percentile(valid_pixels, (2, 98))
                
                # Normalize to 0-1 range
                normalized[i] = np.clip((band - p2) / (p98 - p2), 0, 1)
            else:
                normalized[i] = band
        
        return normalized
    
    def extract_features(self, processed_data):
        """
        Extract features from processed image for classification
        
        Args:
            processed_data: Dictionary containing processed image data
            
        Returns:
            numpy.ndarray: Feature array for classification
        """
        image = processed_data['image']
        height, width = image.shape[1], image.shape[2]
        n_bands = image.shape[0]
        
        # Reshape image for classification (pixels x bands)
        features = image.reshape(n_bands, height * width).T
        
        # Remove pixels with NaN values
        valid_pixels = ~np.isnan(features).any(axis=1)
        features = features[valid_pixels]
        
        # Add texture features if possible
        if n_bands >= 3:
            # Calculate basic texture measures
            gray = np.mean(image[:3], axis=0)
            texture_features = self._calculate_texture_features(gray)
            
            # Reshape texture features
            texture_flat = texture_features.reshape(-1, texture_features.shape[-1])
            texture_valid = texture_flat[valid_pixels]
            
            # Combine spectral and texture features
            features = np.hstack([features, texture_valid])
        
        return {
            'features': features,
            'valid_pixels': valid_pixels,
            'shape': (height, width),
            'n_bands': n_bands
        }
    
    def _calculate_texture_features(self, gray_image):
        """Calculate basic texture features from grayscale image"""
        from scipy.ndimage import generic_filter
        
        # Local standard deviation (texture measure)
        def local_std(x):
            return np.std(x)
        
        # Calculate local standard deviation as texture measure
        texture = generic_filter(gray_image, local_std, size=3)
        
        # Add gradient magnitude
        dy, dx = np.gradient(gray_image)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        
        # Stack texture features
        texture_features = np.stack([texture, gradient_magnitude], axis=-1)
        
        return texture_features

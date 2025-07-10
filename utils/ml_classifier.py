import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class LULCClassifier:
    """
    Machine Learning classifier for Land Use Land Cover classification
    """
    
    def __init__(self):
        self.classifier = None
        self.scaler = StandardScaler()
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
    
    def classify(self, feature_data, method="Random Forest", n_classes=6):
        """
        Classify land cover using the specified method
        
        Args:
            feature_data: Dictionary containing features and metadata
            method: Classification method to use
            n_classes: Number of classes for unsupervised methods
            
        Returns:
            dict: Classification results with labels and metadata
        """
        features = feature_data['features']
        valid_pixels = feature_data['valid_pixels']
        shape = feature_data['shape']
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Classify based on selected method
        if method == "Random Forest":
            labels = self._classify_random_forest(features_scaled, n_classes)
        elif method == "K-Means":
            labels = self._classify_kmeans(features_scaled, n_classes)
        elif method == "Support Vector Machine":
            labels = self._classify_svm(features_scaled, n_classes)
        else:
            raise ValueError(f"Unknown classification method: {method}")
        
        # Create full classification map
        classification_map = np.full(shape, -1, dtype=np.int32)
        classification_map.flat[valid_pixels] = labels
        
        # Get class information
        unique_labels = np.unique(labels)
        used_classes = unique_labels[unique_labels >= 0]
        
        class_info = {
            'names': [self.class_names.get(i, f"Class {i}") for i in used_classes],
            'colors': [self.class_colors.get(i, f"#{i*40:06x}") for i in used_classes],
            'descriptions': [self.class_descriptions.get(i, f"Land cover class {i}") for i in used_classes]
        }
        
        return {
            'classification_map': classification_map,
            'labels': labels,
            'valid_pixels': valid_pixels,
            'shape': shape,
            'n_classes': len(used_classes),
            'class_info': class_info,
            'method': method
        }
    
    def _classify_random_forest(self, features, n_classes):
        """Classify using Random Forest with pseudo-supervised approach"""
        # Create pseudo-labels using K-means first
        kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
        pseudo_labels = kmeans.fit_predict(features)
        
        # Train Random Forest on pseudo-labels
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Use a subset for training to speed up processing
        if len(features) > 50000:
            indices = np.random.choice(len(features), 50000, replace=False)
            X_train = features[indices]
            y_train = pseudo_labels[indices]
        else:
            X_train = features
            y_train = pseudo_labels
        
        rf.fit(X_train, y_train)
        
        # Predict on all features
        labels = rf.predict(features)
        
        # Post-process labels to ensure they follow land cover logic
        labels = self._postprocess_labels(labels, features)
        
        self.classifier = rf
        return labels
    
    def _classify_kmeans(self, features, n_classes):
        """Classify using K-means clustering"""
        kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # Post-process labels
        labels = self._postprocess_labels(labels, features)
        
        self.classifier = kmeans
        return labels
    
    def _classify_svm(self, features, n_classes):
        """Classify using Support Vector Machine with pseudo-supervised approach"""
        # Create pseudo-labels using K-means first
        kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
        pseudo_labels = kmeans.fit_predict(features)
        
        # Use a smaller subset for SVM training due to computational complexity
        if len(features) > 10000:
            indices = np.random.choice(len(features), 10000, replace=False)
            X_train = features[indices]
            y_train = pseudo_labels[indices]
        else:
            X_train = features
            y_train = pseudo_labels
        
        # Train SVM
        svm = SVC(kernel='rbf', random_state=42, probability=True)
        svm.fit(X_train, y_train)
        
        # Predict on all features
        labels = svm.predict(features)
        
        # Post-process labels
        labels = self._postprocess_labels(labels, features)
        
        self.classifier = svm
        return labels
    
    def _postprocess_labels(self, labels, features):
        """Post-process classification labels to improve land cover mapping"""
        # Ensure labels start from 0 and are consecutive
        unique_labels = np.unique(labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        
        processed_labels = np.array([label_mapping[label] for label in labels])
        
        # Apply land cover logic based on spectral characteristics
        if features.shape[1] >= 3:  # If we have at least 3 bands
            processed_labels = self._apply_spectral_rules(processed_labels, features)
        
        return processed_labels
    
    def _apply_spectral_rules(self, labels, features):
        """Apply spectral rules to improve classification accuracy"""
        # Assume first 3 bands are visible spectrum (RGB or similar)
        red_band = features[:, 0] if features.shape[1] > 0 else np.zeros(len(features))
        green_band = features[:, 1] if features.shape[1] > 1 else np.zeros(len(features))
        blue_band = features[:, 2] if features.shape[1] > 2 else np.zeros(len(features))
        
        # Calculate indices
        brightness = (red_band + green_band + blue_band) / 3
        
        # NIR band if available (for NDVI calculation)
        if features.shape[1] >= 4:
            nir_band = features[:, 3]
            ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
        else:
            ndvi = np.zeros(len(features))
        
        # Apply rules
        # Water: Low brightness, low NDVI
        water_mask = (brightness < 0.3) & (ndvi < 0.1)
        labels[water_mask] = 0
        
        # Vegetation: High NDVI
        vegetation_mask = ndvi > 0.4
        labels[vegetation_mask] = 1  # Forest
        
        # Agriculture: Moderate NDVI
        agriculture_mask = (ndvi > 0.2) & (ndvi < 0.4)
        labels[agriculture_mask] = 2  # Agriculture
        
        # Urban: Low NDVI, moderate brightness
        urban_mask = (ndvi < 0.2) & (brightness > 0.3) & (brightness < 0.7)
        labels[urban_mask] = 3  # Urban
        
        # Bare land: Low NDVI, high brightness
        bare_mask = (ndvi < 0.1) & (brightness > 0.4)
        labels[bare_mask] = 4  # Bare land
        
        return labels

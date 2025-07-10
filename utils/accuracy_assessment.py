import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AccuracyAssessment:
    """
    Handles accuracy assessment for LULC classification results
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
    
    def generate_reference_data(self, classification_result, sample_percentage=0.1):
        """
        Generate simulated reference data for accuracy assessment
        In a real application, this would come from ground truth data
        
        Args:
            classification_result: Dictionary containing classification results
            sample_percentage: Percentage of pixels to use for validation
            
        Returns:
            dict: Reference data and predicted data for accuracy assessment
        """
        classification_map = classification_result['classification_map']
        
        # Get valid pixels
        valid_mask = classification_map >= 0
        valid_indices = np.where(valid_mask)
        
        # Sample random pixels for validation
        n_valid = len(valid_indices[0])
        n_sample = int(n_valid * sample_percentage)
        
        if n_sample < 100:
            n_sample = min(100, n_valid)  # Minimum sample size
        
        sample_indices = np.random.choice(n_valid, size=n_sample, replace=False)
        
        # Get predicted labels for sampled pixels
        predicted_labels = classification_map[valid_indices][sample_indices]
        
        # Simulate reference data with some accuracy
        # In reality, this would come from field surveys or high-quality imagery
        reference_labels = self._simulate_reference_data(predicted_labels)
        
        return {
            'predicted': predicted_labels,
            'reference': reference_labels,
            'n_samples': n_sample,
            'sample_percentage': sample_percentage
        }
    
    def _simulate_reference_data(self, predicted_labels):
        """
        Simulate reference data with realistic accuracy patterns
        This creates synthetic ground truth data for demonstration
        """
        reference_labels = predicted_labels.copy()
        
        # Add some classification errors based on typical confusion patterns
        confusion_patterns = {
            # Water can be confused with shadow
            0: {9: 0.05},
            # Forest can be confused with grassland
            1: {5: 0.08, 2: 0.03},
            # Agriculture can be confused with grassland or bare land
            2: {5: 0.10, 4: 0.05, 1: 0.02},
            # Urban can be confused with bare land
            3: {4: 0.07, 0: 0.02},
            # Bare land can be confused with urban or agriculture
            4: {3: 0.06, 2: 0.04},
            # Grassland can be confused with agriculture or forest
            5: {2: 0.09, 1: 0.05},
            # Shadow can be confused with water
            9: {0: 0.03}
        }
        
        for i, pred_class in enumerate(predicted_labels):
            if pred_class in confusion_patterns:
                confusions = confusion_patterns[pred_class]
                for conf_class, prob in confusions.items():
                    if np.random.random() < prob:
                        reference_labels[i] = conf_class
                        break
        
        return reference_labels
    
    def calculate_accuracy_metrics(self, validation_data):
        """
        Calculate comprehensive accuracy metrics
        
        Args:
            validation_data: Dictionary containing predicted and reference data
            
        Returns:
            dict: Comprehensive accuracy metrics
        """
        predicted = validation_data['predicted']
        reference = validation_data['reference']
        
        # Overall accuracy
        overall_accuracy = accuracy_score(reference, predicted)
        
        # Kappa coefficient
        kappa = cohen_kappa_score(reference, predicted)
        
        # Confusion matrix
        unique_classes = np.unique(np.concatenate([predicted, reference]))
        cm = confusion_matrix(reference, predicted, labels=unique_classes)
        
        # Producer's and User's accuracy
        producers_accuracy = {}
        users_accuracy = {}
        
        for i, class_id in enumerate(unique_classes):
            # Producer's accuracy (recall)
            if cm[:, i].sum() > 0:
                producers_accuracy[class_id] = cm[i, i] / cm[:, i].sum()
            else:
                producers_accuracy[class_id] = 0
            
            # User's accuracy (precision)
            if cm[i, :].sum() > 0:
                users_accuracy[class_id] = cm[i, i] / cm[i, :].sum()
            else:
                users_accuracy[class_id] = 0
        
        # F1 scores
        f1_scores = {}
        for class_id in unique_classes:
            if producers_accuracy[class_id] + users_accuracy[class_id] > 0:
                f1_scores[class_id] = 2 * (producers_accuracy[class_id] * users_accuracy[class_id]) / \
                                     (producers_accuracy[class_id] + users_accuracy[class_id])
            else:
                f1_scores[class_id] = 0
        
        # Classification report
        class_names_list = [self.class_names.get(c, f"Class {c}") for c in unique_classes]
        report = classification_report(
            reference, predicted, 
            labels=unique_classes,
            target_names=class_names_list,
            output_dict=True
        )
        
        return {
            'overall_accuracy': overall_accuracy,
            'kappa_coefficient': kappa,
            'confusion_matrix': cm,
            'class_labels': unique_classes,
            'class_names': class_names_list,
            'producers_accuracy': producers_accuracy,
            'users_accuracy': users_accuracy,
            'f1_scores': f1_scores,
            'classification_report': report,
            'n_samples': validation_data['n_samples']
        }
    
    def create_confusion_matrix_plot(self, accuracy_metrics):
        """
        Create an interactive confusion matrix visualization
        
        Args:
            accuracy_metrics: Dictionary containing accuracy metrics
            
        Returns:
            plotly.graph_objects.Figure: Interactive confusion matrix plot
        """
        cm = accuracy_metrics['confusion_matrix']
        class_names = accuracy_metrics['class_names']
        
        # Normalize confusion matrix for better visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        # Create hover text
        hover_text = []
        for i in range(len(class_names)):
            hover_row = []
            for j in range(len(class_names)):
                hover_row.append(
                    f"Reference: {class_names[i]}<br>" +
                    f"Predicted: {class_names[j]}<br>" +
                    f"Count: {cm[i, j]}<br>" +
                    f"Percentage: {cm_normalized[i, j]:.1%}"
                )
            hover_text.append(hover_row)
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=class_names,
            y=class_names,
            hovertemplate='%{text}<extra></extra>',
            text=hover_text,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Normalized Frequency")
        ))
        
        # Add annotations with counts
        annotations = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                annotations.append(
                    dict(
                        x=j, y=i,
                        text=str(cm[i, j]),
                        showarrow=False,
                        font=dict(color="white" if cm_normalized[i, j] > 0.5 else "black")
                    )
                )
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Class',
            yaxis_title='Reference Class',
            annotations=annotations,
            width=800,
            height=600
        )
        
        return fig
    
    def create_accuracy_summary_plot(self, accuracy_metrics):
        """
        Create accuracy summary visualization
        
        Args:
            accuracy_metrics: Dictionary containing accuracy metrics
            
        Returns:
            plotly.graph_objects.Figure: Accuracy summary plot
        """
        class_names = accuracy_metrics['class_names']
        producers_acc = [accuracy_metrics['producers_accuracy'][cls] for cls in accuracy_metrics['class_labels']]
        users_acc = [accuracy_metrics['users_accuracy'][cls] for cls in accuracy_metrics['class_labels']]
        f1_scores = [accuracy_metrics['f1_scores'][cls] for cls in accuracy_metrics['class_labels']]
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Producer\'s Accuracy by Class',
                'User\'s Accuracy by Class',
                'F1 Scores by Class',
                'Overall Metrics'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Producer's accuracy
        fig.add_trace(
            go.Bar(x=class_names, y=producers_acc, name="Producer's Accuracy",
                   marker_color='lightblue'),
            row=1, col=1
        )
        
        # User's accuracy
        fig.add_trace(
            go.Bar(x=class_names, y=users_acc, name="User's Accuracy",
                   marker_color='lightgreen'),
            row=1, col=2
        )
        
        # F1 scores
        fig.add_trace(
            go.Bar(x=class_names, y=f1_scores, name="F1 Score",
                   marker_color='lightcoral'),
            row=2, col=1
        )
        
        # Overall metrics as indicators
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=accuracy_metrics['overall_accuracy'],
                domain={'x': [0, 1], 'y': [0, 0.5]},
                title={'text': "Overall Accuracy"},
                gauge={'axis': {'range': [None, 1]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                {'range': [0.5, 0.8], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 0.9}}
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Classification Accuracy Assessment"
        )
        
        # Update y-axes for percentage display
        for row in [1, 2]:
            for col in [1, 2]:
                if not (row == 2 and col == 2):  # Skip indicator subplot
                    fig.update_yaxes(tickformat='.0%', row=row, col=col)
        
        return fig
    
    def create_accuracy_report(self, accuracy_metrics):
        """
        Create a comprehensive accuracy assessment report
        
        Args:
            accuracy_metrics: Dictionary containing accuracy metrics
            
        Returns:
            str: Formatted accuracy report
        """
        report = []
        report.append("LULC CLASSIFICATION ACCURACY ASSESSMENT REPORT")
        report.append("=" * 55)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL ACCURACY METRICS")
        report.append("-" * 30)
        report.append(f"Overall Accuracy: {accuracy_metrics['overall_accuracy']:.3f} ({accuracy_metrics['overall_accuracy']:.1%})")
        report.append(f"Kappa Coefficient: {accuracy_metrics['kappa_coefficient']:.3f}")
        report.append(f"Number of validation samples: {accuracy_metrics['n_samples']:,}")
        report.append("")
        
        # Kappa interpretation
        kappa = accuracy_metrics['kappa_coefficient']
        if kappa < 0:
            kappa_interp = "Poor (less than chance agreement)"
        elif kappa < 0.20:
            kappa_interp = "Slight agreement"
        elif kappa < 0.40:
            kappa_interp = "Fair agreement"
        elif kappa < 0.60:
            kappa_interp = "Moderate agreement"
        elif kappa < 0.80:
            kappa_interp = "Substantial agreement"
        else:
            kappa_interp = "Almost perfect agreement"
        
        report.append(f"Kappa Interpretation: {kappa_interp}")
        report.append("")
        
        # Class-wise accuracy
        report.append("CLASS-WISE ACCURACY METRICS")
        report.append("-" * 35)
        
        producers_header = "Producer's"
        users_header = "User's" 
        header = f"{'Class':<15} {producers_header:<12} {users_header:<12} {'F1':<8}"
        report.append(header)
        report.append("-" * len(header))
        
        for i, class_id in enumerate(accuracy_metrics['class_labels']):
            class_name = accuracy_metrics['class_names'][i]
            prod_acc = accuracy_metrics['producers_accuracy'][class_id]
            user_acc = accuracy_metrics['users_accuracy'][class_id]
            f1_score = accuracy_metrics['f1_scores'][class_id]
            
            report.append(f"{class_name:<15} {prod_acc:<12.3f} {user_acc:<12.3f} {f1_score:<8.3f}")
        
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 20)
        
        if accuracy_metrics['overall_accuracy'] >= 0.85:
            report.append("• Excellent classification accuracy achieved")
        elif accuracy_metrics['overall_accuracy'] >= 0.70:
            report.append("• Good classification accuracy")
            report.append("• Consider improving training data for better results")
        else:
            report.append("• Classification accuracy needs improvement")
            report.append("• Review training data quality and classification parameters")
            report.append("• Consider additional preprocessing or feature engineering")
        
        if kappa < 0.60:
            report.append("• Kappa coefficient indicates room for improvement")
            report.append("• Consider balancing training data across classes")
        
        return "\n".join(report)
    
    def export_accuracy_results(self, accuracy_metrics, filename_prefix="accuracy_assessment"):
        """
        Export accuracy results to CSV files
        
        Args:
            accuracy_metrics: Dictionary containing accuracy metrics
            filename_prefix: Prefix for output filenames
            
        Returns:
            dict: Dictionary of created filenames
        """
        # Class-wise accuracy table
        class_accuracy_df = pd.DataFrame({
            'Class_ID': accuracy_metrics['class_labels'],
            'Class_Name': accuracy_metrics['class_names'],
            'Producers_Accuracy': [accuracy_metrics['producers_accuracy'][c] for c in accuracy_metrics['class_labels']],
            'Users_Accuracy': [accuracy_metrics['users_accuracy'][c] for c in accuracy_metrics['class_labels']],
            'F1_Score': [accuracy_metrics['f1_scores'][c] for c in accuracy_metrics['class_labels']]
        })
        
        # Confusion matrix
        cm_df = pd.DataFrame(
            accuracy_metrics['confusion_matrix'],
            index=accuracy_metrics['class_names'],
            columns=accuracy_metrics['class_names']
        )
        
        # Overall metrics
        overall_df = pd.DataFrame({
            'Metric': ['Overall_Accuracy', 'Kappa_Coefficient', 'Number_of_Samples'],
            'Value': [
                accuracy_metrics['overall_accuracy'],
                accuracy_metrics['kappa_coefficient'],
                accuracy_metrics['n_samples']
            ]
        })
        
        filenames = {
            'class_accuracy': f"{filename_prefix}_class_accuracy.csv",
            'confusion_matrix': f"{filename_prefix}_confusion_matrix.csv",
            'overall_metrics': f"{filename_prefix}_overall_metrics.csv"
        }
        
        class_accuracy_df.to_csv(filenames['class_accuracy'], index=False)
        cm_df.to_csv(filenames['confusion_matrix'])
        overall_df.to_csv(filenames['overall_metrics'], index=False)
        
        return filenames
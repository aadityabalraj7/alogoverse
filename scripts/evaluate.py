#!/usr/bin/env python3
"""
Duality AI Space Station Hackathon - Evaluation Script
Comprehensive evaluation and visualization of YOLOv8 model performance

Target Objects:
- Toolbox
- Oxygen Tank  
- Fire Extinguisher

Usage:
    python scripts/evaluate.py --model models/space_station_best.pt --config config.yaml
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import json
import cv2
from PIL import Image

def setup_logging():
    """Set up logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"evaluation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def create_confusion_matrix_plot(y_true, y_pred, class_names, output_path):
    """Create and save confusion matrix plot"""
    logger = logging.getLogger(__name__)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Space Station Object Detection', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to: {output_path}")
    return cm

def create_performance_plots(metrics_data, output_dir):
    """Create various performance visualization plots"""
    logger = logging.getLogger(__name__)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Class-wise performance bar chart
    if 'class_metrics' in metrics_data:
        class_metrics = metrics_data['class_metrics']
        classes = list(class_metrics.keys())
        precision = [class_metrics[cls]['precision'] for cls in classes]
        recall = [class_metrics[cls]['recall'] for cls in classes]
        f1_score = [class_metrics[cls]['f1-score'] for cls in classes]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(classes))
        width = 0.25
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Object Classes', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Class-wise Performance Metrics', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'class_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Class performance plot saved to: {output_dir / 'class_performance.png'}")
    
    # Overall metrics summary
    if 'overall_metrics' in metrics_data:
        overall = metrics_data['overall_metrics']
        metrics_names = list(overall.keys())
        metrics_values = list(overall.values())
        
        # Create pie chart for overall performance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        bars = ax1.bar(metrics_names, metrics_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax1.set_title('Overall Model Performance', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=11)
        
        # Donut chart for mAP visualization
        if 'mAP_50' in overall:
            map_score = overall['mAP_50']
            remaining = 1 - map_score
            
            sizes = [map_score, remaining]
            labels = [f'mAP@0.5: {map_score:.3f}', f'Remaining: {remaining:.3f}']
            colors = ['#ff9999', '#66b3ff']
            
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                              startangle=90, wedgeprops=dict(width=0.5))
            ax2.set_title('mAP@0.5 Score Visualization', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'overall_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Overall performance plot saved to: {output_dir / 'overall_performance.png'}")

def analyze_predictions(predictions_dir, ground_truth_dir, class_names, output_dir):
    """Analyze predictions and compare with ground truth"""
    logger = logging.getLogger(__name__)
    
    predictions_path = Path(predictions_dir)
    ground_truth_path = Path(ground_truth_dir)
    output_path = Path(output_dir)
    
    if not predictions_path.exists():
        logger.warning(f"Predictions directory not found: {predictions_path}")
        return None
    
    # Collect prediction and ground truth data
    y_true = []
    y_pred = []
    confidence_scores = []
    
    # Process prediction files
    for pred_file in predictions_path.glob("*.txt"):
        # Read predictions
        if pred_file.stat().st_size > 0:
            with open(pred_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:  # class, x, y, w, h, conf
                        class_id = int(parts[0])
                        confidence = float(parts[5])
                        y_pred.append(class_id)
                        confidence_scores.append(confidence)
        
        # Read corresponding ground truth
        gt_file = ground_truth_path / pred_file.name
        if gt_file.exists() and gt_file.stat().st_size > 0:
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # class, x, y, w, h
                        class_id = int(parts[0])
                        y_true.append(class_id)
    
    if not y_true or not y_pred:
        logger.warning("No valid predictions or ground truth data found")
        return None
    
    # Ensure equal lengths (basic matching)
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    confidence_scores = confidence_scores[:min_len]
    
    # Create confusion matrix
    cm_output = output_path / 'confusion_matrix.png'
    cm = create_confusion_matrix_plot(y_true, y_pred, class_names, cm_output)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Save classification report
    report_output = output_path / 'classification_report.json'
    with open(report_output, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Classification report saved to: {report_output}")
    
    return {
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'class_metrics': {class_names[i]: report[class_names[i]] for i in range(len(class_names)) if class_names[i] in report},
        'overall_metrics': {
            'accuracy': report['accuracy'],
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'macro_f1': report['macro avg']['f1-score']
        }
    }

def evaluate_model_comprehensive(model_path, config_path, output_dir):
    """Comprehensive model evaluation"""
    logger = logging.getLogger(__name__)
    
    # Check if model exists
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        return None
    
    # Load model
    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Check for CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run validation
    logger.info("Running model validation...")
    results = model.val(data=config_path, device=device, project=str(output_path), name='validation')
    
    # Extract metrics
    metrics = {
        'mAP_50': results.box.map50,
        'mAP_50_95': results.box.map,
        'precision': results.box.mp,
        'recall': results.box.mr,
        'class_names': model.names
    }
    
    # Save detailed metrics
    metrics_output = output_path / 'detailed_metrics.json'
    with open(metrics_output, 'w') as f:
        json.dump({k: float(v) if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}, 
                 f, indent=2)
    
    logger.info(f"Detailed metrics saved to: {metrics_output}")
    logger.info(f"mAP@0.5: {metrics['mAP_50']:.4f}")
    logger.info(f"mAP@0.5:0.95: {metrics['mAP_50_95']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    
    return metrics

def create_evaluation_report(evaluation_data, output_path):
    """Create comprehensive evaluation report"""
    logger = logging.getLogger(__name__)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
# Duality AI Space Station Hackathon - Model Evaluation Report
Generated on: {timestamp}

## Model Performance Summary

### Overall Metrics
"""
    
    if 'overall_metrics' in evaluation_data:
        metrics = evaluation_data['overall_metrics']
        for metric, value in metrics.items():
            report += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"
    
    report += "\n### Class-wise Performance\n"
    
    if 'class_metrics' in evaluation_data:
        for class_name, metrics in evaluation_data['class_metrics'].items():
            report += f"\n#### {class_name}\n"
            report += f"- Precision: {metrics['precision']:.4f}\n"
            report += f"- Recall: {metrics['recall']:.4f}\n"
            report += f"- F1-Score: {metrics['f1-score']:.4f}\n"
            report += f"- Support: {metrics['support']}\n"
    
    report += "\n## Benchmarks Comparison\n"
    report += "Expected benchmarks from hackathon documentation:\n"
    report += "- mAP@0.5: 40-50% (Baseline)\n"
    report += "- Precision & Recall: >70% (Best models)\n"
    report += "- Training Loss: Should steadily decrease\n"
    report += "- Inference Speed: <50ms per image\n"
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Evaluation report saved to: {output_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Comprehensive Evaluation for Space Station Object Detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file (.pt)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file (config.yaml)')
    parser.add_argument('--output', type=str, default='results/evaluation',
                       help='Output directory for evaluation results (default: results/evaluation)')
    parser.add_argument('--predictions', type=str,
                       help='Path to predictions directory for detailed analysis')
    parser.add_argument('--ground_truth', type=str,
                       help='Path to ground truth labels directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Duality AI Space Station Hackathon - Evaluation Script")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {args.output}")
    
    try:
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run comprehensive evaluation
        logger.info("Running comprehensive model evaluation...")
        model_metrics = evaluate_model_comprehensive(args.model, args.config, args.output)
        
        if model_metrics is None:
            logger.error("Model evaluation failed")
            sys.exit(1)
        
        # Detailed analysis if prediction data available
        evaluation_data = {}
        class_names = ['Toolbox', 'Oxygen Tank', 'Fire Extinguisher']
        
        if args.predictions and args.ground_truth:
            logger.info("Running detailed prediction analysis...")
            pred_analysis = analyze_predictions(args.predictions, args.ground_truth, 
                                              class_names, args.output)
            if pred_analysis:
                evaluation_data.update(pred_analysis)
                
                # Create performance plots
                logger.info("Creating performance visualizations...")
                create_performance_plots(evaluation_data, output_path / 'plots')
        
        # Create comprehensive report
        logger.info("Creating evaluation report...")
        create_evaluation_report(evaluation_data, output_path / 'evaluation_report.md')
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results available in: {output_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
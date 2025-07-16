#!/usr/bin/env python3
"""
Duality AI Space Station Hackathon - Prediction Script
Perform inference using trained YOLOv8 model for space station object detection

Target Objects:
- Toolbox
- Oxygen Tank  
- Fire Extinguisher

Usage:
    python scripts/predict.py --model models/space_station_best.pt --source dataset/test/images
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import logging
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import json

def setup_logging():
    """Set up logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"prediction_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def predict_batch(model_path, source_path, output_dir, conf_threshold=0.25, iou_threshold=0.7):
    """Perform batch prediction on images"""
    logger = logging.getLogger(__name__)
    
    # Check if model exists
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)
    
    # Load model
    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Check for CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prediction parameters
    predict_params = {
        'source': source_path,
        'conf': conf_threshold,
        'iou': iou_threshold,
        'device': device,
        'project': str(output_path),
        'name': 'predictions',
        'save': True,
        'save_txt': True,
        'save_conf': True,
        'save_json': True,
        'show_labels': True,
        'show_conf': True,
        'line_width': 2,
        'exist_ok': True
    }
    
    logger.info("Prediction parameters:")
    for key, value in predict_params.items():
        logger.info(f"  {key}: {value}")
    
    # Run predictions
    logger.info("Starting predictions...")
    results = model.predict(**predict_params)
    
    logger.info(f"Predictions completed. Results saved to: {output_path}")
    return results

def predict_single(model_path, image_path, output_dir, conf_threshold=0.25, iou_threshold=0.7):
    """Perform prediction on a single image"""
    logger = logging.getLogger(__name__)
    
    # Check if files exist
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        return None
    
    if not Path(image_path).exists():
        logger.error(f"Image file not found: {image_path}")
        return None
    
    # Load model
    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Check for CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load and preprocess image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return None
    
    # Perform prediction
    results = model(image, conf=conf_threshold, iou=iou_threshold, device=device)
    
    # Process results
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Extract box information
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Get class name
                class_name = model.names[cls]
                
                detection = {
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'class_id': cls
                }
                detections.append(detection)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save annotated image
    annotated_image = results[0].plot()
    output_image_path = output_path / f"annotated_{Path(image_path).name}"
    cv2.imwrite(str(output_image_path), annotated_image)
    
    # Save detection results as JSON
    result_data = {
        'image_path': str(image_path),
        'model_path': str(model_path),
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'detections': detections,
        'num_detections': len(detections)
    }
    
    output_json_path = output_path / f"results_{Path(image_path).stem}.json"
    with open(output_json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    logger.info(f"Found {len(detections)} detections in image")
    logger.info(f"Annotated image saved to: {output_image_path}")
    logger.info(f"Results saved to: {output_json_path}")
    
    return result_data

def evaluate_model(model_path, test_data_path, output_dir):
    """Evaluate model performance on test dataset"""
    logger = logging.getLogger(__name__)
    
    # Check if model exists
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        return None
    
    # Load model
    logger.info(f"Loading model for evaluation: {model_path}")
    model = YOLO(model_path)
    
    # Check for CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run validation
    logger.info("Starting model evaluation...")
    results = model.val(data=test_data_path, device=device, project=str(output_path), name='evaluation')
    
    logger.info("Evaluation completed successfully!")
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='YOLOv8 Prediction for Space Station Object Detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file (.pt)')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image file or directory of images')
    parser.add_argument('--output', type=str, default='results/predictions',
                       help='Output directory for predictions (default: results/predictions)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.7,
                       help='IoU threshold for NMS (default: 0.7)')
    parser.add_argument('--single', action='store_true',
                       help='Process single image instead of batch')
    parser.add_argument('--evaluate', type=str,
                       help='Evaluate model on test dataset (provide config.yaml path)')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Duality AI Space Station Hackathon - Prediction Script")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Source: {args.source}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Confidence threshold: {args.conf}")
    logger.info(f"IoU threshold: {args.iou}")
    
    try:
        if args.evaluate:
            # Evaluate model
            logger.info("Running model evaluation...")
            results = evaluate_model(args.model, args.evaluate, args.output)
            logger.info("Evaluation completed successfully!")
            
        elif args.single:
            # Single image prediction
            logger.info("Running single image prediction...")
            results = predict_single(args.model, args.source, args.output, args.conf, args.iou)
            if results:
                logger.info(f"Detected {results['num_detections']} objects")
                for detection in results['detections']:
                    logger.info(f"  {detection['class']}: {detection['confidence']:.3f}")
            
        else:
            # Batch prediction
            logger.info("Running batch prediction...")
            results = predict_batch(args.model, args.source, args.output, args.conf, args.iou)
            logger.info("Batch prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Duality AI Space Station Hackathon - Training Script
Train YOLOv8 model for object detection in space station environment

Target Objects:
- Toolbox
- Oxygen Tank  
- Fire Extinguisher

Usage:
    python scripts/train.py [--config config.yaml] [--model yolov8n.pt] [--epochs 100]
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO
import logging
from datetime import datetime

def setup_logging():
    """Set up logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def validate_dataset(dataset_path):
    """Validate dataset structure"""
    logger = logging.getLogger(__name__)
    
    required_dirs = [
        "train/images", "train/labels",
        "val/images", "val/labels"
    ]
    
    for dir_path in required_dirs:
        full_path = Path(dataset_path) / dir_path
        if not full_path.exists():
            logger.warning(f"Directory not found: {full_path}")
            logger.info(f"Creating directory: {full_path}")
            full_path.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"Found directory: {full_path}")
    
    # Check for test directory (optional)
    test_dir = Path(dataset_path) / "test/images"
    if test_dir.exists():
        logger.info(f"Found test directory: {test_dir}")
    else:
        logger.info("Test directory not found (optional)")

def load_config(config_path):
    """Load configuration from YAML file"""
    logger = logging.getLogger(__name__)
    
    if not Path(config_path).exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from: {config_path}")
    return config

def train_model(config_path, model_name, epochs, resume=False):
    """Train YOLOv8 model"""
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(config_path)
    
    # Validate dataset
    dataset_path = config.get('path', './dataset')
    validate_dataset(dataset_path)
    
    # Initialize model
    logger.info(f"Initializing YOLOv8 model: {model_name}")
    model = YOLO(model_name)
    
    # Check for CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    if device == 'cuda':
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Training parameters
    train_params = {
        'data': config_path,
        'epochs': epochs,
        'imgsz': config.get('imgsz', 640),
        'batch': config.get('batch', 16),
        'workers': config.get('workers', 8),
        'device': device,
        'project': config.get('project', 'runs/detect'),
        'name': config.get('name', 'space_station_detection'),
        'save_period': config.get('save_period', 10),
        'patience': config.get('patience', 50),
        'plots': True,
        'save_json': True,
        'save_txt': True,
        'save_conf': True,
        'exist_ok': False,
        'resume': resume
    }
    
    # Log training parameters
    logger.info("Training parameters:")
    for key, value in train_params.items():
        logger.info(f"  {key}: {value}")
    
    # Start training
    logger.info("Starting training...")
    results = model.train(**train_params)
    
    # Save best model to models directory
    best_model_path = Path(train_params['project']) / train_params['name'] / 'weights' / 'best.pt'
    if best_model_path.exists():
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Copy best model
        import shutil
        final_model_path = models_dir / "space_station_best.pt"
        shutil.copy2(best_model_path, final_model_path)
        logger.info(f"Best model saved to: {final_model_path}")
        
        # Copy last model
        last_model_path = Path(train_params['project']) / train_params['name'] / 'weights' / 'last.pt'
        if last_model_path.exists():
            final_last_path = models_dir / "space_station_last.pt"
            shutil.copy2(last_model_path, final_last_path)
            logger.info(f"Last model saved to: {final_last_path}")
    
    logger.info("Training completed successfully!")
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train YOLOv8 for Space Station Object Detection')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLOv8 model variant (default: yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Duality AI Space Station Hackathon - Training Script")
    logger.info("=" * 60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Resume: {args.resume}")
    
    try:
        # Train model
        results = train_model(args.config, args.model, args.epochs, args.resume)
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
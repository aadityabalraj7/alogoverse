#!/usr/bin/env python3
"""
Duality AI Space Station Hackathon - Sample Data Generator
Creates sample images and labels for testing the training pipeline

This script generates synthetic images with simple geometric shapes
representing the three target objects for pipeline testing purposes.

Usage:
    python scripts/generate_sample_data.py [--num_samples 100] [--output_dir dataset]
"""

import argparse
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import json

def create_sample_image_with_objects(width=640, height=640, num_objects=None):
    """Create a sample image with geometric shapes representing space station objects"""
    
    # Create image with space-like background
    image = Image.new('RGB', (width, height), color=(10, 10, 30))  # Dark space background
    draw = ImageDraw.Draw(image)
    
    # Add some stars for realism
    for _ in range(random.randint(20, 50)):
        x = random.randint(0, width)
        y = random.randint(0, height)
        draw.point((x, y), fill=(255, 255, 255))
    
    annotations = []
    
    # Determine number of objects
    if num_objects is None:
        num_objects = random.randint(1, 4)
    
    for _ in range(num_objects):
        # Randomly choose object type
        object_type = random.randint(0, 2)
        
        # Generate random position and size
        obj_width = random.randint(60, 150)
        obj_height = random.randint(60, 150)
        x = random.randint(0, width - obj_width)
        y = random.randint(0, height - obj_height)
        
        # Create object based on type
        if object_type == 0:  # Toolbox (rectangular)
            color = (100, 100, 100)  # Gray
            draw.rectangle([x, y, x + obj_width, y + obj_height], fill=color, outline=(200, 200, 200), width=2)
            # Add handle
            handle_y = y + obj_height // 4
            draw.rectangle([x + obj_width//4, handle_y, x + 3*obj_width//4, handle_y + 10], 
                         fill=(150, 150, 150))
            
        elif object_type == 1:  # Oxygen Tank (cylindrical/oval)
            color = (0, 100, 200)  # Blue
            draw.ellipse([x, y, x + obj_width, y + obj_height], fill=color, outline=(0, 150, 255), width=2)
            # Add valve on top
            valve_x = x + obj_width // 2 - 5
            valve_y = y - 10
            draw.rectangle([valve_x, valve_y, valve_x + 10, valve_y + 20], fill=(200, 200, 200))
            
        else:  # Fire Extinguisher (cylindrical with red color)
            color = (200, 0, 0)  # Red
            draw.ellipse([x, y, x + obj_width, y + obj_height], fill=color, outline=(255, 100, 100), width=2)
            # Add nozzle
            nozzle_x = x + obj_width // 2 - 3
            nozzle_y = y - 15
            draw.rectangle([nozzle_x, nozzle_y, nozzle_x + 6, nozzle_y + 25], fill=(100, 100, 100))
        
        # Calculate YOLO format annotation (normalized coordinates)
        center_x = (x + obj_width / 2) / width
        center_y = (y + obj_height / 2) / height
        norm_width = obj_width / width
        norm_height = obj_height / height
        
        annotations.append({
            'class': object_type,
            'center_x': center_x,
            'center_y': center_y,
            'width': norm_width,
            'height': norm_height
        })
    
    return image, annotations

def generate_sample_dataset(output_dir, num_samples_per_split):
    """Generate a complete sample dataset"""
    
    output_path = Path(output_dir)
    
    # Create directory structure
    splits = ['train', 'val', 'test']
    for split in splits:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Class names
    class_names = ['Toolbox', 'Oxygen Tank', 'Fire Extinguisher']
    
    # Generate data for each split
    for split in splits:
        num_samples = num_samples_per_split[split]
        print(f"Generating {num_samples} samples for {split} set...")
        
        for i in range(num_samples):
            # Generate image and annotations
            image, annotations = create_sample_image_with_objects()
            
            # Save image
            image_filename = f"sample_{split}_{i:04d}.jpg"
            image_path = output_path / split / 'images' / image_filename
            image.save(image_path, 'JPEG')
            
            # Save YOLO format labels
            label_filename = f"sample_{split}_{i:04d}.txt"
            label_path = output_path / split / 'labels' / label_filename
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    f.write(f"{ann['class']} {ann['center_x']:.6f} {ann['center_y']:.6f} "
                           f"{ann['width']:.6f} {ann['height']:.6f}\n")
        
        print(f"✅ {split} set completed: {num_samples} images and labels")
    
    # Create dataset info file
    dataset_info = {
        'name': 'Sample Space Station Dataset',
        'description': 'Synthetic sample dataset for testing the training pipeline',
        'classes': class_names,
        'num_classes': len(class_names),
        'splits': {
            'train': num_samples_per_split['train'],
            'val': num_samples_per_split['val'],
            'test': num_samples_per_split['test']
        },
        'total_samples': sum(num_samples_per_split.values())
    }
    
    with open(output_path / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n✅ Sample dataset created successfully!")
    print(f"📁 Location: {output_path}")
    print(f"📊 Total samples: {dataset_info['total_samples']}")
    print(f"🎯 Classes: {', '.join(class_names)}")

def create_test_images(output_dir, num_images=5):
    """Create a few test images for immediate testing"""
    
    output_path = Path(output_dir) / 'test_images'
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_images} test images...")
    
    for i in range(num_images):
        # Create image with specific scenarios
        if i == 0:
            # Single object
            image, _ = create_sample_image_with_objects(num_objects=1)
        elif i == 1:
            # Multiple objects
            image, _ = create_sample_image_with_objects(num_objects=3)
        else:
            # Random
            image, _ = create_sample_image_with_objects()
        
        image_path = output_path / f"test_image_{i+1}.jpg"
        image.save(image_path, 'JPEG')
    
    print(f"✅ Test images created in: {output_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate sample dataset for Space Station Object Detection')
    parser.add_argument('--output_dir', type=str, default='dataset',
                       help='Output directory for dataset (default: dataset)')
    parser.add_argument('--train_samples', type=int, default=80,
                       help='Number of training samples (default: 80)')
    parser.add_argument('--val_samples', type=int, default=15,
                       help='Number of validation samples (default: 15)')
    parser.add_argument('--test_samples', type=int, default=15,
                       help='Number of test samples (default: 15)')
    parser.add_argument('--test_images_only', action='store_true',
                       help='Only create test images for quick testing')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Duality AI Space Station Hackathon - Sample Data Generator")
    print("=" * 60)
    
    if args.test_images_only:
        # Only create test images
        create_test_images(args.output_dir, num_images=10)
    else:
        # Create full dataset
        num_samples_per_split = {
            'train': args.train_samples,
            'val': args.val_samples,
            'test': args.test_samples
        }
        
        print(f"Output directory: {args.output_dir}")
        print(f"Training samples: {args.train_samples}")
        print(f"Validation samples: {args.val_samples}")
        print(f"Test samples: {args.test_samples}")
        print(f"Total samples: {sum(num_samples_per_split.values())}")
        print()
        
        generate_sample_dataset(args.output_dir, num_samples_per_split)
        
        # Also create some test images
        create_test_images(args.output_dir, num_images=5)
    
    print("\n🚀 Ready to start training with sample data!")
    print("Note: Replace this sample data with the official hackathon dataset for actual training.")

if __name__ == "__main__":
    main() 
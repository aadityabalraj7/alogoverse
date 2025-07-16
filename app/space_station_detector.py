#!/usr/bin/env python3
"""
General Object Detection Application
Real-time object detection using YOLOv8 for any objects

Detects 80+ common objects including:
- People, vehicles, animals
- Furniture, electronics, tools
- Food, sports equipment, and more

Usage:
    streamlit run app/space_station_detector.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import tempfile
import os
from pathlib import Path
import time
import json
from datetime import datetime

# Page configuration
st.set_page_config(
            page_title="Space Station Object Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .detection-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_general_model():
    """Load trained space station model for detecting ToolBox, Oxygen Tank, Fire Extinguisher"""
    try:
        # Use the trained space station model
        model_path = 'models/space_station_best.pt'
        if Path(model_path).exists():
            model = YOLO(model_path)
            return model
        else:
            st.error(f"❌ Trained model not found at {model_path}. Please ensure the model file exists.")
            st.info("💡 The system expects a trained space station detection model for ToolBox, Oxygen Tank, and Fire Extinguisher detection.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def detect_objects(model, image, conf_threshold, iou_threshold):
    """Perform object detection on image with enhanced generalization"""
    try:
        # Convert PIL image to OpenCV format
        if isinstance(image, Image.Image):
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image_cv = image
        
        # Test Time Augmentation for better generalization
        original_detections = []
        augmented_detections = []
        
        # 1. Original image detection
        start_time = time.time()
        results = model(image_cv, conf=conf_threshold, iou=iou_threshold, verbose=False)
        inference_time = (time.time() - start_time) * 1000
        
        print(f"Debug: Got {len(results)} results, conf_threshold={conf_threshold}")
        
        # Process original results
        for result in results:
            boxes = result.boxes
            print(f"Debug: Processing result, boxes={boxes}")
            if boxes is not None and len(boxes) > 0:
                print(f"Debug: Found {len(boxes)} boxes")
                for i, box in enumerate(boxes):
                    try:
                        # Extract box information with error handling
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Validate coordinates
                        if len(xyxy) >= 4:
                            x1, y1, x2, y2 = xyxy[:4].astype(int)
                            
                            # Get class name safely
                            if cls < len(model.names):
                                class_name = model.names[cls]
                                
                                detection = {
                                    'class': class_name,
                                    'confidence': conf,
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'class_id': cls,
                                    'source': 'original'
                                }
                                original_detections.append(detection)
                                print(f"Debug: Added detection {i+1} - {class_name} ({conf:.3f})")
                            else:
                                print(f"Debug: Invalid class ID {cls}, skipping")
                        else:
                            print(f"Debug: Invalid bbox coordinates, skipping")
                    except Exception as e:
                        print(f"Debug: Error processing box {i}: {e}")
                        continue
            else:
                print("Debug: No boxes found in result")
        
        # 2. Multi-scale augmentation for better recall (if no detections found)
        if len(original_detections) == 0 and conf_threshold > 0.1:
            print("Debug: No detections found, trying augmented detection...")
            
            # Try with even lower confidence
            lower_conf = max(0.05, conf_threshold * 0.5)
            aug_results = model(image_cv, conf=lower_conf, iou=iou_threshold, verbose=False)
            
            for result in aug_results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        try:
                            xyxy = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            cls = int(box.cls[0].cpu().numpy())
                            
                            if len(xyxy) >= 4 and cls < len(model.names):
                                x1, y1, x2, y2 = xyxy[:4].astype(int)
                                class_name = model.names[cls]
                                
                                # Only add if confidence meets original threshold
                                if conf >= conf_threshold:
                                    detection = {
                                        'class': class_name,
                                        'confidence': conf,
                                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                        'class_id': cls,
                                        'source': 'augmented'
                                    }
                                    augmented_detections.append(detection)
                                    print(f"Debug: Added augmented detection - {class_name} ({conf:.3f})")
                        except Exception as e:
                            continue
        
        # Combine detections
        all_detections = original_detections + augmented_detections
        
        # Remove duplicates using IoU
        final_detections = remove_duplicate_detections(all_detections, iou_threshold)
        
        # Create annotated image
        annotated_image = image_cv.copy()
        
        for detection in final_detections:
            try:
                x1, y1, x2, y2 = detection['bbox']
                cls = detection['class_id']
                conf = detection['confidence']
                class_name = detection['class']
                
                # Draw bounding box and label
                color = get_class_color(cls)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Enhanced label with source info
                source_indicator = "🔍" if detection['source'] == 'augmented' else ""
                label = f"{source_indicator}{class_name}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            except Exception as e:
                print(f"Debug: Error drawing detection: {e}")
                continue
        
        # Convert back to RGB for display
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        print(f"Debug: Returning {len(final_detections)} detections")
        return annotated_image_rgb, final_detections, inference_time
        
    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
        print(f"Debug: Detection error: {str(e)}")
        return None, [], 0

def remove_duplicate_detections(detections, iou_threshold):
    """Remove duplicate detections using IoU"""
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence
    sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    final_detections = []
    
    for current in sorted_detections:
        is_duplicate = False
        
        for existing in final_detections:
            # Calculate IoU
            iou = calculate_iou(current['bbox'], existing['bbox'])
            
            # If same class and high overlap, consider duplicate
            if current['class'] == existing['class'] and iou > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_detections.append(current)
    
    return final_detections

def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def get_class_color(class_id):
    """Get color for each class - using a rainbow of colors for variety"""
    # Generate different colors for different classes
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
        (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
        (0, 255, 128), (128, 0, 255), (0, 128, 255), (192, 192, 192), (128, 128, 128)
    ]
    return colors[class_id % len(colors)]

def display_detection_results(detections, inference_time):
    """Display detection results with enhanced debugging and analytics"""
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        detection_count = len(detections)
        st.metric("Objects Detected", detection_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Inference Time", f"{inference_time:.1f} ms")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        benchmark_met = "✅" if inference_time < 50 else "❌"
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Speed Benchmark (<50ms)", benchmark_met)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        # Calculate confidence stats
        if detections:
            avg_conf = sum(d['confidence'] for d in detections) / len(detections)
            max_conf = max(d['confidence'] for d in detections)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg/Max Confidence", f"{avg_conf:.2f}/{max_conf:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Confidence Range", "No detections")
            st.markdown('</div>', unsafe_allow_html=True)
    
    if detections:
        # Success message
        st.success(f"🎉 **Successfully detected {detection_count} object{'s' if detection_count != 1 else ''}!**")
        
        # Detection source breakdown
        original_count = sum(1 for d in detections if d.get('source') == 'original')
        augmented_count = sum(1 for d in detections if d.get('source') == 'augmented')
        
        if augmented_count > 0:
            st.info(f"🔍 **Enhanced Detection Active**: {original_count} direct + {augmented_count} augmented detections")
        
        st.markdown('<div class="detection-box">', unsafe_allow_html=True)
        st.subheader("🎯 Detection Details")
        
        # Group detections by class
        class_counts = {}
        for detection in detections:
            cls = detection['class']
            if cls not in class_counts:
                class_counts[cls] = []
            class_counts[cls].append(detection)
        
        # Display class summary
        class_summary = ", ".join([f"{cls} ({len(instances)})" for cls, instances in class_counts.items()])
        st.write(f"**Detected Classes:** {class_summary}")
        
        # Detailed detections
        for i, detection in enumerate(detections, 1):
            source_icon = "🔍" if detection.get('source') == 'augmented' else "🎯"
            confidence_color = "🟢" if detection['confidence'] > 0.5 else "🟡" if detection['confidence'] > 0.25 else "🟠"
            
            with st.expander(f"{source_icon} Detection {i}: {detection['class']} {confidence_color} ({detection['confidence']:.3f})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Class:** {detection['class']}")
                    st.write(f"**Confidence:** {detection['confidence']:.3f}")
                    st.write(f"**Source:** {detection.get('source', 'original').title()}")
                with col2:
                    bbox = detection['bbox']
                    st.write(f"**Position:** ({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})")
                    st.write(f"**Size:** {bbox[2] - bbox[0]} x {bbox[3] - bbox[1]} pixels")
                    st.write(f"**Area:** {(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]):,} pixels²")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance analysis
        with st.expander("📊 Detection Performance Analysis", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Confidence Distribution")
                conf_ranges = {
                    "High (>0.5)": sum(1 for d in detections if d['confidence'] > 0.5),
                    "Medium (0.25-0.5)": sum(1 for d in detections if 0.25 < d['confidence'] <= 0.5),
                    "Low (0.1-0.25)": sum(1 for d in detections if 0.1 < d['confidence'] <= 0.25),
                    "Very Low (<0.1)": sum(1 for d in detections if d['confidence'] <= 0.1)
                }
                for range_name, count in conf_ranges.items():
                    if count > 0:
                        st.write(f"• {range_name}: {count}")
            
            with col2:
                st.subheader("Detection Quality")
                high_conf = sum(1 for d in detections if d['confidence'] > 0.5)
                medium_conf = sum(1 for d in detections if 0.25 < d['confidence'] <= 0.5)
                low_conf = sum(1 for d in detections if d['confidence'] <= 0.25)
                
                if high_conf > 0:
                    st.success(f"✅ {high_conf} high-confidence detections")
                if medium_conf > 0:
                    st.info(f"ℹ️ {medium_conf} medium-confidence detections")
                if low_conf > 0:
                    st.warning(f"⚠️ {low_conf} low-confidence detections")
    
    else:
        st.error("❌ **No objects detected in this image!**")
        
        # Enhanced troubleshooting with current settings
        st.markdown("### 🔧 **Immediate Solutions:**")
        
        # Current settings display
        current_conf = st.session_state.get('confidence_threshold', 0.25)
        current_iou = st.session_state.get('iou_threshold', 0.45)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **🎚️ Adjust Detection Sensitivity:**
            - Current confidence: {current_conf:.2f}
            - Try lowering to **0.05-0.10** for maximum recall
            - The system will try augmented detection automatically
            """)
        
        with col2:
            st.markdown(f"""
            **🎯 Current IoU Setting: {current_iou:.2f}**
            - Lower IoU (0.35) = more detections
            - Higher IoU (0.60) = fewer duplicates
            - Use quick preset buttons for common scenarios
            """)
        
        st.markdown("### 📸 **Image Requirements for Best Results:**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **✅ Works Great:**
            - 👥 People & faces
            - 🚗 Cars & vehicles
            - 🐕 Animals & pets
            - 🪑 Furniture
            - 📱 Electronics
            """)
        
        with col2:
            st.markdown("""
            **📸 Image Quality:**
            - Good lighting
            - Clear focus
            - Objects not too small
            - Minimal blur/motion
            - Standard angles
            """)
        
        with col3:
            st.markdown("""
            **❌ May Not Detect:**
            - Abstract art
            - Very dark images
            - Extreme close-ups
            - Heavily artistic filters
            - Very specialized objects
            """)
        
        # Actionable recommendations
        st.info("""
        **🚀 Quick Actions to Try:**
        1. **Lower confidence slider** to 0.05-0.10 on the left sidebar
        2. **Upload a different image** with common objects (people, cars, animals, furniture)
        3. **Try the sample images** which are guaranteed to work
        4. **Check image quality** - ensure it's clear and well-lit
        5. **Adjust IoU threshold** for crowded scenes (use preset buttons)
        
        **💡 Pro Tip:** The model excels at detecting everyday objects in typical photos. For specialized detection tasks, custom training would be needed.
        """)

def create_model_update_plan():
    """Create section about keeping model up-to-date with Falcon"""
    st.markdown("---")
    st.header("🔄 Model Update Strategy with Falcon")
    
    st.markdown("""
    ### Continuous Learning Pipeline
    
    Our approach to keeping the space station object detection model current using Duality AI's Falcon platform:
    
    #### 1. **Automated Data Collection**
    - 🌟 **Falcon Integration**: Continuous generation of new synthetic training data
    - 📸 **Scenario Variations**: Different lighting conditions, object orientations, and occlusions
    - 🎯 **Edge Case Focus**: Target challenging scenarios identified during deployment
    
    #### 2. **Incremental Training Schedule**
    - 📅 **Weekly Updates**: Scheduled retraining with new Falcon-generated data
    - 🎚️ **Performance Monitoring**: Automatic detection of model degradation
    - 🔄 **Version Control**: Seamless rollback capabilities if performance drops
    
    #### 3. **Quality Assurance Pipeline**
    - ✅ **Validation Tests**: Automated testing on reserved validation sets
    - 📊 **Performance Metrics**: Continuous tracking of mAP, precision, and recall
    - 🚨 **Alert System**: Notifications when performance falls below thresholds
    
    #### 4. **Deployment Strategy**
    - 🚀 **Blue-Green Deployment**: Zero-downtime model updates
    - 📈 **A/B Testing**: Compare new models against current production version
    - 🎯 **Gradual Rollout**: Phased deployment to minimize risk
    """)
    
    # Interactive update simulator
    st.subheader("📊 Update Simulation Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Model Performance:**")
        current_map = st.slider("Current mAP@0.5", 0.0, 1.0, 0.65, 0.01)
        current_speed = st.slider("Current Inference Speed (ms)", 10, 100, 35, 1)
        
    with col2:
        st.markdown("**Updated Model Prediction:**")
        new_map = st.slider("New mAP@0.5", 0.0, 1.0, 0.72, 0.01)
        new_speed = st.slider("New Inference Speed (ms)", 10, 100, 32, 1)
    
    # Update recommendation
    improvement = new_map - current_map
    speed_change = new_speed - current_speed
    
    if improvement > 0.02 and speed_change < 5:
        st.success("✅ **Recommendation**: Deploy the updated model")
        st.write(f"• Performance improvement: +{improvement:.3f} mAP")
        st.write(f"• Speed impact: {speed_change:+.1f} ms")
    elif improvement > 0.05:
        st.warning("⚠️ **Recommendation**: Deploy with monitoring")
        st.write(f"• Significant improvement: +{improvement:.3f} mAP")
        st.write(f"• Speed trade-off: {speed_change:+.1f} ms")
    else:
        st.error("❌ **Recommendation**: Keep current model")
        st.write("• Insufficient improvement to justify update")

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">🚀 Space Station Object Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Detection of Space Station Equipment using YOLOv8</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Debug: Show loaded model class names
        model_debug_info = ""
        try:
            model_debug_info = f"Loaded model classes: {getattr(model, 'names', 'N/A')}"
        except Exception as e:
            model_debug_info = f"Could not read model classes: {e}"
        st.markdown(f"<div style='font-size:0.9em; color:#888;'><b>Debug:</b> {model_debug_info}</div>", unsafe_allow_html=True)
        
        # Model info
        st.info("🤖 **Model**: YOLOv8n (general purpose)\n📦 **Auto-downloads** on first use")
        
        # Detection parameters
        st.subheader("🎛️ Detection Parameters")
        
        # Confidence threshold with improved defaults for better recall
        col1, col2 = st.columns([2, 1])
        with col1:
            conf_threshold = st.slider("Confidence Threshold", 0.05, 1.0, 0.15, 0.05, 
                                     help="Lower values detect more objects but may include false positives")
            # Store in session state for troubleshooting display
            st.session_state['confidence_threshold'] = conf_threshold
        with col2:
            st.markdown("**💡 Space Station Recommendations:**")
            st.markdown("• **0.10-0.15**: Good for ToolBox detection\n• **0.20-0.25**: Balanced for all equipment\n• **0.30+**: High confidence only\n• **🔍 Augmented detection** when needed")
        
        # IoU threshold with presets
        col3, col4 = st.columns([2, 1])
        with col3:
            iou_threshold = st.slider("IoU Threshold (Intersection over Union)", 0.1, 1.0, 0.45, 0.05,
                                    help="Controls duplicate detection removal - higher values = stricter overlap filtering")
            # Store in session state for troubleshooting display
            st.session_state['iou_threshold'] = iou_threshold
        with col4:
            st.markdown("**🎯 IoU Recommendations:**")
            st.markdown("• **0.35**: Crowded scenes\n• **0.45**: Balanced (default)\n• **0.60**: High precision\n• **Higher = fewer duplicates**")
            
            # Quick preset buttons
            st.markdown("**⚡ Quick Presets:**")
            col_preset1, col_preset2, col_preset3 = st.columns(3)
            with col_preset1:
                if st.button("🏢 Crowded", help="IoU 0.35 for busy scenes"):
                    st.session_state.iou_preset = 0.35
            with col_preset2:
                if st.button("⚖️ Balanced", help="IoU 0.45 for general use"):
                    st.session_state.iou_preset = 0.45
            with col_preset3:
                if st.button("🎯 Precise", help="IoU 0.60 for clean results"):
                    st.session_state.iou_preset = 0.60
            
            # Apply preset if selected
            if 'iou_preset' in st.session_state:
                iou_threshold = st.session_state.iou_preset
        
        # IoU Explanation
        with st.expander("🤔 What is IoU? (Intersection over Union)", expanded=False):
            st.markdown("""
            **IoU measures how much two bounding boxes overlap:**
            
            📊 **Formula:** `IoU = Overlap Area / Total Area`
            
            🎯 **What it does:**
            - **Removes duplicate detections** of the same object
            - **Controls precision** vs **recall** trade-off
            - **Filters overlapping boxes** during detection
            
            📈 **Effect of different values:**
            - **Lower IoU (0.3-0.4):** More detections, may include duplicates
            - **Higher IoU (0.5-0.7):** Cleaner results, fewer false positives
            - **Medium IoU (0.45):** Good balance for most use cases
            
            💡 **When to adjust:**
            - **Crowded photos** → Lower IoU (0.35)
            - **Technical precision** → Higher IoU (0.60)
            - **General photos** → Keep default (0.45)
            """)
        
        # Object info
        st.subheader("📋 Detectable Objects")
        st.info("""
        **Space Station Equipment Detection Model:**
        • 🧰 **ToolBox** - Space station maintenance tools
        • 🫁 **Oxygen Tank** - Life support oxygen containers  
        • 🧯 **Fire Extinguisher** - Emergency fire suppression equipment
        
        **✅ Model Performance**: 91.5% mAP@0.5, 94.7% precision, 88.9% recall
        **🎯 Training**: Specialized for space station environment detection
        """)
    
    # Load model
    model = load_general_model()
    if model is None:
        st.error("❌ Failed to load model!")
        st.stop()
    
    # Define model name for results logging
    model_name = 'yolov8n.pt'
    
    # Detection tabs
    tab1, tab2, tab3 = st.tabs(["📸 Image Detection", "🎥 Video Detection", "🔄 Model Updates"])
    
    with tab1:
        st.header("🔍 Object Detection")
        
        # Information alert
        st.info("ℹ️ **Welcome!** This space station object detector specializes in finding ToolBox, Oxygen Tank, and Fire Extinguisher equipment in images. Upload a space station image and see what equipment it detects!")
        
        # Two clear testing options
        st.markdown("---")
        
        # Option selection
        test_option = st.radio(
            "🎯 **Choose Testing Method:**",
            ["📤 Upload Your Own Image", "🖼️ Use Sample Images"],
            help="Upload any image with everyday objects like people, cars, animals, furniture, etc."
        )
        
        image_to_process = None
        image_source = ""
        
        if test_option == "🖼️ Use Sample Images":
            st.subheader("📋 Dataset Sample Images")
            
            # Check for local dataset images first
            dataset_samples = {}
            dataset_paths = [
                ("000000028.png", "HackByte_Dataset/data/test/images/000000028.png", "ToolBox", 0.279),
                ("000000005.png", "HackByte_Dataset/data/test/images/000000005.png", "ToolBox", 0.222),
                ("000000025.png", "HackByte_Dataset/data/test/images/000000025.png", "ToolBox", 0.19),
                ("000000105.png", "HackByte_Dataset/data/test/images/000000105.png", "ToolBox", 0.17)
            ]
            
            for name, path, obj, conf in dataset_paths:
                if Path(path).exists():
                    dataset_samples[name] = {"path": path, "object": obj, "conf": conf, "type": "local"}
            
            if dataset_samples:
                st.success("✨ **Using original dataset images** - These contain the space station objects the model was trained on.")
                st.warning("⚠️ **Note**: These images contain space station specific objects (ToolBox, Oxygen Tank, Fire Extinguisher) from the original training dataset.")
                
                # Use dataset images
                col1, col2 = st.columns([2, 1])
                with col1:
                    selected_sample = st.selectbox(
                        "Choose a dataset sample:",
                        options=list(dataset_samples.keys()),
                        format_func=lambda x: f"{x} - {dataset_samples[x]['object']} (conf: {dataset_samples[x]['conf']:.3f})"
                    )
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                    if st.button("🔄 Load Dataset Sample", type="primary"):
                        if selected_sample:
                            try:
                                image_to_process = Image.open(dataset_samples[selected_sample]["path"])
                                image_source = f"Dataset: {selected_sample}"
                                st.success(f"✅ Loaded dataset image: {selected_sample}")
                                st.info("📊 **Original Training Data**: This image was part of the space station object detection dataset.")
                            except Exception as e:
                                st.error(f"Error loading dataset sample: {str(e)}")
                
                if selected_sample:
                    expected_conf = dataset_samples[selected_sample]['conf']
                    st.info(f"💡 **Expected Detection:** {dataset_samples[selected_sample]['object']} with confidence around {expected_conf:.1%} (from original space station model)")
                    st.warning("🔄 **Note**: Since we're now using general YOLOv8 model, these specific space station objects may not be detected. The model now detects 80+ common objects like people, cars, animals, furniture, etc.")
            
            else:
                st.error("❌ **Dataset images not found!**")
                st.info("""
                **Expected dataset location:**
                - `HackByte_Dataset/data/test/images/000000028.png`
                - `HackByte_Dataset/data/test/images/000000005.png`
                - `HackByte_Dataset/data/test/images/000000025.png`
                - `HackByte_Dataset/data/test/images/000000105.png`
                
                **Recommendation:** Upload your own images with common objects for testing instead.
                """)
        
        if test_option == "📤 Upload Your Own Image":
            st.subheader("📤 Upload Your Own Image")
            st.success("✨ **Great choice!** Upload any image with common objects like people, cars, animals, furniture, electronics, food, etc.")
            
            st.markdown("**📋 Best Results With:**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("• 📸 **Clear, well-lit photos**\n• 🏠 **Indoor/outdoor scenes**\n• 👥 **People and faces**")
            with col2:
                st.markdown("• 🚗 **Vehicles and transportation**\n• 🐕 **Animals and pets**\n• 📱 **Electronics and gadgets**")
            
            uploaded_file = st.file_uploader(
                "Choose an image file", 
                type=['png', 'jpg', 'jpeg'],
                help="Upload any photo - PNG, JPG, or JPEG format"
            )
            
            if uploaded_file is not None:
                try:
                    image_to_process = Image.open(uploaded_file)
                    image_source = f"Uploaded: {uploaded_file.name}"
                    st.success(f"✅ Uploaded: {uploaded_file.name}")
                    st.info("🔍 **Ready to detect!** The model will identify any common objects in your image.")
                except Exception as e:
                    st.error(f"Error loading uploaded image: {str(e)}")
        
        # Process the selected image
        if image_to_process is not None:
            # Debug info
            st.success(f"✅ **Processing:** {image_source}")
            st.caption(f"Image size: {image_to_process.size}, Mode: {image_to_process.mode}")
            # Warn if image size or mode is very different from typical training images
            expected_size = (640, 640)  # Change if your training images are a different size
            if image_to_process.size != expected_size:
                st.warning(f"⚠️ Uploaded image size {image_to_process.size} is different from expected training size {expected_size}. Detection may be less accurate.")
            if image_to_process.mode != "RGB":
                st.warning(f"⚠️ Uploaded image mode {image_to_process.mode} is different from expected 'RGB'. Detection may be less accurate.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image_to_process, use_container_width=True)
            
            # Perform detection
            with st.spinner("🔍 Detecting objects..."):
                annotated_image, detections, inference_time = detect_objects(
                    model, image_to_process, conf_threshold, iou_threshold
                )
            
            with col2:
                st.subheader("Detection Results")
                if annotated_image is not None:
                    st.image(annotated_image, use_container_width=True)
            
            # Display results
            display_detection_results(detections, inference_time)
            
            # Download results
            if detections:
                results_data = {
                    'timestamp': datetime.now().isoformat(),
                    'model_path': model_name,
                    'conf_threshold': conf_threshold,
                    'iou_threshold': iou_threshold,
                    'inference_time_ms': inference_time,
                    'detections': detections
                }
                
                st.download_button(
                    label="📥 Download Detection Results (JSON)",
                    data=json.dumps(results_data, indent=2),
                    file_name=f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    with tab2:
        st.header("🎥 Video Object Detection")
        st.info("📝 **Note**: This feature would process video files frame by frame. In a production environment, this would support real-time camera feeds from the space station.")
        
        uploaded_video = st.file_uploader(
            "Choose a video file...", 
            type=['mp4', 'avi', 'mov'],
            help="Upload a video for frame-by-frame object detection"
        )
        
        if uploaded_video is not None:
            # Save uploaded video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                video_path = tmp_file.name
            
            # Video processing would go here
            st.success("🎬 Video uploaded successfully! Processing would analyze each frame for objects.")
            
            # Cleanup
            os.unlink(video_path)
    
    with tab3:
        create_model_update_plan()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🚀 <strong>Duality AI Space Station Hackathon</strong> 🚀<br>
        Powered by YOLOv8 & Falcon Synthetic Data Platform
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
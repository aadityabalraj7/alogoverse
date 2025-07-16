#!/bin/bash

# Duality AI Space Station Hackathon - Quick Start Script
# This script demonstrates the complete pipeline with sample data

echo "🚀 Duality AI Space Station Hackathon - Quick Start Demo"
echo "========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Step 1: Check if environment is set up
print_info "Step 1: Checking environment setup..."

if conda info --envs | grep -q "EDU"; then
    print_status "EDU environment found"
    print_info "Activating EDU environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate EDU
else
    print_warning "EDU environment not found. Setting up environment..."
    if [ -f "ENV_SETUP/setup_env.sh" ]; then
        chmod +x ENV_SETUP/setup_env.sh
        ./ENV_SETUP/setup_env.sh
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate EDU
    else
        print_error "Environment setup script not found!"
        exit 1
    fi
fi

# Step 2: Generate sample data if dataset is empty
print_info "Step 2: Checking dataset..."

if [ ! -d "dataset/train/images" ] || [ -z "$(ls -A dataset/train/images 2>/dev/null)" ]; then
    print_warning "No training data found. Generating sample dataset..."
    python scripts/generate_sample_data.py --train_samples 50 --val_samples 10 --test_samples 10
    print_status "Sample dataset generated"
else
    print_status "Dataset found"
fi

# Step 3: Test training pipeline (quick demo)
print_info "Step 3: Testing training pipeline (quick demo - 5 epochs)..."

python scripts/train.py --config config.yaml --model yolov8n.pt --epochs 5

if [ $? -eq 0 ]; then
    print_status "Training pipeline test completed"
else
    print_error "Training pipeline test failed"
    exit 1
fi

# Step 4: Test prediction pipeline
print_info "Step 4: Testing prediction pipeline..."

# Check if model exists
if [ -f "models/space_station_best.pt" ]; then
    # Create test images if they don't exist
    if [ ! -d "dataset/test_images" ]; then
        python scripts/generate_sample_data.py --test_images_only
    fi
    
    # Test single image prediction
    TEST_IMAGE=$(find dataset/test_images -name "*.jpg" | head -1)
    if [ -n "$TEST_IMAGE" ]; then
        print_info "Testing single image prediction..."
        python scripts/predict.py --model models/space_station_best.pt --source "$TEST_IMAGE" --single
        print_status "Single image prediction test completed"
    fi
    
    # Test batch prediction
    print_info "Testing batch prediction..."
    python scripts/predict.py --model models/space_station_best.pt --source dataset/test_images
    print_status "Batch prediction test completed"
    
else
    print_warning "Trained model not found. Skipping prediction tests."
fi

# Step 5: Test evaluation pipeline
print_info "Step 5: Testing evaluation pipeline..."

if [ -f "models/space_station_best.pt" ]; then
    python scripts/evaluate.py --model models/space_station_best.pt --config config.yaml
    print_status "Evaluation pipeline test completed"
else
    print_warning "Trained model not found. Skipping evaluation test."
fi

# Step 6: Display results
print_info "Step 6: Displaying results..."

echo ""
echo "📊 Quick Start Demo Results:"
echo "=================================="

if [ -d "results" ]; then
    echo "📁 Results directory: results/"
    find results -type f -name "*.log" | head -3 | while read logfile; do
        echo "   📄 Log: $logfile"
    done
    
    if [ -d "results/predictions" ]; then
        echo "   🎯 Predictions: results/predictions/"
    fi
    
    if [ -d "results/evaluation" ]; then
        echo "   📈 Evaluation: results/evaluation/"
    fi
fi

if [ -d "models" ]; then
    echo "🤖 Model directory: models/"
    find models -name "*.pt" | while read model; do
        echo "   🧠 Model: $model"
    done
fi

echo ""
echo "🚀 Next Steps:"
echo "=============="
echo "1. Replace sample data with official hackathon dataset"
echo "2. Run full training: python scripts/train.py --config config.yaml --epochs 100"
echo "3. Evaluate performance: python scripts/evaluate.py --model models/space_station_best.pt --config config.yaml"
echo "4. Launch interactive app: streamlit run app/space_station_detector.py"
echo "5. Submit your results following hackathon guidelines"

echo ""
echo "📚 Documentation:"
echo "=================="
echo "• README.md - Complete project documentation"
echo "• config.yaml - Model configuration"
echo "• scripts/ - Training, prediction, and evaluation scripts"
echo "• app/ - Interactive Streamlit application"

echo ""
print_status "Quick start demo completed successfully! 🎉"
echo "Ready for the Duality AI Space Station Hackathon! 🚀" 
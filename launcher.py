#!/usr/bin/env python3
"""
Cool vs Uncool Image Classifier - Launcher Script

This script provides an easy way to launch the different components of the system.
"""

import os
import sys
import subprocess
import argparse

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'PIL', 'numpy', 'matplotlib', 
        'sklearn', 'tqdm', 'tkinter'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'sklearn':
                import sklearn
            elif package == 'tkinter':
                import tkinter
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install them with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def launch_labeling_app():
    """Launch the interactive labeling application"""
    print("🚀 Starting Cool vs Uncool Labeling App...")
    print("📝 Use this to label your images as cool or uncool")
    print("⌨️  Keyboard shortcuts: C=cool, U=uncool, ←→=navigate, Space=skip")
    print("-" * 60)
    
    try:
        subprocess.run([sys.executable, "labeling_app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching labeling app: {e}")
    except FileNotFoundError:
        print("❌ labeling_app.py not found!")

def launch_predictor():
    """Launch the prediction application"""
    print("🚀 Starting Cool vs Uncool Predictor...")
    print("🔮 Use this to test your trained model on new images")
    print("-" * 60)
    
    try:
        subprocess.run([sys.executable, "predictor.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching predictor: {e}")
    except FileNotFoundError:
        print("❌ predictor.py not found!")

def train_model():
    """Train the model using command line"""
    print("🚀 Starting Model Training...")
    print("🧠 This will train your CNN model using labeled data")
    print("-" * 60)
    
    try:
        subprocess.run([sys.executable, "train.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error training model: {e}")
    except FileNotFoundError:
        print("❌ train.py not found!")

def show_status():
    """Show current status of the system"""
    print("📊 Cool vs Uncool Classifier Status")
    print("=" * 50)
    
    # Check for labels file
    labels_file = "labels.json"
    if os.path.exists(labels_file):
        import json
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        
        cool_count = sum(1 for l in labels.values() if l == 1)
        uncool_count = sum(1 for l in labels.values() if l == 0)
        total_labeled = len(labels)
        
        print(f"📝 Labeled Images: {total_labeled}")
        print(f"   🔥 Cool: {cool_count}")
        print(f"   ❌ Uncool: {uncool_count}")
    else:
        print("📝 No labeled data found")
    
    # Check for trained model
    model_file = "cool_uncool_model.pth"
    if os.path.exists(model_file):
        print(f"🧠 Trained Model: ✅ Found ({model_file})")
    else:
        print("🧠 Trained Model: ❌ Not found")
    
    # Check for images
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    image_files = [f for f in os.listdir('.') if f.lower().endswith(image_extensions)]
    print(f"🖼️  Total Images: {len(image_files)}")
    
    print("\n💡 Next Steps:")
    if not os.path.exists(labels_file):
        print("   1. Run 'python launcher.py label' to start labeling images")
    elif not os.path.exists(model_file):
        print("   1. Run 'python launcher.py train' to train your model")
    else:
        print("   1. Run 'python launcher.py predict' to test your model")
        print("   2. Run 'python launcher.py train' to retrain with more data")

def main():
    parser = argparse.ArgumentParser(description='Cool vs Uncool Image Classifier Launcher')
    parser.add_argument('command', nargs='?', choices=['label', 'predict', 'train', 'status', 'check'],
                       help='Command to run')
    
    args = parser.parse_args()
    
    print("🎨 Cool vs Uncool Image Classifier")
    print("=" * 50)
    
    if args.command == 'check' or not args.command:
        if not check_dependencies():
            return
        if not args.command:
            show_status()
    
    elif args.command == 'label':
        if check_dependencies():
            launch_labeling_app()
    
    elif args.command == 'predict':
        if check_dependencies():
            launch_predictor()
    
    elif args.command == 'train':
        if check_dependencies():
            train_model()
    
    elif args.command == 'status':
        show_status()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Cool vs Uncool Image Classifier Training Script

This script provides a command-line interface for training the cool/uncool classifier
using labeled data from the labeling app.
"""

import os
import json
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional dependency
from model import CoolUncoolCNN, CoolUncoolTrainer, CoolUncoolDataset, get_data_transforms, plot_training_history

def load_labeled_data(image_directory):
    """Load labeled data from the JSON file"""
    labels_file = os.path.join(image_directory, "labels.json")
    
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    # Get image paths and labels
    image_paths = []
    label_list = []
    
    for filename, label in labels.items():
        image_path = os.path.join(image_directory, filename)
        if os.path.exists(image_path):
            image_paths.append(image_path)
            label_list.append(label)
    
    return image_paths, label_list

def create_data_loaders(image_paths, labels, batch_size=16, train_split=0.8):
    """Create training and validation data loaders"""
    # Get transforms
    train_transform, val_transform = get_data_transforms()
    
    # Create dataset
    dataset = CoolUncoolDataset(image_paths, labels, transform=train_transform)
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def evaluate_model(model, val_loader, device):
    """Evaluate model and return detailed metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
    
    return all_predictions, all_labels, accuracy

def plot_confusion_matrix(predictions, labels, save_path="confusion_matrix.png"):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['Uncool', 'Cool'])
    plt.yticks(tick_marks, ['Uncool', 'Cool'])
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train Cool vs Uncool Image Classifier')
    parser.add_argument('--data_dir', type=str, default='/Users/jovgav/Desktop/MLClass/vogue_images',
                       help='Directory containing images and labels.json')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Fraction of data to use for training')
    parser.add_argument('--model_name', type=str, default='cool_uncool_model',
                       help='Name for the saved model')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu, cuda, or auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Data directory: {args.data_dir}")
    
    # Load labeled data
    try:
        image_paths, labels = load_labeled_data(args.data_dir)
        print(f"Loaded {len(image_paths)} labeled images")
        
        # Check label distribution
        cool_count = sum(1 for l in labels if l == 1)
        uncool_count = sum(1 for l in labels if l == 0)
        print(f"Cool images: {cool_count}")
        print(f"Uncool images: {uncool_count}")
        
        if len(image_paths) < 10:
            print("Warning: Very few labeled images. Consider labeling more images for better results.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the labeling app first to create labels.json")
        return
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        image_paths, labels, 
        batch_size=args.batch_size, 
        train_split=args.train_split
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model and trainer
    model = CoolUncoolCNN(num_classes=2)
    trainer = CoolUncoolTrainer(model, device)
    
    # Adjust learning rate if specified
    if args.lr != 0.001:
        trainer.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print(f"\nStarting training for {args.epochs} epochs...")
    train_losses, train_accs, val_losses, val_accs = trainer.train(
        train_loader, val_loader, epochs=args.epochs
    )
    
    # Save model
    model_path = os.path.join(args.data_dir, f"{args.model_name}.pth")
    trainer.save_model(model_path)
    
    # Evaluate model
    print("\nEvaluating model...")
    predictions, true_labels, accuracy = evaluate_model(model, val_loader, device)
    
    print(f"\nFinal Results:")
    print(f"Validation Accuracy: {accuracy:.2%}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, 
                              target_names=['Uncool', 'Cool']))
    
    # Plot results
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    plot_confusion_matrix(predictions, true_labels)
    
    print(f"\nModel saved to: {model_path}")
    print("Training complete!")

if __name__ == "__main__":
    main()

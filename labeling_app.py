import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import os
import json
import random
from model import CoolUncoolCNN, CoolUncoolTrainer, CoolUncoolDataset, get_data_transforms
import torch
from torch.utils.data import DataLoader, random_split
import threading

class CoolUncoolLabelingApp:
    """Interactive GUI for labeling images as cool/uncool"""
    
    def __init__(self, root, image_directory):
        self.root = root
        self.image_directory = image_directory
        self.labels_file = os.path.join(image_directory, "labels.json")
        
        print(f"Image directory: {image_directory}")  # Debug print
        
        # Load existing labels or create new
        self.labels = self.load_labels()
        
        # Get all image files
        self.image_files = self.get_image_files()
        print(f"Found {len(self.image_files)} image files")  # Debug print
        
        # Current image index
        self.current_index = 0
        
        # Setup GUI
        self.setup_gui()
        
        # Load first image
        self.load_current_image()
        
        # Update stats
        self.update_stats()
    
    def get_image_files(self):
        """Get all image files from directory"""
        extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
        files = []
        for file in os.listdir(self.image_directory):
            if file.lower().endswith(extensions):
                files.append(file)
        return sorted(files)
    
    def load_labels(self):
        """Load existing labels from JSON file"""
        if os.path.exists(self.labels_file):
            with open(self.labels_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_labels(self):
        """Save labels to JSON file"""
        with open(self.labels_file, 'w') as f:
            json.dump(self.labels, f, indent=2)
    
    def setup_gui(self):
        """Setup the GUI components"""
        self.root.title("Cool vs Uncool Image Labeling Tool")
        self.root.geometry("1000x700")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Stats frame
        stats_frame = ttk.LabelFrame(main_frame, text="Labeling Progress", padding="5")
        stats_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.stats_label = ttk.Label(stats_frame, text="")
        self.stats_label.grid(row=0, column=0, sticky=tk.W)
        
        # Image frame
        image_frame = ttk.LabelFrame(main_frame, text="Current Image", padding="5")
        image_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Image label
        self.image_label = ttk.Label(image_frame)
        self.image_label.grid(row=0, column=0, padx=10, pady=10)
        
        # Image info
        self.image_info_label = ttk.Label(image_frame, text="")
        self.image_info_label.grid(row=1, column=0, pady=(0, 10))
        
        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Navigation buttons
        nav_frame = ttk.Frame(control_frame)
        nav_frame.grid(row=0, column=0, sticky=tk.W)
        
        ttk.Button(nav_frame, text="← Previous", command=self.previous_image).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(nav_frame, text="Next →", command=self.next_image).grid(row=0, column=1, padx=5)
        ttk.Button(nav_frame, text="Random", command=self.random_image).grid(row=0, column=2, padx=5)
        
        # Label buttons
        label_frame = ttk.Frame(control_frame)
        label_frame.grid(row=0, column=1, sticky=tk.E)
        
        self.cool_button = ttk.Button(label_frame, text="🔥 COOL", command=lambda: self.label_image(1), 
                                    style="Cool.TButton")
        self.cool_button.grid(row=0, column=0, padx=5)
        
        self.uncool_button = ttk.Button(label_frame, text="❌ UNCOOL", command=lambda: self.label_image(0),
                                      style="Uncool.TButton")
        self.uncool_button.grid(row=0, column=1, padx=5)
        
        self.skip_button = ttk.Button(label_frame, text="Skip", command=self.skip_image)
        self.skip_button.grid(row=0, column=2, padx=5)
        
        # Training frame
        training_frame = ttk.LabelFrame(main_frame, text="Model Training", padding="5")
        training_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Training controls
        train_control_frame = ttk.Frame(training_frame)
        train_control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(train_control_frame, text="Train Model", command=self.start_training).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(train_control_frame, text="Load Model", command=self.load_model).grid(row=0, column=1, padx=5)
        ttk.Button(train_control_frame, text="Test Model", command=self.test_model).grid(row=0, column=2, padx=5)
        
        # Training progress
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(training_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.training_status_label = ttk.Label(training_frame, text="Ready to train")
        self.training_status_label.grid(row=2, column=0, pady=(5, 0))
        
        # Configure styles
        style = ttk.Style()
        style.configure("Cool.TButton", foreground="green", font=("Arial", 12, "bold"))
        style.configure("Uncool.TButton", foreground="red", font=("Arial", 12, "bold"))
        
        # Keyboard bindings
        self.root.bind('<Left>', lambda e: self.previous_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<space>', lambda e: self.skip_image())
        self.root.bind('<c>', lambda e: self.label_image(1))
        self.root.bind('<u>', lambda e: self.label_image(0))
        self.root.bind('<r>', lambda e: self.random_image())
        
        # Focus on root for keyboard events
        self.root.focus_set()
    
    def load_current_image(self):
        """Load and display the current image"""
        if not self.image_files:
            messagebox.showwarning("No Images", "No image files found in the directory!")
            return
        
        current_file = self.image_files[self.current_index]
        image_path = os.path.join(self.image_directory, current_file)
        
        print(f"Loading image: {current_file}")  # Debug print
        
        try:
            # Load and resize image
            image = Image.open(image_path)
            print(f"Original image size: {image.size}")  # Debug print
            
            # Calculate display size (max 600x400)
            max_width, max_height = 600, 400
            image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            print(f"Resized image size: {image.size}")  # Debug print
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            print("PhotoImage created successfully")  # Debug print
            
            # Update image label
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep a reference
            
            # Update image info
            current_label = self.labels.get(current_file, "Unlabeled")
            label_text = "🔥 COOL" if current_label == 1 else "❌ UNCOOL" if current_label == 0 else "❓ UNLABELED"
            
            info_text = f"File: {current_file}\nLabel: {label_text}\nImage {self.current_index + 1} of {len(self.image_files)}"
            self.image_info_label.configure(text=info_text)
            
            # Update button states
            if current_label == 1:
                self.cool_button.configure(state="disabled")
                self.uncool_button.configure(state="normal")
            elif current_label == 0:
                self.cool_button.configure(state="normal")
                self.uncool_button.configure(state="disabled")
            else:
                self.cool_button.configure(state="normal")
                self.uncool_button.configure(state="normal")
            
            print("Image loaded and displayed successfully")  # Debug print
                
        except Exception as e:
            print(f"Error loading image: {str(e)}")  # Debug print
            messagebox.showerror("Error", f"Could not load image: {str(e)}")
    
    def label_image(self, label):
        """Label the current image"""
        current_file = self.image_files[self.current_index]
        self.labels[current_file] = label
        self.save_labels()
        self.load_current_image()
        self.update_stats()
    
    def skip_image(self):
        """Skip the current image"""
        self.next_image()
    
    def previous_image(self):
        """Go to previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
    
    def next_image(self):
        """Go to next image"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_current_image()
    
    def random_image(self):
        """Go to a random image"""
        self.current_index = random.randint(0, len(self.image_files) - 1)
        self.load_current_image()
    
    def update_stats(self):
        """Update the statistics display"""
        total_images = len(self.image_files)
        labeled_images = len([f for f in self.image_files if f in self.labels])
        cool_images = len([f for f in self.image_files if self.labels.get(f) == 1])
        uncool_images = len([f for f in self.image_files if self.labels.get(f) == 0])
        
        stats_text = f"Total: {total_images} | Labeled: {labeled_images} | Cool: {cool_images} | Uncool: {uncool_images}"
        self.stats_label.configure(text=stats_text)
    
    def start_training(self):
        """Start training the model in a separate thread"""
        labeled_files = [f for f in self.image_files if f in self.labels]
        
        if len(labeled_files) < 10:
            messagebox.showwarning("Insufficient Data", 
                                 "Please label at least 10 images before training!")
            return
        
        # Start training in separate thread
        training_thread = threading.Thread(target=self.train_model)
        training_thread.daemon = True
        training_thread.start()
    
    def train_model(self):
        """Train the model"""
        try:
            # Update status
            self.root.after(0, lambda: self.training_status_label.configure(text="Preparing data..."))
            
            # Prepare data
            labeled_files = [f for f in self.image_files if f in self.labels]
            image_paths = [os.path.join(self.image_directory, f) for f in labeled_files]
            labels = [self.labels[f] for f in labeled_files]
            
            # Get transforms
            train_transform, val_transform = get_data_transforms()
            
            # Create dataset
            dataset = CoolUncoolDataset(image_paths, labels, transform=train_transform)
            
            # Split dataset
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            # Create model and trainer
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = CoolUncoolCNN(num_classes=2)
            trainer = CoolUncoolTrainer(model, device)
            
            # Update status
            self.root.after(0, lambda: self.training_status_label.configure(text="Training model..."))
            
            # Train model
            train_losses, train_accs, val_losses, val_accs = trainer.train(train_loader, val_loader, epochs=20)
            
            # Save model
            model_path = os.path.join(self.image_directory, "cool_uncool_model.pth")
            trainer.save_model(model_path)
            
            # Update status
            final_acc = val_accs[-1] if val_accs else 0
            self.root.after(0, lambda: self.training_status_label.configure(
                text=f"Training complete! Final accuracy: {final_acc:.1f}%"))
            
            # Show completion message
            self.root.after(0, lambda: messagebox.showinfo("Training Complete", 
                f"Model training completed!\nFinal validation accuracy: {final_acc:.1f}%\nModel saved to: {model_path}"))
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            self.root.after(0, lambda: self.training_status_label.configure(text=error_msg))
            self.root.after(0, lambda: messagebox.showerror("Training Error", error_msg))
    
    def load_model(self):
        """Load a trained model"""
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")]
        )
        
        if model_path:
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = CoolUncoolCNN(num_classes=2)
                trainer = CoolUncoolTrainer(model, device)
                trainer.load_model(model_path)
                
                messagebox.showinfo("Model Loaded", f"Model loaded successfully from {model_path}")
                self.training_status_label.configure(text="Model loaded and ready for testing")
                
            except Exception as e:
                messagebox.showerror("Load Error", f"Could not load model: {str(e)}")
    
    def test_model(self):
        """Test the model on current image"""
        # This would require loading a trained model and making predictions
        messagebox.showinfo("Test Model", "Model testing feature coming soon!")

def main():
    """Main function to run the application"""
    # Get image directory
    image_directory = "/Users/jovgav/Desktop/MLClass/vogue_images"
    
    if not os.path.exists(image_directory):
        print(f"Directory {image_directory} does not exist!")
        return
    
    # Create and run the application
    root = tk.Tk()
    app = CoolUncoolLabelingApp(root, image_directory)
    
    # Add instructions
    instructions = """
Keyboard Shortcuts:
- Left/Right arrows: Navigate images
- C: Label as COOL
- U: Label as UNCOOL  
- Space: Skip image
- R: Random image
    """
    
    print(instructions)
    
    root.mainloop()

if __name__ == "__main__":
    main()

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import os
import torch
import torchvision.transforms as transforms
from model import CoolUncoolCNN, CoolUncoolTrainer
import numpy as np

class CoolUncoolPredictor:
    """GUI for predicting cool/uncool on new images"""
    
    def __init__(self, root, model_path=None):
        self.root = root
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup GUI
        self.setup_gui()
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def setup_gui(self):
        """Setup the GUI components"""
        self.root.title("Cool vs Uncool Image Predictor")
        self.root.geometry("800x600")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Model frame
        model_frame = ttk.LabelFrame(main_frame, text="Model", padding="5")
        model_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Model controls
        ttk.Button(model_frame, text="Load Model", command=self.load_model_dialog).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(model_frame, text="Load Image", command=self.load_image_dialog).grid(row=0, column=1, padx=5)
        ttk.Button(model_frame, text="Predict", command=self.predict_image).grid(row=0, column=2, padx=5)
        
        self.model_status_label = ttk.Label(model_frame, text="No model loaded")
        self.model_status_label.grid(row=1, column=0, columnspan=3, pady=(5, 0))
        
        # Image frame
        image_frame = ttk.LabelFrame(main_frame, text="Image", padding="5")
        image_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Image display
        self.image_label = ttk.Label(image_frame)
        self.image_label.grid(row=0, column=0, padx=10, pady=10)
        
        # Image info
        self.image_info_label = ttk.Label(image_frame, text="No image loaded")
        self.image_info_label.grid(row=1, column=0, pady=(0, 10))
        
        # Prediction frame
        prediction_frame = ttk.LabelFrame(main_frame, text="Prediction", padding="5")
        prediction_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Prediction results
        self.prediction_label = ttk.Label(prediction_frame, text="No prediction yet", font=("Arial", 14, "bold"))
        self.prediction_label.grid(row=0, column=0, pady=10)
        
        self.confidence_label = ttk.Label(prediction_frame, text="")
        self.confidence_label.grid(row=1, column=0, pady=(0, 10))
        
        # Batch prediction frame
        batch_frame = ttk.LabelFrame(main_frame, text="Batch Prediction", padding="5")
        batch_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(batch_frame, text="Predict Folder", command=self.predict_folder).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(batch_frame, text="Sort Images", command=self.sort_images).grid(row=0, column=1, padx=5)
        
        self.batch_status_label = ttk.Label(batch_frame, text="")
        self.batch_status_label.grid(row=1, column=0, columnspan=2, pady=(5, 0))
        
        # Current image path
        self.current_image_path = None
    
    def load_model_dialog(self):
        """Open dialog to load model"""
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")]
        )
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load a trained model"""
        try:
            self.model = CoolUncoolCNN(num_classes=2)
            trainer = CoolUncoolTrainer(self.model, self.device)
            trainer.load_model(model_path)
            
            self.model_path = model_path
            self.model_status_label.configure(text=f"Model loaded: {os.path.basename(model_path)}")
            
            messagebox.showinfo("Model Loaded", f"Model loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load model: {str(e)}")
            self.model_status_label.configure(text="Error loading model")
    
    def load_image_dialog(self):
        """Open dialog to load image"""
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp"), ("All files", "*.*")]
        )
        
        if image_path:
            self.load_image(image_path)
    
    def load_image(self, image_path):
        """Load and display an image"""
        try:
            # Load and resize image
            image = Image.open(image_path)
            
            # Calculate display size (max 500x400)
            max_width, max_height = 500, 400
            image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update image label
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep a reference
            
            # Update image info
            filename = os.path.basename(image_path)
            info_text = f"File: {filename}\nPath: {image_path}"
            self.image_info_label.configure(text=info_text)
            
            self.current_image_path = image_path
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")
    
    def predict_image(self):
        """Predict cool/uncool for the current image"""
        if not self.model:
            messagebox.showwarning("No Model", "Please load a model first!")
            return
        
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please load an image first!")
            return
        
        try:
            # Load and preprocess image
            image = Image.open(self.current_image_path).convert('RGB')
            
            # Transform for model
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Update prediction display
            if predicted_class == 1:
                prediction_text = "🔥 COOL"
                color = "green"
            else:
                prediction_text = "❌ UNCOOL"
                color = "red"
            
            self.prediction_label.configure(text=prediction_text, foreground=color)
            self.confidence_label.configure(text=f"Confidence: {confidence:.1%}")
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Could not make prediction: {str(e)}")
    
    def predict_folder(self):
        """Predict cool/uncool for all images in a folder"""
        if not self.model:
            messagebox.showwarning("No Model", "Please load a model first!")
            return
        
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if not folder_path:
            return
        
        try:
            # Get all image files
            extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(extensions)]
            
            if not image_files:
                messagebox.showwarning("No Images", "No image files found in the selected folder!")
                return
            
            # Create results directory
            results_dir = os.path.join(folder_path, "predictions")
            cool_dir = os.path.join(results_dir, "cool")
            uncool_dir = os.path.join(results_dir, "uncool")
            
            os.makedirs(cool_dir, exist_ok=True)
            os.makedirs(uncool_dir, exist_ok=True)
            
            # Transform for model
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.model.eval()
            cool_count = 0
            uncool_count = 0
            
            self.batch_status_label.configure(text="Processing images...")
            self.root.update()
            
            for i, filename in enumerate(image_files):
                try:
                    # Load and preprocess image
                    image_path = os.path.join(folder_path, filename)
                    image = Image.open(image_path).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0).to(self.device)
                    
                    # Make prediction
                    with torch.no_grad():
                        outputs = self.model(image_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        predicted_class = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0][predicted_class].item()
                    
                    # Copy to appropriate folder
                    if predicted_class == 1 and confidence > 0.6:  # Only high confidence cool
                        import shutil
                        shutil.copy2(image_path, os.path.join(cool_dir, filename))
                        cool_count += 1
                    elif predicted_class == 0 and confidence > 0.6:  # Only high confidence uncool
                        import shutil
                        shutil.copy2(image_path, os.path.join(uncool_dir, filename))
                        uncool_count += 1
                    
                    # Update progress
                    progress = (i + 1) / len(image_files) * 100
                    self.batch_status_label.configure(text=f"Processed {i+1}/{len(image_files)} images ({progress:.1f}%)")
                    self.root.update()
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue
            
            # Show results
            self.batch_status_label.configure(text=f"Complete! Cool: {cool_count}, Uncool: {uncool_count}")
            messagebox.showinfo("Batch Prediction Complete", 
                f"Processed {len(image_files)} images\nCool: {cool_count}\nUncool: {uncool_count}\nResults saved to: {results_dir}")
            
        except Exception as e:
            messagebox.showerror("Batch Prediction Error", f"Could not process folder: {str(e)}")
            self.batch_status_label.configure(text="Error processing folder")
    
    def sort_images(self):
        """Sort images in the current directory based on predictions"""
        if not self.model:
            messagebox.showwarning("No Model", "Please load a model first!")
            return
        
        # Use the same directory as the model
        if not self.model_path:
            messagebox.showwarning("No Model Path", "Please load a model first!")
            return
        
        base_dir = os.path.dirname(self.model_path)
        
        try:
            # Get all image files
            extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
            image_files = [f for f in os.listdir(base_dir) 
                          if f.lower().endswith(extensions)]
            
            if not image_files:
                messagebox.showwarning("No Images", "No image files found in the directory!")
                return
            
            # Create sorted directories
            sorted_dir = os.path.join(base_dir, "sorted_images")
            cool_dir = os.path.join(sorted_dir, "cool")
            uncool_dir = os.path.join(sorted_dir, "uncool")
            uncertain_dir = os.path.join(sorted_dir, "uncertain")
            
            os.makedirs(cool_dir, exist_ok=True)
            os.makedirs(uncool_dir, exist_ok=True)
            os.makedirs(uncertain_dir, exist_ok=True)
            
            # Transform for model
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.model.eval()
            cool_count = 0
            uncool_count = 0
            uncertain_count = 0
            
            self.batch_status_label.configure(text="Sorting images...")
            self.root.update()
            
            for i, filename in enumerate(image_files):
                try:
                    # Load and preprocess image
                    image_path = os.path.join(base_dir, filename)
                    image = Image.open(image_path).convert('RGB')
                    image_tensor = transform(image).unsqueeze(0).to(self.device)
                    
                    # Make prediction
                    with torch.no_grad():
                        outputs = self.model(image_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        predicted_class = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0][predicted_class].item()
                    
                    # Copy to appropriate folder based on confidence
                    import shutil
                    if confidence > 0.7:  # High confidence
                        if predicted_class == 1:
                            shutil.copy2(image_path, os.path.join(cool_dir, filename))
                            cool_count += 1
                        else:
                            shutil.copy2(image_path, os.path.join(uncool_dir, filename))
                            uncool_count += 1
                    else:  # Low confidence - uncertain
                        shutil.copy2(image_path, os.path.join(uncertain_dir, filename))
                        uncertain_count += 1
                    
                    # Update progress
                    progress = (i + 1) / len(image_files) * 100
                    self.batch_status_label.configure(text=f"Sorted {i+1}/{len(image_files)} images ({progress:.1f}%)")
                    self.root.update()
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue
            
            # Show results
            self.batch_status_label.configure(text=f"Sorting complete! Cool: {cool_count}, Uncool: {uncool_count}, Uncertain: {uncertain_count}")
            messagebox.showinfo("Image Sorting Complete", 
                f"Sorted {len(image_files)} images\nCool: {cool_count}\nUncool: {uncool_count}\nUncertain: {uncertain_count}\nResults saved to: {sorted_dir}")
            
        except Exception as e:
            messagebox.showerror("Sorting Error", f"Could not sort images: {str(e)}")
            self.batch_status_label.configure(text="Error sorting images")

def main():
    """Main function to run the predictor"""
    root = tk.Tk()
    
    # Try to load model from default location
    default_model_path = "/Users/jovgav/Desktop/MLClass/vogue_images/cool_uncool_model.pth"
    
    app = CoolUncoolPredictor(root, default_model_path if os.path.exists(default_model_path) else None)
    
    root.mainloop()

if __name__ == "__main__":
    main()

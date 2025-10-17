#!/usr/bin/env python3
"""
Command-line version of the labeling app
This works without GUI and displays images using matplotlib
"""

import os
import json
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

class CommandLineLabeler:
    def __init__(self):
        self.image_dir = os.getcwd()
        self.labels_file = "labels.json"
        self.labels = self.load_labels()
        self.image_files = self.get_image_files()
        
        if not self.image_files:
            print("No image files found!")
            return
        
        self.current_index = 0
        self.run()
    
    def get_image_files(self):
        """Get all image files"""
        extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
        files = []
        for file in os.listdir(self.image_dir):
            if file.lower().endswith(extensions):
                files.append(file)
        return sorted(files)
    
    def load_labels(self):
        """Load existing labels"""
        if os.path.exists(self.labels_file):
            try:
                with open(self.labels_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading labels: {e}")
        return {}
    
    def save_labels(self):
        """Save labels"""
        try:
            with open(self.labels_file, 'w') as f:
                json.dump(self.labels, f, indent=2)
            print("Labels saved!")
        except Exception as e:
            print(f"Error saving labels: {e}")
    
    def show_image(self):
        """Display current image"""
        if not self.image_files:
            return
        
        current_file = self.image_files[self.current_index]
        image_path = os.path.join(self.image_dir, current_file)
        
        try:
            # Load and display image
            img = mpimg.imread(image_path)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.title(f"Image {self.current_index + 1} of {len(self.image_files)}\nFile: {current_file}")
            plt.axis('off')
            
            # Show current label
            current_label = self.labels.get(current_file, "Unlabeled")
            label_text = "🔥 COOL" if current_label == 1 else "❌ UNCOOL" if current_label == 0 else "❓ UNLABELED"
            plt.figtext(0.5, 0.02, f"Current Label: {label_text}", ha='center', fontsize=12)
            
            plt.tight_layout()
            plt.show()
            
            return current_file
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def show_stats(self):
        """Show labeling statistics"""
        total = len(self.image_files)
        labeled = len(self.labels)
        cool = sum(1 for l in self.labels.values() if l == 1)
        uncool = sum(1 for l in self.labels.values() if l == 0)
        
        print(f"\n📊 Statistics:")
        print(f"   Total images: {total}")
        print(f"   Labeled: {labeled}")
        print(f"   Cool: {cool}")
        print(f"   Uncool: {uncool}")
        print(f"   Remaining: {total - labeled}")
    
    def run(self):
        """Main labeling loop"""
        print("🎨 Cool vs Uncool Command-Line Labeler")
        print("=" * 50)
        
        while True:
            self.show_stats()
            
            # Show current image
            current_file = self.show_image()
            if not current_file:
                break
            
            print(f"\nCurrent image: {current_file}")
            print("Commands:")
            print("  c = Label as COOL")
            print("  u = Label as UNCOOL")
            print("  n = Next image")
            print("  p = Previous image")
            print("  r = Random image")
            print("  s = Skip (next)")
            print("  q = Quit")
            
            while True:
                command = input("\nEnter command: ").lower().strip()
                
                if command == 'c':
                    self.labels[current_file] = 1
                    self.save_labels()
                    print("✅ Labeled as COOL")
                    break
                elif command == 'u':
                    self.labels[current_file] = 0
                    self.save_labels()
                    print("✅ Labeled as UNCOOL")
                    break
                elif command == 'n':
                    if self.current_index < len(self.image_files) - 1:
                        self.current_index += 1
                    else:
                        print("Already at last image")
                    break
                elif command == 'p':
                    if self.current_index > 0:
                        self.current_index -= 1
                    else:
                        print("Already at first image")
                    break
                elif command == 'r':
                    self.current_index = random.randint(0, len(self.image_files) - 1)
                    break
                elif command == 's':
                    if self.current_index < len(self.image_files) - 1:
                        self.current_index += 1
                    break
                elif command == 'q':
                    print("Goodbye!")
                    return
                else:
                    print("Invalid command. Try again.")

def main():
    try:
        labeler = CommandLineLabeler()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

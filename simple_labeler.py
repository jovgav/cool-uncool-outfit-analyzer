#!/usr/bin/env python3
"""
Super Simple Image Labeler
Shows image info and lets you label without matplotlib windows
"""

import os
import json
import random

class SimpleLabeler:
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
            print("✅ Labels saved!")
        except Exception as e:
            print(f"Error saving labels: {e}")
    
    def show_stats(self):
        """Show labeling statistics"""
        total = len(self.image_files)
        labeled = len(self.labels)
        cool = sum(1 for l in self.labels.values() if l == 1)
        uncool = sum(1 for l in self.labels.values() if l == 0)
        
        print(f"\n📊 PROGRESS: {labeled}/{total} labeled | Cool: {cool} | Uncool: {uncool}")
    
    def run(self):
        """Main labeling loop"""
        print("\n" + "="*60)
        print("🎨 COOL vs UNCOOL IMAGE LABELER")
        print("="*60)
        print("I'll show you image filenames and you can label them.")
        print("You can open the images in Preview/Finder to see them.")
        print("="*60)
        
        while True:
            self.show_stats()
            
            if self.current_index >= len(self.image_files):
                print("✅ All images processed!")
                break
            
            current_file = self.image_files[self.current_index]
            
            # Show current label
            current_label = self.labels.get(current_file, "UNLABELED")
            if current_label == 1:
                label_text = "COOL"
            elif current_label == 0:
                label_text = "UNCOOL"
            else:
                label_text = "UNLABELED"
            
            print(f"\n📸 IMAGE {self.current_index + 1} of {len(self.image_files)}")
            print(f"📁 File: {current_file}")
            print(f"🏷️  Current Label: {label_text}")
            
            # Open image in default viewer
            image_path = os.path.join(self.image_dir, current_file)
            print(f"🖼️  Opening image in Preview...")
            os.system(f"open '{image_path}'")
            
            print(f"\nCOMMANDS:")
            print(f"  c = Label as COOL")
            print(f"  u = Label as UNCOOL")
            print(f"  n = Next image")
            print(f"  p = Previous image")
            print(f"  r = Random image")
            print(f"  s = Skip (next)")
            print(f"  q = Quit and save")
            
            while True:
                command = input(f"\nEnter command (c/u/n/p/r/s/q): ").lower().strip()
                
                if command == 'c':
                    self.labels[current_file] = 1
                    self.save_labels()
                    print("🔥 LABELED AS COOL!")
                    self.current_index += 1
                    break
                elif command == 'u':
                    self.labels[current_file] = 0
                    self.save_labels()
                    print("❌ LABELED AS UNCOOL!")
                    self.current_index += 1
                    break
                elif command == 'n':
                    if self.current_index < len(self.image_files) - 1:
                        self.current_index += 1
                        print("➡️ Next image")
                    else:
                        print("⚠️ Already at last image")
                    break
                elif command == 'p':
                    if self.current_index > 0:
                        self.current_index -= 1
                        print("⬅️ Previous image")
                    else:
                        print("⚠️ Already at first image")
                    break
                elif command == 'r':
                    self.current_index = random.randint(0, len(self.image_files) - 1)
                    print("🎲 Random image")
                    break
                elif command == 's':
                    if self.current_index < len(self.image_files) - 1:
                        self.current_index += 1
                        print("⏭️ Skipped to next image")
                    break
                elif command == 'q':
                    print("💾 Saving and exiting...")
                    self.save_labels()
                    print("👋 Goodbye!")
                    return
                else:
                    print("❌ Invalid command. Try: c, u, n, p, r, s, or q")

def main():
    try:
        labeler = SimpleLabeler()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted. Saving labels...")
        if 'labeler' in locals():
            labeler.save_labels()
        print("👋 Goodbye!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# Directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'data', 'dataset')

# Characters to generate (0-9, A-Z)
chars = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Image size for CNN input
IMG_SIZE = 32
NUM_SAMPLES_PER_CLASS = 50

def get_font():
    # Try common Windows fonts. Fall back to default if none found.
    font_paths = [
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\ariblk.ttf",
        "C:\\Windows\\Fonts\\tahoma.ttf",
        "C:\\Windows\\Fonts\\consola.ttf",
        "C:\\Windows\\Fonts\\verdana.ttf"
    ]
    for path in font_paths:
        if os.path.exists(path):
            return path
    return None

def generate_dataset():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        
    font_path = get_font()
    print(f"Using font path: {font_path if font_path else 'Default (may be small)'}")
        
    for char in chars:
        char_dir = os.path.join(DATASET_DIR, char)
        if not os.path.exists(char_dir):
            os.makedirs(char_dir)
            
        print(f"Generating data for character '{char}'...")
        for i in range(NUM_SAMPLES_PER_CLASS):
            # Start with a white background or slightly noisy background
            bg_color = random.randint(220, 255)
            img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=bg_color)
            draw = ImageDraw.Draw(img)
            
            # Select random font size
            font_size = random.randint(20, 28)
            
            if font_path:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                except Exception:
                    font = ImageFont.load_default()
            else:
                font = ImageFont.load_default()
                
            # Random text color (dark)
            text_color = random.randint(0, 50)
            
            # Calculate position to center text with random offset
            bbox = draw.textbbox((0, 0), char, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            x_offset = (IMG_SIZE - text_w) // 2 + random.randint(-4, 4)
            y_offset = (IMG_SIZE - text_h) // 2 + random.randint(-4, 4)
            
            draw.text((x_offset, y_offset), char, font=font, fill=text_color)
            
            # Data Augmentation
            # 1. Random Rotation
            angle = random.randint(-15, 15)
            img = img.rotate(angle, fillcolor=bg_color)
            
            # 2. Add some noise or blur occasionally
            if random.random() > 0.5:
                img = img.filter(ImageFilter.BoxBlur(radius=random.uniform(0.1, 0.8)))
                
            # 3. Add pixel noise
            if random.random() > 0.5:
                pixels = np.array(img)
                noise = np.random.normal(loc=0, scale=10, size=pixels.shape)
                pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(pixels)
            
            file_path = os.path.join(char_dir, f"{char}_{i}.jpg")
            img.save(file_path)

if __name__ == "__main__":
    print(f"Generating synthetic dataset to: {DATASET_DIR}")
    generate_dataset()
    print("Dataset generation complete.")

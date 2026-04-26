import cv2

def preprocess_image(image_path_or_img):
    if isinstance(image_path_or_img, str):
        img = cv2.imread(image_path_or_img)
        if img is None:
            raise ValueError(f"Could not read image: {image_path_or_img}")
    else:
        img = image_path_or_img.copy()
        
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Noise reduction using Bilateral Filter
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # 3. Edge detection
    edged = cv2.Canny(bfilter, 30, 200)
    
    return img, gray, bfilter, edged

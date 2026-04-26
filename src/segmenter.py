import cv2
import numpy as np

def segment_characters(plate_img):
    # Depending on the plate, characters might be black on white or white on black.
    # Usually standard plates are black text on white bg.
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_images = []
    bounding_boxes = []
    
    plate_h, plate_w = plate_img.shape[:2]
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        # Filter for typical character dimensions
        if 0.15 < aspect_ratio < 1.0 and h > plate_h * 0.4 and h < plate_h * 0.95:
            bounding_boxes.append((x, y, w, h))
            
    # Sort bounding boxes from left to right
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    
    for (x, y, w, h) in bounding_boxes:
        # Add some padding
        pad = 2
        y1 = max(0, y - pad)
        y2 = min(plate_h, y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(plate_w, x + w + pad)
        
        roi = thresh[y1:y2, x1:x2]
        squared = make_square(roi)
        # Resize to (32, 32)
        resized = cv2.resize(squared, (32, 32), interpolation=cv2.INTER_AREA)
        # Apply slight dilation to make lines clearer
        kernel = np.ones((2,2), np.uint8)
        resized = cv2.dilate(resized, kernel, iterations=1)
        
        char_images.append((resized, (x, y, w, h)))
        
    return char_images

def make_square(img):
    h, w = img.shape
    size = max(h, w) + 4 # add padding to square
    new_img = np.zeros((size, size), np.uint8)
    
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    
    new_img[y_off:y_off+h, x_off:x_off+w] = img
    return new_img

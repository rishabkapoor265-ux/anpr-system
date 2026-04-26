import cv2
import numpy as np

def detect_plate(img, edged):
    # Find contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = keypoints[0] if len(keypoints) == 2 else keypoints[1]
    
    # Sort contours based on area and keep top 10
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    location = None
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # If contour has 4 points, we assume we have found the plate
        if len(approx) == 4:
            location = approx
            break
            
    if location is None:
        return None, None
        
    # Masking the detected plate
    mask = np.zeros(img.shape[:2], np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    
    # Crop the image
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_plate = img[x1:x2+1, y1:y2+1]
    
    return cropped_plate, location

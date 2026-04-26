import os
import cv2
import numpy as np
import tensorflow as tf

from .preprocess import preprocess_image
from .plate_detector import detect_plate
from .segmenter import segment_characters

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'char_cnn.h5')
CLASS_INDICES_PATH = os.path.join(BASE_DIR, 'class_indices.txt')

class LicensePlateRecognizer:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        
        self.classes = []
        if os.path.exists(CLASS_INDICES_PATH):
            with open(CLASS_INDICES_PATH, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
        else:
            # Fallback to standard alphanumeric
            self.classes = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def predict_char(self, char_img):
        img_array = np.array(char_img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0) # Batch dim
        img_array = np.expand_dims(img_array, axis=-1) # Channel dim
        
        predictions = self.model.predict(img_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        return self.classes[class_idx]

    def recognize(self, image_path_or_img):
        # 1. Preprocess
        original_img, gray, bfilter, edged = preprocess_image(image_path_or_img)
        
        # 2. Detect Plate
        plate_img, plate_contour = detect_plate(original_img, edged)
        if plate_img is None:
            return None, "No plate detected", original_img, edged, None
            
        # 3. Segment Characters
        char_data = segment_characters(plate_img)
        if not char_data:
            return plate_img, "Failed to segment characters", original_img, edged, plate_contour
            
        # 4. Recognize Characters
        plate_text = ""
        labeled_plate = plate_img.copy()
        if len(labeled_plate.shape) == 2:
            labeled_plate = cv2.cvtColor(labeled_plate, cv2.COLOR_GRAY2BGR)
            
        for char_img, (x, y, w, h) in char_data:
            char_pred = self.predict_char(char_img)
            plate_text += char_pred
            # Draw box and text on the plate image
            cv2.rectangle(labeled_plate, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(labeled_plate, char_pred, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        return labeled_plate, plate_text, original_img, edged, plate_contour

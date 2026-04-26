import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, 'test_images')

if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)

# Create a synthetic car rear image (very crude)
# Background: dark gray car body
img = np.ones((400, 600, 3), dtype=np.uint8) * 100

# Draw some car features (tail lights)
cv2.rectangle(img, (50, 150), (150, 250), (0, 0, 200), -1)
cv2.rectangle(img, (450, 150), (550, 250), (0, 0, 200), -1)

# Draw a license plate (white background, black text)
plate_x1, plate_y1 = 200, 200
plate_x2, plate_y2 = 400, 260
cv2.rectangle(img, (plate_x1, plate_y1), (plate_x2, plate_y2), (255, 255, 255), -1)

# Add some noise to make it realistic for the edge detector
noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
img = cv2.add(img, noise)

# Ensure the plate is white again
cv2.rectangle(img, (plate_x1, plate_y1), (plate_x2, plate_y2), (255, 255, 255), -1)

# Write text to the plate (e.g., "MH12AB1234")
text = "MH12AB34"
font = cv2.FONT_HERSHEY_SIMPLEX
# Use putText to write characters. We space them out slightly.
char_x = plate_x1 + 10
for char in text:
    cv2.putText(img, char, (char_x, plate_y2 - 15), font, 0.8, (0, 0, 0), 2)
    char_x += 22

# Save the image
save_path = os.path.join(TEST_DIR, 'sample_car.jpg')
cv2.imwrite(save_path, img)
print(f"Saved test image to {save_path}")

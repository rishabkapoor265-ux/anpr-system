# 🚗 AI-Powered Automatic Number Plate Recognition (ANPR)

A premium, modular ANPR system built with Python, OpenCV, and TensorFlow. This system localized vehicle license plates and performs character recognition using a custom-trained Convolutional Neural Network (CNN).

![ANPR Dashboard Mockup](https://raw.githubusercontent.com/username/repo/main/assets/readme_banner.png) *(Note: Replace with actual screenshot)*

## ✨ Features
- **Real-time Plate Localization**: Uses OpenCV contour analysis and edge detection.
- **Character Segmentation**: Sophisticated logic to isolate individual characters from the plate.
- **CNN-Based Recognition**: Custom-trained model to accurately identify alphanumeric characters.
- **Premium UI Dashboard**: Built with Streamlit, featuring glassmorphism design and smooth animations.
- **Modular Pipeline**: Clean separation of preprocessing, detection, segmentation, and recognition.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/anpr-system.git
   cd anpr-system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

## 🧠 Model Training
If you want to train the model from scratch:
1. **Build the dataset**: `python src/build_dataset.py`
2. **Train the model**: `python src/model.py`

The trained model will be saved as `char_cnn.h5`.

## 🛠️ Tech Stack
- **Backend**: Python, OpenCV, TensorFlow/Keras
- **Frontend**: Streamlit, Custom CSS
- **Processing**: NumPy, Scipy, PIL

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

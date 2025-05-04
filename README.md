# Arabic Sign Language Recognition AI

A computer vision project that recognizes Arabic sign language gestures in real-time using hand tracking technology.

## 📖 Overview

This project utilizes machine learning and computer vision to recognize and translate Arabic Sign Language gestures in real-time. The system tracks hand movements through a webcam feed, extracts key landmarks, and classifies them into corresponding Arabic letters.

## ✨ Features

- Real-time Arabic sign language detection
- Support for all 28 Arabic alphabet letters
- Custom dataset creation tool
- Hand landmark tracking using MediaPipe
- Machine learning classification with Random Forest algorithm
- Visual feedback with hand tracking visualization

## 📁 Project Structure

```
├── collect_imgs.py        # Data collection script for creating image dataset
├── create_dataset.py      # Processes images to extract hand landmarks
├── data.pickle            # Processed dataset with hand landmarks features
├── inference_classifier.py # Real-time sign language recognition script
├── model.p                # Trained Random Forest classifier model
└── train_classifier.py    # Script for training the classification model
```

## 🔍 How It Works

1. **Data Collection**: The system uses `collect_imgs.py` to capture images of hand gestures through a webcam
2. **Feature Extraction**: `create_dataset.py` processes these images using MediaPipe Hands to extract hand landmarks
3. **Training**: `train_classifier.py` trains a Random Forest classifier on the extracted features
4. **Recognition**: `inference_classifier.py` provides real-time recognition of Arabic sign language gestures

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- OpenCV
- MediaPipe
- scikit-learn
- NumPy
- PIL (Pillow)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/arabic-sign-language-recognition.git
   cd arabic-sign-language-recognition
   ```

2. Install the required dependencies:
   ```bash
   pip install opencv-python mediapipe scikit-learn numpy pillow
   ```

3. Make sure you have an Arabic supporting font (e.g., arial.ttf) in your system or update the font path in the inference script

## 📊 Usage

### Creating Your Own Dataset

1. Run the data collection script:
   ```bash
   python collect_imgs.py
   ```
   - Press 'q' when ready to start collecting images for each class
   - The script will collect 300 images per class for 4 classes by default (adjust these parameters in the script if needed)

2. Process the collected images to extract hand landmarks:
   ```bash
   python create_dataset.py
   ```
   - This will create a `data.pickle` file containing the processed features

### Training the Model

Train the classifier on your dataset:
```bash
python train_classifier.py
```
The script will:
- Split the data into training and testing sets
- Train a Random Forest classifier
- Display the classification accuracy
- Save the trained model as `model.p`

### Running the Recognition System

Execute the real-time recognition script:
```bash
python inference_classifier.py
```
- Position your hand in front of the webcam
- The system will display the recognized Arabic letter in real-time

## 🔧 Customization

- To recognize more gestures: Update `number_of_classes` in `collect_imgs.py` and corresponding labels in `inference_classifier.py`
- To improve accuracy: Collect more training data or adjust the machine learning model in `train_classifier.py`
- To change the visualization: Modify the drawing parameters in `inference_classifier.py`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
made by : Bouderbala Mohamed Islem , Abed Ahmed.

## 🙏 Acknowledgments

- [MediaPipe](https://github.com/google/mediapipe) for the hand tracking technology
- [scikit-learn](https://scikit-learn.org/) for the machine learning implementation

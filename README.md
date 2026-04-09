```markdown
# Flood Detection Using Convolutional Neural Network

## Overview
This repository contains a flood detection system that utilizes a Convolutional Neural Network (CNN) to classify aerial and satellite images as either "Flooded" or "Non-Flooded." The model is integrated into a web application using Flask, providing a straightforward interface for image upload and prediction. 

## Dataset
The dataset is a combination of multiple sources from Kaggle, ensuring variation in geography, time of day, and weather conditions. 
* Total images: 1016 (508 flooded, 508 non-flooded).
* Sources include the Aerial Imagery Dataset (FloodNet Challenge), Flood Segmentation Dataset, and Louisiana flood 2016 dataset.

## Technology Stack
* **Frontend:** HTML, CSS, JavaScript
* **Backend:** Python, Flask
* **Deep Learning Framework:** PyTorch (Model saved as `cnn_model.pth`)

## Model Architecture
The CNN consists of the following layers:
1. **Conv2D (conv1):** 3 input channels, 16 filters, 3x3 kernel, stride 1, padding 1, ReLU activation. Output: (16, 128, 128).
2. **Conv2D (conv2):** 16 input channels, 32 filters, 3x3 kernel, stride 1, padding 1, ReLU activation. Output: (32, 128, 128).
3. **MaxPooling2D (pool):** 2x2 kernel, stride 2. Output: (32, 64, 64).
4. **Flatten Layer:** Reshapes the tensor for the fully connected layers.
5. **Dense (fc1):** 128 units, ReLU activation.
6. **Dense (fc2):** 2 units (binary classification output).

The model uses Cross-Entropy Loss and the Adam Optimizer with a learning rate of 0.001.

## Performance Metrics
Based on the latest training run (10 epochs):
* **Test Accuracy:** 92.00%
* **Test Loss:** 0.2455

## Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd flooddetection1
   ```

2. **Install dependencies:**
   Ensure you have Python installed. Install the required packages using the `requirements.txt` file.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask application:**
   ```bash
   python app.py
   ```

4. **Access the application:**
   Open a web browser and navigate to `http://127.0.0.1:5000/`.

## Usage
1. Click the "Choose a file" button on the web interface.
2. Select an aerial or satellite image (.jpg, .png).
3. Click "Detect" to process the image.
4. The system will display the image and the predicted status: "Flooded" or "Non-Flooded."

```
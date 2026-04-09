<div align="center">
  <h1>🌊 Flood Detection AI Web Service</h1>
  <p>An end-to-end Machine Learning Engineering project that integrates a custom PyTorch Convolutional Neural Network with a sleek, interactive Flask API.</p>
</div>

<br />

<div align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5" />
  <img src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white" alt="CSS3" />
</div>

<br />

## 🔍 Overview
This project bridges the gap between deep learning and software engineering. It utilizes a custom-built Convolutional Neural Network (CNN) trained to classify geographical topology and satellite imagery as either **"Flooded"** or **"Safe/Clear"**. 

Instead of remaining an enclosed Jupyter Notebook, the trained model (`cnn_model.pth`) is deployed locally via a dynamic, REST-driven Flask web application, featuring an intuitive, modern Glassmorphism User Interface.

## 🧠 Neural Network Architecture
Built dynamically using PyTorch, the CNN follows a highly optimized pipeline for real-time inference:
- **Data Augmentation:** Utilizes `torchvision.transforms` for dynamic image reduction ($128 \times 128$), normalization, and stochastic horizontal flipping.
- **Conv2d Layer 1:** 3 input channels $\rightarrow$ 16 output channels (3x3 kernel, padding=1).
- **Conv2d Layer 2:** 16 input channels $\rightarrow$ 32 output channels (3x3 kernel, padding=1).
- **Pooling Pipeline:** Uses $2 \times 2$ Max Pooling, effectively halving spatial dimensions.
- **Dense Classifier:** The final tensors are flattened statically ($32 \times 64 \times 64$) into a dense Linear layer of 128 inputs before compressing into a final Binary Classification node.

## 📊 Training & Performance Metrics
The model was recently retrained on an augmented `FinalFloodDataset` (1,016 images sourced from Kaggle's FloodNet and Louisiana 2016 incidents). It ran utilizing **Nvidia CUDA GPU Acceleration**.

**Final Benchmark Summary:**
- **Test Accuracy:** `92.00%` 🏆
- **Test Loss:** `0.2455`

<details>
<summary><b>Click to expand full Epoch Training Logs</b></summary>
<br>

```text
Using device: cuda
Epoch [1/10], Loss: 0.9873 | Validation Loss: 0.3840
Epoch [2/10], Loss: 0.3604 | Validation Loss: 0.3336
Epoch [3/10], Loss: 0.2905 | Validation Loss: 0.4061
Epoch [4/10], Loss: 0.2241 | Validation Loss: 0.2920
Epoch [5/10], Loss: 0.2345 | Validation Loss: 0.2938
Epoch [6/10], Loss: 0.1742 | Validation Loss: 0.3501
Epoch [7/10], Loss: 0.1575 | Validation Loss: 0.4039
Epoch [8/10], Loss: 0.1362 | Validation Loss: 0.2323
Epoch [9/10], Loss: 0.0954 | Validation Loss: 0.2735
Epoch [10/10], Loss: 0.0726 | Validation Loss: 0.2919
```

</details>

## 🎨 User Interface Enhancements
The frontend was rebuilt to incorporate industry-standard UX/UI practices:
- **Glassmorphism:** Employs CSS backdrop-filters to create a frosted glass aesthetic above an animated background.
- **Asynchronous Fetch API:** Image analysis runs without page reloads using a JavaScript `fetch()` pipe connected directly to Flask's `/predict` route.
- **Dynamic Feedback:** Analysis renders conditionally colored output (`#ff4d4d` for Flood, `#34d399` for Safe) for instant user comprehension.

## 🛠️ Installation & Setup (Local Environment)

**1. Clone the repository & download tracking files:**
*(Note: Because this repo tracks the 67MB `.pth` file, you must have Git LFS installed).*
```bash
git clone https://github.com/Udit-Kumar-k/FloodDetection1.git
cd FloodDetection1
git lfs pull
```

**2. Install runtime dependencies:**
```bash
pip install -r requirements.txt
```

**3. Launch the Server:**
*(Ensure the application isn't attempting to retrain by verifying `cnntrain.py` execution guards).*
```bash
python app.py
```

**4. Interact:**
Open your browser and navigate to `http://127.0.0.1:5000/`. Upload any geographical image to witness real-time neural inference.
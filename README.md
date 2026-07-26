# 🏭 DefectForge AI: Steel Defect Analyzer

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Analytics-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-89.17%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Computer Vision and Deep Learning system designed to automate surface defect detection, classification, and severity assessment in hot-rolled steel manufacturing. Built with PyTorch and deployed via an interactive Streamlit Web Dashboard (**DefectForge AI**).

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Defect Taxonomy & Severity Matrix](#-defect-taxonomy--severity-matrix)
- [Model Architecture](#-model-architecture)
- [Performance & Results](#-performance--results)
- [Web Dashboard UI](#-web-dashboard-ui)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
  - [1. Launch Web Application](#1-launch-web-application)
  - [2. Train Model](#2-train-model)
  - [3. Evaluate Model](#3-evaluate-model)
  - [4. Predict via CLI](#4-predict-via-cli)
- [Future Enhancements](#-future-enhancements)
- [Project Context & Acknowledgments](#-project-context--acknowledgments)
- [License](#-license)

---

## 📸 Overview

In industrial steel rolling processes, surface quality directly impacts structural integrity and material reliability. Traditional manual quality control is labor-intensive, subject to observer fatigue, and prone to classification errors.

**DefectForge AI** addresses this challenge by providing an automated real-time inspection pipeline using **SteelCNN**—a deep convolutional neural network trained on Northeastern University's **NEU-DET** dataset. The system identifies 6 primary defect classes, calculates prediction confidence, assesses risk severity, and rejects out-of-distribution or ambiguous images with a confidence threshold safety mechanism.

---

## ✨ Key Features

- 🎯 **Multi-Class Classification**: Identifies 6 distinct hot-rolled steel surface defect categories.
- 🛡️ **Confidence Safeguard**: Implements a 70% confidence threshold to filter ambiguous samples or non-steel surfaces.
- ⚠️ **Severity Risk Rating**: Categorizes detected defects into **Low**, **Medium**, or **High** severity levels for rapid quality triage.
- 📊 **Interactive Streamlit UI**: Multi-page dashboard featuring real-time image analysis, per-class accuracy metrics, interactive Plotly charts, and defect reference cards.
- ⚙️ **Modular & Scalable**: Clean separation between data pipelines (`dataset.py`), model architecture (`model.py`), training loops (`train.py`), evaluation (`evaluate.py`), and deployment (`app.py`).

---

## 🔬 Defect Taxonomy & Severity Matrix

The model classifies defects based on the **NEU-DET Surface Defect Database**:

| Defect Class | Abbr. | Description | Severity | Model Accuracy |
| :--- | :---: | :--- | :---: | :---: |
| **Crazing** | `Cr` | Fine network of surface cracks resulting from thermal or mechanical stress | **Medium** | **100.0%** |
| **Inclusion** | `In` | Foreign non-metallic particles embedded into the steel matrix | **High** | **60.0%** |
| **Patches** | `Pa` | Localized discolored or irregular surface region | **Low** | **98.33%** |
| **Pitted Surface** | `Ps` | Small cavities or pits caused by corrosion or chemical reactions | **Medium** | **91.67%** |
| **Rolled-in Scale** | `Rs` | Oxide scale mechanically pressed into surface during hot rolling | **Medium** | **100.0%** |
| **Scratches** | `Sc` | Linear abrasive grooves caused by mechanical contact or tooling | **Low** | **85.0%** |

---

## 🧠 Model Architecture

The core classifier is **SteelCNN**, a custom PyTorch Convolutional Neural Network optimized for single-channel and RGB industrial surface images resized to $224 \times 224$:

```
Input Image (3 x 224 x 224)
  │
  ├──► Conv2D(3 ➔ 16, kernel=3, padding=1) ──► ReLU ──► MaxPool2d(2x2)  # [16 x 112 x 112]
  │
  ├──► Conv2D(16 ➔ 32, kernel=3, padding=1) ──► ReLU ──► MaxPool2d(2x2) # [32 x 56 x 56]
  │
  ├──► Flatten() [32 * 56 * 56 = 100,352]
  │
  ├──► Linear(100352 ➔ 128) ──► ReLU
  │
  └──► Linear(128 ➔ 6) ──► Output Logits / Softmax
```

### Hyperparameters & Settings
- **Optimizer**: Adam ($\text{lr} = 0.001$)
- **Loss Function**: Cross-Entropy Loss
- **Input Resolution**: $224 \times 224$ pixels
- **Batch Size**: 8 (Training), 16 (Validation)

---

## 📈 Performance & Results

Tested on **360 validation images** from the NEU-DET benchmark dataset:

- **Overall Accuracy**: **89.17%** (321 / 360 correct predictions)
- **Top Performing Classes**: Crazing (100%), Rolled-in Scale (100%), Patches (98.33%)

```json
{
  "overall_accuracy": 89.17,
  "total_images": 360,
  "correct_predictions": 321,
  "class_accuracy": {
    "crazing": 100.0,
    "rolled-in_scale": 100.0,
    "patches": 98.33,
    "pitted_surface": 91.67,
    "scratches": 85.0,
    "inclusion": 60.0
  }
}
```

---

## 💻 Web Dashboard UI

The application UI is built with **Streamlit** and customized with CSS styled in **JetBrains Mono** for an industrial terminal aesthetic. Key pages include:

1. 📊 **Dashboard**: High-level KPIs, overall accuracy metrics, per-class horizontal bar chart, distribution donut chart, and defect class reference cards.
2. 🔍 **Analyze Defect**: Image uploader with real-time inference, confidence breakdown, probability distribution graph, severity rating, and low-confidence warning system.
3. 📉 **Model Performance**: Evaluation overview and per-class performance stats.
4. ℹ️ **About**: Project information, dataset details, and architecture specifications.

---

## 📁 Repository Structure

```
steel-defect-analyzer/
├── app/
│   └── app.py                     # Streamlit web application dashboard
├── src/
│   ├── dataset.py                 # PyTorch Dataset implementation for NEU-DET
│   ├── model.py                   # SteelCNN network definition
│   ├── train.py                   # Model training script
│   ├── evaluate.py                # Model validation script (generates accuracy_results.json)
│   ├── predict.py                 # CLI inference script
│   └── utils.py                   # Helper utilities
├── models/
│   └── steel_cnn.pth              # Saved PyTorch model weights (~51 MB)
├── data/
│   └── NEU-DET/                   # Dataset directory (train & validation splits)
├── notebooks/
│   └── exploration.ipynb          # Exploratory Data Analysis notebook
├── outputs/
│   ├── accuracy_results.json      # Structured evaluation output
│   ├── Internship_Project_Report.md # Formal internship report
│   ├── images/                    # UI assets & export images
│   └── logs/                      # Training & run logs
├── .streamlit/
│   └── config.toml                # Streamlit configuration settings
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation
```

---

## 🛠️ Getting Started

### Prerequisites

- **Python 3.8+**
- **pip** package manager
- Recommended: NVIDIA GPU with CUDA support for accelerated training (CPU supported for inference).

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/steel-defect-analyzer.git
   cd steel-defect-analyzer
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate

   # On Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Launch Web Application

To run the interactive Streamlit UI dashboard:

```bash
streamlit run app/app.py
```

Then open your browser and navigate to `http://localhost:8501`.

### 2. Train Model

To train or retrain the `SteelCNN` model:

```bash
python src/train.py
```

*The trained model weights will be saved to `models/steel_cnn.pth`.*

### 3. Evaluate Model

To run model evaluation against the validation dataset and generate performance JSON metrics:

```bash
python src/evaluate.py
```

### 4. Predict via CLI

To run inference on a single image from command line:

```bash
python src/predict.py --image path/to/steel_sample.jpg
```

---

## 🔮 Future Enhancements

- [ ] **Transfer Learning**: Evaluate deeper architectures (e.g., ResNet-50, EfficientNet-B0) to improve challenging classes like *Inclusion*.
- [ ] **Defect Localization**: Implement bounding-box object detection (YOLOv8 / Faster R-CNN) or segmentation (U-Net) to pinpoint defect coordinates on large steel plates.
- [ ] **Edge AI Deployment**: Export model to ONNX runtime or TensorRT for low-latency edge deployment in rolling mills.
- [ ] **Live Camera Feed**: Integrate real-time industrial camera streaming via OpenCV.

---

## 👨‍💻 Project Context & Acknowledgments

This project was developed as part of an industrial internship:

- **Organization**: **Tata Steel Ltd.**
- **Intern**: Rishav Singh (Siksha 'O' Anusandhan University)
- **Dataset**: NEU-DET (Northeastern University Surface Defect Database)

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

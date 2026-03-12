# Handwritten Digit & Character Recognition using CNN

A complete Deep Learning project for recognizing handwritten digits (0-9) and uppercase letters (A-Z) using Convolutional Neural Networks trained on the MNIST and EMNIST datasets. Features a FastAPI backend and a modern Next.js frontend with canvas drawing capabilities.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![Next.js](https://img.shields.io/badge/Next.js-16+-black.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Training the Models](#-training-the-models)
- [Running the Application](#-running-the-application)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Technologies Used](#-technologies-used)

## ✨ Features

- **Neural Path Visualization**: Cinematic 10-second staged animation of the forward pass
- **Explainable AI UI**: Visualize connections, weights, and signal pulses in real-time
- **Staged Inference Flow**: Phased animation: Input → Connections → Pulses → Decision Path
- **Interactive Canvas**: Draw digits/characters directly in the browser
- **Image Upload**: Upload photos of handwritten digits or characters
- **Real-time Predictions**: Instant recognition with confidence scores
- **Robust Preprocessing**: Advanced canvas preprocessing with center-of-mass centering
- **Comprehensive Evaluation**: Confusion matrix, classification report, and visualizations

## 📁 Project Structure

```
digit-recognizer/
├── backend/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_model.py       # Fully-connected ANN for digits
│   │   ├── cnn_model.py            # CNN architecture for digits
│   │   └── char_cnn.py             # CNN architecture for characters
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── preprocess.py           # Data preprocessing
│   │   ├── canvas_preprocess.py    # Canvas image preprocessing
│   │   ├── load_emnist.py          # EMNIST dataset loader
│   │   └── visualization.py        # Plotting utilities
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_digits.py         # Digit model training pipeline
│   │   └── train_characters.py     # Character model training pipeline
│   ├── checkpoints/                # Saved model weights
│   ├── outputs/                    # Training visualizations
│   ├── logs/                       # TensorBoard logs
│   ├── inference_api.py            # FastAPI server
│   └── requirements.txt            # Python dependencies
├── frontend/
│   ├── app/
│   │   ├── components/
│   │   │   ├── DrawingCanvas.tsx
│   │   │   ├── FileUpload.tsx
│   │   │   ├── PredictionDisplay.tsx
│   │   │   ├── ForwardPassVisualizer.tsx   # Cinematic Neural Network Visualizer
│   │   │   └── CharacterPredictionDisplay.tsx
│   │   ├── characters/
│   │   │   └── page.tsx            # Character recognition page
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx                # Digit recognition page
│   ├── public/
│   └── package.json
├── notebooks/
│   ├── 01_eda.ipynb                        # Exploratory Data Analysis
│   ├── 02_digit_model_training.ipynb       # Digit model training notebook
│   ├── 03_character_model_training.ipynb   # Character model training notebook
│   └── 04_model_comparison.ipynb           # Model comparison analysis
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd digit-recognizer/backend
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd digit-recognizer/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## 🎓 Training the Models

### Training the Digit Recognition Model

1. Activate the virtual environment and run the training script:
   ```bash
   cd backend
   python -m training.train_digits
   ```

2. The script will:
   - Load and preprocess the MNIST dataset
   - Perform exploratory data analysis
   - Train both baseline ANN and CNN models
   - Generate visualizations in the `outputs/` directory
   - Save model checkpoints to `checkpoints/`

### Training the Character Recognition Model

1. Run the character training script:
   ```bash
   python -m training.train_characters
   ```

2. The script will:
   - Load and preprocess the EMNIST Letters dataset
   - Train the character CNN model
   - Generate visualizations in `outputs/characters/`
   - Save model checkpoints to `checkpoints/`

### Expected Training Output

- Sample images and class distribution plots
- Training history curves
- Confusion matrices
- Classification reports

## 🖥️ Running the Application

### Start the Backend

```bash
cd backend
uvicorn inference_api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Start the Frontend

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:3000`

- **Digit Recognition**: `http://localhost:3000/`
- **Character Recognition**: `http://localhost:3000/characters`

## 🔍 Neural Path Visualization

The project includes custom **Explainable AI (XAI)** tools to visualize how the CNN arrives at its decision. 

- **10s Cinematic Sequence**: A staged 7-phase JOURNEY explaining the forward pass.
- **Dynamic Connection Growth**: Watch the neural connections form layer-by-layer.
- **Activation Pulses**: High-speed energy pulses represent the signal propagating through the network.
- **Path Highlighting**: The final 2 seconds dim all "noise" and highlight the winning decision path.
- **Anchored Results**: Predicted output neurons are connected to their labels via dynamic dotted lines.

*(To view: Click "VIEW ANALYSIS" after any prediction)*

## 🧠 Model Architecture

### Baseline ANN (Digits)
```
Input (784) → Dense(128, ReLU) → Dropout(0.2) → Dense(64, ReLU) → Dropout(0.2) → Dense(10, Softmax)
```

### CNN Model (Digits)
```
Input (28×28×1)
    → Conv2D(32, 3×3, ReLU) → BatchNorm → MaxPool(2×2)
    → Conv2D(64, 3×3, ReLU) → BatchNorm → MaxPool(2×2)
    → Conv2D(64, 3×3, ReLU) → BatchNorm
    → Flatten → Dense(128, ReLU) → Dropout(0.5)
    → Dense(10, Softmax)
```

### Character CNN (A-Z)
```
Input (28×28×1)
    → Conv2D(32, 3×3, ReLU) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
    → Conv2D(64, 3×3, ReLU) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
    → Conv2D(128, 3×3, ReLU) → BatchNorm → Dropout(0.25)
    → Flatten → Dense(256, ReLU) → Dropout(0.5)
    → Dense(26, Softmax)
```

### Hyperparameters

| Parameter | Digits | Characters |
|-----------|--------|------------|
| Optimizer | Adam | Adam |
| Learning Rate | 0.001 | 0.001 |
| Loss Function | Categorical Crossentropy | Categorical Crossentropy |
| Batch Size | 128 | 128 |
| Epochs | 15 | 20 |
| Validation Split | 20% | 20% |

## 📊 Results

| Model | Dataset | Test Accuracy |
|-------|---------|---------------|
| Baseline ANN | MNIST (Digits) | ~97% |
| CNN | MNIST (Digits) | ~99% |
| Character CNN | EMNIST (Letters) | ~94% |

The CNN significantly outperforms the baseline due to its ability to learn spatial hierarchies and local patterns in images. Character recognition is more challenging due to the higher number of classes (26 vs 10) and similar-looking letters.

## 🛠️ Technologies Used

### Backend
- **TensorFlow/Keras** - Deep learning framework
- **FastAPI** - Modern web framework for APIs
- **NumPy** - Numerical computing
- **OpenCV** - Image processing
- **Matplotlib & Seaborn** - Data visualization
- **scikit-learn** - Classification metrics

### Frontend
- **Next.js 16** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **HTML5 Canvas** - Drawing interface

## 📂 Jupyter Notebooks

The `notebooks/` directory contains detailed Jupyter notebooks for exploration and training:

| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Exploratory Data Analysis on MNIST/EMNIST |
| `02_digit_model_training.ipynb` | Step-by-step digit model training |
| `03_character_model_training.ipynb` | Step-by-step character model training |
| `04_model_comparison.ipynb` | Comparison of different model architectures |

## 📝 License

This project is for educational purposes as part of a university course on Deep Learning.

---

**Made with ❤️ for ML/DL Course Project**

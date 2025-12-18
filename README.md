# Face Classification Project

Deep Learning-Based Face Classification using CNNs and Transfer Learning on the LFW Dataset.

## Results

| Model | Test Accuracy | Top-3 Accuracy |
|-------|---------------|----------------|
| **InceptionV3** | **98.06%** | **100%** |
| ResNet50 | ~95% (expected) | - |
| EfficientNetB0 | ~93% (expected) | - |

## Quick Start

**Important:** After cloning, you must preprocess data and train models before evaluating.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess Data (Required)
```bash
python src/data_preprocessing.py
```
This downloads the LFW dataset and prepares it for training.

### 3. Train a Model (Required)
```bash
# Train InceptionV3 (recommended - best accuracy, ~7-10 min)
python src/train.py --model inceptionv3

# Or train all models
python src/train.py --model resnet50
python src/train.py --model inceptionv3
python src/train.py --model efficientnetb0
```

### 4. Evaluate a Model
```bash
# Evaluate single model
python src/evaluate.py --model inceptionv3

# Evaluate all models (must train all first)
python src/evaluate.py --model all
```

### 5. Run the GUI
```bash
streamlit run gui/streamlit_app.py
```

## How to Check Model Accuracy

### Option 1: Run Evaluation Script
```bash
python src/evaluate.py --model inceptionv3
```
This will output:
- Overall accuracy
- Top-3 accuracy
- Per-class precision, recall, F1-score
- Confusion matrix (saved to `docs/results/`)

### Option 2: Check Saved Reports
After evaluation, results are saved to:
- `docs/results/{model}_metrics_report.txt` - Detailed metrics
- `docs/results/{model}_confusion_matrix.png` - Visual confusion matrix
- `docs/results/{model}_training_history.png` - Training curves

### Option 3: Use the GUI
```bash
streamlit run gui/streamlit_app.py
```
Upload an image or use webcam to test the model interactively.

## Project Structure

```
face_classification_project/
├── src/
│   ├── data_preprocessing.py    # Dataset loading and preprocessing
│   ├── model_resnet.py          # ResNet50 architecture
│   ├── model_inception.py       # InceptionV3 architecture
│   ├── model_efficientnet.py    # EfficientNetB0 architecture
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Model evaluation
│   └── gradcam.py               # Grad-CAM visualization
├── models/                      # Trained model weights
├── data/processed/              # Preprocessed dataset
├── docs/results/                # Evaluation results and plots
├── gui/                         # Streamlit web interface
└── requirements.txt             # Python dependencies
```

## Key Features

- **Transfer Learning**: Pre-trained ImageNet weights for faster convergence
- **Class Balancing**: Automatic class weighting for imbalanced data
- **Two-Phase Training**: Frozen base + fine-tuning for optimal results
- **Data Augmentation**: Rotation, shift, flip, zoom, brightness variation
- **Interactive GUI**: Web interface with webcam support

## Technologies

- Python 3.10+
- TensorFlow 2.x / Keras
- Streamlit (GUI)
- OpenCV, NumPy, Scikit-learn

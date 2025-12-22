# Face Classification Project

Deep Learning-Based Face Classification using CNNs and Transfer Learning on the LFW Dataset.

## Results

| Model | Test Accuracy | Top-3 Accuracy |
|-------|---------------|----------------|
| **InceptionV3** | **98.06%** | **100%** |
| ResNet50 | ~95% | **100%** |
| EfficientNetB0 | ~93% | **100%** |


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

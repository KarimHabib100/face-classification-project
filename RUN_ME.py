"""
This code will:
1. Install all required packages
2. Download and preprocess the LFW dataset (automatic)
3. Train all 3 models (ResNet50, InceptionV3, EfficientNetB0)
4. Save models and training results

"""

import subprocess
import sys
import os
import time

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def print_status(text):
    print(f"✓ {text}")

def print_error(text):
    print(f"✗ {text}")

def run_command(command, description):
    """Run a command and show progress"""
    print(f"\n→ {description}...")
    start_time = time.time()
    
    result = subprocess.run(command, shell=False)
    
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print_status(f"{description} completed in {elapsed/60:.1f} minutes")
        return True
    else:
        print_error(f"{description} failed!")
        return False

def check_gpu():
    """Check if GPU is available"""
    print_header("Checking GPU Availability")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print_status(f"GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"  - {gpu}")
            return True
        else:
            print("⚠ No GPU detected. Training will use CPU (slower).")
            return False
    except:
        print("⚠ TensorFlow not installed yet. Will check GPU after installation.")
        return False

def main():
    print_header("FACE CLASSIFICATION PROJECT - FULL TRAINING")
    print("System: Windows")
    print("Python: 3.10.6")
    print("GPU: NVIDIA GTX 1650")
    print("Mode: Full Training (All 3 models, 30 epochs each)")
    
    print("\n⏱ Estimated total time: 8-10 hours")
    print("💡 Tip: You can leave this running overnight\n")
    
    # Confirmation
    response = input("Ready to start? This will take several hours. (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("\nExiting. Run again when ready!")
        return
    
    # Step 1: Install packages
    print_header("STEP 1/4: Installing Required Packages")
    print("This will install TensorFlow, Keras, OpenCV, and other dependencies...")
    print("(This may take 10-15 minutes)\n")
    
    if not run_command(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--upgrade"],
        "Package installation"
    ):
        print_error("Failed to install packages. Please check your internet connection.")
        print("\nTry running manually:")
        print("  pip install -r requirements.txt")
        return
    
    # Check GPU after installation
    check_gpu()
    
    # Step 2: Preprocess data
    print_header("STEP 2/4: Downloading and Preprocessing Dataset")
    print("This will:")
    print("  - Download LFW dataset (auto, ~200MB)")
    print("  - Preprocess images (resize, normalize)")
    print("  - Split into train/val/test sets")
    print("  - Save processed data\n")
    
    if not run_command(
        [sys.executable, "src/data_preprocessing.py"],
        "Data preprocessing"
    ):
        print_error("Data preprocessing failed!")
        return
    
    # Verify data file was created
    if not os.path.exists("data/processed/face_data.pkl"):
        print_error("Processed data file not found!")
        return
    
    print_status("Dataset ready! Found: data/processed/face_data.pkl")
    
    # Step 3: Train all three models
    print_header("STEP 3/4: Training All Three Models")
    print("Training order:")
    print("  1. ResNet50 (2-3 hours)")
    print("  2. InceptionV3 (2-3 hours)")
    print("  3. EfficientNetB0 (2-3 hours)")
    print("\nEach model trains in 2 phases:")
    print("  - Phase 1: Initial training (30 epochs)")
    print("  - Phase 2: Fine-tuning (30 epochs)\n")
    
    models = ['resnet50', 'inceptionv3', 'efficientnetb0']
    trained_models = []
    
    for i, model in enumerate(models, 1):
        print(f"\n{'#'*70}")
        print(f"# MODEL {i}/3: {model.upper()}")
        print(f"{'#'*70}\n")
        
        start = time.time()
        
        success = run_command(
            [sys.executable, "src/train.py", 
             "--model", model,
             "--initial_epochs", "30",
             "--finetune_epochs", "30"],
            f"Training {model.upper()}"
        )
        
        elapsed = time.time() - start
        
        if success:
            trained_models.append(model)
            print(f"\n✓ {model.upper()} completed in {elapsed/3600:.1f} hours")
            
            # Check if model file was created
            model_path = f"models/{model}/{model}_final.h5"
            if os.path.exists(model_path):
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"✓ Model saved: {model_path} ({size_mb:.1f} MB)")
            else:
                print(f"⚠ Model file not found at {model_path}")
        else:
            print(f"\n⚠ {model.upper()} training failed. Continuing with next model...")
        
        # Show progress
        if i < len(models):
            remaining = len(models) - i
            print(f"\n📊 Progress: {i}/{len(models)} models complete")
            print(f"   Remaining: {remaining} model(s)")
    
    # Step 4: Summary
    print_header("STEP 4/4: Training Complete!")
    
    if len(trained_models) == 0:
        print_error("No models were trained successfully.")
        print("\nPlease check the error messages above.")
        return
    
    print(f"✓ Successfully trained {len(trained_models)}/{len(models)} models:")
    for model in trained_models:
        print(f"  - {model.upper()}")
    
    print("\n" + "="*70)
    print("  🎉 ALL DONE!")
    print("="*70)
    
    print("\nYour trained models are in the 'models/' folder:")
    for model in trained_models:
        model_path = f"models/{model}/{model}_final.h5"
        if os.path.exists(model_path):
            print(f"  ✓ {model_path}")
    
    print("\nTraining plots saved in 'docs/results/':")
    for model in trained_models:
        plot_path = f"docs/results/{model}_training_history.png"
        if os.path.exists(plot_path):
            print(f"  ✓ {plot_path}")
    
    print("\n" + "="*70)
    print("  NEXT STEPS")
    print("="*70)
    
    print("\n1. Create evaluation script (src/evaluate.py) to see model performance")
    print("2. Create GUI (gui/streamlit_app.py) to test the models")
    print("3. Generate Grad-CAM visualizations")
    
    print("\nWould you like me to provide these additional scripts?")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user.")
        print("You can resume by running this script again.")
        print("Already trained models are saved and won't be lost.")
    except Exception as e:
        print(f"\n\n✗ An error occurred: {e}")
        print("\nPlease share this error message for help.")
        import traceback
        traceback.print_exc()
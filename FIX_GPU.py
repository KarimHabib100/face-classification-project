"""
GPU FIX Code
This Code will:
1. Check current TensorFlow installation
2. Install correct version for GPU support
3. Test GPU detection
"""

import subprocess
import sys

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

print_header("GPU FIX FOR NVIDIA GTX 1650")

print("Step 1: Checking current TensorFlow installation...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow version: {tf.__version__}")
    
    # Check for GPU
    print("\nStep 2: Checking GPU devices...")
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"✓ GPU DETECTED! Found {len(gpus)} device(s):")
        for gpu in gpus:
            print(f"  - {gpu}")
        print("\n✓ Your GPU is working! No fix needed.")
        print("You can continue with: python RUN_ME.py")
        sys.exit(0)
    else:
        print("✗ No GPU detected.")
        print("\nThis usually means TensorFlow doesn't have GPU support.")
        print("Let's fix this...\n")
        
except ImportError:
    print("✗ TensorFlow not installed yet.")
    print("Installing TensorFlow with GPU support...\n")

# Uninstall existing TensorFlow
print_header("Step 3: Removing Current TensorFlow")
print("Uninstalling any existing TensorFlow versions...")
run_command(f"{sys.executable} -m pip uninstall tensorflow tensorflow-gpu -y")
print("✓ Cleaned up old installations\n")

# Install TensorFlow with GPU support
print_header("Step 4: Installing TensorFlow with GPU Support")
print("Installing tensorflow[and-cuda]==2.15.0...")
print("(This includes CUDA and cuDNN automatically)\n")

success, stdout, stderr = run_command(
    f"{sys.executable} -m pip install tensorflow[and-cuda]==2.15.0"
)

if not success:
    print("⚠ Installation with [and-cuda] failed. Trying alternative method...\n")
    success, stdout, stderr = run_command(
        f"{sys.executable} -m pip install tensorflow-gpu==2.15.0"
    )

if success:
    print("✓ TensorFlow installed successfully!\n")
else:
    print("✗ Installation failed. Error:")
    print(stderr)
    print("\nTrying basic TensorFlow installation...")
    run_command(f"{sys.executable} -m pip install tensorflow==2.15.0")

# Test GPU again
print_header("Step 5: Testing GPU Detection")
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    print("\nGPU devices:")
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"✓ SUCCESS! GPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
        
        # Try to use GPU
        print("\nTesting GPU computation...")
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
        print("✓ GPU computation test passed!")
        
        print("\n" + "="*70)
        print("  ✓ GPU IS WORKING!")
        print("="*70)
        print("\nYou can now run: python RUN_ME.py")
        print("Training will use your GTX 1650 GPU (much faster!)\n")
        
    else:
        print("✗ Still no GPU detected.")
        print("\n" + "="*70)
        print("  MANUAL FIX REQUIRED")
        print("="*70)
        print("\nYour GPU needs CUDA Toolkit and cuDNN installed.")
        print("\nOption 1 - Install CUDA manually:")
        print("  1. Download CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive")
        print("  2. Download cuDNN 8.6: https://developer.nvidia.com/cudnn")
        print("  3. Install both, then run this script again")
        
        print("\nOption 2 - Continue with CPU (slower but works):")
        print("  Training will work but take longer (~30 hours vs 8 hours)")
        print("  Just run: python RUN_ME.py")
        
        print("\nOption 3 - Use Google Colab (FREE GPU):")
        print("  Upload your code to Google Colab for free GPU access")
        
        response = input("\nContinue with CPU training? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            print("\n✓ Proceeding with CPU training.")
            print("Run: python RUN_ME.py")
        else:
            print("\nPlease install CUDA 11.8 and cuDNN 8.6, then run:")
            print("  python FIX_GPU.py")

except Exception as e:
    print(f"\n✗ Error during GPU test: {e}")
    import traceback
    traceback.print_exc()
    
print("\n" + "="*70 + "\n")
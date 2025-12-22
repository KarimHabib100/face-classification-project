import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import pickle
import argparse
import os

class GradCAM:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        
        # Get the base model layer
        base_model = None
        for layer in model.layers:
            if 'resnet' in layer.name.lower() or 'inception' in layer.name.lower() or 'efficientnet' in layer.name.lower():
                base_model = layer
                break
        
        if base_model is None:
            raise ValueError("Could not find base model")
        
        # Find last conv layer in base model
        layer_name = None
        for layer in reversed(base_model.layers):
            if 'conv' in layer.name.lower() and len(layer.output_shape) == 4:
                layer_name = layer.name
                break
        
        if layer_name is None:
            raise ValueError("Could not find convolutional layer")
        
        self.layer_name = layer_name
        self.base_model = base_model
        print(f"✓ Using layer: {layer_name}")
        
        # Create gradient model
        self.grad_model = keras.Model(
            inputs=model.input,
            outputs=[base_model.get_layer(layer_name).output, model.output]
        )
    
    def compute_heatmap(self, image, class_idx=None, eps=1e-8):
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            conv_outputs, predictions = self.grad_model(inputs)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            class_channel = predictions[:, class_idx]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()
        
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + eps)
        
        return heatmap
    
    def overlay_heatmap(self, heatmap, image, alpha=0.4):
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        superimposed = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
        return superimposed

def generate_gradcam_samples(model_name='resnet50', num_samples=6):
    print(f"\n{'='*60}")
    print(f"GENERATING GRAD-CAM FOR {model_name.upper()}")
    print(f"{'='*60}\n")
    
    print("Loading data...")
    with open('data/processed/face_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_test = data['X_test']
    y_test = data['y_test_original']
    class_names = data['class_names']
    
    print("Loading model...")
    model = keras.models.load_model(f'models/{model_name}/{model_name}_final.h5')
    
    print("Initializing Grad-CAM...")
    try:
        gradcam = GradCAM(model, model_name)
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Skipping this model.")
        return
    
    os.makedirs(f'docs/results/gradcam_{model_name}', exist_ok=True)
    
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    print(f"\nGenerating {num_samples} visualizations...")
    
    success_count = 0
    for i, idx in enumerate(indices):
        try:
            image = X_test[idx]
            true_label = y_test[idx]
            
            pred = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
            pred_class = np.argmax(pred)
            pred_conf = pred[pred_class]
            
            heatmap = gradcam.compute_heatmap(image)
            
            display_image = image.copy()
            if display_image.max() <= 1.0:
                display_image = (display_image * 255).astype(np.uint8)
            
            superimposed = gradcam.overlay_heatmap(heatmap, display_image)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(display_image)
            axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            axes[2].imshow(superimposed)
            axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
            axes[2].axis('off')
            
            info = f"True: {class_names[true_label]}\nPredicted: {class_names[pred_class]} ({pred_conf:.1%})"
            plt.suptitle(info, fontsize=11, y=0.02)
            plt.tight_layout(rect=[0, 0.08, 1, 1])
            
            save_path = f'docs/results/gradcam_{model_name}/sample_{i+1}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            success_count += 1
            print(f"  ✓ Sample {i+1}/{num_samples}")
            
        except Exception as e:
            print(f"  ✗ Sample {i+1} failed: {e}")
            continue
    
    print(f"\n✓ Generated {success_count}/{num_samples} visualizations")
    print(f"✓ Saved to: docs/results/gradcam_{model_name}/")

def create_gradcam_grid(model_name='resnet50'):
    print(f"\nCreating Grad-CAM grid for {model_name.upper()}...")
    
    with open('data/processed/face_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_test = data['X_test']
    y_test = data['y_test_original']
    class_names = data['class_names']
    
    model = keras.models.load_model(f'models/{model_name}/{model_name}_final.h5')
    
    try:
        gradcam = GradCAM(model, model_name)
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Skipping grid for this model.")
        return
    
    indices = np.random.choice(len(X_test), 9, replace=False)
    
    fig, axes = plt.subplots(3, 6, figsize=(20, 10))
    
    success_count = 0
    for i, idx in enumerate(indices):
        try:
            row = i // 3
            col = (i % 3) * 2
            
            image = X_test[idx]
            true_label = y_test[idx]
            
            pred = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
            pred_class = np.argmax(pred)
            
            heatmap = gradcam.compute_heatmap(image)
            
            display_image = image.copy()
            if display_image.max() <= 1.0:
                display_image = (display_image * 255).astype(np.uint8)
            
            superimposed = gradcam.overlay_heatmap(heatmap, display_image)
            
            axes[row, col].imshow(display_image)
            axes[row, col].set_title(f'True: {class_names[true_label][:15]}', fontsize=9)
            axes[row, col].axis('off')
            
            axes[row, col+1].imshow(superimposed)
            axes[row, col+1].set_title(f'Pred: {class_names[pred_class][:15]}', fontsize=9)
            axes[row, col+1].axis('off')
            
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ Sample {i+1} failed: {e}")
            # Hide the axes for failed samples
            axes[row, col].axis('off')
            axes[row, col+1].axis('off')
            continue
    
    plt.suptitle(f'{model_name.upper()} - Grad-CAM Visualizations', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    save_path = f'docs/results/{model_name}_gradcam_grid.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Grid saved: {save_path} ({success_count}/9 samples)")

def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'inceptionv3', 'efficientnetb0', 'all'])
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--grid', action='store_true')
    
    args = parser.parse_args()
    
    models = ['resnet50', 'inceptionv3', 'efficientnetb0'] if args.model == 'all' else [args.model]
    
    print("\n" + "="*60)
    print("  GRAD-CAM VISUALIZATION GENERATOR")
    print("="*60)
    
    for model in models:
        print(f"\n{'#'*60}")
        print(f"# {model.upper()}")
        print(f"{'#'*60}")
        
        try:
            if args.grid:
                create_gradcam_grid(model)
            else:
                generate_gradcam_samples(model, args.samples)
        except Exception as e:
            print(f"\n✗ Failed to process {model}: {e}")
            continue
    
    print("\n" + "="*60)
    print("  ✓ GRAD-CAM GENERATION COMPLETE")
    print("="*60)
    print("\nCheck docs/results/ for all visualizations!")

if __name__ == "__main__":

    main()

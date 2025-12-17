import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

from model_resnet import create_resnet50_model, compile_model as compile_resnet, unfreeze_and_fine_tune as finetune_resnet
from model_inception import create_inceptionv3_model, compile_model as compile_inception, unfreeze_and_fine_tune as finetune_inception
from model_efficientnet import create_efficientnet_model, compile_model as compile_efficientnet, unfreeze_and_fine_tune as finetune_efficientnet

class OptimizedTrainer:
    def __init__(self, model_name='resnet50', data_path='data/processed/face_data.pkl'):
        self.model_name = model_name
        self.model = None
        self.base_model = None
        
        os.makedirs(f'models/{model_name}', exist_ok=True)
        os.makedirs('docs/results', exist_ok=True)
        
        print(f"Loading data from {data_path}...")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        self.class_names = data['class_names']
        self.num_classes = len(self.class_names)
        
        print(f"✓ Loaded!")
        print(f"  Train: {len(self.X_train)}, Val: {len(self.X_val)}, Test: {len(self.X_test)}")
        print(f"  Classes: {self.num_classes}")
        
        # Data augmentation
        self.datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
    def create_model(self):
        print(f"\nCreating {self.model_name}...")
        
        input_shape = self.X_train.shape[1:]
        
        if self.model_name == 'resnet50':
            self.model, self.base_model = create_resnet50_model(input_shape, self.num_classes)
            self.model = compile_resnet(self.model, learning_rate=0.0005)  # Lower LR
        elif self.model_name == 'inceptionv3':
            self.model, self.base_model = create_inceptionv3_model(input_shape, self.num_classes)
            self.model = compile_inception(self.model, learning_rate=0.0005)
        elif self.model_name == 'efficientnetb0':
            self.model, self.base_model = create_efficientnet_model(input_shape, self.num_classes)
            self.model = compile_efficientnet(self.model, learning_rate=0.0005)
        
        print(f"✓ Created! Parameters: {self.model.count_params():,}")
        
    def train_initial(self, epochs=50, batch_size=16):
        print(f"\n{'='*60}")
        print(f"PHASE 1: Initial Training")
        print(f"  Epochs: {epochs}, Batch: {batch_size}")
        print(f"{'='*60}\n")
        
        callbacks = [
            ModelCheckpoint(
                f'models/{self.model_name}/{self.model_name}_initial_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        start = time.time()
        
        history = self.model.fit(
            self.datagen.flow(self.X_train, self.y_train, batch_size=batch_size),
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            steps_per_epoch=len(self.X_train) // batch_size,
            callbacks=callbacks,
            verbose=2  # Less verbose output
        )
        
        elapsed = (time.time() - start) / 60
        print(f"\n✓ Phase 1 done in {elapsed:.1f} min")
        
        return history
    
    def train_finetune(self, epochs=50, batch_size=8):
        print(f"\n{'='*60}")
        print(f"PHASE 2: Fine-Tuning")
        print(f"  Epochs: {epochs}, Batch: {batch_size}")
        print(f"{'='*60}\n")
        
        if self.model_name == 'resnet50':
            self.model = finetune_resnet(self.model, self.base_model, learning_rate=0.00005)
        elif self.model_name == 'inceptionv3':
            self.model = finetune_inception(self.model, self.base_model, learning_rate=0.00005)
        elif self.model_name == 'efficientnetb0':
            self.model = finetune_efficientnet(self.model, self.base_model, learning_rate=0.00005)
        
        callbacks = [
            ModelCheckpoint(
                f'models/{self.model_name}/{self.model_name}_finetune_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-8,
                verbose=1
            )
        ]
        
        start = time.time()
        
        history = self.model.fit(
            self.datagen.flow(self.X_train, self.y_train, batch_size=batch_size),
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            steps_per_epoch=len(self.X_train) // batch_size,
            callbacks=callbacks,
            verbose=2
        )
        
        elapsed = (time.time() - start) / 60
        print(f"\n✓ Phase 2 done in {elapsed:.1f} min")
        
        return history
    
    def plot_and_save(self, history1, history2):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        acc = history1.history['accuracy'] + history2.history['accuracy']
        val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
        loss = history1.history['loss'] + history2.history['loss']
        val_loss = history1.history['val_loss'] + history2.history['val_loss']
        
        epochs_range = range(1, len(acc) + 1)
        split = len(history1.history['accuracy'])
        
        axes[0].plot(epochs_range, acc, 'b-', label='Train')
        axes[0].plot(epochs_range, val_acc, 'r-', label='Val')
        axes[0].axvline(x=split, color='g', linestyle='--', label='Fine-tune')
        axes[0].set_title(f'{self.model_name.upper()} - Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(epochs_range, loss, 'b-', label='Train')
        axes[1].plot(epochs_range, val_loss, 'r-', label='Val')
        axes[1].axvline(x=split, color='g', linestyle='--')
        axes[1].set_title(f'{self.model_name.upper()} - Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'docs/results/{self.model_name}_training_history.png', dpi=300)
        plt.close()
        
        self.model.save(f'models/{self.model_name}/{self.model_name}_final.h5')
        
        print(f"\n✓ Model saved!")
        print(f"  Final val accuracy: {val_acc[-1]:.4f} ({val_acc[-1]*100:.2f}%)")
    
    def train_complete(self, initial_epochs=50, finetune_epochs=50):
        start_time = time.time()
        
        print(f"\n{'#'*60}")
        print(f"# TRAINING: {self.model_name.upper()}")
        print(f"{'#'*60}\n")
        
        self.create_model()
        h1 = self.train_initial(epochs=initial_epochs)
        h2 = self.train_finetune(epochs=finetune_epochs)
        self.plot_and_save(h1, h2)
        
        total_min = (time.time() - start_time) / 60
        print(f"\n{'#'*60}")
        print(f"# COMPLETE! Time: {total_min:.1f} min ({total_min/60:.1f} hrs)")
        print(f"{'#'*60}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'inceptionv3', 'efficientnetb0'])
    parser.add_argument('--initial_epochs', type=int, default=50)
    parser.add_argument('--finetune_epochs', type=int, default=50)
    
    args = parser.parse_args()
    
    trainer = OptimizedTrainer(model_name=args.model)
    trainer.train_complete(args.initial_epochs, args.finetune_epochs)

if __name__ == "__main__":
    main()
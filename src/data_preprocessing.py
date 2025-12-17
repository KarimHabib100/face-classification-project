import numpy as np
import cv2
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle
import os

class DataPreprocessor:
    def __init__(self, min_faces_per_person=70, resize=(160, 160)):
        """
        FINAL OPTIMIZED VERSION
        - min_faces=70 for balanced classes (gives ~7-10 classes)
        - Larger images (160x160) for better features
        """
        self.min_faces_per_person = min_faces_per_person
        self.resize = resize
        self.class_names = None
        
    def load_dataset(self):
        print("="*60)
        print("LOADING LFW DATASET - FINAL OPTIMIZED VERSION")
        print("="*60)
        print(f"\nSettings:")
        print(f"  - Min faces per person: {self.min_faces_per_person}")
        print(f"  - Target image size: {self.resize}")
        print("\nDownloading...\n")
        
        lfw_people = fetch_lfw_people(
            min_faces_per_person=self.min_faces_per_person,
            resize=None,
            color=True
        )
        
        X = lfw_people.images
        y = lfw_people.target
        self.class_names = lfw_people.target_names
        
        print(f"\n✓ Dataset loaded!")
        print(f"  Total images: {X.shape[0]}")
        print(f"  Number of people: {len(self.class_names)}")
        print(f"  Original shape: {X.shape[1:]}")
        
        print(f"\nPeople in dataset:")
        unique, counts = np.unique(y, return_counts=True)
        for i, (u, c) in enumerate(zip(unique, counts)):
            print(f"  {i+1}. {self.class_names[i]:25s}: {c:4d} images")
        
        print(f"\nTotal training examples: {len(X)}")
        
        return X, y
    
    def preprocess_images(self, X):
        print("\n" + "="*60)
        print("PREPROCESSING")
        print("="*60)
        print(f"\nResizing to {self.resize}...")
        print("NOTE: Images kept in [0, 255] range for model preprocess_input")

        processed_images = []

        for i, img in enumerate(X):
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(X)} images processed ({(i+1)/len(X)*100:.1f}%)", end='\r')

            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            img_resized = cv2.resize(img, self.resize)
            # Keep as float32 in [0, 255] range - model's preprocess_input will handle normalization
            processed_images.append(img_resized.astype('float32'))

        print(f"\n✓ All images preprocessed!")
        return np.array(processed_images)
    
    def split_data(self, X, y, test_size=0.2, val_size=0.15):
        print("\n" + "="*60)
        print("SPLITTING DATASET")
        print("="*60)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        num_classes = len(np.unique(y))
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
        
        print(f"\n✓ Split complete!")
        print(f"  Training:   {X_train.shape[0]:4d} images ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {X_val.shape[0]:4d} images ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test:       {X_test.shape[0]:4d} images ({len(X_test)/len(X)*100:.1f}%)")
        
        return (X_train, y_train_cat), (X_val, y_val_cat), (X_test, y_test_cat), y_test
    
    def save_processed_data(self, data, filename):
        os.makedirs('data/processed', exist_ok=True)
        filepath = f'data/processed/{filename}'
        
        print(f"\nSaving to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✓ Saved! ({size_mb:.2f} MB)")
    
    def process_and_save_all(self):
        print("\n" + "#"*60)
        print("# FACE CLASSIFICATION - DATA PIPELINE")
        print("#"*60 + "\n")
        
        X, y = self.load_dataset()
        X_processed = self.preprocess_images(X)
        train_data, val_data, test_data, y_test_orig = self.split_data(X_processed, y)
        
        self.save_processed_data({
            'X_train': train_data[0],
            'y_train': train_data[1],
            'X_val': val_data[0],
            'y_val': val_data[1],
            'X_test': test_data[0],
            'y_test': test_data[1],
            'y_test_original': y_test_orig,
            'class_names': self.class_names
        }, 'face_data.pkl')
        
        print("\n" + "="*60)
        print("COMPLETE!")
        print("="*60)
        print(f"\nSummary:")
        print(f"  People: {len(self.class_names)}")
        print(f"  Total images: {len(X_processed)}")
        print(f"  Image size: {self.resize}")
        print(f"  Training samples: {len(train_data[0])}")
        print(f"\nReady for training!")
        print("="*60 + "\n")

if __name__ == "__main__":
    # OPTIMIZED: 70 faces minimum, larger images
    preprocessor = DataPreprocessor(min_faces_per_person=70, resize=(160, 160))
    preprocessor.process_and_save_all()
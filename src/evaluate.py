import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow import keras
import os

class ModelEvaluator:
    def __init__(self, model_name, data_path='data/processed/face_data.pkl'):
        self.model_name = model_name
        self.data_path = data_path
        self.model = None
        
        os.makedirs('docs/results', exist_ok=True)
        self.load_data()
        
    def load_data(self):
        print(f"Loading test data from {self.data_path}...")
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        self.y_test_original = data['y_test_original']
        self.class_names = data['class_names']
        
        print(f"✓ Test data loaded: {len(self.X_test)} samples, {len(self.class_names)} classes")
        
    def load_model(self):
        # Try native Keras format first, then fallback to H5
        model_path_keras = f'models/{self.model_name}/{self.model_name}_final.keras'
        model_path_h5 = f'models/{self.model_name}/{self.model_name}_final.h5'

        if os.path.exists(model_path_keras):
            model_path = model_path_keras
        elif os.path.exists(model_path_h5):
            model_path = model_path_h5
        else:
            raise FileNotFoundError(f"No model found at {model_path_keras} or {model_path_h5}")

        print(f"\nLoading model from {model_path}...")

        self.model = keras.models.load_model(model_path)
        print("✓ Model loaded successfully!")
    
    def predict(self):
        print("\nMaking predictions on test set...")
        self.y_pred_proba = self.model.predict(self.X_test, verbose=1)
        self.y_pred = np.argmax(self.y_pred_proba, axis=1)
        print("✓ Predictions complete!")
    
    def calculate_metrics(self):
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        
        accuracy = accuracy_score(self.y_test_original, self.y_pred)
        print(f"\n✓ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        top3_acc = self.calculate_top_k_accuracy(k=3)
        print(f"✓ Top-3 Accuracy: {top3_acc:.4f} ({top3_acc*100:.2f}%)")
        
        print("\n" + "-"*60)
        print("DETAILED CLASSIFICATION REPORT")
        print("-"*60)
        print(classification_report(self.y_test_original, self.y_pred, 
                                   target_names=self.class_names, zero_division=0))
        
        return accuracy, top3_acc
    
    def calculate_top_k_accuracy(self, k=3):
        top_k_preds = np.argsort(self.y_pred_proba, axis=1)[:, -k:]
        correct = sum(1 for i, true_label in enumerate(self.y_test_original) 
                     if true_label in top_k_preds[i])
        return correct / len(self.y_test_original)
    
    def plot_confusion_matrix(self):
        print("\nGenerating confusion matrix...")
        
        cm = confusion_matrix(self.y_test_original, self.y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Normalized Count'})
        
        plt.title(f'{self.model_name.upper()} - Confusion Matrix', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = f'docs/results/{self.model_name}_confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {save_path}")
        plt.close()
    
    def save_metrics_report(self, accuracy, top3_acc):
        report_path = f'docs/results/{self.model_name}_metrics_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"EVALUATION REPORT: {self.model_name.upper()}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"Top-3 Accuracy: {top3_acc:.4f} ({top3_acc*100:.2f}%)\n\n")
            f.write("="*60 + "\n")
            f.write("DETAILED CLASSIFICATION REPORT:\n")
            f.write("="*60 + "\n\n")
            f.write(classification_report(self.y_test_original, self.y_pred, 
                                        target_names=self.class_names, zero_division=0))
        
        print(f"✓ Metrics report saved: {report_path}")
    
    def evaluate_complete(self):
        print(f"\n{'#'*60}")
        print(f"# EVALUATING: {self.model_name.upper()}")
        print(f"{'#'*60}\n")
        
        self.load_model()
        self.predict()
        accuracy, top3_acc = self.calculate_metrics()
        self.plot_confusion_matrix()
        self.save_metrics_report(accuracy, top3_acc)
        
        print(f"\n{'#'*60}")
        print(f"# EVALUATION COMPLETE")
        print(f"{'#'*60}\n")
        
        return accuracy, top3_acc

def compare_all_models():
    print("\n" + "="*60)
    print("COMPARING ALL THREE MODELS")
    print("="*60 + "\n")
    
    models = ['resnet50', 'inceptionv3', 'efficientnetb0']
    results = {}
    
    for model_name in models:
        evaluator = ModelEvaluator(model_name)
        accuracy, top3_acc = evaluator.evaluate_complete()
        results[model_name] = {'accuracy': accuracy, 'top3_accuracy': top3_acc}
    
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"\n{'Model':<20} {'Accuracy':<15} {'Top-3 Accuracy':<15}")
    print("-"*50)
    for model, metrics in results.items():
        print(f"{model.upper():<20} {metrics['accuracy']:<15.4f} {metrics['top3_accuracy']:<15.4f}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    models_upper = [m.upper() for m in models]
    accuracies = [results[m]['accuracy'] for m in models]
    top3_accs = [results[m]['top3_accuracy'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x + width/2, top3_accs, width, label='Top-3 Accuracy', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_upper)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('docs/results/model_comparison.png', dpi=300)
    print("\n✓ Comparison plot saved: docs/results/model_comparison.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate face classification model')
    parser.add_argument('--model', type=str, default='all',
                       help='Model to evaluate: resnet50, inceptionv3, efficientnetb0, or all')
    
    args = parser.parse_args()
    
    if args.model == 'all':
        compare_all_models()
    else:
        evaluator = ModelEvaluator(args.model)
        evaluator.evaluate_complete()

if __name__ == "__main__":
    main()
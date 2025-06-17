# ==============================================================================
# Bangladeshi Food Classification - Complete Pipeline
# Modified for Apple Silicon (M1/M2/M3) and Research-focused Evaluation
# ==============================================================================

# ================================
# STEP 1: Setup and Installation
# ================================
# Ensure these are installed in your Python environment.
# pip install torch torchvision transformers datasets pillow
# pip install accelerate wandb scikit-learn matplotlib seaborn

# ================================
# STEP 2: Imports and Setup
# ================================
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# Suppress excessive PIL logging
import logging
logging.getLogger("PIL").setLevel(logging.WARNING)


# ================================
# STEP 3: Configuration
# ================================
class Config:
    """
    Configuration class for all hyperparameters and paths.
    """
    # Dataset paths - Modified for local structure
    dataset_path = "./data"
    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "validation")

    # Model configuration
    model_name = "google/vit-base-patch16-224"  # Vision Transformer is a strong baseline

    # Training parameters
    batch_size = 32  # Adjusted for M1 Pro VRAM
    num_epochs = 15  # Increased for potentially better convergence
    learning_rate = 3e-5 # A common learning rate for fine-tuning ViT
    weight_decay = 0.01
    warmup_ratio = 0.1

    # Image parameters
    image_size = 224

    # Output directory for model, logs, and plots
    output_dir = "./bangladeshi-food-classifier-vit"

    # Device: Automatically use Apple's MPS for GPU acceleration if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Apple MPS for GPU acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using NVIDIA CUDA for GPU acceleration.")
    else:
        device = torch.device("cpu")
        print("⚠️ MPS or CUDA not available. Using CPU. This will be slow.")

config = Config()

# Create output directory if it doesn't exist
Path(config.output_dir).mkdir(parents=True, exist_ok=True)


# ================================
# STEP 4: Custom Dataset Class
# ================================
class BangladeshiFoodDataset(Dataset):
    """
    Custom PyTorch Dataset for loading Bangladeshi food images.
    """
    def __init__(self, data_dir, processor, augment=False):
        self.data_dir = data_dir
        self.processor = processor
        self.augment = augment
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        self.class_names = sorted([d.name for d in os.scandir(data_dir) if d.is_dir()])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        print(f"Found {len(self.class_names)} classes in '{data_dir}'.")
        
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        
        print(f"-> Found {len(self.image_paths)} total images.")
        
        # Define transforms
        # Normalization is handled by the Hugging Face processor
        common_transforms = [
            transforms.Resize((config.image_size, config.image_size)),
        ]
        
        # Augmentation for the training set
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.3),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomPerspective(0.2),
                *common_transforms
            ])
        else:
            self.transform = transforms.Compose(common_transforms)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            # Apply augmentations/resizing
            image = self.transform(image)
            # Process with model-specific processor (handles normalization, tensor conversion)
            processed = self.processor(images=image, return_tensors="pt")
            
            return {
                'pixel_values': processed['pixel_values'].squeeze(0), # Remove batch dim
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
        except Exception as e:
            print(f"Error loading image {image_path}: {e}. Skipping.")
            # Return a dummy sample if an image is corrupted
            dummy_pixel_values = torch.zeros((3, config.image_size, config.image_size))
            return {'pixel_values': dummy_pixel_values, 'labels': torch.tensor(0, dtype=torch.long)}


# ================================
# STEP 5: Data Loading
# ================================
def setup_data():
    """
    Initializes the image processor and creates the datasets.
    """
    print("\nSetting up data...")
    processor = AutoImageProcessor.from_pretrained(config.model_name)
    
    train_dataset = BangladeshiFoodDataset(config.train_dir, processor, augment=True)
    val_dataset = BangladeshiFoodDataset(config.val_dir, processor, augment=False)
    
    # Save class names for later use in inference
    with open(os.path.join(config.output_dir, "class_names.json"), "w") as f:
        json.dump(train_dataset.class_names, f)
        
    return train_dataset, val_dataset, processor


# ================================
# STEP 6: Model Setup
# ================================
def setup_model(num_classes):
    """
    Loads the pre-trained model and configures it for the number of classes.
    """
    print(f"\nLoading model: {config.model_name}")
    model = AutoModelForImageClassification.from_pretrained(
        config.model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,  # Necessary to replace the classifier head
    ).to(config.device)
    
    return model


# ================================
# STEP 7: Training Setup
# ================================
def compute_metrics(eval_pred):
    """
    Computes and returns metrics for evaluation (essential for research).
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def setup_training(model, train_dataset, val_dataset):
    """
    Sets up the TrainingArguments and the Trainer.
    """
    # Set up Weights & Biases for experiment tracking
    os.environ["WANDB_PROJECT"] = "Bangladeshi-Food-Classification"
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        
        # Evaluation and logging
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=50,
        logging_steps=50,
        
        # Model saving and loading
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        # W&B reporting
        report_to="wandb",
        run_name=f"vit-lr{config.learning_rate}-epochs{config.num_epochs}",
        
        # Dataloader settings
        dataloader_num_workers=2,
        remove_unused_columns=False,
        
        # Disable distributed training features
        local_rank=-1,
        ddp_backend=None,
        save_safetensors=True,
        save_only_model=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)]
    )
    
    return trainer


# ================================
# STEP 8: Evaluation and Visualization
# ================================
def evaluate_and_visualize(trainer, val_dataset, class_names):
    """
    Performs final evaluation and generates plots and reports for the paper.
    """
    print("\nPerforming final evaluation and generating artifacts...")
    
    # Get predictions
    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    
    # 1. Generate and save the classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    report_path = os.path.join(config.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Classification Report\n")
        f.write("=====================\n\n")
        f.write(report)
    print("\nClassification Report:")
    print(report)
    print(f"-> Saved to {report_path}")

    # 2. Generate and save the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(config.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.show()
    print(f"-> Confusion matrix saved to {cm_path}")

    # 3. Plot and save training history
    logs = trainer.state.log_history
    train_loss = [log['loss'] for log in logs if 'loss' in log]
    eval_loss = [log['eval_loss'] for log in logs if 'eval_loss' in log]
    eval_f1 = [log['eval_f1'] for log in logs if 'eval_f1' in log]
    eval_accuracy = [log['eval_accuracy'] for log in logs if 'eval_accuracy' in log]
    
    epochs = [log['epoch'] for log in logs if 'eval_loss' in log]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(epochs, [log['eval_loss'] for log in logs if 'eval_loss' in log], 'o-', label='Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Metrics')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, [log['eval_accuracy'] for log in logs if 'eval_accuracy' in log], 'o-', label='Validation Accuracy')
    ax2.plot(epochs, [log['eval_f1'] for log in logs if 'eval_f1' in log], 'o-', label='Validation F1-Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Metric')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    history_path = os.path.join(config.output_dir, "training_history.png")
    plt.savefig(history_path, dpi=300)
    plt.show()
    print(f"-> Training history plot saved to {history_path}")

    # 4. Create and save sample predictions
    create_sample_predictions(trainer.model, val_dataset.processor, val_dataset, class_names)

def create_sample_predictions(model, processor, val_dataset, class_names, num_samples=10):
    """
    Generates and saves a plot of sample predictions.
    """
    model.eval()
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        image_path = val_dataset.image_paths[idx]
        true_label = class_names[val_dataset.labels[idx]]
        
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt").to(config.device)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            confidence = probabilities.max().item()
            predicted_class_id = probabilities.argmax().item()
        
        predicted_label = class_names[predicted_class_id]
        
        axes[i].imshow(image)
        axes[i].axis('off')
        color = 'green' if predicted_label == true_label else 'red'
        title = f'True: {true_label}\nPred: {predicted_label}\nConf: {confidence:.3f}'
        axes[i].set_title(title, color=color, fontsize=10)
    
    plt.tight_layout()
    samples_path = os.path.join(config.output_dir, "sample_predictions.png")
    plt.savefig(samples_path, dpi=300)
    plt.show()
    print(f"-> Sample predictions plot saved to {samples_path}")


# ================================
# STEP 9: Main Training Pipeline
# ================================
def main():
    """
    Main function to run the entire pipeline.
    """
    print("=====================================================")
    print("=== Starting Bangladeshi Food Classification Task ===")
    print("=====================================================")
    
    # Setup data
    train_dataset, val_dataset, processor = setup_data()
    
    # Setup model
    num_classes = len(train_dataset.class_names)
    model = setup_model(num_classes)
    
    # Setup training
    trainer = setup_training(model, train_dataset, val_dataset)
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    print("Training complete!")
    
    # Save the best model and processor
    print("\nSaving best model and processor...")
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)
    print(f"-> Model saved to {config.output_dir}")
    
    # Final evaluation and visualization
    evaluate_and_visualize(trainer, val_dataset, train_dataset.class_names)
    
    print("\n=================================================")
    print("=== Pipeline Finished Successfully! ===")
    print("=================================================")


# ================================
# STEP 10: Run the Pipeline
# ================================
if __name__ == "__main__":
    # Verify that the dataset directory exists before starting
    if not os.path.exists(config.train_dir) or not os.path.exists(config.val_dir):
        print("\n" + "="*50)
        print("ERROR: Dataset not found!")
        print(f"Please ensure your data is in the correct directory structure:")
        print(f"  - Current Directory")
        print(f"    └── data/")
        print(f"        ├── train/")
        print(f"        │   ├── Biriyani/")
        print(f"        │   ├── Dal/")
        print(f"        │   └── ... (43 more class folders)")
        print(f"        └── validation/")
        print(f"            ├── Biriyani/")
        print(f"            ├── Dal/")
        print(f"            └── ... (43 more class folders)")
        print("="*50 + "\n")
    else:
        main()
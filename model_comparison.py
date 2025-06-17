# ==============================================================================
# Model Comparison Study for Bangladeshi Food Classification
# ==============================================================================

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score
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
# Configuration
# ================================
class Config:
    """
    Configuration class for model comparison study.
    """
    # Dataset paths
    dataset_path = "./data"
    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "validation")
    
    # Models to compare
    models = {
        "vit": "google/vit-base-patch16-224",
        "resnet": "microsoft/resnet-50",
        "convnext": "facebook/convnext-base-224",
        "efficientnet": "google/efficientnet-b0"
    }
    
    # Training parameters
    batch_size = 16
    num_epochs = 30
    learning_rate = 1e-5
    weight_decay = 0.05
    warmup_ratio = 0.2
    
    # Image parameters
    image_size = 224
    
    # Output directory for results
    output_dir = "./model_comparison_results"
    
    # Device configuration
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

# Create output directory
Path(config.output_dir).mkdir(parents=True, exist_ok=True)

# ================================
# Dataset Class
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
        common_transforms = [
            transforms.Resize((config.image_size, config.image_size)),
        ]
        
        # Enhanced augmentation for training
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
            image = self.transform(image)
            processed = self.processor(images=image, return_tensors="pt")
            return {
                'pixel_values': processed['pixel_values'].squeeze(0),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
        except Exception as e:
            print(f"Error loading image {image_path}: {e}. Skipping.")
            dummy_pixel_values = torch.zeros((3, config.image_size, config.image_size))
            return {'pixel_values': dummy_pixel_values, 'labels': torch.tensor(0, dtype=torch.long)}

# ================================
# Training Functions
# ================================
def compute_metrics(eval_pred):
    """
    Custom compute_metrics function for the Trainer.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

def train_and_evaluate_model(model_name, model_path, train_dataset, val_dataset, processor):
    """
    Train and evaluate a single model.
    """
    print(f"\n{'='*50}")
    print(f"Training {model_name} model...")
    print(f"{'='*50}")
    
    # Setup model
    model = AutoModelForImageClassification.from_pretrained(
        model_path,
        num_labels=len(train_dataset.class_names),
        ignore_mismatched_sizes=True
    ).to(config.device)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(config.output_dir, model_name),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=os.path.join(config.output_dir, model_name, "logs"),
        logging_steps=10,
        save_total_limit=2,
        gradient_accumulation_steps=4,
        fp16=False  # Disabled for MPS compatibility
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,  # Pass compute_metrics here
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train model
    trainer.train()
    
    # Evaluate model
    eval_results = trainer.evaluate()
    
    # Save results
    results = {
        'model_name': model_name,
        'accuracy': eval_results.get('eval_accuracy', 0.0),  # Use get() with default
        'loss': eval_results.get('eval_loss', 0.0),
        'class_names': train_dataset.class_names
    }
    
    # Save model and results
    model_save_path = os.path.join(config.output_dir, model_name)
    trainer.save_model(model_save_path)
    with open(os.path.join(model_save_path, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def compare_models():
    """
    Train and evaluate all models, then generate comparison report.
    Each model will use its own specific processor for data preparation.
    """
    all_results = []
    
    # Loop through each model to train and evaluate
    for model_name, model_path in config.models.items():
        
        # Load the correct processor for the CURRENT model
        print(f"\n{'='*20} Preparing data for: {model_name} {'='*20}")
        processor = AutoImageProcessor.from_pretrained(model_path)
        
        # Create fresh datasets with the correct processor
        train_dataset = BangladeshiFoodDataset(config.train_dir, processor, augment=True)
        val_dataset = BangladeshiFoodDataset(config.val_dir, processor, augment=False)
        
        # Train and evaluate the current model
        model_results = train_and_evaluate_model(
            model_name, model_path, train_dataset, val_dataset, processor
        )
        all_results.append(model_results)
    
    # Create comparison DataFrame
    df = pd.DataFrame(all_results)
    
    # Save comparison results
    df.to_csv(os.path.join(config.output_dir, 'model_comparison.csv'), index=False)
    
    # Generate comparison plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='model_name', y='accuracy')
    plt.title('Model Comparison - Accuracy', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.0) # Set y-axis to be from 0 to 1 for accuracy
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'model_comparison.png'))
    plt.show()
    
    print("\nModel comparison complete!")
    print(f"Results saved to: {config.output_dir}")
    print("\n--- Final Model Accuracies ---")
    print(df[['model_name', 'accuracy']].to_string(index=False))

if __name__ == "__main__":
    if not os.path.exists(config.train_dir):
        print("ERROR: Dataset not found! Please check the `dataset_path` in the Config class.")
    else:
        compare_models()
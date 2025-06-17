# ==============================================================================
# Parameter-Efficient Fine-Tuning (LoRA) for Bangladeshi Food Classification
# ==============================================================================

import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
# --- LoRA / PEFT Imports ---
from peft import get_peft_model, LoraConfig, TaskType

# Suppress excessive PIL logging
import logging
logging.getLogger("PIL").setLevel(logging.WARNING)

# ================================
# Configuration
# ================================
class Config:
    # Dataset paths
    dataset_path = "./data"
    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "validation")
    
    # Model to fine-tune with LoRA
    model_name_or_path = "google/vit-base-patch16-224"
    model_short_name = "vit_lora" # For output directory naming
    
    # Training parameters
    batch_size = 16
    num_epochs = 30
    learning_rate = 1e-3  # LoRA can often use a higher learning rate
    weight_decay = 0.01
    warmup_ratio = 0.1
    
    # Output directory for results
    output_dir = os.path.join("./lora_comparison_results", model_short_name)
    
    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

config = Config()
Path(config.output_dir).mkdir(parents=True, exist_ok=True)

# Helper function to see the benefit of LoRA
def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} || All params: {all_param} || "
        f"Trainable %: {100 * trainable_params / all_param:.2f}"
    )

# Use the same Dataset class as before
class BangladeshiFoodDataset(Dataset):
    def __init__(self, data_dir, processor, augment=False):
        self.processor = processor
        self.image_paths = []
        self.labels = []
        self.class_names = sorted([d.name for d in os.scandir(data_dir) if d.is_dir()])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        
        # Using a standard set of augmentations
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ])
        else:
            self.transform = transforms.Compose([]) # For validation, processor handles resizing

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.augment:
            image = self.transform(image)
        # The processor handles resizing, normalization, and conversion to tensors
        processed = self.processor(images=image, return_tensors="pt")
        return {
            'pixel_values': processed['pixel_values'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ================================
# Main LoRA Fine-Tuning Pipeline
# ================================
def main():
    print(f"--- Starting LoRA Fine-Tuning for {config.model_name_or_path} ---")

    # 1. Load the processor and create datasets
    processor = AutoImageProcessor.from_pretrained(config.model_name_or_path)
    train_dataset = BangladeshiFoodDataset(config.train_dir, processor, augment=True)
    val_dataset = BangladeshiFoodDataset(config.val_dir, processor, augment=False)
    num_classes = len(train_dataset.class_names)

    # 2. Load the base model
    model = AutoModelForImageClassification.from_pretrained(
        config.model_name_or_path,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    print("\n--- Model Parameters Before LoRA ---")
    print_trainable_parameters(model)

    # 3. Configure and apply LoRA
    print("\n--- Applying LoRA Adapters ---")
    lora_config = LoraConfig(
        r=16,  # Rank of the update matrices. Higher rank means more parameters.
        lora_alpha=32,  # Alpha scaling factor.
        target_modules=["query", "value"],  # Apply LoRA to query and value layers in attention blocks
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.IMAGE_CLASSIFICATION,
    )
    lora_model = get_peft_model(model, lora_config)
    
    print("\n--- Model Parameters After LoRA ---")
    print_trainable_parameters(lora_model)

    # 4. Set up Training Arguments and Trainer
    training_args = TrainingArguments(
        output_dir=config.output_dir,
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
        logging_dir=os.path.join(config.output_dir, "logs"),
        logging_steps=50,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 5. Train the model
    print("\n--- Starting Training ---")
    trainer.train()
    
    # 6. Evaluate and save results
    print("\n--- Evaluating Best Model ---")
    eval_results = trainer.evaluate()
    
    results_summary = {
        'model_name': config.model_short_name,
        'base_model': config.model_name_or_path,
        'accuracy': eval_results['eval_accuracy'],
        'loss': eval_results['eval_loss']
    }
    
    with open(os.path.join(config.output_dir, 'lora_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=4)
        
    print("\nLoRA fine-tuning complete!")
    print(f"Final Accuracy: {results_summary['accuracy']:.4f}")
    print(f"Results saved to: {config.output_dir}")

    # The trainer saves the adapter weights, not the full model
    lora_model.save_pretrained(os.path.join(config.output_dir, "final_lora_checkpoint"))


if __name__ == "__main__":
    if not os.path.exists(config.train_dir):
        print("ERROR: Dataset not found! Please check the `dataset_path` in the Config class.")
    else:
        main()
# Fine-tuning OpenAI CLIP for Bangladeshi Food Classification
# CLIP (Contrastive Language-Image Pre-training) by OpenAI

# ================================
# STEP 1: Installation
# ================================

"""
pip install torch torchvision transformers
pip install clip-by-openai
pip install open_clip_torch
pip install pillow matplotlib seaborn scikit-learn
pip install accelerate datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import clip
import open_clip
from tqdm import tqdm

# For Google Colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except:
    IN_COLAB = False

# ================================
# STEP 2: Configuration
# ================================

class Config:
    # Dataset paths
    if IN_COLAB:
        dataset_path = "/content/drive/MyDrive/dataset"  # Adjust this path
    else:
        dataset_path = "./data" # Corrected path to match your structure
    
    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "validation")
    
    # CLIP model options:
    # Option 1: Original OpenAI CLIP
    model_name = "ViT-B/32"  # Options: "ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"
    use_openai_clip = True
    
    # Option 2: Open CLIP (more models available)
    # model_name = "ViT-B-32"
    # pretrained = "laion2b_s34b_b79k"
    # use_openai_clip = False
    
    # Training parameters
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-5
    weight_decay = 0.01
    warmup_epochs = 2
    
    # Output directory
    output_dir = "./clip-bangladeshi-food"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# ================================
# STEP 3: Dataset Class for CLIP
# ================================

class CLIPFoodDataset(Dataset):
    def __init__(self, data_dir, preprocess, class_descriptions=None):
        self.data_dir = data_dir
        self.preprocess = preprocess
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        self.class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        
        # Create text descriptions for each class
        if class_descriptions is None:
            # Generate simple descriptions if none are provided
            self.class_descriptions = [f"a photo of {class_name.replace('_', ' ')}" 
                                       for class_name in self.class_names]
        else:
            # Use provided descriptions, ensuring order matches class_names
            self.class_descriptions = [class_descriptions.get(name, f"a photo of {name.replace('_', ' ')}") for name in self.class_names]

        print(f"Found {len(self.class_names)} classes in {os.path.basename(data_dir)} directory.")
        
        # Collect all image paths
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        
        print(f"Total images: {len(self.image_paths)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # Load and preprocess image
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            image = self.preprocess(image)
            
            # Get corresponding text description
            label = self.labels[idx]
            text_description = self.class_descriptions[label]
            
            return {
                'image': image,
                'text': text_description,
                'label': label,
                'image_path': image_path
            }
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            # Return dummy data to prevent training crash
            dummy_image = Image.new('RGB', (224, 224), color='white')
            return {
                'image': self.preprocess(dummy_image),
                'text': self.class_descriptions[0],
                'label': 0,
                'image_path': self.image_paths[idx]
            }

# ================================
# STEP 4: Enhanced Class Descriptions
# ================================

def create_bangladeshi_food_descriptions():
    """Create detailed descriptions for all 45 Bangladeshi food classes."""
    # This dictionary now contains all 45 descriptions.
    descriptions = {
        'Bakorkhani': 'A traditional, crispy, layered flatbread from Bangladesh.',
        'Balosai': 'A sweet, syrupy, donut-like dessert, often coated in sugar.',
        'Beef Kala Bhuna': 'A famous dark and tender beef curry from Chittagong, cooked with many spices.',
        'Beguni': 'Sliced eggplant fritters, battered in chickpea flour and deep-fried.',
        'Biryani': 'Aromatic rice dish with meat (chicken or mutton), potatoes, and fragrant spices.',
        'Chola_bhuna': 'A spicy and savory chickpea curry, often eaten as a snack or side dish.',
        'Cream Jam': 'A milk-based sweet, similar to Gulab Jamun but with a creamier texture.',
        'Dim Bhuna': 'A spicy egg curry where boiled eggs are cooked in a thick, flavorful gravy.',
        'Doi': 'Traditional Bangladeshi sweet yogurt.',
        'Dudh Puli PItha': 'A sweet dumpling made from rice flour, coconut, and milk.',
        'Egg_omlete': 'A simple pan-fried omelette, often with onions, chilies, and cilantro.',
        'Faluda': 'A cold dessert made with vermicelli, basil seeds, jelly, and ice cream.',
        'Golap Jam': 'A rose-flavored, deep-fried milk-solid sweet soaked in syrup.',
        'Hafsi Sondesh': 'A unique, dark, halwa-like sweet made from flour and nuts.',
        'Haleem': 'A thick stew of wheat, barley, lentils, and meat, slow-cooked to perfection.',
        'Hilsha_fish': 'The national fish of Bangladesh, often cooked in a mustard gravy (Shorshe Ilish).',
        'Jilapi': 'A deep-fried, coil-shaped sweet pretzel soaked in saffron-infused syrup.',
        'Kabab': 'Grilled or skewered meat, marinated in spices.',
        'Kacha Chana': 'A soft, uncooked cottage cheese sweet, often plain or lightly sweetened.',
        'Kacha Golla': 'A very soft, delicate milk-based sweet from the Natore region.',
        'Kalojam': 'A dark, oblong milk-solid sweet, similar to Gulab Jamun but darker and denser.',
        'Kasmiri Chomchom': 'A variety of chomchom sweet, often with a colorful, kasmiri-style topping.',
        'Khichuri': 'A comforting one-pot dish of rice and lentils cooked together, often with vegetables.',
        'Lemon Barfi': 'A fudge-like sweet with a zesty lemon flavor.',
        'Malaikari': 'A creamy and rich prawn curry cooked in coconut milk.',
        'Maoyaa Laddo': 'A type of laddoo made with mawa (milk solids).',
        'Mashed_potato': 'A simple dish of boiled and mashed potatoes, often spiced (Aloo Bhorta).',
        'Morog_polao': 'A fragrant pilaf cooked with chicken and aromatic spices, less spicy than biryani.',
        'Nehari': 'A slow-cooked stew of beef or lamb shanks with a rich, gelatinous gravy.',
        'Nokshi Pitha': 'Artistically designed rice cakes, often fried and soaked in syrup.',
        'Pakon Pitha': 'A diamond-shaped, fried rice cake that is crispy outside and soft inside.',
        'Panta Ilish': 'A traditional celebratory dish of fermented rice with fried hilsa fish.',
        'Patishapta': 'A thin crepe made from rice flour, filled with coconut or kheer.',
        'Payesh': 'A sweet and creamy rice pudding, flavored with cardamom and nuts.',
        'Phuchka': 'A popular street food; hollow, crispy shells filled with spiced potatoes and tamarind water.',
        'Porota': 'A layered, pan-fried flatbread, a staple for breakfast.',
        'Rajvog': 'A large, saffron-infused version of a rosogolla, often with a nutty filling.',
        'Ravogsai': 'A type of sweet, likely a regional variation.',
        'Roshmalai': 'Soft cheese dumplings soaked in a thickened, sweetened milk.',
        'Rosogolla': 'Spongy, white cheese balls soaked in a light sugar syrup.',
        'S.P.Chomchom': 'A specific variety of the chomchom sweet.',
        'Sada Chomchom': 'A plain, white version of the chomchom sweet.',
        'Sajer Pitha': 'A type of rice cake, often made for special occasions.',
        'Shamok Pitha': 'A snail-shaped rice flour dumpling, can be sweet or savory.',
        'Shemai': 'Sweet vermicelli cooked in milk, a festive dessert especially for Eid.'
    }
    return descriptions

# ================================
# STEP 5: CLIP Model Setup
# ================================

def setup_clip_model():
    """Load CLIP model and preprocessing"""
    print(f"Loading CLIP model: {config.model_name}")
    
    if config.use_openai_clip:
        # Use original OpenAI CLIP
        model, preprocess = clip.load(config.model_name, device=config.device)
        tokenizer = clip.tokenize
    else:
        # Use Open CLIP
        model, _, preprocess = open_clip.create_model_and_transforms(
            config.model_name, pretrained=config.pretrained, device=config.device
        )
        tokenizer = open_clip.get_tokenizer(config.model_name)
    
    return model, preprocess, tokenizer

# ================================
# STEP 6: CLIP Fine-tuning Class (Using a Classification Head)
# ================================

class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes, freeze_clip=False):
        super().__init__()
        self.clip_model = clip_model
        self.num_classes = num_classes
        self.freeze_clip = freeze_clip # Store freeze_clip state
        
        # Freeze CLIP parameters if specified
        if self.freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Get CLIP embedding dimension
        with torch.no_grad():
            dummy_image = torch.randn(1, 3, 224, 224).to(config.device)
            image_features = self.clip_model.encode_image(dummy_image)
            embed_dim = image_features.shape[-1]
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def forward(self, images):
        # Get image features from CLIP
        # Conditionally enable gradients based on the freeze_clip flag
        with torch.no_grad() if self.freeze_clip else torch.enable_grad():
            image_features = self.clip_model.encode_image(images)
        
        # Classification
        # The classifier part always requires gradients
        logits = self.classifier(image_features.float())
        
        return logits

# ================================
# STEP 7: Zero-shot Classification (Baseline)
# ================================

def evaluate_zero_shot(model, tokenizer, val_dataset):
    """Evaluate CLIP's zero-shot performance"""
    print("Evaluating zero-shot performance...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    # Prepare text descriptions
    text_descriptions = val_dataset.class_descriptions
    text_tokens = tokenizer(text_descriptions, context_length=77, truncate=True).to(config.device)
    
    # Get text embeddings
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Evaluate on validation set
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Zero-shot evaluation"):
            images = batch['image'].to(config.device)
            labels = batch['label']
            
            # Get image features
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarities
            similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            predictions = similarities.argmax(dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate accuracy
    zero_shot_accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Zero-shot accuracy: {zero_shot_accuracy:.4f}")
    
    return zero_shot_accuracy

# ================================
# STEP 8: Training Function
# ================================

def train_clip_classifier(model, train_loader, val_loader, num_classes):
    """Train CLIP-based classifier"""
    
    # Create classifier, set freeze_clip=False for fine-tuning
    classifier = CLIPClassifier(model, num_classes, freeze_clip=False).to(config.device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(classifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_accuracy = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(config.num_epochs):
        # Training phase
        classifier.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Training]")
        for batch in progress_bar:
            images = batch['image'].to(config.device)
            labels = batch['label'].to(config.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = classifier(images)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Validation phase
        classifier.eval()
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Validation]"):
                images = batch['image'].to(config.device)
                labels = batch['label']
                
                logits = classifier(images)
                predictions = logits.argmax(dim=-1)
                
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.numpy())
        
        # Calculate metrics
        avg_loss = epoch_loss / len(train_loader)
        val_accuracy = accuracy_score(val_labels, val_predictions)
        
        train_losses.append(avg_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(classifier.state_dict(), f"{config.output_dir}/best_clip_classifier.pth")
            print(f"New best accuracy: {best_accuracy:.4f}. Model saved.")
        
        scheduler.step()
    
    return classifier, train_losses, val_accuracies, best_accuracy

# ================================
# STEP 9: Evaluation and Visualization
# ================================

def evaluate_final_model(classifier, val_loader, class_names):
    """Final evaluation with detailed metrics"""
    classifier.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final Evaluation"):
            images = batch['image'].to(config.device)
            labels = batch['label']
            
            logits = classifier(images)
            predictions = logits.argmax(dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, target_names=class_names, zero_division=0)
    
    print(f"\nFinal Accuracy: {accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print(report)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - CLIP Fine-tuned', fontsize=20)
    plt.xlabel('Predicted', fontsize=16)
    plt.ylabel('Actual', fontsize=16)
    plt.xticks(rotation=90, ha='center')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/confusion_matrix_clip.png", dpi=300)
    plt.show()
    
    return accuracy

def plot_training_history(train_losses, val_accuracies):
    """Plot training history"""
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(train_losses, 'b-', label='Training Loss')
    ax1.set_title('Training Loss per Epoch', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy per Epoch', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{config.output_dir}/training_history_clip.png", dpi=300)
    plt.show()

# ================================
# STEP 10: Main Training Pipeline
# ================================

def main():
    print("Starting CLIP Fine-tuning for Bangladeshi Food Classification...")
    print(f"Using device: {config.device}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load CLIP model
    clip_model, preprocess, tokenizer = setup_clip_model()
    
    # Create enhanced descriptions
    food_descriptions = create_bangladeshi_food_descriptions()
    
    # Setup datasets
    print("\nSetting up datasets...")
    train_dataset = CLIPFoodDataset(config.train_dir, preprocess, food_descriptions)
    val_dataset = CLIPFoodDataset(config.val_dir, preprocess, food_descriptions)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.class_names)}")
    
    # Evaluate zero-shot performance first
    zero_shot_acc = evaluate_zero_shot(clip_model, tokenizer, val_dataset)
    
    # Fine-tune the model
    print("\nStarting fine-tuning...")
    classifier, train_losses, val_accuracies, best_accuracy = train_clip_classifier(
        clip_model, train_loader, val_loader, len(train_dataset.class_names)
    )
    
    # Load the best performing model for final evaluation
    print("\nLoading best model for final evaluation...")
    classifier.load_state_dict(torch.load(f"{config.output_dir}/best_clip_classifier.pth"))
    
    # Final evaluation
    print("Final evaluation on validation set...")
    final_accuracy = evaluate_final_model(classifier, val_loader, train_dataset.class_names)
    
    # Plot training history
    plot_training_history(train_losses, val_accuracies)
    
    # Save results
    results = {
        'model_name': config.model_name,
        'zero_shot_accuracy': zero_shot_acc,
        'best_fine_tuned_accuracy': best_accuracy,
        'final_accuracy': final_accuracy,
        'class_names': train_dataset.class_names,
        'class_descriptions': train_dataset.class_descriptions
    }
    
    with open(f"{config.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n--- Results Summary ---")
    print(f"Zero-shot accuracy: {zero_shot_acc:.4f}")
    print(f"Best fine-tuned accuracy: {best_accuracy:.4f}")
    print(f"Improvement over zero-shot: {best_accuracy - zero_shot_acc:.4f}")
    
    return classifier, clip_model, preprocess, train_dataset.class_names

# ================================
# STEP 11: Inference Function
# ================================

def predict_with_clip(image_path, classifier, clip_model, preprocess, class_names):
    """Make prediction on a single image"""
    classifier.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).to(config.device)
    
    # Make prediction
    with torch.no_grad():
        logits = classifier(image_tensor)
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class_id = probabilities.argmax().item()
        confidence = probabilities.max().item()
    
    predicted_class = class_names[predicted_class_id]
    
    print(f"Prediction for '{os.path.basename(image_path)}':")
    print(f"  - Class: {predicted_class}")
    print(f"  - Confidence: {confidence:.2%}")
    return predicted_class, confidence, probabilities[0].cpu().numpy()

# ================================
# STEP 12: Run Training
# ================================

if __name__ == "__main__":
    # Check dataset path
    if not os.path.exists(config.train_dir):
        print(f"Error: Dataset not found at '{config.train_dir}'")
        print("Please update the dataset_path in the Config class or check the directory structure.")
    else:
        # Run the training pipeline
        classifier, clip_model, preprocess, class_names = main()
        
        print("\n--- Training completed successfully! ---")
        print("You can now use predict_with_clip() for inference on new images.")
        
        # Example of how to run inference:
        # val_images = val_dataset.image_paths
        # if val_images:
        #    random_image_path = np.random.choice(val_images)
        #    predict_with_clip(random_image_path, classifier, clip_model, preprocess, class_names)
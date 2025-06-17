import ollama
import os
import base64
import pandas as pd
import numpy as np
import random
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.autonotebook import tqdm
from pathlib import Path

# ================================
# --- CONFIGURATION ---
# ================================
class LLaVAConfig:
    # Model name as it appears in Ollama
    MODEL_NAME = "llava:7b"
    
    # Paths
    DATASET_PATH = "./data"
    TRAIN_DIR = os.path.join(DATASET_PATH, "train") # Needed for few-shot examples
    VAL_DIR = os.path.join(DATASET_PATH, "validation")
    OUTPUT_DIR = "llava-results"

    # Evaluation settings
    # Set to a small number (e.g., 5) for a quick test run, or None to run on all images
    MAX_IMAGES_PER_CLASS_TO_TEST = 5 
    
    # Few-shot settings
    # Number of example images to provide in the prompt for few-shot learning
    NUM_FEW_SHOT_EXAMPLES = 3 # (e.g., k=3)

# Initialize configuration
config = LLaVAConfig()

# ================================
# --- HELPER FUNCTIONS ---
# ================================
def encode_image(image_path: str) -> str:
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_llava_response(response: str, class_names: list) -> str:
    """Parses the text response from LLaVA to find the most likely class name."""
    response_lower = response.lower()
    # First, check for an exact match
    for name in class_names:
        if name.lower() in response_lower:
            return name
    # If no exact match, return the raw response for manual checking
    return "unparsed_response"

# ================================
# --- CORE CLASSIFICATION LOGIC ---
# ================================
def classify_image_with_llava(client, image_path: str, class_names: list, few_shot_examples: list = None):
    """
    Classifies a single image using the local LLaVA model via Ollama.
    Supports both zero-shot and in-context few-shot learning.
    """
    image_b64 = encode_image(image_path)
    all_images_b64 = [image_b64]
    
    class_list_str = ", ".join(class_names)
    
    if few_shot_examples:
        # --- In-Context Few-Shot Prompt ---
        prompt_text = "Here are some examples of Bangladeshi food.\n"
        example_images_b64 = []
        for i, (ex_path, ex_label) in enumerate(few_shot_examples):
            prompt_text += f"Example {i+1} is '{ex_label}'.\n"
            example_images_b64.append(encode_image(ex_path))
        
        prompt_text += f"\nBased on these examples, what food is in the final image? Choose from this list: {class_list_str}. Respond with only the best class name."
        # Prepend example images to the list
        all_images_b64 = example_images_b64 + all_images_b64
    else:
        # --- Zero-Shot Prompt ---
        prompt_text = f"What type of Bangladeshi food is in this image? Choose from the following list: {class_list_str}. Respond with only the single best class name."

    try:
        response = client.generate(
            model=config.MODEL_NAME,
            prompt=prompt_text,
            images=all_images_b64,
            stream=False
        )
        prediction_raw = response['response']
        prediction = parse_llava_response(prediction_raw, class_names)
        return {'success': True, 'prediction': prediction}
    except Exception as e:
        return {'success': False, 'prediction': None, 'error': str(e)}

# ================================
# --- EVALUATION WORKFLOWS ---
# ================================
def run_evaluation(client, mode: str, class_names: list, train_image_map: dict = None):
    """A general function to run evaluation for either zero-shot or few-shot."""
    print(f"\n--- Starting {mode.upper()} Evaluation ---")
    
    image_paths = []
    for class_name in class_names:
        class_dir = os.path.join(config.VAL_DIR, class_name)
        class_images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.lower().endswith(('.png', '.jpg', 'jpeg'))]
        if config.MAX_IMAGES_PER_CLASS_TO_TEST:
            image_paths.extend(random.sample(class_images, min(len(class_images), config.MAX_IMAGES_PER_CLASS_TO_TEST)))
        else:
            image_paths.extend(class_images)

    results = []
    pbar = tqdm(image_paths, desc=f"LLaVA {mode.upper()} Evaluation")
    for image_path in pbar:
        ground_truth = Path(image_path).parent.name
        examples = None
        if mode == 'few-shot':
            # Select k random examples from other classes for in-context learning
            examples = []
            possible_classes = [c for c in class_names if c != ground_truth]
            for _ in range(config.NUM_FEW_SHOT_EXAMPLES):
                chosen_class = random.choice(possible_classes)
                chosen_image = random.choice(train_image_map[chosen_class])
                examples.append((chosen_image, chosen_class))

        result = classify_image_with_llava(client, image_path, class_names, few_shot_examples=examples)
        results.append({
            'image_path': image_path,
            'ground_truth': ground_truth,
            'prediction': result['prediction'],
            'success': result.get('success', False)
        })

    df = pd.DataFrame(results)
    output_csv_path = os.path.join(config.OUTPUT_DIR, f"llava_results_{mode}.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"✅ {mode.upper()} evaluation complete. Results saved to {output_csv_path}")
    return df

def analyze_results(df, class_names, mode: str):
    """Analyzes results and saves report and confusion matrix."""
    print(f"\n--- Analyzing {mode.upper()} Results ---")
    df.dropna(subset=['prediction', 'ground_truth'], inplace=True)
    
    y_true = df['ground_truth']
    y_pred = df['prediction']
    
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=class_names, digits=4, zero_division=0)

    report_content = f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n" + report
    print(report_content)
    
    report_path = os.path.join(config.OUTPUT_DIR, f"llava_report_{mode}.txt")
    with open(report_path, 'w') as f:
        f.write(f"LLaVA {mode.upper()} Classification Report\n" + "="*50 + "\n" + report_content)
    print(f"✅ Report saved to {report_path}")

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(22, 18))
    sns.heatmap(cm, annot=False, cmap='viridis', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'LLaVA Confusion Matrix - {mode.upper()}', fontsize=20)
    plt.ylabel('True Label', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(config.OUTPUT_DIR, f"llava_cm_{mode}.png")
    plt.savefig(cm_path, dpi=300)
    print(f"✅ Confusion matrix saved to {cm_path}\n")
    plt.close()

# ================================
# --- MAIN EXECUTION ---
# ================================
def main():
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    client = ollama.Client()
    
    class_names = sorted([d.name for d in os.scandir(config.VAL_DIR) if d.is_dir()])
    
    # --- Run Zero-Shot Evaluation ---
    zero_shot_df = run_evaluation(client, 'zero-shot', class_names)
    analyze_results(zero_shot_df, class_names, 'zero-shot')
    
    # --- Run Few-Shot Evaluation ---
    # First, map all training images to their class for quick lookups
    train_image_map = {c: [os.path.join(config.TRAIN_DIR, c, img) for img in os.listdir(os.path.join(config.TRAIN_DIR, c))] for c in class_names}
    few_shot_df = run_evaluation(client, 'few-shot', class_names, train_image_map)
    analyze_results(few_shot_df, class_names, 'few-shot')

if __name__ == "__main__":
    main()
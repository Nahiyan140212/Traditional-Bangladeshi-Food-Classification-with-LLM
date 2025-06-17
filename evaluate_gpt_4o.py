# ==============================================================================
# GPT-4o API Integration for Bangladeshi Food Classification
# For Inference and Evaluation (Zero-Shot Baseline)
# ==============================================================================
# WARNING: THIS SCRIPT USES THE PAID OPENAI API. BE MINDFUL OF COSTS.
# ==============================================================================

# ================================
# STEP 1: Installation
# ================================
# pip install openai pandas scikit-learn matplotlib seaborn pillow

import openai
import json
import os
import base64
from PIL import Image
import pandas as pd
from typing import List, Dict
import time
from tqdm.autonotebook import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ================================
# STEP 2: Configuration
# ================================
class GPT4oConfig:
    # API Key: Best practice is to set this as an environment variable
    # In your terminal: export OPENAI_API_KEY='your-key-here'
    api_key = os.getenv("OPENAI_API_KEY")

    # Model configuration
    model = "gpt-4o"  # Current model for vision tasks
    max_tokens = 50
    temperature = 0.0 # Zero temperature for deterministic classification

    # Define the path to your training data directory
    train_dir_for_classes = "./data/train"

    # Get class names by listing the directories inside the train folder
    # We sort them to ensure a consistent order every time
    try:
        food_classes = sorted([d.name for d in os.scandir(train_dir_for_classes) if d.is_dir()])
        print(f"✅ Found {len(food_classes)} classes automatically.")
    except FileNotFoundError:
        print(f"❌ ERROR: Directory not found at '{train_dir_for_classes}'. Please check the path.")
        food_classes = [] # Set to empty list to avoid further errors
        
    # Dataset path for evaluation
    val_dir = "./data/validation"
    
    # Path to save results
    output_dir = "./gpt4o-results"
    results_csv = os.path.join(output_dir, "gpt4o_classification_results.csv")

config = GPT4oConfig()
Path(config.output_dir).mkdir(parents=True, exist_ok=True)


# ================================
# STEP 3: API Interaction Functions
# ================================
def encode_image(image_path: str) -> str:
    """Encodes image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def classify_image_with_gpt4o(client, image_path: str, food_classes: List[str]) -> Dict:
    """
    Classifies a single image using the GPT-4o API with a robust JSON response format.
    """
    try:
        base64_image = encode_image(image_path)
        classes_str = ", ".join(food_classes)

        # New, more robust prompt asking for JSON output
        prompt = f"""
        You are a food classification expert for Bangladeshi cuisine. Analyze the image
        and classify it into ONE of the following categories: {classes_str}.

        Your response MUST be a JSON object with a single key "prediction".
        Example: {{"prediction": "Biryani"}}
        """

        response = client.chat.completions.create(
            model=config.model,
            # This enables JSON mode for more reliable output
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        },
                    ],
                }
            ],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        
        # Parse the JSON response
        response_data = json.loads(response.choices[0].message.content)
        prediction = response_data.get("prediction", "parsing_error")

        return {'success': True, 'prediction': prediction, 'error': None}
    
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return {'success': False, 'prediction': None, 'error': str(e)}

# def classify_image_with_gpt4o(client, image_path: str, food_classes: List[str]) -> Dict:
#     """Classifies a single image using the GPT-4o API."""
#     try:
#         base64_image = encode_image(image_path)
#         classes_str = ", ".join(food_classes)

#         prompt = f"""
#         You are a food classification expert specializing in Bangladeshi cuisine.
#         Analyze the provided image and classify it into ONLY ONE of the following categories:
#         ---
#         {classes_str}
#         ---
#         Your response must be a single word or phrase exactly as it appears in the list above. Do not add any extra text, explanation, or punctuation.
#         """

#         response = client.chat.completions.create(
#             model=config.model,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": prompt},
#                         {
#                             "type": "image_url",
#                             "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
#                         },
#                     ],
#                 }
#             ],
#             max_tokens=config.max_tokens,
#             temperature=config.temperature,
#         )
#         prediction = response.choices[0].message.content.strip().lower()

#         # Simple validation to find the closest match
#         if prediction not in food_classes:
#             for cls in food_classes:
#                 if prediction in cls or cls in prediction:
#                     prediction = cls
#                     break

#         return {'success': True, 'prediction': prediction, 'error': None}
    
#     except Exception as e:
#         print(f"\nAn error occurred: {e}")
#         return {'success': False, 'prediction': None, 'error': str(e)}

# ================================
# STEP 4: Batch Processing and Evaluation
# ================================
def run_batch_classification(client, max_images_per_class: int = 5):
    """
    Runs classification on a sample of the dataset and saves results to a CSV.
    """
    print("--- Starting GPT-4o Batch Classification ---")
    print(f"⚠️ Cost Warning: Processing up to {max_images_per_class} images per class.")
    
    results = []
    image_paths_to_process = []
    
    # Collect image paths
    for class_name in config.food_classes:
        class_dir = os.path.join(config.val_dir, class_name)
        if os.path.isdir(class_dir):
            images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_paths_to_process.extend(images[:max_images_per_class])

    # Process images with a progress bar
    for image_path in tqdm(image_paths_to_process, desc="Classifying with GPT-4o"):
        ground_truth = os.path.basename(os.path.dirname(image_path))
        result = classify_image_with_gpt4o(client, image_path, config.food_classes)
        
        results.append({
            'image_path': image_path,
            'ground_truth': ground_truth,
            'prediction': result['prediction'],
            'success': result['success'],
            'error': result['error']
        })
        time.sleep(1) # Delay to respect rate limits

    df = pd.DataFrame(results)
    df.to_csv(config.results_csv, index=False)
    print(f"\nClassification complete. Results saved to {config.results_csv}")
    return df

def analyze_results_from_csv(csv_path: str):
    """Analyzes the results from the saved CSV file and generates a report."""
    if not os.path.exists(csv_path):
        print(f"Error: Results file not found at {csv_path}")
        return

    print("\n--- Analyzing GPT-4o Classification Results ---")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['prediction', 'ground_truth']) # Ignore failed API calls
    
    y_true = df['ground_truth']
    y_pred = df['prediction']
    
    # 1. Generate and save Classification Report
    report = classification_report(y_true, y_pred, labels=config.food_classes, digits=4, zero_division=0)
    report_path = os.path.join(config.output_dir, "gpt4o_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("GPT-4o Zero-Shot Classification Report\n")
        f.write("="*40 + "\n\n")
        f.write(report)
    print("GPT-4o Classification Report:")
    print(report)
    print(f"-> Report saved to {report_path}")

    # 2. Plot and Save Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=config.food_classes)
    plt.figure(figsize=(20, 17))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=config.food_classes, yticklabels=config.food_classes)
    plt.title('GPT-4o Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    cm_path = os.path.join(config.output_dir, "gpt4o_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"-> Confusion matrix saved to {cm_path}")


# ================================
# STEP 5: Main Execution
# ================================
def main():
    if not config.api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set it in your terminal: export OPENAI_API_KEY='your-key'")
        return

    # Initialize OpenAI client with minimal configuration
    client = openai.OpenAI(api_key=config.api_key)
    
    # --- Part 1: Run classification and save to CSV ---
    # To re-run analysis without calling the API again, comment out this line.
    run_batch_classification(client, max_images_per_class=5)
    
    # --- Part 2: Analyze results from the saved CSV ---
    analyze_results_from_csv(config.results_csv)


if __name__ == "__main__":
    # Ensure your class list is populated before running
    if len(config.food_classes) < 45:
         print("⚠️ WARNING: Please update the `food_classes` list in the GPT4oConfig class with all 45 of your food names.")
    else:
        main()
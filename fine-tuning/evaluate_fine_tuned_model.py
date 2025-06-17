import openai
import os
import base64
from PIL import Image
import pandas as pd
from typing import List, Dict
import time
from tqdm.autonotebook import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ================================
# --- CONFIGURATION ---
# ================================
class FineTunedConfig:
    # ⚠️ IMPORTANT: Paste your custom model ID here after fine-tuning is complete.
    # It will look like: ft:gpt-4o-2024-05-13:your-org::xxxxxxxx
    # FINE_TUNED_MODEL_ID = "ft:gpt-4o-2024-08-06:personal::BjEnXtK7"
    FINE_TUNED_MODEL_ID = "ft:gpt-4o-2024-08-06:personal:few-shot-fine-tuning:BjFG4FMO"
    # API Key from environment variable
    API_KEY = os.getenv("OPENAI_API_KEY")

    # Paths
    VALIDATION_DIR = "../data/validation"
    OUTPUT_DIR = "finetuned-results/few_shot_fine_tuning"
    RESULTS_CSV = os.path.join(OUTPUT_DIR, "few_shot_finetuned_classification_results.csv")
    ANALYSIS_REPORT_PATH = os.path.join(OUTPUT_DIR, 'few_shot_finetuned_detailed_report.txt')
    CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, 'few_shot_finetuned_confusion_matrix.png')
    
    # API settings
    MAX_TOKENS = 10
    TEMPERATURE = 0.0

# Initialize the configuration
config = FineTunedConfig()

# ================================
# --- HELPER & API FUNCTIONS ---
# ================================
def encode_image(image_path: str) -> str:
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def classify_with_finetuned_model(client, image_path: str) -> Dict:
    """Classifies a single image using your custom fine-tuned model."""
    try:
        base64_image = encode_image(image_path)
        
        # The prompt is simple because the model is now specialized
        prompt_text = "What type of Bangladeshi food is in this image?"
        
        response = client.chat.completions.create(
            model=config.FINE_TUNED_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}
                        },
                    ],
                }
            ],
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
        )
        prediction = response.choices[0].message.content.strip()
        return {'success': True, 'prediction': prediction, 'error': None}
    
    except Exception as e:
        print(f"\nAn error occurred processing {os.path.basename(image_path)}: {e}")
        return {'success': False, 'prediction': None, 'error': str(e)}

# ================================
# --- CORE WORKFLOW FUNCTIONS ---
# ================================
def run_full_evaluation(client):
    """
    Iterates through the entire validation set, classifies each image,
    and saves the results incrementally to a CSV file.
    """
    print("--- Starting Full Validation Set Evaluation ---")
    
    image_paths = []
    class_names = sorted([d.name for d in os.scandir(config.VALIDATION_DIR) if d.is_dir()])
    for class_name in class_names:
        class_dir = os.path.join(config.VALIDATION_DIR, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', 'jpeg')):
                image_paths.append(os.path.join(class_dir, img_name))
                
    print(f"Found {len(image_paths)} images to evaluate across {len(class_names)} classes.")

    results = []
    pbar = tqdm(image_paths, desc="Evaluating with Fine-Tuned Model")
    for image_path in pbar:
        ground_truth = Path(image_path).parent.name
        result = classify_with_finetuned_model(client, image_path)
        
        results.append({
            'image_path': image_path,
            'ground_truth': ground_truth,
            'prediction': result['prediction'],
            'success': result['success'],
            'error': result['error']
        })
        time.sleep(0.5) # Be kind to the API and avoid rate limits

    df = pd.DataFrame(results)
    df.to_csv(config.RESULTS_CSV, index=False)
    print(f"\nEvaluation complete. Raw results saved to {config.RESULTS_CSV}")
    return df, class_names

def analyze_results(df, class_names):
    """Analyzes the results DataFrame and generates the final report and plot."""
    print("\n--- Analyzing Fine-Tuned Model Performance ---")

    df.dropna(subset=['ground_truth', 'prediction'], inplace=True)
    df['ground_truth_lower'] = df['ground_truth'].str.lower()
    df['prediction_lower'] = df['prediction'].str.lower()

    y_true = df['ground_truth_lower']
    y_pred = df['prediction_lower']
    
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=sorted(y_true.unique()), digits=4, zero_division=0)

    report_content = f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n" + report
    print(report_content)
    with open(config.ANALYSIS_REPORT_PATH, 'w') as f:
        f.write("="*50 + "\nDetailed Classification Report (Fine-Tuned Model)\n" + "="*50 + "\n\n" + report_content)
    print(f"\n✅ Detailed text report saved to: {config.ANALYSIS_REPORT_PATH}")

    cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))
    plt.figure(figsize=(22, 18))
    sns.heatmap(cm, annot=False, cmap='Greens', xticklabels=sorted(y_true.unique()), yticklabels=sorted(y_true.unique()))
    plt.title('Confusion Matrix - Fine-Tuned GPT-4o', fontsize=20)
    plt.ylabel('True Label', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=300)
    print(f"✅ Confusion matrix plot saved to: {config.CONFUSION_MATRIX_PATH}")
    plt.show()

# ================================
# --- MAIN EXECUTION ---
# ================================
def main():
    if not config.API_KEY:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set.")
        return
        
    if "ft:gpt-4o" not in config.FINE_TUNED_MODEL_ID:
        print("❌ ERROR: Please paste your actual fine-tuned model ID into the `FINE_TUNED_MODEL_ID` variable in the script.")
        return

    # Ensure the output directory exists
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    client = openai.OpenAI(api_key=config.API_KEY)

    # Run the full evaluation and save the results
    results_df, class_names = run_full_evaluation(client)
    
    # Analyze the results to generate the final report
    analyze_results(results_df, class_names)

if __name__ == "__main__":
    main()
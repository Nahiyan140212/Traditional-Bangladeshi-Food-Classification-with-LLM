import os
import json
import random

# --- CONFIGURATION ---
TRAIN_DATA_DIR = "../data/train"
# MAX_IMAGES_PER_CLASS = 50 # for fine tuning 
MAX_IMAGES_PER_CLASS = 5 # for few shot learning
OUTPUT_FILE = "finetuning_data_5shot_lowres.jsonl"

# ⚠️ IMPORTANT: Replace these with your S3 bucket name and region.
BUCKET_NAME = "my-bangla-food-dataset-2025"
BUCKET_REGION = "us-east-1" 
BASE_IMAGE_URL = f"https://{BUCKET_NAME}.s3.{BUCKET_REGION}.amazonaws.com"
# --- END CONFIGURATION ---

def create_finetuning_dataset():
    """
    Scans the local dataset, takes a random subset of images for each class,
    and creates a .jsonl file with low-resolution image URLs for a cheap experiment.
    """
    print(f"--- Starting CHEAP EXPERIMENT data preparation ---")
    print(f"Creating a small dataset with {MAX_IMAGES_PER_CLASS} images per class.")
    
    total_images_processed = 0
    with open(OUTPUT_FILE, 'w') as f:
        for class_name in sorted(os.listdir(TRAIN_DATA_DIR)):
            class_dir = os.path.join(TRAIN_DATA_DIR, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            print(f"Processing class: {class_name}")
            
            image_files = [img for img in os.listdir(class_dir) if img.lower().endswith(('.png', '.jpg', 'jpeg'))]
            # Shuffle the list to get a random sample of images
            random.shuffle(image_files)
            
            # Take only the first N images after shuffling
            image_subset = image_files[:MAX_IMAGES_PER_CLASS]

            for image_name in image_subset:
                # Construct the full public S3 URL
                image_url = f"{BASE_IMAGE_URL}/data/train/{class_name}/{image_name}"
                
                # Set image detail to "low" for massive token savings ---
                data_entry = {
                    "messages": [
                        {"role": "system", "content": "You are an expert that identifies the Bangladeshi food both traditional and hybrid in an image."},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}}
                        ]},
                        {"role": "assistant", "content": class_name}
                    ]
                }
                
                f.write(json.dumps(data_entry) + "\n")
                total_images_processed += 1
                    
    print(f"\n✅ Successfully created SMALL, LOW-RESOLUTION dataset at: {OUTPUT_FILE}")
    print(f"Total images in dataset: {total_images_processed}")

if __name__ == "__main__":
    create_finetuning_dataset()
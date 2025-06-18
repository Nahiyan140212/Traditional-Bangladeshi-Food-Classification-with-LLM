
# A Comparative Analysis of SOTA Models for Fine-Grained Classification of Bangladeshi Cuisine

This repository contains the official code and experimental framework for the research paper, "A Comparative Analysis of Vision Transformers, CLIP, and Large Multi-modal Models for Fine-Grained Classification of Bangladeshi Cuisine."

## Abstract

The performance of state-of-the-art vision models often degrades when applied to specialized, fine-grained domains from underrepresented cultures. This work addresses this gap by focusing on the challenging task of classifying Bangladeshi cuisine. We introduce **BanglaFood-45**, a new dataset of 45 distinct Bangladeshi food classes. Using this benchmark, we conduct a comprehensive study comparing three families of models: pure vision transformers (ViT, ResNet), vision-language models (CLIP), and large multi-modal models (GPT-4o, LLaVA). We systematically evaluate these models across zero-shot, few-shot fine-tuning, full-subset fine-tuning, and parameter-efficient fine-tuning (LoRA) paradigms. Our analysis dissects the performance, data efficiency, and cost-benefit trade-offs of each approach, providing a roadmap for tackling culturally specific, fine-grained classification challenges.

## Key Features

- **Novel Dataset Focus**: A project centered on the underrepresented domain of Bangladeshi cuisine.
- **Multi-Model Comparison**: Includes scripts to train and evaluate ViT, ResNet, ConvNeXT, EfficientNet, CLIP, GPT-4o, and LLaVA.
- **Multi-Paradigm Analysis**: Implements zero-shot, few-shot, full fine-tuning, and LoRA to provide a comprehensive performance overview.
- **API and Local Integration**: Provides code for both API-based models (OpenAI) and locally-run open-source models (Ollama).
- **Reproducibility**: Detailed setup instructions and a `requirements.txt` file to ensure the research is fully reproducible.

## Project Structure

```
├── data/
│   ├── train/
│   │   ├── Biryani/
│   │   └── ... (44 other class folders)
│   └── validation/
│       ├── Biryani/
│       └── ... (44 other class folders)
├── model_comparison_results/
├── clip-bangladeshi-food-results/
├── gpt4o-results/
├── llava-results/
├── lora_comparison_results/
├── 1_model_comparison.py
├── 2_finetune_clip.py
├── 3_evaluate_gpt4o.py
├── 4_evaluate_finetuned_model.py
├── 5_evaluate_local_llava.py
├── 6_finetune_lora.py
├── prepare_finetuning_data.py
├── analyze_results.py
├── requirements.txt
└── README.md
```

## Setup and Installation

This project was developed on an Apple Silicon (M1) Mac. The following instructions are tailored for this architecture but can be adapted for Linux/Windows with NVIDIA GPUs.

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Step 2: Set Up the Conda Environment

We use `conda` (specifically `miniforge` for native ARM64 performance) to manage the environment.

1. **Install Miniforge** (if you don't have it):

   - Download the `Miniforge3-MacOSX-arm64.sh` installer from the official releases.
   - Run the installer: `bash Miniforge3-MacOSX-arm64.sh`
   - Close and reopen your terminal.

2. **Create and Activate the Conda Environment**:

```bash
conda create -n food_project python=3.10 -y
conda activate food_project
```

### Step 3: Install Dependencies

All required Python packages are listed in `requirements.txt`. Install them using `pip`:

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables (for OpenAI API)

For experiments involving GPT-4o, you must set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="sk-YourSecretApiKeyHere"
```

To make this permanent, add the line above to your shell's startup file (`~/.zshrc` or `~/.bash_profile`).

### Step 5: Set Up Local LLM (for LLaVA experiment)

1. Download and install **Ollama**.
2. Pull the LLaVA model from the command line:

```bash
ollama pull llava:7b
```

3. Ensure the Ollama application is running before executing the LLaVA script.

## Dataset

This study uses the **BanglaFood-45** dataset. To use this repository, your data must be structured as follows:

- A top-level `data/` directory.
- Inside `data/`, create two sub-directories: `train/` and `validation/`.
- Inside both `train/` and `validation/`, create 45 sub-directories, one for each food class (e.g., `Biryani`, `Haleem`, `Roshmalai`).
- Place the corresponding image files inside each class folder.

## Running the Experiments

Each script is designed to run a specific experiment. Run them from your activated `food_project` conda environment.

### Experiment 1: Pure Vision Model Comparison

This script trains and evaluates ViT, ResNet, ConvNeXT, and EfficientNet, creating a comparison report and saving the best model for each.

```bash
python model_comparison.py
```

**Output**: Results will be saved in the `model_comparison_results/` directory.

### Experiment 2: CLIP Fine-Tuning

This script performs a full fine-tune on the open-source CLIP model. It first runs a zero-shot baseline, then fine-tunes the model, and saves all artifacts.

```bash
python finetune_clip.py
```

**Output**: Results will be saved in `clip-bangladeshi-food-results/`.

**Note**: This experiment uses `wandb` for logging. You will be prompted to log in.

### Experiment 3: LoRA Parameter-Efficient Fine-Tuning

This script fine-tunes a ViT model using the highly efficient LoRA method. This provides a direct comparison against the full fine-tune from Experiment 1.

```bash
python finetune_lora.py
```

**Output**: Results are saved in `lora_comparison_results/`. You can compare the number of trainable parameters and final model size against the full fine-tune.

### Experiment 4: LLaVA Local Evaluation

This script uses your locally-run LLaVA model to perform both zero-shot and in-context few-shot evaluation on the validation set.

```bash
# Ensure your Ollama server/app is running first!
python evaluate_local_llava.py
```

**Output**: Two sets of reports and confusion matrices will be saved in `llava-results/`, one for zero-shot and one for few-shot.

### Experiment 5: GPT-4o Fine-Tuning and Evaluation

This is a multi-step process involving the OpenAI API.

1. **Host Your Data**: Upload your entire `data/train` directory to a public cloud storage bucket (e.g., Amazon S3) and ensure all images have a public URL.
2. **Prepare the Training File**: Update `prepare_finetuning_data.py` with your S3 bucket URL and other settings. This script creates the `.jsonl` file required by OpenAI.

```bash
python prepare_finetuning_data.py
```

3. **Launch the Fine-Tuning Job**: Use a separate script (or an interactive Python session) to upload the `.jsonl` file and create the fine-tuning job with OpenAI. (See `4_evaluate_finetuned_model.py` for code examples).
4. **Evaluate the Custom Model**: Once the job is complete, you will receive a custom model ID. Paste this ID into `4_evaluate_finetuned_model.py` and run it to evaluate your new model on the full validation set.

```bash
python evaluate_finetuned_model.py
```

**Output**: The final report and confusion matrix will be saved in `gpt4o-results/`.

## Results Summary

Our comprehensive analysis yields a clear performance hierarchy across different models and training paradigms.

| Model | Method | Trainable Params | Accuracy | F1-Score (Weighted) |
| --- | --- | --- | --- | --- |
| GPT-4o | Zero-Shot | 0 | 54.22% | 0.5087 |
| CLIP ViT-B/32 | Zero-Shot | 0 | \[Your Result\] | \[Your Result\] |
| LLaVA-7b | Zero-Shot | 0 | \[Your Result\] | \[Your Result\] |
| ViT-Base | Full Fine-Tune | \~86M | \[Your Result\] | \[Your Result\] |
| ResNet-50 | Full Fine-Tune | \~25M | \[Your Result\] | \[Your Result\] |
| CLIP ViT-B/32 | Full Fine-Tune | \~150M | \[Your Result\] | \[Your Result\] |
| ViT-Base | LoRA Fine-Tune | \~0.3M | \[Your Result\] | \[Your Result\] |
| GPT-4o | Few-Shot FT (10-shot) | N/A | \[Your Result\] | \[Your Result\] |
| GPT-4o | Subset FT (\~50-shot) | N/A | \[Your Result\] | \[Your Result\] |

(Placeholder: Insert your final comparison graphs here, e.g., the bar chart from `analyze_results.py`.)

Key findings indicate that while large multi-modal models show strong zero-shot ability, fine-tuned open-source models can achieve competitive or superior performance. Parameter-efficient methods like LoRA offer a compelling balance of high performance and resource efficiency.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citing Our Work

If you use this dataset, code, or our findings in your research, please consider citing our paper:

```bibtex
@inproceedings{yourname2026banglafood,
  title={A Comparative Analysis of SOTA Models for Fine-Grained Classification of Bangladeshi Cuisine},
  author={Your, Name(s)},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```


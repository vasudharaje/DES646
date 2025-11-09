# DES646

# AI-Powered Design Inspiration Generator

An AI-based concept generation tool built for product and industrial designers.  
It converts a simple text prompt into multiple design concept images, then uses AI to analyze and cluster them into meaningful design directions.  
The system supports refinement, exploration, and moodboard export—making it a practical ideation tool rather than just an image generator.


## Features

- Multi-image generation using Stable Diffusion  
- CLIP-based semantic clustering  
- Prompt refinement for fast iteration  
- Streamlit UI for interactive usage  
- Moodboard export for presentations or research

## How It Works

User Prompt  
    ↓
Stable Diffusion (Diffusers)  
    ↓ 
generates multiple concept images  
    ↓  
CLIP (SentenceTransformer)  
    ↓  
embeds each image semantically  
    ↓
KMeans Clustering  
    ↓ groups images into design directions ↓  
Streamlit UI  
    ↓ displays clusters + refinement options ↓  
Moodboard Export  

## Project Structure

AI_Design_Inspiration_Generator/
│
├── app/
│ └── streamlit_app.py # Main UI
│
├── models/
│ ├── generate_image.py # Stable Diffusion pipeline
│ └── semantic_cluster.py # CLIP embeddings + clustering
│
├── data/
│ └── prompts.csv # Sample prompt dataset
│
├── outputs/ # Generated images (auto-created)
│
├── test_env.py # Quick environment verification
├── requirements.txt
├── .env # Hugging Face token (NOT committed)
├── .env.example
└── .gitignore


## Setup & Installation

1. Clone the repo (or open your folder in VS Code)
```bash
git clone <your-repo-url>
cd AI_Design_Inspiration_Generator
```

2. Create & activate virtual environment
Mac/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

```bash
## Requirements
- Python 3.9+
- CUDA-enabled GPU recommended (or use CPU with reduced options)
- HuggingFace token (if using HF models)
- Recommended: run on Colab or HuggingFace Spaces for easier GPU access
```

4. Set up your Hugging Face token
Step A: Create .env

```bash
cp .env.example .env      # mac/linux
Copy-Item .env.example .env   # windows
```

Step B: Paste your token into .env
```bash
HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXX
```

## Running the App

Running the App:
```bash
python models/generate_image.py --prompt "modern lamp concept" --num 1
```

Run Streamlit UI:
```bash
streamlit run streamlit_app.py
```

## Usage Guide
Enter a design prompt
e.g. “minimalist electric kettle, soft curves, matte finish”
Adjust generation settings (number of images, guidance scale).
Click Generate.
View variations + clustered design directions.
Refine with text modifiers
e.g. “more organic shape”, “add wood texture”.
Export a moodboard ZIP of any cluster.

## Why It’s Better Than Standard Image Generators
Unlike DALL·E or Midjourney, IdeoForge does not only generate images — it:
-Produces many concept directions
-Clusters & organizes them
-Helps refine ideas iteratively
-Outputs moodboards
-Supports industrial design workflows
-Emphasizes variation, form language, and inspiration
This makes it a design ideation tool, not just a text-to-image generator.

## License
For academic project use only (adapt as required).

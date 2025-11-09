# models/generate_image.py
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # loads .env from project root
import os
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

def get_pipeline(model_name="runwayml/stable-diffusion-v1-5", hf_token=None):
    # Set environment or pass hf_token explicitly
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN", None)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
        revision="fp16" if torch.cuda.is_available() else None,
        safety_checker=None,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    return pipe

def generate_images(pipe, prompt, num_images=4, guidance_scale=7.5, num_inference_steps=30, seed=None, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    images = []
    for i in range(num_images):
        generator = torch.Generator(device="cuda") if torch.cuda.is_available() else None
        if seed is not None:
            generator = torch.Generator(device="cuda") if torch.cuda.is_available() else torch.Generator()
            generator.manual_seed(seed + i)
        result = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, generator=generator)
        img = result.images[0]
        filename = f"{out_dir}/gen_{i+1}.png"
        img.save(filename)
        images.append(filename)
    return images

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--num", type=int, default=4)
    args = parser.parse_args()
    TOKEN = os.environ.get("HF_TOKEN", None)
    p = get_pipeline(hf_token=TOKEN)
    print("Generating...")
    imgs = generate_images(p, args.prompt, num_images=args.num)
    print("Saved:", imgs)

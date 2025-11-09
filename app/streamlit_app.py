# app/streamlit_app.py
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # loads .env from project root
import streamlit as st
import os
from models.generate_image import get_pipeline, generate_images
from models.semantic_cluster import embed_images, cluster_embeddings, organize_by_cluster
from PIL import Image
import io
import zipfile
import base64

st.set_page_config(page_title="Design Inspiration Generator", layout="wide")
st.title("AI-Powered Design Inspiration Generator")

with st.sidebar:
    st.header("Settings")
    num_images = st.slider("Number of images", 1, 8, 4)
    guidance = st.slider("Guidance scale", 3.0, 12.0, 7.5)
    steps = st.slider("Diffusion steps", 10, 50, 28)
    clusters = st.slider("Clusters for grouping", 1, 5, 3)

prompt = st.text_area("Enter your prompt", "A futuristic minimalist electric kettle concept, brushed metal, 3/4 view")
refine = st.text_input("Refinement (e.g. more minimalist, add wood texture)", "")

if st.button("Generate"):
    final_prompt = prompt + (", " + refine if refine.strip() != "" else "")
    st.info("Loading model... (may take ~30s on first run)")
    HF_TOKEN = os.environ.get("HF_TOKEN", None)
    pipe = get_pipeline(hf_token=HF_TOKEN)
    outdir = "outputs"
    images = generate_images(pipe, final_prompt, num_images=num_images, guidance_scale=guidance, num_inference_steps=steps, out_dir=outdir)
    st.success(f"Generated {len(images)} images.")
    cols = st.columns(len(images))
    for c,img_path in zip(cols, images):
        c.image(Image.open(img_path).resize((300,300)))
        c.write(os.path.basename(img_path))

    # clustering
    emb = embed_images(images)
    preds = cluster_embeddings(emb, n_clusters=clusters)
    clustered = organize_by_cluster(images, preds)
    st.write("### Clustered results")
    for k,v in clustered.items():
        st.write(f"**Cluster {k}**")
        row = st.columns(len(v))
        for c,img_path in zip(row, v):
            c.image(Image.open(img_path).resize((220,220)))
            c.write(os.path.basename(img_path))

    # moodboard export: create zip of selected cluster
    st.write("### Export")
    cluster_choice = st.selectbox("Choose cluster to export as moodboard", options=list(clustered.keys()))
    if st.button("Create moodboard zip"):
        selected = clustered[cluster_choice]
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for path in selected:
                zf.write(path, arcname=os.path.basename(path))
        b64 = base64.b64encode(zip_buf.getvalue()).decode()
        href = f'<a href="data:application/zip;base64,{b64}" download="moodboard_cluster_{cluster_choice}.zip">Download moodboard zip</a>'
        st.markdown(href, unsafe_allow_html=True)

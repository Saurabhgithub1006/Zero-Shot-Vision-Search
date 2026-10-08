# Hugging Face Space image: CPU-only Streamlit app.
# PINECONE_API_KEY is provided at runtime as a Space secret, never baked into the image.
FROM python:3.12-slim

# Spaces run containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    DATASET=a2d2
WORKDIR /home/user/app

COPY --chown=user requirements-space.txt .
RUN pip install --no-cache-dir --user -r requirements-space.txt

# Bake the SigLIP weights into the image so cold starts don't re-download ~3.5 GB
RUN python -c "from transformers import AutoModel, AutoProcessor; \
m='google/siglip-so400m-patch14-384'; AutoModel.from_pretrained(m, use_safetensors=True); AutoProcessor.from_pretrained(m)"

COPY --chown=user src ./src
COPY --chown=user app.py .
COPY --chown=user assets/a2d2/manifest.csv ./assets/a2d2/manifest.csv

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
